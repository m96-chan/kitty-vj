//! Transitions — the Mixer shape: two frames in, one out.
//!
//! This is the arity that could not have been retrofitted onto a
//! one-frame trait, which is why it gets implemented early rather than
//! left for last. Everything else in the project reads one frame or
//! none; a transition needs the outgoing picture and the incoming one at
//! the same time.
//!
//! A terminal cannot alpha-blend two glyphs, so a mix is a per-cell
//! decision: show A or show B. The masks are monotonic in `t` and stable
//! per cell (seed-hashed, no per-frame reshuffle), so a transition
//! accumulates cells instead of boiling them.

use crate::rng::{hash3, unit_f64};

/// Two frames in, one out. Implementors decide, per cell, which side
/// wins at progress `t`.
pub trait Transition {
    fn name(&self) -> &'static str;

    /// true = this cell shows the incoming frame.
    fn shows_b(&self, x: u16, y: u16, w: u16, h: u16, t: f64) -> bool;

    /// The same decision, told the luminance of the incoming frame at
    /// this cell, `luma_b` in [0,1].
    ///
    /// Some of the originals key their mask off the picture rather than
    /// off position — `LUMA` dissolves out of the incoming highlights —
    /// and position alone cannot say where the highlights are. The
    /// default throws the key away, so every position-only transition
    /// needs nothing; `shows_b` stays the whole contract for them.
    ///
    /// The mixer still calls plain `shows_b`, hence the dead code until
    /// the key is wired through.
    #[allow(dead_code)]
    fn shows_b_keyed(&self, x: u16, y: u16, w: u16, h: u16, t: f64, _luma_b: f64) -> bool {
        self.shows_b(x, y, w, h, t)
    }

    /// Whether the key changes the answer. Reading a cell's luminance
    /// means sampling the incoming frame for every cell, including the
    /// ones that end up showing the outgoing one — work the mask
    /// otherwise skips — so the caller gets to ask before paying it.
    #[allow(dead_code)]
    fn keyed(&self) -> bool {
        false
    }

    /// Additive white bloom over the cut, [0,1]. The original `FLASH`
    /// blew the frame to white at the cut and let it fall back; a
    /// per-cell A-or-B mask cannot brighten anything, so the transition
    /// only reports the amount and the caller lifts the colours.
    #[allow(dead_code)]
    fn whiteout(&self, _t: f64) -> f64 {
        0.0
    }

    /// How long this transition wants to run, in beats. Over there the
    /// long ones read as a change of chapter and the short ones as an
    /// edit; that distinction lives with the transition, not the caller.
    fn beats(&self) -> f64 {
        1.0
    }
}

/// GLSL `smoothstep`, because the ported masks are written in its terms.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per-cell hash threshold — the terminal-native crossfade. Where the
/// original could ramp alpha, a grid dissolves.
pub struct Fade;

/// Fourteen slats sweeping in alternating directions.
pub struct Wipe;

/// Blocks displaced with a colour fringe, over in a beat.
pub struct GlitchCut;

/// The incoming frame arrives out of its own highlights: bright cells
/// cross first, shadows last.
pub struct Luma;

/// Hard cut on the half beat, with a bloom for the caller to add.
pub struct Flash;

/// A centre-out reveal standing in for two plates scaling through each
/// other.
pub struct Zoom;

/// The new frame slides the old one out, one clean edge, no fringe.
pub struct Push;

const SLATS: f64 = 14.0;

impl Transition for Fade {
    fn name(&self) -> &'static str {
        "FADE"
    }

    fn shows_b(&self, x: u16, y: u16, _w: u16, _h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        unit_f64(hash3(x as u64, y as u64, 3)) < t
    }

    fn beats(&self) -> f64 {
        2.0 // a chapter change
    }
}

impl Transition for Wipe {
    fn name(&self) -> &'static str {
        "WIPE"
    }

    fn shows_b(&self, x: u16, _y: u16, w: u16, _h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        let u = (x as f64 + 0.5) / w.max(1) as f64;
        let slat = (u * SLATS).floor();
        // Alternate slats sweep the other way, and each starts a little
        // late so the edge is ragged rather than a ruler.
        let local = (t * 1.35 - unit_f64(hash3(slat as u64, 11, 0)) * 0.3).clamp(0.0, 1.0);
        let within = (u * SLATS).fract();
        let p = if (slat as u64).is_multiple_of(2) {
            within
        } else {
            1.0 - within
        };
        p < local
    }
}

impl Transition for GlitchCut {
    fn name(&self) -> &'static str {
        "GLITCH"
    }

    fn shows_b(&self, _x: u16, y: u16, _w: u16, h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        // 30 bands, each flipping at its own point in the transition.
        let band = (y as f64 / h.max(1) as f64 * 30.0).floor() as u64;
        t > unit_f64(hash3(band, 13, 0)) * 0.55 + 0.2
    }
}

impl Luma {
    /// The original blends with `smoothstep(0, 0.4, t*1.4 - (1-l))`, so
    /// a cell whose incoming luminance is `l` starts crossing once the
    /// ramp has climbed past its own darkness. A grid has no blend, so
    /// the cell flips where that factor passes the halfway mark — which
    /// keeps the ordering the original had, bright before dark, and
    /// nothing else about it.
    fn reveals(t: f64, luma_b: f64) -> bool {
        let l = luma_b.clamp(0.0, 1.0);
        smoothstep(0.0, 0.4, t * 1.4 - (1.0 - l)) >= 0.5
    }
}

impl Transition for Luma {
    fn name(&self) -> &'static str {
        "LUMA"
    }

    /// Unkeyed, there is no picture to key off, so every cell is taken
    /// for mid grey and the whole frame turns over at once. A caller
    /// that wants the effect calls `shows_b_keyed`.
    fn shows_b(&self, _x: u16, _y: u16, _w: u16, _h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        Self::reveals(t, 0.5)
    }

    fn shows_b_keyed(&self, _x: u16, _y: u16, _w: u16, _h: u16, t: f64, luma_b: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        Self::reveals(t, luma_b)
    }

    fn keyed(&self) -> bool {
        true
    }

    fn beats(&self) -> f64 {
        2.0 // a chapter change
    }
}

impl Transition for Flash {
    fn name(&self) -> &'static str {
        "FLASH"
    }

    /// No mask at all — the cut lands whole on the half beat. What sold
    /// it over there was the bloom either side, not the geometry.
    fn shows_b(&self, _x: u16, _y: u16, _w: u16, _h: u16, t: f64) -> bool {
        t >= 0.5
    }

    /// `pow(1 - |2t - 1|, 2) * 0.9`: nothing at either end, brightest
    /// on the cut itself, so the seam hides inside the blowout.
    fn whiteout(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        (1.0 - (2.0 * t - 1.0).abs()).powi(2) * 0.9
    }
}

impl Transition for Zoom {
    fn name(&self) -> &'static str {
        "ZOOM"
    }

    /// The original scaled both plates through each other, the outgoing
    /// one growing past the frame while the incoming one settled into
    /// it. A mask cannot resample anything — it only picks a side per
    /// cell — so what survives the port is the shape the scale drew: a
    /// disc opening from the centre on the same `smoothstep(0.12, 0.92)`
    /// timing. A caller that wants the plates to actually move has to
    /// scale them itself; that part is not expressible here.
    fn shows_b(&self, x: u16, y: u16, w: u16, h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        let dx = (x as f64 + 0.5) / w.max(1) as f64 - 0.5;
        let dy = (y as f64 + 0.5) / h.max(1) as f64 - 0.5;
        // Normalised so a corner sits at 1 and the disc has to grow the
        // whole way to finish, whatever the terminal's aspect.
        let r = (dx * dx + dy * dy).sqrt() / 0.5f64.hypot(0.5);
        r <= smoothstep(0.12, 0.92, t)
    }

    fn beats(&self) -> f64 {
        2.0 // a chapter change
    }
}

impl Transition for Push {
    fn name(&self) -> &'static str {
        "PUSH"
    }

    /// The incoming frame shoves the outgoing one off the side. One
    /// edge, eased but never ragged — that straightness is the whole
    /// difference between this and `WIPE`, which deliberately breaks its
    /// edge up into slats.
    fn shows_b(&self, x: u16, _y: u16, w: u16, _h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        let u = (x as f64 + 0.5) / w.max(1) as f64;
        u > 1.0 - smoothstep(0.0, 1.0, t)
    }
}

/// The set, in the order the operator cycles them.
pub const TRANSITIONS: [&dyn Transition; 7] =
    [&Fade, &Wipe, &GlitchCut, &Luma, &Flash, &Zoom, &Push];

#[cfg(test)]
mod tests {
    use super::*;

    /// Keys a cell would plausibly carry: black, mid grey, white.
    const KEYS: [f64; 3] = [0.0, 0.5, 1.0];

    #[test]
    fn endpoints_are_pure() {
        for tr in TRANSITIONS {
            for x in 0..40u16 {
                for y in 0..20u16 {
                    assert!(!tr.shows_b(x, y, 40, 20, 0.0), "{} at t=0", tr.name());
                    assert!(tr.shows_b(x, y, 40, 20, 1.0), "{} at t=1", tr.name());
                    for k in KEYS {
                        assert!(
                            !tr.shows_b_keyed(x, y, 40, 20, 0.0, k),
                            "{} at t=0, key {k}",
                            tr.name()
                        );
                        assert!(
                            tr.shows_b_keyed(x, y, 40, 20, 1.0, k),
                            "{} at t=1, key {k}",
                            tr.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_mask_is_monotonic() {
        // A cell that has switched to B must never switch back — that is
        // the difference between a transition and a boil. A keyed mask
        // holds to the same rule: the cell's key does not change over a
        // transition, so a fixed key must still only ever gain cells.
        for tr in TRANSITIONS {
            for x in 0..40u16 {
                for y in 0..20u16 {
                    let mut shown = false;
                    for i in 0..=20 {
                        let now = tr.shows_b(x, y, 40, 20, i as f64 / 20.0);
                        assert!(!shown || now, "{} flickered back at {i}", tr.name());
                        shown = now;
                    }
                    for k in KEYS {
                        let mut shown = false;
                        for i in 0..=20 {
                            let now = tr.shows_b_keyed(x, y, 40, 20, i as f64 / 20.0, k);
                            assert!(
                                !shown || now,
                                "{} flickered back at {i}, key {k}",
                                tr.name()
                            );
                            shown = now;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mid_transition_shows_both_sides() {
        for tr in TRANSITIONS {
            // FLASH is a cut by definition; there is no halfway state to
            // look at, which is the point of it.
            if tr.name() == "FLASH" {
                continue;
            }
            let mut a = 0;
            let mut b = 0;
            for x in 0..40u16 {
                for y in 0..20u16 {
                    // A luminance ramp across the frame, so the keyed
                    // masks get a picture with both highlights and
                    // shadows in it; the rest ignore it.
                    let key = x as f64 / 39.0;
                    if tr.shows_b_keyed(x, y, 40, 20, 0.5, key) {
                        b += 1;
                    } else {
                        a += 1;
                    }
                }
            }
            assert!(a > 0 && b > 0, "{} is a hard cut at t=0.5", tr.name());
        }
    }

    #[test]
    fn wipe_slats_run_in_opposite_directions() {
        // Early on, slat 0 has covered its left edge and slat 1 its right.
        let t = 0.35;
        let w = 140u16; // ten cells per slat
        let left_of_slat0 = Wipe.shows_b(1, 0, w, 20, t);
        let right_of_slat1 = Wipe.shows_b(19, 0, w, 20, t);
        assert!(
            left_of_slat0 || right_of_slat1,
            "neither slat swept from its own side"
        );
    }

    #[test]
    fn tiny_areas_do_not_panic() {
        for tr in TRANSITIONS {
            for (w, h) in [(0u16, 0u16), (1, 1), (1, 9), (9, 1)] {
                let _ = tr.shows_b(0, 0, w, h, 0.5);
                let _ = tr.shows_b_keyed(0, 0, w, h, 0.5, 1.0);
            }
        }
    }

    #[test]
    fn only_luma_reads_the_key() {
        for tr in TRANSITIONS {
            if tr.name() == "LUMA" {
                continue;
            }
            for t in [0.25, 0.5, 0.75] {
                for x in 0..40u16 {
                    for y in 0..20u16 {
                        assert_eq!(
                            tr.shows_b(x, y, 40, 20, t),
                            tr.shows_b_keyed(x, y, 40, 20, t, 1.0),
                            "{} changed its answer for a key it does not use",
                            tr.name()
                        );
                    }
                }
            }
            assert!(!tr.keyed(), "{} advertises a key it ignores", tr.name());
        }
        assert!(Luma.keyed());
    }

    #[test]
    fn luma_reveals_bright_cells_before_dark_ones() {
        // Same cell, same instant: the only thing separating them is the
        // brightness of the incoming frame.
        let t = 0.5;
        assert!(Luma.shows_b_keyed(0, 0, 40, 20, t, 1.0), "highlight held");
        assert!(!Luma.shows_b_keyed(0, 0, 40, 20, t, 0.0), "shadow jumped");
        // And the ordering holds all the way down the ramp: a brighter
        // cell is never the later one.
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let mut lit = false;
            for j in 0..=20 {
                let now = Luma.shows_b_keyed(0, 0, 40, 20, t, j as f64 / 20.0);
                assert!(!lit || now, "a brighter cell lagged a darker one at t={t}");
                lit = now;
            }
        }
    }

    #[test]
    fn flash_is_a_hard_cut_on_the_half() {
        for x in 0..40u16 {
            for y in 0..20u16 {
                assert!(!Flash.shows_b(x, y, 40, 20, 0.499));
                assert!(Flash.shows_b(x, y, 40, 20, 0.5));
            }
        }
        // The bloom is the part that sells it, and it lives on the cut.
        assert_eq!(Flash.whiteout(0.0), 0.0);
        assert_eq!(Flash.whiteout(1.0), 0.0);
        assert!((Flash.whiteout(0.5) - 0.9).abs() < 1e-9);
        assert!(Flash.whiteout(0.25) < Flash.whiteout(0.4));
        // Everything else leaves the colours alone.
        for tr in TRANSITIONS {
            if tr.name() != "FLASH" {
                assert_eq!(tr.whiteout(0.5), 0.0, "{} blooms", tr.name());
            }
        }
    }

    #[test]
    fn zoom_opens_from_the_centre() {
        let (w, h) = (40u16, 20u16);
        let t = 0.5;
        assert!(Zoom.shows_b(w / 2, h / 2, w, h, t), "centre lagged");
        assert!(!Zoom.shows_b(0, 0, w, h, t), "corner led");
        // Every cell that has turned over is nearer the centre than
        // every cell that has not.
        let d = |x: u16, y: u16| {
            let dx = (x as f64 + 0.5) / w as f64 - 0.5;
            let dy = (y as f64 + 0.5) / h as f64 - 0.5;
            (dx * dx + dy * dy).sqrt()
        };
        let (mut inside, mut outside) = (0.0f64, f64::MAX);
        for x in 0..w {
            for y in 0..h {
                if Zoom.shows_b(x, y, w, h, t) {
                    inside = inside.max(d(x, y));
                } else {
                    outside = outside.min(d(x, y));
                }
            }
        }
        assert!(inside <= outside, "the disc is not a disc");
    }

    #[test]
    fn push_edge_is_hard_and_sweeps_one_way() {
        let (w, h) = (40u16, 20u16);
        for i in 1..20 {
            let t = i as f64 / 20.0;
            // No ragged edge: every row is cut at the same column, and
            // the cut is a single boundary, not a scatter.
            let row: Vec<bool> = (0..w).map(|x| Push.shows_b(x, 0, w, h, t)).collect();
            for y in 1..h {
                for x in 0..w {
                    assert_eq!(
                        row[x as usize],
                        Push.shows_b(x, y, w, h, t),
                        "row {y} differs"
                    );
                }
            }
            assert_eq!(
                row.windows(2).filter(|p| p[0] != p[1]).count(),
                usize::from(row[0] != row[w as usize - 1]),
                "more than one edge at t={t}"
            );
            // And the incoming frame arrives from the right.
            assert!(row[w as usize - 1] >= row[0], "swept the wrong way");
        }
        // The edge only ever travels one direction.
        let covered = |t: f64| (0..w).filter(|&x| Push.shows_b(x, 0, w, h, t)).count();
        for i in 0..20 {
            assert!(covered(i as f64 / 20.0) <= covered((i + 1) as f64 / 20.0));
        }
    }

    #[test]
    fn long_transitions_take_a_chapter_and_short_ones_an_edit() {
        for tr in TRANSITIONS {
            let want = match tr.name() {
                "FADE" | "LUMA" | "ZOOM" => 2.0,
                _ => 1.0,
            };
            assert_eq!(tr.beats(), want, "{} runs the wrong length", tr.name());
        }
    }
}
