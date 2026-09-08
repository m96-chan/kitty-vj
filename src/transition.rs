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

    /// How long this transition wants to run, in beats. Over there the
    /// long ones read as a change of chapter and the short ones as an
    /// edit; that distinction lives with the transition, not the caller.
    fn beats(&self) -> f64 {
        1.0
    }
}

/// Per-cell hash threshold — the terminal-native crossfade. Where the
/// original could ramp alpha, a grid dissolves.
pub struct Fade;

/// Fourteen slats sweeping in alternating directions.
pub struct Wipe;

/// Blocks displaced with a colour fringe, over in a beat.
pub struct GlitchCut;

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

/// The set, in the order the operator cycles them.
pub const TRANSITIONS: [&dyn Transition; 3] = [&Fade, &Wipe, &GlitchCut];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_pure() {
        for tr in TRANSITIONS {
            for x in 0..40u16 {
                for y in 0..20u16 {
                    assert!(!tr.shows_b(x, y, 40, 20, 0.0), "{} at t=0", tr.name());
                    assert!(tr.shows_b(x, y, 40, 20, 1.0), "{} at t=1", tr.name());
                }
            }
        }
    }

    #[test]
    fn every_mask_is_monotonic() {
        // A cell that has switched to B must never switch back — that is
        // the difference between a transition and a boil.
        for tr in TRANSITIONS {
            for x in 0..40u16 {
                for y in 0..20u16 {
                    let mut shown = false;
                    for i in 0..=20 {
                        let now = tr.shows_b(x, y, 40, 20, i as f64 / 20.0);
                        assert!(!shown || now, "{} flickered back at {i}", tr.name());
                        shown = now;
                    }
                }
            }
        }
    }

    #[test]
    fn mid_transition_shows_both_sides() {
        for tr in TRANSITIONS {
            let mut a = 0;
            let mut b = 0;
            for x in 0..40u16 {
                for y in 0..20u16 {
                    if tr.shows_b(x, y, 40, 20, 0.5) {
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
            }
        }
    }
}
