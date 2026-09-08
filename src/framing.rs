//! Framing — which fit a plate gets, and where inside it we are looking.
//!
//! A Modulator in the `pass.rs` sense: it reads the clock and writes
//! numbers, never pixels. Over there the fit was decided before any
//! effect ran — a plate was *covered*, *closed up on*, or *contained* —
//! and everything downstream sampled whatever that decision produced.
//! Getting it wrong is not a subtle loss: a portrait plate cover-fitted
//! into a wide frame throws away the head, which is usually the only
//! thing in the picture, and a landscape plate contained into a wide
//! frame is a small picture with bars round it for no reason.
//!
//! Unlike `KenBurns`, this modulator is **stateful**. The original's
//! focus point does not jump to a new aim, it eases toward it across
//! frames, and easing has to remember where it was. So `update` advances
//! the drift and `cam` reads the current value back, where `KenBurns` is
//! a pure function of `t`. It stays deterministic all the same: every
//! aim comes from a seed-hashed roll counted by re-aims, never from wall
//! time, so the same seed and the same sequence of updates reproduce the
//! same framing exactly.
//!
//! Two things deliberately do **not** live here. Which fit a given plate
//! gets is a scene decision — `Fit::roll` offers the original's odds and
//! the scene calls it. And the rotation angle is currently always zero:
//! this project has deferred non-axis-aligned rotation because a tilted
//! image on a cell grid reads as noise rather than as a tilt. The
//! rotated-bounding-box maths are ported and tested anyway, so that a
//! future pixel-tier plate can set an angle and have the fit stay whole.

// Nothing wires this module into a render path yet — the scene picks the
// fit, and that lands separately. Delete this the moment it does, so the
// usual dead-code pressure applies again.
#![allow(dead_code)]

use crate::camera::{Cam, Modulator};
use crate::drive::Drive;
use crate::rng::{hash3, unit_f64};

/// Contain leaves a hair of air around the plate rather than butting it
/// against the frame edge — the original's 6%.
const CONTAIN_MARGIN: f64 = 0.94;
/// Contain damping. Nothing important may leave the frame in this mode,
/// because in contain the *whole* picture is the subject; so the camera's
/// pan is cut to a third and its zoom to a bit over half. Cover can
/// afford a wild camera since it is already throwing away edges.
const CONTAIN_PAN_DAMP: f64 = 0.3;
const CONTAIN_ZOOM_DAMP: f64 = 0.55;

/// Focus easing time constant, seconds. Slow enough that a re-aim reads
/// as the camera looking somewhere else, not as a cut.
const FOCUS_TAU: f64 = 1.1;
/// Horizontal aim never strays far from centre — off-centre framing
/// looks like a mistake more often than it looks like a choice.
const FOCUS_X: (f64, f64) = (0.34, 0.66);
/// Portraits aim high. A portrait plate is nearly always a person, and a
/// face sits in the top third; aiming at the geometric centre of a
/// portrait frames a torso.
const FOCUS_Y_PORTRAIT: (f64, f64) = (0.13, 0.34);
/// Landscape aims just above centre — high enough for horizons and
/// heads, low enough not to crop into sky.
const FOCUS_Y_LANDSCAPE: (f64, f64) = (0.28, 0.58);

/// Close-up zoom, first roll and every re-roll after. The original used
/// a slightly wider band once it was already close.
const CLOSE_FIRST: (f64, f64) = (1.7, 2.8);
const CLOSE_REROLL: (f64, f64) = (1.6, 2.9);

/// Phrase-edge detection thresholds on `Drive::phrase`. The pulse is 1.0
/// on the edge and decays with a 60 ms tau, so by the next frame at 30 fps
/// it is still ~0.58: a 0.35 trip point catches the edge at any sane frame
/// rate, and the hysteresis down to 0.12 stops the same phrase firing
/// twice while the pulse is still ringing. `phrase` is read raw rather
/// than through `gphrase()` because this is a question about where we are
/// in the bar, not about how hard the music is playing.
const PHRASE_HI: f64 = 0.35;
const PHRASE_LO: f64 = 0.12;

/// How a plate is fitted into the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Fill the frame, crop whatever hangs over. The default, and what a
    /// 16:9 plate wants almost always.
    Cover,
    /// Cover, then dive in further and re-aim every phrase. This is the
    /// one that makes a still image feel edited.
    Closeup,
    /// Fit the whole plate inside the frame over a blurred backdrop.
    /// Reserved for plates whose shape fights the frame — normally
    /// portraits — because it is the only fit that shows all of one.
    Contain,
}

impl Fit {
    /// The original's odds, and the only place they belong. A 16:9 plate
    /// already matches the frame, so it covers ~5 times in 6 and closes
    /// up otherwise; it never contains, because there would be nothing
    /// to contain it away from. A portrait contains only ~1 time in 5 —
    /// the bars are a cost, paid only when the crop would hurt more —
    /// and otherwise covers or closes up like anything else.
    pub fn roll(seed: u64, is_portrait: bool) -> Fit {
        let r = unit_f64(hash3(seed, 0xF17, 0));
        if is_portrait {
            if r < 0.20 {
                Fit::Contain
            } else if r < 0.70 {
                Fit::Cover
            } else {
                Fit::Closeup
            }
        } else if r < 5.0 / 6.0 {
            Fit::Cover
        } else {
            Fit::Closeup
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Fit::Cover => "COVER",
            Fit::Closeup => "CLOSEUP",
            Fit::Contain => "CONTAIN",
        }
    }

    /// True when the plate will not fill the frame, so the caller owes
    /// the frame a backdrop. `Cam` alone cannot say this, which is why
    /// `Placement` exists.
    pub fn letterboxed(&self) -> bool {
        matches!(self, Fit::Contain)
    }
}

/// The bounding box of a `w`×`h` rectangle turned by `angle` radians.
///
/// Both fits are the same question asked from opposite ends: cover turns
/// the *frame* into image space and demands the image cover that box;
/// contain turns the *image* into frame space and demands the frame hold
/// that box. One function serves both. At `angle == 0` — which is every
/// caller today — it returns the rectangle unchanged, so the axis-aligned
/// path costs nothing but a `sin`/`cos`.
pub fn rotated_bbox(w: f64, h: f64, angle: f64) -> (f64, f64) {
    let (s, c) = angle.sin_cos();
    let (s, c) = (s.abs(), c.abs());
    (w * c + h * s, w * s + h * c)
}

/// Where a plate lands, in the terms the sampler wants.
///
/// The contract matches the halfblock plate pass: for an output pixel
/// `(px, py)` measured from the top left of a `dst` grid,
///
/// ```text
/// sx = (px - dst.0 / 2) / scale + src.0 / 2 + drift_x
/// sy = (py - dst.1 / 2) / scale + src.1 / 2 + drift_y
/// ```
///
/// For `Cover` and `Closeup` that is guaranteed to land inside the plate
/// for every pixel in the frame. For `Contain` it is guaranteed *not* to
/// for some of them — that is the point of the mode — so a contain
/// consumer must test the sample against the plate bounds and fall
/// through to the backdrop instead of clamping to the edge, which would
/// smear the border pixels outward into a streak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// Output pixels per source pixel.
    pub scale: f64,
    /// Offset of the sampled centre from the plate centre, in source
    /// pixels. Already clamped: for cover no corner can open up, for
    /// contain the plate cannot slide out of its own margin.
    pub drift_x: f64,
    pub drift_y: f64,
    pub fit: Fit,
    /// The plate does not fill the frame; draw a backdrop behind it.
    pub letterboxed: bool,
}

/// The framing modulator: an eased focus point, a fit, and the geometry
/// to land one in the other.
pub struct Framing {
    seed: u64,
    fit: Fit,
    /// Plate rotation against the frame, radians. Always 0 today — see
    /// the module docs — but carried through the fit maths so that
    /// switching it on later is a one-line change at the call site.
    angle: f64,
    /// Current focus, normalised image space.
    fx: f64,
    fy: f64,
    /// Where the focus is heading.
    tx: f64,
    ty: f64,
    /// Close-up zoom on top of cover. Unused by the other fits.
    close: f64,
    /// Re-aims so far; seeds each fresh roll.
    aim: u64,
    /// Phrase edges seen so far; seeds the coin flip that decides
    /// whether a phrase re-aims at all.
    phrases: u64,
    /// Phrase-pulse edge detector state.
    armed: bool,
}

impl Framing {
    /// Start a framing. `seed` distinguishes plates or scenes; the first
    /// aim is rolled immediately so frame one is already composed rather
    /// than starting dead centre and sliding.
    pub fn new(seed: u64, fit: Fit, is_portrait: bool) -> Self {
        let mut f = Self {
            seed,
            fit,
            angle: 0.0,
            fx: 0.5,
            fy: 0.5,
            tx: 0.5,
            ty: 0.5,
            close: 1.0,
            aim: 0,
            phrases: 0,
            armed: false,
        };
        f.reaim(is_portrait);
        // The opening close-up is drawn from the tighter first-roll band.
        f.close = lerp_roll(CLOSE_FIRST, unit_f64(hash3(seed, 0, 4)));
        f.fx = f.tx;
        f.fy = f.ty;
        f
    }

    /// Give the plate a tilt. Nothing does today; see the module docs.
    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    pub fn fit(&self) -> Fit {
        self.fit
    }

    /// Where the focus is easing toward, normalised image space.
    pub fn target(&self) -> (f64, f64) {
        (self.tx, self.ty)
    }

    /// The current close-up zoom over cover. 1.0 unless the fit is
    /// `Closeup`.
    pub fn close_zoom(&self) -> f64 {
        match self.fit {
            Fit::Closeup => self.close,
            _ => 1.0,
        }
    }

    /// Advance the drift and handle re-aiming. `dt` is real seconds;
    /// `is_portrait` is a property of the plate on screen right now, so it
    /// is passed per frame rather than stored — a plate swap changes it.
    ///
    /// A phrase is 16 beats, so the original's "re-aim every 16 beats"
    /// and its "re-aim on a phrase" are the same edge. A close-up takes
    /// every one of them, because a close-up that sits still is just a
    /// crop; cover and contain take half, so the aim mostly holds and the
    /// change is an event when it comes.
    pub fn update(&mut self, dt: f64, d: &Drive, is_portrait: bool) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };

        if d.phrase >= PHRASE_HI && !self.armed {
            self.armed = true;
            let take = match self.fit {
                Fit::Closeup => true,
                _ => unit_f64(hash3(self.seed, self.phrases, 9)) < 0.5,
            };
            self.phrases += 1;
            if take {
                self.reaim(is_portrait);
            }
        } else if d.phrase < PHRASE_LO {
            self.armed = false;
        }

        // Exponential approach: a fixed fraction of the remaining
        // distance per second, so it is frame-rate independent and never
        // overshoots.
        let k = 1.0 - (-dt / FOCUS_TAU).exp();
        self.fx += (self.tx - self.fx) * k;
        self.fy += (self.ty - self.fy) * k;
    }

    /// Roll a fresh aim, and for a close-up a fresh depth with it. The
    /// depth snaps where the focus eases — on a phrase that snap is the
    /// edit, and easing it would turn a cut into a slow push.
    fn reaim(&mut self, is_portrait: bool) {
        let r = |k: u64| unit_f64(hash3(self.seed, self.aim, k));
        self.tx = lerp_roll(FOCUS_X, r(1));
        self.ty = lerp_roll(
            if is_portrait {
                FOCUS_Y_PORTRAIT
            } else {
                FOCUS_Y_LANDSCAPE
            },
            r(2),
        );
        if self.fit == Fit::Closeup {
            self.close = lerp_roll(CLOSE_REROLL, r(3));
        }
        self.aim = self.aim.wrapping_add(1);
    }

    /// This framing's own contribution, without geometry.
    fn own_cam(&self) -> Cam {
        Cam {
            zoom: self.close_zoom(),
            fx: self.fx,
            fy: self.fy,
        }
    }

    /// Solve the fit. `src` is the plate in pixels; `dst` is the output
    /// sample grid, which for the halfblock tier is cells across by twice
    /// the cells down. `cam` is whatever the other modulators composed to
    /// — pass `Cam::default()` for none — and is folded in on top of this
    /// framing's own aim: its zoom multiplies, its focus displaces.
    pub fn place(&self, src: (f64, f64), dst: (f64, f64), cam: Cam) -> Placement {
        // Degenerate sizes are a real case (a 1×1 plate, a two-cell
        // terminal). Floor everything so no division can produce an
        // infinity and no clamp range can come out inverted.
        let (iw, ih) = (guard(src.0), guard(src.1));
        let (w, h) = (guard(dst.0), guard(dst.1));

        let own = self.own_cam();
        let mut zoom = if cam.zoom.is_finite() { cam.zoom } else { 1.0 } * own.zoom;
        let mut fx = own.fx + (cam.fx - 0.5);
        let mut fy = own.fy + (cam.fy - 0.5);
        if self.fit == Fit::Contain {
            zoom = 1.0 + (zoom - 1.0) * CONTAIN_ZOOM_DAMP;
            fx = 0.5 + (fx - 0.5) * CONTAIN_PAN_DAMP;
            fy = 0.5 + (fy - 0.5) * CONTAIN_PAN_DAMP;
        }

        let (scale, allow_x, allow_y) = match self.fit {
            Fit::Cover | Fit::Closeup => {
                // The frame, turned into image space, is the box the
                // plate has to cover.
                let (bw, bh) = rotated_bbox(w, h, self.angle);
                // Cover means cover: a zoom below 1 would open a corner,
                // so it is floored rather than honoured.
                let scale = (bw / iw).max(bh / ih) * zoom.max(1.0);
                // Half of the source region the frame needs, in source
                // pixels. Whatever is left over is how far the aim may
                // travel before an edge shows.
                (
                    scale,
                    (iw / 2.0 - bw / (2.0 * scale)).max(0.0),
                    (ih / 2.0 - bh / (2.0 * scale)).max(0.0),
                )
            }
            Fit::Contain => {
                // The plate, turned into frame space, is the box the
                // frame has to hold.
                let (bw, bh) = rotated_bbox(iw, ih, self.angle);
                let fit_s = (w / bw).min(h / bh);
                // Capped at the bare fit: a contain that crops is just a
                // bad cover. The damping above makes this rare; the cap
                // makes "the whole plate is visible" a guarantee rather
                // than a tendency.
                let scale = (fit_s * CONTAIN_MARGIN * zoom.max(0.0)).clamp(fit_s * 1e-3, fit_s);
                // Leftover letterbox margin, converted from frame pixels
                // to source pixels so both fits clamp in the same units.
                (
                    scale,
                    (w - bw * scale).max(0.0) / (2.0 * scale),
                    (h - bh * scale).max(0.0) / (2.0 * scale),
                )
            }
        };

        let cx = (fx * iw).clamp(iw / 2.0 - allow_x, iw / 2.0 + allow_x);
        let cy = (fy * ih).clamp(ih / 2.0 - allow_y, ih / 2.0 + allow_y);

        Placement {
            scale,
            drift_x: cx - iw / 2.0,
            drift_y: cy - ih / 2.0,
            fit: self.fit,
            letterboxed: self.fit.letterboxed(),
        }
    }
}

impl Modulator for Framing {
    fn name(&self) -> &'static str {
        self.fit.name()
    }

    /// The clock arguments are ignored on purpose: this modulator's state
    /// is advanced by `update`, because easing cannot be recomputed from
    /// `t` alone. Note also that a `Contain` fit reports zoom 1.0 here —
    /// its real scale is geometric and only `place` can produce it.
    fn cam(&self, _t: f64, _d: &Drive) -> Cam {
        self.own_cam()
    }
}

/// A roll in `[0,1)` mapped onto a `(lo, hi)` band.
fn lerp_roll((lo, hi): (f64, f64), r: f64) -> f64 {
    lo + (hi - lo) * r
}

/// Keep a dimension positive and finite. A zero-width plate or a
/// zero-width frame is a caller bug, but it must not be a panic or a NaN
/// halfway down the render.
fn guard(v: f64) -> f64 {
    if v.is_finite() { v.max(1.0) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LANDSCAPE: (f64, f64) = (1920.0, 1080.0);
    const PORTRAIT: (f64, f64) = (900.0, 1600.0);
    /// A halfblock frame: 160 cells across, 45 down = 90 sample rows.
    const FRAME: (f64, f64) = (160.0, 90.0);

    /// Where an output pixel samples the plate, per the `Placement`
    /// contract.
    fn sample_at(p: &Placement, src: (f64, f64), dst: (f64, f64), px: f64, py: f64) -> (f64, f64) {
        (
            (px - dst.0 / 2.0) / p.scale + src.0 / 2.0 + p.drift_x,
            (py - dst.1 / 2.0) / p.scale + src.1 / 2.0 + p.drift_y,
        )
    }

    /// Where a plate pixel lands in the frame — the inverse.
    fn lands_at(p: &Placement, src: (f64, f64), dst: (f64, f64), sx: f64, sy: f64) -> (f64, f64) {
        (
            (sx - src.0 / 2.0 - p.drift_x) * p.scale + dst.0 / 2.0,
            (sy - src.1 / 2.0 - p.drift_y) * p.scale + dst.1 / 2.0,
        )
    }

    fn corners(w: f64, h: f64) -> [(f64, f64); 4] {
        [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
    }

    /// Run a framing over a stretch of beats at ~150 BPM, driving the
    /// real `Drive` so phrase edges arrive the way they do in the show.
    fn advance(f: &mut Framing, d: &mut Drive, beat: &mut f64, beats: f64, portrait: bool) {
        let steps = (beats / 0.05).round() as i32;
        for _ in 0..steps {
            *beat += 0.05;
            d.update(0.02, *beat, None);
            f.update(0.02, d, portrait);
        }
    }

    #[test]
    fn cover_fills_the_frame_from_either_aspect() {
        for src in [LANDSCAPE, PORTRAIT] {
            let mut f = Framing::new(4, Fit::Cover, src.1 > src.0);
            let mut d = Drive::default();
            let mut beat = 0.0;
            // Let the aim drift as far off centre as it will go.
            advance(&mut f, &mut d, &mut beat, 40.0, src.1 > src.0);
            let p = f.place(src, FRAME, Cam::default());
            for (px, py) in corners(FRAME.0, FRAME.1) {
                let (sx, sy) = sample_at(&p, src, FRAME, px, py);
                assert!(
                    sx >= -1e-6 && sx <= src.0 + 1e-6 && sy >= -1e-6 && sy <= src.1 + 1e-6,
                    "corner opened up for {src:?}: sampled ({sx}, {sy})"
                );
            }
        }
    }

    #[test]
    fn cover_never_opens_a_corner_however_the_camera_moves() {
        let f = Framing::new(9, Fit::Cover, false);
        // Including a nonsense sub-1 zoom, which cover must floor.
        for zoom in [0.5, 1.0, 1.4, 3.0] {
            for fx in [0.0, 0.5, 1.0] {
                let cam = Cam { zoom, fx, fy: fx };
                let p = f.place(PORTRAIT, FRAME, cam);
                for (px, py) in corners(FRAME.0, FRAME.1) {
                    let (sx, sy) = sample_at(&p, PORTRAIT, FRAME, px, py);
                    assert!(
                        sx >= -1e-6 && sx <= PORTRAIT.0 + 1e-6,
                        "x opened at zoom {zoom} fx {fx}: {sx}"
                    );
                    assert!(
                        sy >= -1e-6 && sy <= PORTRAIT.1 + 1e-6,
                        "y opened at zoom {zoom} fx {fx}: {sy}"
                    );
                }
            }
        }
    }

    #[test]
    fn closeup_is_tighter_than_cover() {
        let cover = Framing::new(12, Fit::Cover, false).place(LANDSCAPE, FRAME, Cam::default());
        let close = Framing::new(12, Fit::Closeup, false).place(LANDSCAPE, FRAME, Cam::default());
        assert!(
            close.scale > cover.scale * 1.6,
            "close-up should dive in: {} vs {}",
            close.scale,
            cover.scale
        );
        assert!(close.scale < cover.scale * 2.9);
    }

    #[test]
    fn contain_never_crops_and_stays_in_its_margin() {
        let src = PORTRAIT;
        let f = Framing::new(21, Fit::Contain, true);
        // Even with a camera that would blow a cover fit apart.
        for cam in [
            Cam::default(),
            Cam {
                zoom: 2.2,
                fx: 0.05,
                fy: 0.95,
            },
        ] {
            let p = f.place(src, FRAME, cam);
            assert!(p.letterboxed, "contain owes the caller a backdrop");
            for (sx, sy) in corners(src.0, src.1) {
                let (px, py) = lands_at(&p, src, FRAME, sx, sy);
                assert!(
                    px >= -1e-6 && px <= FRAME.0 + 1e-6 && py >= -1e-6 && py <= FRAME.1 + 1e-6,
                    "plate corner ({sx}, {sy}) fell outside the frame at ({px}, {py})"
                );
            }
            // And the offset never exceeds the leftover letterbox space.
            let (bw, bh) = rotated_bbox(src.0, src.1, 0.0);
            let mx = (FRAME.0 - bw * p.scale) / 2.0;
            let my = (FRAME.1 - bh * p.scale) / 2.0;
            assert!(mx >= -1e-9 && my >= -1e-9, "negative margin {mx} {my}");
            assert!(p.drift_x.abs() * p.scale <= mx + 1e-6, "x {}", p.drift_x);
            assert!(p.drift_y.abs() * p.scale <= my + 1e-6, "y {}", p.drift_y);
        }
    }

    #[test]
    fn contain_damps_the_camera() {
        // Same wild camera, two fits: contain must move less.
        let cam = Cam {
            zoom: 2.0,
            fx: 0.1,
            fy: 0.1,
        };
        let contain = Framing::new(5, Fit::Contain, true);
        let calm = contain.place(PORTRAIT, FRAME, cam);
        let still = contain.place(PORTRAIT, FRAME, Cam::default());
        // Zoom is damped to 0.55 of its excess, then capped by the fit.
        assert!(calm.scale <= still.scale * 1.56);
        assert!(calm.scale >= still.scale);
    }

    #[test]
    fn rotated_bbox_matches_the_plain_box_at_zero() {
        let (w, h) = rotated_bbox(1920.0, 1080.0, 0.0);
        assert!((w - 1920.0).abs() < 1e-9 && (h - 1080.0).abs() < 1e-9);
        // Half a turn is the same box.
        let (w, h) = rotated_bbox(1920.0, 1080.0, std::f64::consts::PI);
        assert!((w - 1920.0).abs() < 1e-6 && (h - 1080.0).abs() < 1e-6);
        // A quarter turn swaps it.
        let (w, h) = rotated_bbox(1920.0, 1080.0, std::f64::consts::FRAC_PI_2);
        assert!((w - 1080.0).abs() < 1e-6 && (h - 1920.0).abs() < 1e-6);
    }

    #[test]
    fn rotated_bbox_grows_at_forty_five_degrees() {
        let r2 = std::f64::consts::SQRT_2;
        let (w, h) = rotated_bbox(100.0, 100.0, std::f64::consts::FRAC_PI_4);
        assert!((w - 100.0 * r2).abs() < 1e-6, "w {w}");
        assert!((h - 100.0 * r2).abs() < 1e-6, "h {h}");
        // A rectangle turns into a square box of half the perimeter.
        let (w, h) = rotated_bbox(200.0, 100.0, std::f64::consts::FRAC_PI_4);
        assert!((w - 150.0 * r2).abs() < 1e-6 && (h - 150.0 * r2).abs() < 1e-6);
    }

    #[test]
    fn a_tilted_cover_still_lands_whole() {
        // Not reachable today (the angle is always 0), but the maths are
        // the reason the fit can be switched on later.
        let angle = std::f64::consts::FRAC_PI_4;
        let f = Framing::new(3, Fit::Cover, false).with_angle(angle);
        let flat = Framing::new(3, Fit::Cover, false);
        let tilted = f.place(LANDSCAPE, FRAME, Cam::default());
        let plain = flat.place(LANDSCAPE, FRAME, Cam::default());
        assert!(
            tilted.scale > plain.scale,
            "a turned frame needs more plate: {} vs {}",
            tilted.scale,
            plain.scale
        );
        // The rotated frame box must still fit inside the plate.
        let (bw, bh) = rotated_bbox(FRAME.0, FRAME.1, angle);
        assert!(bw / tilted.scale <= LANDSCAPE.0 + 1e-6);
        assert!(bh / tilted.scale <= LANDSCAPE.1 + 1e-6);
    }

    #[test]
    fn closeup_reaims_on_the_phrase_and_not_between() {
        let mut f = Framing::new(7, Fit::Closeup, false);
        let mut d = Drive::default();
        let mut beat = 0.0;
        let before = (f.target(), f.close_zoom());

        advance(&mut f, &mut d, &mut beat, 15.0, false);
        assert_eq!(
            (f.target(), f.close_zoom()),
            before,
            "nothing may re-aim inside a phrase"
        );

        advance(&mut f, &mut d, &mut beat, 2.0, false); // over beat 16
        let after = (f.target(), f.close_zoom());
        assert_ne!(after, before, "the 16-beat boundary must re-aim");

        advance(&mut f, &mut d, &mut beat, 10.0, false);
        assert_eq!(
            (f.target(), f.close_zoom()),
            after,
            "and then hold until the next one"
        );
    }

    #[test]
    fn cover_reaims_on_some_phrases_but_not_all() {
        // The coin flip is per phrase; over a few minutes both outcomes
        // must show up, or the flip is not doing anything.
        let mut f = Framing::new(31, Fit::Cover, false);
        let mut d = Drive::default();
        let mut beat = 0.0;
        let mut changes = 0;
        let mut last = f.target();
        for _ in 0..24 {
            advance(&mut f, &mut d, &mut beat, 16.0, false);
            if f.target() != last {
                changes += 1;
                last = f.target();
            }
        }
        assert!(changes > 3, "cover never re-aimed ({changes})");
        assert!(changes < 21, "cover re-aimed on every phrase ({changes})");
    }

    #[test]
    fn focus_eases_toward_its_target_instead_of_snapping() {
        let mut f = Framing::new(2, Fit::Closeup, false);
        let mut d = Drive::default();
        let mut beat = 0.0;
        advance(&mut f, &mut d, &mut beat, 16.1, false); // force a re-aim
        let start = f.cam(0.0, &d);
        let (tx, ty) = f.target();
        let d0 = (start.fx - tx).hypot(start.fy - ty);
        assert!(d0 > 1e-4, "the re-aim should leave somewhere to travel");

        // One 60 fps frame closes ~1.5% of the gap: moved, nowhere near
        // arrived.
        f.update(1.0 / 60.0, &d, false);
        let step = f.cam(0.0, &d);
        let d1 = (step.fx - tx).hypot(step.fy - ty);
        assert!(d1 < d0, "focus should approach its target");
        assert!(
            d1 > d0 * 0.9,
            "focus snapped instead of easing: {d0} -> {d1}"
        );

        // Nine tau later it is effectively there.
        for _ in 0..600 {
            f.update(1.0 / 60.0, &d, false);
        }
        let done = f.cam(0.0, &d);
        assert!((done.fx - tx).hypot(done.fy - ty) < 1e-3);
    }

    #[test]
    fn focus_stays_inside_the_documented_bands() {
        for portrait in [false, true] {
            let (lo, hi) = if portrait {
                FOCUS_Y_PORTRAIT
            } else {
                FOCUS_Y_LANDSCAPE
            };
            for seed in 0..40u64 {
                let mut f = Framing::new(seed, Fit::Closeup, portrait);
                let mut d = Drive::default();
                let mut beat = 0.0;
                for _ in 0..6 {
                    advance(&mut f, &mut d, &mut beat, 16.0, portrait);
                    let c = f.cam(0.0, &d);
                    assert!(
                        (FOCUS_X.0 - 1e-9..=FOCUS_X.1 + 1e-9).contains(&c.fx),
                        "fx {} out of band",
                        c.fx
                    );
                    assert!(
                        (lo - 1e-9..=hi + 1e-9).contains(&c.fy),
                        "fy {} out of band for portrait={portrait}",
                        c.fy
                    );
                }
            }
        }
    }

    #[test]
    fn portrait_focus_sits_higher_than_landscape() {
        let mut hi = 0.0;
        let mut lo = 0.0;
        for seed in 0..64u64 {
            hi += Framing::new(seed, Fit::Cover, true).target().1;
            lo += Framing::new(seed, Fit::Cover, false).target().1;
        }
        assert!(
            hi / 64.0 < lo / 64.0,
            "portraits must aim at the head: {} vs {}",
            hi / 64.0,
            lo / 64.0
        );
    }

    #[test]
    fn fit_roll_follows_the_original_odds() {
        let n = 4000u64;
        let (mut cover, mut contain) = (0, 0);
        for s in 0..n {
            match Fit::roll(s, false) {
                Fit::Cover => cover += 1,
                Fit::Contain => contain += 1,
                _ => {}
            }
        }
        let p = cover as f64 / n as f64;
        assert!((p - 5.0 / 6.0).abs() < 0.03, "landscape cover rate {p}");
        assert_eq!(contain, 0, "a 16:9 plate has nothing to contain it from");

        let mut contain = 0;
        for s in 0..n {
            if Fit::roll(s, true) == Fit::Contain {
                contain += 1;
            }
        }
        let p = contain as f64 / n as f64;
        assert!((p - 0.20).abs() < 0.03, "portrait contain rate {p}");
    }

    #[test]
    fn everything_is_deterministic_for_a_seed() {
        assert_eq!(Fit::roll(99, true), Fit::roll(99, true));
        let run = || {
            let mut f = Framing::new(77, Fit::Closeup, true);
            let mut d = Drive::default();
            let mut beat = 0.0;
            let mut out = Vec::new();
            for _ in 0..3 {
                advance(&mut f, &mut d, &mut beat, 17.0, true);
                let c = f.cam(0.0, &d);
                out.push((c.zoom, c.fx, c.fy));
                out.push({
                    let p = f.place(PORTRAIT, FRAME, c);
                    (p.scale, p.drift_x, p.drift_y)
                });
            }
            out
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn degenerate_sizes_do_not_panic() {
        let sizes = [
            (1.0, 1.0),
            (1.0, 4000.0),
            (4000.0, 1.0),
            (0.0, 0.0),
            (-5.0, 3.0),
            (f64::NAN, 10.0),
        ];
        for fit in [Fit::Cover, Fit::Closeup, Fit::Contain] {
            let mut f = Framing::new(1, fit, true);
            f.update(0.016, &Drive::default(), true);
            // A wild dt must not poison the state either.
            f.update(f64::NAN, &Drive::default(), true);
            f.update(-1.0, &Drive::default(), true);
            for src in sizes {
                for dst in sizes {
                    let p = f.place(src, dst, Cam::default());
                    assert!(p.scale.is_finite() && p.scale > 0.0, "scale {}", p.scale);
                    assert!(p.drift_x.is_finite() && p.drift_y.is_finite());
                }
            }
        }
    }
}
