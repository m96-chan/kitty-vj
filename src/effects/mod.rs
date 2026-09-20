//! ASCII-tier effects. Each renders straight into the ratatui buffer.
//!
//! Determinism contract: an effect may only derive randomness from
//! `FrameCtx` (beat time) and cell coordinates via `rng::hash3` — never
//! from wall time. Same beat + same size = same frame.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

mod cam;
mod collapse;
mod cube;
mod imgdust;
mod plate;
mod pulse;
mod rain;
mod sparks;
mod text;
mod tunnel;

pub use cam::CamFx;
pub use collapse::Collapse;
pub use cube::Cube;
pub use imgdust::ImgDust;
pub use plate::PlateFx;
pub use pulse::Pulse;
pub use rain::Rain;
pub use sparks::Sparks;
pub use text::{TextOverlay, draw_text};
pub use tunnel::Tunnel;

/// Everything an effect is allowed to know about the world.
pub struct FrameCtx {
    /// Cumulative beats. Fractional part is beat phase.
    pub beat: f64,
    /// Phase within one beat, [0, 1).
    pub phase: f64,
    /// Phase within a 4-beat bar, [0, 4).
    pub bar_phase: f64,
    /// Real time since the last frame, seconds. For the stateful pieces
    /// an effect owns — the framing's focus easing — which were being
    /// fed a made-up sixtieth of a second and so ran at the wrong rate
    /// whenever the frame rate was not exactly 60.
    pub dt: f64,
    /// Visual clock in seconds — wall time, not beat time. Effects whose
    /// original was written against a clock (a hue cycle per second, a
    /// slow camera drift) must read this, or they change speed with the
    /// tempo.
    pub vt: f64,
    /// Global intensity fader, [0, 1]. Keyboard now, MIDI CC later.
    pub intensity: f64,
    /// Drive signals — grid pulses, kick measure, onset flinch. Effects
    /// ported from EasyPngVJ are written against these, not raw phase.
    pub drive: crate::drive::Drive,
}

impl FrameCtx {
    /// Beat time quantized to 1/16 beats — for groove-locked flicker.
    pub fn tick16(&self) -> u64 {
        crate::pass::tick16(self.beat)
    }
}

/// `v` wrapped into `[0, n)` — the half-open range `phase` and
/// `bar_phase` document.
///
/// `f64::rem_euclid` alone does **not** give that range: for a `v` a
/// hair below zero it computes `v % n + n`, which rounds up to exactly
/// `n`. Backspin and jog scrub can put the visual beat there, and a
/// `bar_phase` of 4.0 indexes a four-entry table out of bounds. So the
/// wrap is one function and the contract is true wherever it is used.
pub fn wrap(v: f64, n: f64) -> f64 {
    let r = v.rem_euclid(n);
    if r < n { r } else { 0.0 }
}

pub trait Effect {
    fn name(&self) -> &'static str;
    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx);
    /// Keys the app didn't consume are offered to the active effect.
    /// Return true if handled.
    fn on_key(&mut self, _code: KeyCode) -> bool {
        false
    }
    /// A new scene has been rolled. Units that care about the cast list
    /// — the framing an artwork gets, the seed its moves derive from —
    /// take it here; everything else ignores it. Defaulted so adding a
    /// scene concept does not touch every effect.
    fn on_scene(&mut self, _fit: crate::framing::Fit, _seed: u64) {}

    /// The scene asked for a transition of a particular length; this is
    /// the index it resolved to. Only units that mix two frames care.
    fn on_transition(&mut self, _index: usize) {}

    /// Extra HUD text (e.g. current plate name).
    fn status(&self) -> Option<String> {
        None
    }
}

/// Every cell effect's name, in construction order, plates included.
/// The single authority the scene tables and the collision tests check
/// against — each used to keep its own copy, and a copy is worse than
/// no test: it keeps passing while a rename turns a scene's choice into
/// a silent no-op. `build_effects` in main.rs is bridge-tested to
/// produce exactly these names, so a rename in an `Effect::name()`
/// breaks a test instead of a gig.
pub const CELL_NAMES: &[&str] = &[
    "PULSE", "RAIN", "TUNNEL", "COLLAPSE", "CUBE", "SPARKS", "IMGDUST", "PLATE", "CAM",
];

/// The classic luminance ramp, dark to bright.
pub const RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

pub fn ramp_glyph(v: f64) -> char {
    let i = (v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f64).round() as usize;
    RAMP[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rem_euclid` alone leaks the upper bound: jog scrub and backspin
    /// can leave the visual beat a hair below zero, and `(-1e-18).rem_
    /// euclid(4.0)` rounds up to exactly 4.0. `bar_phase` indexes a
    /// four-entry digit table in COLLAPSE, so that was a panic.
    #[test]
    fn the_phase_range_is_half_open_even_just_below_zero() {
        for v in [-1e-18f64, -1e-16, -f64::MIN_POSITIVE, -0.0] {
            assert!(v.rem_euclid(4.0) == 4.0 || v.rem_euclid(4.0) == 0.0);
            assert!(wrap(v, 4.0) < 4.0, "bar_phase {v:e} -> {}", wrap(v, 4.0));
            assert!(wrap(v, 1.0) < 1.0, "phase {v:e} -> {}", wrap(v, 1.0));
        }
        // Ordinary values are untouched.
        assert_eq!(wrap(2.5, 4.0), 2.5);
        assert_eq!(wrap(-1.0, 4.0), 3.0);
        assert_eq!(wrap(5.5, 4.0), 1.5);
        // And nothing non-finite escapes into an array index.
        assert_eq!(wrap(f64::NAN, 4.0), 0.0);
        assert_eq!(wrap(f64::INFINITY, 4.0), 0.0);
    }
}
