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

/// The classic luminance ramp, dark to bright.
pub const RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

pub fn ramp_glyph(v: f64) -> char {
    let i = (v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f64).round() as usize;
    RAMP[i]
}
