//! ASCII-tier effects. Each renders straight into the ratatui buffer.
//!
//! Determinism contract: an effect may only derive randomness from
//! `FrameCtx` (beat time) and cell coordinates via `rng::hash3` — never
//! from wall time. Same beat + same size = same frame.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

mod collapse;
mod pulse;
mod rain;
mod tunnel;

pub use collapse::Collapse;
pub use pulse::Pulse;
pub use rain::Rain;
pub use tunnel::Tunnel;

/// Everything an effect is allowed to know about the world.
pub struct FrameCtx {
    /// Cumulative beats. Fractional part is beat phase.
    pub beat: f64,
    /// Phase within one beat, [0, 1).
    pub phase: f64,
    /// Phase within a 4-beat bar, [0, 4).
    pub bar_phase: f64,
    /// Global intensity fader, [0, 1]. Keyboard now, MIDI CC later.
    pub intensity: f64,
}

impl FrameCtx {
    /// Beat time quantized to 1/16 beats — for groove-locked flicker.
    pub fn tick16(&self) -> u64 {
        (self.beat.max(0.0) * 16.0) as u64
    }
}

pub trait Effect {
    fn name(&self) -> &'static str;
    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx);
}

/// The classic luminance ramp, dark to bright.
pub const RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

pub fn ramp_glyph(v: f64) -> char {
    let i = (v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f64).round() as usize;
    RAMP[i]
}
