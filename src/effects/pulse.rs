//! PULSE — radial luminance ramp breathing on the beat, with an
//! expanding ring on each beat and a heavier hit on the bar.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx, ramp_glyph};

pub struct Pulse;

impl Effect for Pulse {
    fn name(&self) -> &'static str {
        "PULSE"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        let (w, h) = (area.width as f64, area.height as f64);
        if w < 2.0 || h < 2.0 {
            return;
        }
        let (cx, cy) = (w / 2.0, h / 2.0);
        // Terminal cells are ~1:2, so weigh y double for a circular field.
        let max_d = (cx * cx + (cy * 2.0) * (cy * 2.0)).sqrt();

        // Grid pulses instead of raw phase: the decay curve and the bar
        // emphasis now come from the shared drive signals, so a
        // breakdown calms this the same way it calms everything else.
        let decay = ctx.drive.gbeat();
        let bar_boost = 0.35 * ctx.drive.gbar();
        let energy = (0.25 + 0.75 * decay + bar_boost) * (0.4 + 0.6 * ctx.intensity);

        // Ring expands outward across the beat.
        let ring_pos = ctx.phase * max_d;

        for y in 0..area.height {
            for x in 0..area.width {
                let dx = x as f64 + 0.5 - cx;
                let dy = (y as f64 + 0.5 - cy) * 2.0;
                let d = (dx * dx + dy * dy).sqrt();

                let field = energy * (1.0 - d / max_d).max(0.0).powf(1.5);
                let ring = (1.0 - ((d - ring_pos).abs() / 2.5)).max(0.0) * decay;
                let v = (field + ring * 0.8).min(1.0);

                let cell = &mut buf[(area.x + x, area.y + y)];
                cell.set_char(ramp_glyph(v));
                let lum = (v * 255.0) as u8;
                cell.set_fg(Color::Rgb(lum, lum / 2 + 96, 255 - lum / 3));
            }
        }
    }
}
