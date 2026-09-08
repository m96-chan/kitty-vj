//! TUNNEL — concentric box-drawing rectangles rushing outward, one ring
//! born per beat.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};

pub struct Tunnel;

const SPACING: f64 = 6.0;

impl Effect for Tunnel {
    fn name(&self) -> &'static str {
        "TUNNEL"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        let (w, h) = (area.width as f64, area.height as f64);
        if w < 4.0 || h < 4.0 {
            return;
        }
        let (cx, cy) = (w / 2.0, h / 2.0);
        let max_d = cx.max(cy * 2.0);

        // Rings travel outward one SPACING per beat.
        let travel = ctx.beat * SPACING;
        let thickness = 0.5 + ctx.intensity * 1.2;

        for y in 0..area.height {
            for x in 0..area.width {
                let dx = x as f64 + 0.5 - cx;
                let dy = (y as f64 + 0.5 - cy) * 2.0;
                // Chebyshev distance = rectangular rings.
                let d = dx.abs().max(dy.abs());

                let r = (d - travel).rem_euclid(SPACING);
                if r > thickness {
                    continue;
                }

                // Pick a box-drawing glyph by which edge of the ring we're on.
                let ch = if (dx.abs() - dy.abs()).abs() < 1.2 {
                    '┼'
                } else if dx.abs() > dy.abs() {
                    '│'
                } else {
                    '─'
                };

                // Newborn rings (near center) glow, then dim as they fly out.
                let age = (d / max_d).clamp(0.0, 1.0);
                let flash = (1.0 - ctx.phase).powi(2);
                let v = ((1.0 - age * 0.7) * (0.45 + 0.55 * flash)).min(1.0);
                let lum = (v * 255.0) as u8;

                let cell = &mut buf[(area.x + x, area.y + y)];
                cell.set_char(ch);
                cell.set_fg(Color::Rgb(255 - lum / 2, lum, (lum / 2) + 80));
            }
        }
    }
}
