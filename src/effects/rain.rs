//! RAIN — glyph waterfall. Each column falls at its own speed, quantized
//! to 16ths, with a bright head and a fading tail.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::rng::{hash3, unit_f64};

const GLYPHS: &[char] = &[
    '0', '1', '7', '9', 'A', 'K', 'T', 'V', 'X', 'Z', '$', '%', '&', '*', '+', '<', '>', '/', '\\',
    '|', '=', '?',
];

pub struct Rain;

impl Effect for Rain {
    fn name(&self) -> &'static str {
        "RAIN"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        let h = area.height as f64;
        if h < 2.0 {
            return;
        }
        let tail = (h * 0.5).max(4.0);
        let span = h + tail;

        for x in 0..area.width {
            let col = x as u64;
            // Per-column personality, stable across frames.
            let speed = 4.0 + unit_f64(hash3(col, 1, 0)) * 8.0; // rows per beat
            let offset = unit_f64(hash3(col, 2, 0)) * span;
            let pass_len = span / speed; // beats per full pass

            // Quantize fall position to 16ths so motion locks to the grid.
            let t = (ctx.beat * 16.0).floor() / 16.0;
            let head = (t * speed + offset).rem_euclid(span);
            let pass = ((t * speed + offset) / span) as u64;

            let boost = 0.5 + 0.5 * ctx.intensity;
            for y in 0..area.height {
                let dist = head - y as f64; // how far behind the head this row is
                if dist < 0.0 || dist > tail {
                    continue;
                }
                let fade = (1.0 - dist / tail).powf(1.5) * boost;
                // Glyph reshuffles every pass and flickers on 16ths.
                let g = hash3(col, y as u64, pass ^ (ctx.tick16() % 4));
                let ch = GLYPHS[(g as usize) % GLYPHS.len()];

                let cell = &mut buf[(area.x + x, area.y + y)];
                cell.set_char(ch);
                if dist < 1.0 {
                    cell.set_fg(Color::Rgb(220, 255, 220)); // head
                } else {
                    let lum = (fade * 220.0) as u8;
                    cell.set_fg(Color::Rgb(lum / 4, lum, lum / 3));
                }
            }
            let _ = pass_len;
        }
    }
}
