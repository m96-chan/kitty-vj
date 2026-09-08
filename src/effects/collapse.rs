//! COLLAPSE — a giant beat counter (1..4) with glitch: random glyph
//! substitution whose density rides the intensity fader. Low intensity is
//! a stable loop, high intensity is collapse.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::rng::{hash3, unit_f64};

/// 5x7 bitmap digits, row-major, '#' = on.
const DIGITS: [[&str; 7]; 4] = [
    [
        "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
    ],
    [
        ".###.", "#...#", "....#", "..##.", ".#...", "#....", "#####",
    ],
    [
        ".###.", "#...#", "....#", ".###.", "....#", "#...#", ".###.",
    ],
    [
        "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#.",
    ],
];

const GLITCH_GLYPHS: &[char] = &[
    '▓', '▒', '░', '█', '#', '%', '/', '\\', 'X', '?', '!', '~', '@', '$',
];

pub struct Collapse;

impl Effect for Collapse {
    fn name(&self) -> &'static str {
        "COLLAPSE"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        if area.width < 10 || area.height < 8 {
            return;
        }
        let beat_in_bar = ctx.bar_phase as usize; // 0..3
        let digit = &DIGITS[beat_in_bar];

        // Scale the 5x7 bitmap to fill ~60% of the area.
        let scale_x = ((area.width as f64 * 0.6) / 5.0).max(1.0) as u16;
        let scale_y = ((area.height as f64 * 0.7) / 7.0).max(1.0) as u16;
        let dw = 5 * scale_x;
        let dh = 7 * scale_y;
        let ox = area.x + (area.width.saturating_sub(dw)) / 2;
        let oy = area.y + (area.height.saturating_sub(dh)) / 2;

        let flash = (1.0 - ctx.phase).powi(3);
        let lum = (120.0 + 135.0 * flash) as u8;

        for y in 0..dh.min(area.height) {
            for x in 0..dw.min(area.width) {
                let bit = digit[(y / scale_y) as usize].as_bytes()[(x / scale_x) as usize] == b'#';
                if bit {
                    let cell = &mut buf[(ox + x, oy + y)];
                    cell.set_char('█');
                    cell.set_fg(Color::Rgb(lum, lum, lum));
                }
            }
        }

        // Glitch pass: density rides the fader, flicker locked to 16ths.
        let density = ctx.intensity.powi(2) * 0.45;
        if density <= 0.0 {
            return;
        }
        let tick = ctx.tick16();
        for y in 0..area.height {
            for x in 0..area.width {
                let h = hash3(x as u64, y as u64, tick);
                if unit_f64(h) < density {
                    let g = hash3(y as u64, x as u64, tick ^ 0xff);
                    let cell = &mut buf[(area.x + x, area.y + y)];
                    cell.set_char(GLITCH_GLYPHS[(g as usize) % GLITCH_GLYPHS.len()]);
                    let r = 128 + (g >> 8) as u8 / 2;
                    let b = 128 + (g >> 16) as u8 / 2;
                    cell.set_fg(Color::Rgb(r, 64, b));
                }
            }
        }
    }
}
