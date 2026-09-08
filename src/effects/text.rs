//! Text overlay — big-glyph caption composited over whatever effect is
//! playing, EasyPngVJ's `?text=` brought to the grid. Beat-jittered,
//! scale punched on the downbeat, chroma-fringed at high intensity.
//! Not an `Effect`: the app draws it on top when toggled.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::FrameCtx;
use crate::font;
use crate::rng::{hash3, unit_f64};

pub struct TextOverlay {
    pub text: String,
    pub visible: bool,
}

impl TextOverlay {
    pub fn new(text: String) -> Self {
        Self {
            text,
            visible: false,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub fn render(&self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        if !self.visible || self.text.is_empty() {
            return;
        }
        draw_text(buf, area, ctx, &self.text, 0.5, 1.0, 1.0, true);
    }
}

/// Draw one line of big-glyph text.
///
/// `y_frac` places the line's centre in the area (0.5 = middle), `scale`
/// trims the fitted size (lyric context lines ride smaller than the
/// current one), `lit` is the fraction of the line drawn bright — the
/// rest is dimmed, which is how a sung word fills in. `fx` enables the
/// beat jitter and chroma fringe; context lines want it off.
#[allow(clippy::too_many_arguments)]
pub fn draw_text(
    buf: &mut Buffer,
    area: Rect,
    ctx: &FrameCtx,
    text: &str,
    y_frac: f64,
    scale: f64,
    lit: f64,
    fx: bool,
) {
    {
        if text.is_empty() || area.width < 12 || area.height < 9 {
            return;
        }
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len() as u16;

        // Fit: glyphs are 5 wide + 1 gap; x-scale doubled vs y for cell aspect.
        let sx = ((((area.width * 9 / 10) / (n * 6)) as f64 * scale) as u16).max(1);
        let sy = ((sx / 2).max(1)).min((area.height * 7 / 10) / 7).max(1);
        let tw = n * 6 * sx - sx;
        let th = 7 * sy;

        // Beat jitter, quantized to 16ths so it snaps rather than swims.
        let tick = ctx.tick16();
        let (jx, jy) = if fx {
            (
                ((unit_f64(hash3(tick, 200, 0)) - 0.5) * 3.0 * ctx.intensity) as i32,
                ((unit_f64(hash3(tick, 201, 0)) - 0.5) * 2.0 * ctx.intensity) as i32,
            )
        } else {
            (0, 0)
        };

        let ox = (area.x + (area.width.saturating_sub(tw)) / 2) as i32 + jx;
        let cy = area.y as f64 + area.height as f64 * y_frac;
        let oy = (cy - th as f64 / 2.0) as i32 + jy;

        let flash = (1.0 - ctx.phase).powi(2);
        let lum = (170.0 + 85.0 * flash) as u8;
        let chroma = fx && ctx.intensity > 0.4;
        // Glyphs past the lit fraction are drawn dim.
        let lit_upto = (chars.len() as f64 * lit.clamp(0.0, 1.0)).round() as usize;

        let mut plot = |x: i32, y: i32, color: Color, ch: char| {
            if x < area.x as i32
                || y < area.y as i32
                || x >= (area.x + area.width) as i32
                || y >= (area.y + area.height) as i32
            {
                return;
            }
            let cell = &mut buf[(x as u16, y as u16)];
            cell.set_char(ch);
            cell.set_fg(color);
        };

        for (i, &c) in chars.iter().enumerate() {
            let bright = i < lit_upto;
            let body = if bright {
                Color::Rgb(lum, lum, lum)
            } else {
                Color::Rgb(lum / 3, lum / 3, lum / 3)
            };
            let g = font::glyph(c);
            let gx = ox + (i as i32) * (6 * sx as i32);
            for (row, line) in g.iter().enumerate() {
                for (col, b) in line.bytes().enumerate() {
                    if b != b'#' {
                        continue;
                    }
                    for dy in 0..sy as i32 {
                        for dx in 0..sx as i32 {
                            let x = gx + col as i32 * sx as i32 + dx;
                            let y = oy + row as i32 * sy as i32 + dy;
                            // Chroma fringe first so the body wins overlaps.
                            if chroma && bright {
                                let off = 1 + (flash * 2.0) as i32;
                                plot(x - off, y, Color::Rgb(255, 60, 90), '░');
                                plot(x + off, y, Color::Rgb(60, 200, 255), '░');
                            }
                            plot(x, y, body, if bright { '█' } else { '▒' });
                        }
                    }
                }
            }
        }
    }
}
