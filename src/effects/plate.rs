//! PLATE — halfblock tier. Artwork sampled cover-fit into `▀` cells
//! (separate fg/bg = double vertical resolution), with the EasyPngVJ
//! moves that survive the trip to a terminal: ken burns drift, zoom
//! punch, chroma split, slice, invert flash. Plate changes land on a
//! bar line, queued not executed.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::assets::Plate;
use crate::rng::{hash3, unit_f64};

/// Bars before the rotation moves on by itself.
const AUTO_BARS: i64 = 8;

pub struct PlateFx {
    plates: Vec<Plate>,
    current: usize,
    pending: Option<usize>,
    last_bar: i64,
}

impl PlateFx {
    pub fn new(plates: Vec<Plate>) -> Self {
        Self {
            plates,
            current: 0,
            pending: None,
            last_bar: 0,
        }
    }
}

impl Effect for PlateFx {
    fn name(&self) -> &'static str {
        "PLATE"
    }

    fn on_key(&mut self, code: KeyCode) -> bool {
        let n = self.plates.len();
        match code {
            KeyCode::Char('n') => {
                self.pending = Some((self.pending.unwrap_or(self.current) + 1) % n);
                true
            }
            KeyCode::Char('p') => {
                self.pending = Some((self.pending.unwrap_or(self.current) + n - 1) % n);
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        let mut s = self.plates.get(self.current)?.name.clone();
        if let Some(p) = self.pending {
            s.push_str(" → ");
            s.push_str(&self.plates[p].name);
        }
        Some(s)
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        if self.plates.is_empty() || area.width < 2 || area.height < 2 {
            return;
        }

        // Bar-line bookkeeping: apply a queued change, auto-rotate on phrase.
        let bar = (ctx.beat / 4.0).floor() as i64;
        if bar != self.last_bar {
            if let Some(next) = self.pending.take() {
                self.current = next;
            } else if bar % AUTO_BARS == 0 && bar > self.last_bar {
                self.current = (self.current + 1) % self.plates.len();
            }
            self.last_bar = bar;
        }

        let img = &self.plates[self.current].img;
        let (sw, sh) = (img.width() as f64, img.height() as f64);
        // Halfblock pixel grid: one column per cell, two rows per cell.
        let (tw, th) = (area.width as f64, area.height as f64 * 2.0);

        // Cover fit, then the clocked camera on top.
        let cover = (tw / sw).max(th / sh);
        let punch = 1.0 + 0.10 * (1.0 - ctx.phase).powi(2) * (0.3 + 0.7 * ctx.intensity);
        let zoom = cover * punch;
        // Ken burns: slow deterministic drift of the focus point.
        let drift_x = 0.06 * (ctx.beat * 0.071).sin() * sw;
        let drift_y = 0.05 * (ctx.beat * 0.047).cos() * sh;

        // Chroma split rides the beat decay.
        let chroma = 2.5 * (1.0 - ctx.phase).powi(2) * ctx.intensity;

        // Slice: on 16ths past half intensity, one horizontal band shears.
        let tick = ctx.tick16();
        let slice_on = ctx.intensity > 0.5 && unit_f64(hash3(tick, 7, 0)) < ctx.intensity - 0.25;
        let (slice_y0, slice_y1, slice_dx) = if slice_on {
            let y0 = unit_f64(hash3(tick, 8, 0)) * th * 0.8;
            let hgt = th * (0.08 + 0.15 * unit_f64(hash3(tick, 9, 0)));
            let dx = (unit_f64(hash3(tick, 10, 0)) - 0.5) * tw * 0.25;
            (y0, y0 + hgt, dx)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Invert flash on the downbeat when the fader is hot.
        let invert = ctx.intensity > 0.75 && ctx.bar_phase < 0.12;

        let sample = |x: f64, y: f64| -> (u8, u8, u8) {
            let sx = (x / zoom + sw / 2.0 + drift_x).clamp(0.0, sw - 1.0) as u32;
            let sy = (y / zoom + sh / 2.0 + drift_y).clamp(0.0, sh - 1.0) as u32;
            let p = img.get_pixel(sx, sy).0;
            (p[0], p[1], p[2])
        };

        let px_at = |px: f64, py: f64| -> (u8, u8, u8) {
            // Centered output coords, sheared if inside the slice band.
            let x = px - tw / 2.0
                + if py >= slice_y0 && py < slice_y1 {
                    slice_dx
                } else {
                    0.0
                };
            let y = py - th / 2.0;
            let (_, g, b) = sample(x, y);
            let (r, _, _) = sample(x + chroma, y);
            if invert {
                (255 - r, 255 - g, 255 - b)
            } else {
                (r, g, b)
            }
        };

        for cy in 0..area.height {
            for cx in 0..area.width {
                let top = px_at(cx as f64 + 0.5, cy as f64 * 2.0 + 0.5);
                let bot = px_at(cx as f64 + 0.5, cy as f64 * 2.0 + 1.5);
                let cell = &mut buf[(area.x + cx, area.y + cy)];
                cell.set_char('▀');
                cell.set_fg(Color::Rgb(top.0, top.1, top.2));
                cell.set_bg(Color::Rgb(bot.0, bot.1, bot.2));
            }
        }
    }
}
