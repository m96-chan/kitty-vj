//! PLATE — halfblock tier. Artwork sampled cover-fit into `▀` cells
//! (separate fg/bg = double vertical resolution), with the EasyPngVJ
//! moves that survive the trip to a terminal: ken burns drift, zoom
//! punch, chroma split, slice, invert flash. Plate changes land on a
//! bar line, queued not executed.

use std::rc::Rc;

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
    plates: Rc<Vec<Plate>>,
    current: usize,
    pending: Option<usize>,
    last_bar: i64,
    /// The plate we are transitioning away from, and when the mix
    /// started. A swap is a Mixer: both plates are on screen at once.
    outgoing: Option<usize>,
    swap_beat: f64,
    /// Camera modulator — parameters, not pixels; re-rolled per plate.
    ken: crate::camera::KenBurns,
    /// Phrase dive, layered on top of the drift.
    punch: crate::camera::PunchIn,
    /// Which mixer runs on the next plate swap.
    trans: usize,
}

impl PlateFx {
    pub fn new(plates: Rc<Vec<Plate>>) -> Self {
        Self {
            plates,
            current: 0,
            pending: None,
            last_bar: 0,
            outgoing: None,
            swap_beat: 0.0,
            ken: crate::camera::KenBurns::roll(0, 0.0),
            punch: crate::camera::PunchIn::new(1),
            trans: 0,
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
            KeyCode::Char('t') => {
                self.trans = (self.trans + 1) % crate::transition::TRANSITIONS.len();
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
        let mut s = format!(
            "{} {}",
            crate::transition::TRANSITIONS[self.trans].name(),
            self.plates.get(self.current)?.name
        );
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
            let next = if let Some(next) = self.pending.take() {
                Some(next)
            } else if bar % AUTO_BARS == 0 && bar > self.last_bar {
                Some((self.current + 1) % self.plates.len())
            } else {
                None
            };
            if let Some(next) = next {
                // Keep the old plate around so the transition has two
                // frames to mix; roll a fresh camera move for the new one.
                self.outgoing = Some(self.current);
                self.swap_beat = ctx.beat;
                self.current = next;
                self.ken = crate::camera::KenBurns::roll(next as u64, ctx.beat);
            }
            self.last_bar = bar;
        }

        // Transition progress. Past the end the outgoing plate is done.
        let tr = crate::transition::TRANSITIONS[self.trans];
        let mix = if self.outgoing.is_some() {
            let p = (ctx.beat - self.swap_beat) / tr.beats();
            if p >= 1.0 {
                self.outgoing = None;
                1.0
            } else {
                p.max(0.0)
            }
        } else {
            1.0
        };

        let img = &self.plates[self.current].img;
        let (sw, sh) = (img.width() as f64, img.height() as f64);
        // Halfblock pixel grid: one column per cell, two rows per cell.
        let (tw, th) = (area.width as f64, area.height as f64 * 2.0);

        // Cover fit, then the modulator's camera on top. Ken Burns keeps
        // drifting through a breakdown by design; the punch is the beat.
        let cam = {
            use crate::camera::Modulator;
            // Two modulators compose by multiplying zoom and taking the
            // dive's focus while it is actually diving.
            let base = self.ken.cam(ctx.beat, &ctx.drive);
            let dive = self.punch.cam(ctx.beat, &ctx.drive);
            let k = (dive.zoom - 1.0).clamp(0.0, 1.0);
            crate::camera::Cam {
                zoom: base.zoom * dive.zoom,
                fx: base.fx + (dive.fx - base.fx) * k,
                fy: base.fy + (dive.fy - base.fy) * k,
            }
        };
        let cover = (tw / sw).max(th / sh);
        let punch = 1.0 + 0.10 * ctx.drive.gbeat() * (0.3 + 0.7 * ctx.intensity);
        let zoom = cover * punch * cam.zoom;
        let drift_x = (cam.fx - 0.5) * sw;
        let drift_y = (cam.fy - 0.5) * sh;

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

        // The outgoing plate is sampled with the same camera — a cut
        // that also moved the camera reads as two separate events.
        let old = self.outgoing.map(|i| &self.plates[i].img);

        let sample_from = |im: &image::RgbImage, x: f64, y: f64| -> (u8, u8, u8) {
            let (iw, ih) = (im.width() as f64, im.height() as f64);
            let sx = (x / zoom + iw / 2.0 + drift_x).clamp(0.0, iw - 1.0) as u32;
            let sy = (y / zoom + ih / 2.0 + drift_y).clamp(0.0, ih - 1.0) as u32;
            let p = im.get_pixel(sx, sy).0;
            (p[0], p[1], p[2])
        };
        let sample = |x: f64, y: f64| -> (u8, u8, u8) { sample_from(img, x, y) };

        let px_at_img = |im: &image::RgbImage, px: f64, py: f64| -> (u8, u8, u8) {
            let x = px - tw / 2.0
                + if py >= slice_y0 && py < slice_y1 {
                    slice_dx
                } else {
                    0.0
                };
            let y = py - th / 2.0;
            let (_, g, b) = sample_from(im, x, y);
            let (r, _, _) = sample_from(im, x + chroma, y);
            if invert {
                (255 - r, 255 - g, 255 - b)
            } else {
                (r, g, b)
            }
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
                // The Mixer decides, per cell, which plate wins.
                let show_new = old.is_none() || tr.shows_b(cx, cy, area.width, area.height, mix);
                let (top, bot) = match (show_new, old) {
                    (true, _) => (
                        px_at(cx as f64 + 0.5, cy as f64 * 2.0 + 0.5),
                        px_at(cx as f64 + 0.5, cy as f64 * 2.0 + 1.5),
                    ),
                    (false, Some(o)) => (
                        px_at_img(o, cx as f64 + 0.5, cy as f64 * 2.0 + 0.5),
                        px_at_img(o, cx as f64 + 0.5, cy as f64 * 2.0 + 1.5),
                    ),
                    (false, None) => unreachable!("no outgoing plate"),
                };
                let cell = &mut buf[(area.x + cx, area.y + cy)];
                cell.set_char('▀');
                cell.set_fg(Color::Rgb(top.0, top.1, top.2));
                cell.set_bg(Color::Rgb(bot.0, bot.1, bot.2));
            }
        }
    }
}
