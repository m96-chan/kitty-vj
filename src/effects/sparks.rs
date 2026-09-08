//! SPARKS — particle bursts. One burst per beat, a double burst on the
//! bar. Particles are closed-form (position is a pure function of beat
//! time and hash), so there is no simulation state and determinism
//! survives time running backwards.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::rng::{hash3, unit_f64};

/// Beats a burst stays visible.
const LIFE: f64 = 3.0;

pub struct Sparks;

impl Effect for Sparks {
    fn name(&self) -> &'static str {
        "SPARKS"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        let (w, h) = (area.width as f64, area.height as f64);
        if w < 4.0 || h < 4.0 {
            return;
        }
        let max_r = (w / 2.0).min(h); // in cell-x units; y is halved later

        let now = ctx.beat;
        let first = (now - LIFE).ceil().max(0.0) as i64;
        let last = now.floor() as i64;

        for b in first..=last {
            let age = now - b as f64; // 0..LIFE
            if age < 0.0 {
                continue;
            }
            let on_bar = b % 4 == 0;
            let count = ((30.0 + 90.0 * ctx.intensity) * if on_bar { 2.0 } else { 1.0 }) as u64;

            // Burst origin drifts around the middle third, per-burst hash.
            let ox = w * (0.35 + 0.3 * unit_f64(hash3(b as u64, 100, 0)));
            let oy = h * (0.35 + 0.3 * unit_f64(hash3(b as u64, 101, 0)));

            let fade = (1.0 - age / LIFE).max(0.0);
            for j in 0..count {
                let ang = unit_f64(hash3(b as u64, j, 1)) * std::f64::consts::TAU;
                let spd = (0.25 + 0.75 * unit_f64(hash3(b as u64, j, 2))) * max_r / LIFE;
                // Decelerating flight plus a little gravity.
                let r = spd * age * (1.0 - 0.25 * age / LIFE);
                let px = ox + ang.cos() * r;
                let py = oy + ang.sin() * r * 0.5 + 0.35 * age * age; // cell aspect + fall
                if px < 0.0 || py < 0.0 || px >= w || py >= h {
                    continue;
                }

                let v = fade * (0.4 + 0.6 * unit_f64(hash3(b as u64, j, 3)));
                let g = match (v * 4.0) as u8 {
                    0 => '.',
                    1 => ':',
                    2 => '+',
                    _ => '*',
                };
                // Per-burst hue.
                let hue = unit_f64(hash3(b as u64, 102, 0));
                let lum = (v * 255.0) as u8;
                let (rr, gg, bb) = if hue < 0.33 {
                    (lum, (lum as f64 * 0.6) as u8, 40)
                } else if hue < 0.66 {
                    (40, lum, (lum as f64 * 0.7) as u8)
                } else {
                    ((lum as f64 * 0.7) as u8, 60, lum)
                };

                let cell = &mut buf[(area.x + px as u16, area.y + py as u16)];
                cell.set_char(g);
                cell.set_fg(Color::Rgb(rr, gg, bb));
            }
        }
    }
}
