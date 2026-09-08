//! IMGDUST — the plate blown apart into its own dust. Every particle
//! carries a fixed random UV into the artwork, takes its colour from that
//! texel and its *depth* from that texel's luminance, so on the beat the
//! bright parts of the picture come off the plane and fly at the camera
//! while the dark parts stay pinned. The plate is still there underneath,
//! dimmed to 0.36, so the frame reads as the artwork disintegrating rather
//! than as sparks over a photo.
//!
//! The original (EasyPngVJ's `imgdust`) did all of this in a vertex
//! shader over a few hundred thousand GL_POINTS — the most expensive mode
//! in the rig. In a cell grid it is the *cheapest* idea to express: every
//! halfblock pixel already is a colour with a luminance, so the port is
//! the shader's own math run over a capped particle budget and composited
//! into `▀` cells with a z-test, exactly like [`super::Cube`].
//!
//! Math kept from the vertex shader, term for term:
//!
//! ```text
//! p     = ((u-0.5)*1560, (0.5-v)*877, -300)
//! p.z  += (lum - 0.30) * spread * (150 + 1100*beat)   // the signature move
//! p.xy *= 1 + 0.07*beat + 0.02*bass
//! p.y  += 6*sin(t*0.9 + seed*12)                      // slow drift
//! rgb   = c * (1.15 + 1.9*beat + 0.9*bass)
//! alpha = 0.35 + 0.65*lum
//! ```
//!
//! `beat` is [`crate::drive::Drive::gbeat`] and `bass` is `drive.thump`,
//! per the drive contract — a ported effect is a function of the drive
//! signals, not of raw phase. Two adaptations the terminal forces:
//! `spread` rides the intensity fader (there is no uniform panel here),
//! and depth is clamped at a near plane instead of clipped, because a
//! particle that overshoots the camera should turn into a fat blob of
//! dust, not vanish on the loudest beat.
//!
//! Determinism contract (see [`super`]): every particle's UV and drift
//! seed come from `rng::hash3` over the particle *index* only, and the
//! motion comes from `ctx.beat` / `ctx.drive`. Nothing here reads wall
//! time, and nothing accumulates between frames, so the same beat at the
//! same size is the same buffer — the blend order is fixed by the loop.

use std::rc::Rc;

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::assets::Plate;
use crate::rng::{hash3, unit_f64};

/// The plate's extent in world units, straight from the original.
const PLATE_W: f64 = 1560.0;
const PLATE_H: f64 = 877.0;

/// Where the undisturbed plane sits in front of the camera.
const BASE_Z: f64 = 300.0;

/// Luminance that stays put. Below it the pixel sinks away, above it the
/// pixel comes at you — 0.30 is roughly "darker than mid grey".
const LUM_PIVOT: f64 = 0.30;

/// Depth floor. The original relies on GL clipping; we would rather keep
/// the brightest dust on screen as a near, magnified blob.
const NEAR_Z: f64 = 60.0;

/// Nearer than this a particle stamps 2x2 pixels instead of 1, so the
/// dust that flew at you reads as chunks rather than lonely dots.
const STAMP_Z: f64 = 165.0;

/// How far the plate dims behind its own dust.
const PLATE_DIM: f64 = 0.36;

/// `spread` as a function of the intensity fader. Bounded so that even a
/// full blast only just reaches the near plane.
const SPREAD_MIN: f64 = 0.10;
const SPREAD_GAIN: f64 = 0.25;

/// One particle per output pixel, up to here. A fullscreen halfblock grid
/// is ~120k pixels; a MacBook Air does not want 120k texel fetches per
/// frame, and the dimmed plate behind fills the gaps anyway.
const MAX_PARTICLES: usize = 28_000;

/// Bars before the rotation moves on by itself. Matches PLATE.
const AUTO_BARS: i64 = 8;

pub struct ImgDust {
    plates: Rc<Vec<Plate>>,
    current: usize,
    pending: Option<usize>,
    last_bar: i64,
    /// Halfblock z/color buffer, reused across frames.
    zbuf: Vec<Option<(f64, [u8; 3])>>,
}

/// The frame-constant half of the shader's uniforms: same for every
/// particle in a frame, so it is computed once and handed down.
#[derive(Clone, Copy)]
struct Blast {
    /// Time driving the slow drift. Beat time here, seconds over there.
    t: f64,
    /// `drive.gbeat()` — the grid pulse the original called `beat`.
    beat: f64,
    /// `drive.thump` — the kick measure the original called `bass`.
    bass: f64,
    /// How hard the depth term throws. Rides the intensity fader.
    spread: f64,
}

/// One particle's position in plate space, before projection. Pure: this
/// is the vertex shader's body and nothing else, which is why the tests
/// can poke at it directly.
fn particle_pos(u: f64, v: f64, seed: f64, lum: f64, b: Blast) -> [f64; 3] {
    let mut x = (u - 0.5) * PLATE_W;
    let mut y = (0.5 - v) * PLATE_H;
    // The signature move: brightness is depth, and the beat is the blast.
    let z = -BASE_Z + (lum - LUM_PIVOT) * b.spread * (150.0 + 1100.0 * b.beat);
    let swell = 1.0 + 0.07 * b.beat + 0.02 * b.bass;
    x *= swell;
    y *= swell;
    y += 6.0 * (b.t * 0.9 + seed * 12.0).sin();
    [x, y, z]
}

/// Alpha-over, with the plate (or a farther particle) as the ground.
fn blend(bg: [u8; 3], fg: [f64; 3], a: f64) -> [u8; 3] {
    let mix = |b: u8, f: f64| (b as f64 * (1.0 - a) + f * a).clamp(0.0, 255.0) as u8;
    [mix(bg[0], fg[0]), mix(bg[1], fg[1]), mix(bg[2], fg[2])]
}

fn luminance(c: [u8; 3]) -> f64 {
    (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64) / 255.0
}

impl ImgDust {
    pub fn new(plates: Rc<Vec<Plate>>) -> Self {
        Self {
            plates,
            current: 0,
            pending: None,
            last_bar: 0,
            zbuf: Vec::new(),
        }
    }
}

impl Effect for ImgDust {
    fn name(&self) -> &'static str {
        "IMGDUST"
    }

    fn on_key(&mut self, code: KeyCode) -> bool {
        let n = self.plates.len();
        if n == 0 {
            return false;
        }
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

        let (w, h) = (area.width as usize, area.height as usize);
        let (pw, ph) = (w, h * 2); // halfblock pixel grid
        self.zbuf.clear();
        self.zbuf.resize(pw * ph, None);

        let (cx, cy) = (pw as f64 / 2.0, ph as f64 / 2.0);
        // Focal length chosen so the undisturbed plate exactly covers the
        // grid: the backdrop below and the resting dust then agree pixel
        // for pixel, and the beat is the only thing that moves anything.
        let fit = (pw as f64 / PLATE_W).max(ph as f64 / PLATE_H);
        let focal = fit * BASE_Z;

        let img = &self.plates[self.current].img;
        let (sw, sh) = (img.width() as f64, img.height() as f64);
        let texel = |u: f64, v: f64| -> [u8; 3] {
            let sx = (u * (sw - 1.0)).clamp(0.0, sw - 1.0) as u32;
            let sy = (v * (sh - 1.0)).clamp(0.0, sh - 1.0) as u32;
            img.get_pixel(sx, sy).0
        };

        // Backdrop: the plate itself, dimmed, at infinite depth so every
        // particle wins the z-test against it and blends over it.
        for py in 0..ph {
            let v = 0.5 - (cy - (py as f64 + 0.5)) / (fit * PLATE_H);
            for px in 0..pw {
                let u = 0.5 + ((px as f64 + 0.5) - cx) / (fit * PLATE_W);
                let c = texel(u, v);
                let dim = [
                    (c[0] as f64 * PLATE_DIM) as u8,
                    (c[1] as f64 * PLATE_DIM) as u8,
                    (c[2] as f64 * PLATE_DIM) as u8,
                ];
                self.zbuf[py * pw + px] = Some((f64::INFINITY, dim));
            }
        }

        let blast = Blast {
            t: ctx.beat,
            beat: ctx.drive.gbeat(),
            bass: ctx.drive.thump,
            spread: SPREAD_MIN + SPREAD_GAIN * ctx.intensity,
        };
        let gain = 1.15 + 1.9 * blast.beat + 0.9 * blast.bass;
        let count = (pw * ph).min(MAX_PARTICLES) as u64;

        for i in 0..count {
            // Fixed random UV per particle — the plate's own texel is this
            // particle's colour and depth for the whole run.
            let u = unit_f64(hash3(i, 1, 0));
            let v = unit_f64(hash3(i, 2, 0));
            let seed = unit_f64(hash3(i, 3, 0));
            let c = texel(u, v);
            let lum = luminance(c);

            let p = particle_pos(u, v, seed, lum, blast);
            let dist = (-p[2]).max(NEAR_Z);
            let scale = focal / dist;
            let sx = cx + p[0] * scale;
            let sy = cy - p[1] * scale;
            if sx < 0.0 || sy < 0.0 || sx >= pw as f64 || sy >= ph as f64 {
                continue;
            }

            let lit = [c[0] as f64 * gain, c[1] as f64 * gain, c[2] as f64 * gain];
            let alpha = 0.35 + 0.65 * lum;
            let stamp = if dist < STAMP_Z { 1 } else { 0 };

            for oy in 0..=stamp {
                for ox in 0..=stamp {
                    let (x, y) = (sx as usize + ox, sy as usize + oy);
                    if x >= pw || y >= ph {
                        continue;
                    }
                    let idx = y * pw + x;
                    if let Some((z, _)) = self.zbuf[idx]
                        && z <= dist
                    {
                        continue;
                    }
                    let bg = self.zbuf[idx].map_or([0, 0, 0], |(_, c)| c);
                    self.zbuf[idx] = Some((dist, blend(bg, lit, alpha)));
                }
            }
        }

        // Composite the pixel grid into ▀ cells; untouched cells stay put.
        for y in 0..h {
            for x in 0..w {
                let top = self.zbuf[(y * 2) * pw + x].map(|(_, c)| c);
                let bot = self.zbuf[(y * 2 + 1) * pw + x].map(|(_, c)| c);
                if top.is_none() && bot.is_none() {
                    continue;
                }
                let t = top.unwrap_or([0, 0, 0]);
                let b = bot.unwrap_or([0, 0, 0]);
                let cell = &mut buf[(area.x + x as u16, area.y + y as u16)];
                cell.set_char('▀');
                cell.set_fg(Color::Rgb(t[0], t[1], t[2]));
                cell.set_bg(Color::Rgb(b[0], b[1], b[2]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::Drive;
    use image::RgbImage;

    /// A plate with structure in it: a colour ramp so luminance varies
    /// across the frame and the depth term actually has something to bite.
    fn plate(name: &str) -> Plate {
        let img = RgbImage::from_fn(32, 18, |x, y| {
            image::Rgb([(x * 8) as u8, (y * 14) as u8, ((x + y) * 5) as u8])
        });
        Plate {
            name: name.to_string(),
            img,
        }
    }

    fn plates(n: usize) -> Rc<Vec<Plate>> {
        Rc::new((0..n).map(|i| plate(&format!("p{i}"))).collect())
    }

    /// `decay` is the seconds since the last beat edge: ~0 is the hit,
    /// half a second later the pulse is gone.
    fn ctx(beat: f64, decay: f64) -> FrameCtx {
        let mut drive = Drive::default();
        drive.update(decay, beat, None);
        FrameCtx {
            beat,
            phase: beat.rem_euclid(1.0),
            bar_phase: beat.rem_euclid(4.0),
            intensity: 0.7,
            drive,
        }
    }

    fn render(fx: &mut ImgDust, w: u16, h: u16, c: &FrameCtx) -> Buffer {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        fx.render(&mut buf, area, c);
        buf
    }

    #[test]
    fn same_beat_same_buffer() {
        let c = ctx(8.0, 0.01);
        let mut a = ImgDust::new(plates(2));
        let mut b = ImgDust::new(plates(2));
        assert_eq!(render(&mut a, 40, 12, &c), render(&mut b, 40, 12, &c));

        // And re-rendering the same frame does not drift.
        let first = render(&mut a, 40, 12, &c);
        let second = render(&mut a, 40, 12, &c);
        assert_eq!(first, second);
    }

    #[test]
    fn writes_something() {
        let c = ctx(8.0, 0.01);
        let mut fx = ImgDust::new(plates(1));
        let buf = render(&mut fx, 40, 12, &c);
        assert!(
            buf.content().iter().any(|cell| cell.symbol() == "▀"),
            "the plate backdrop alone should have filled the area"
        );
    }

    #[test]
    fn tiny_areas_do_not_panic() {
        let c = ctx(4.0, 0.01);
        for w in 0..6u16 {
            for h in 0..6u16 {
                let mut fx = ImgDust::new(plates(1));
                let area = Rect::new(0, 0, w, h);
                let mut buf = Buffer::empty(area);
                fx.render(&mut buf, area, &c);
            }
        }
    }

    #[test]
    fn empty_plate_list_renders_nothing() {
        let c = ctx(8.0, 0.01);
        let mut fx = ImgDust::new(Rc::new(Vec::new()));
        let before = Buffer::empty(Rect::new(0, 0, 20, 6));
        let after = render(&mut fx, 20, 6, &c);
        assert_eq!(before, after);
        assert!(!fx.on_key(KeyCode::Char('n')));
        assert!(!fx.on_key(KeyCode::Char('p')));
        assert_eq!(fx.status(), None);
    }

    #[test]
    fn bigger_beat_displaces_more() {
        let blast = |beat: f64| Blast {
            t: 3.0,
            beat,
            bass: 0.0,
            spread: 1.0,
        };
        // A bright texel (lum 0.9) on the beat comes at the camera, and
        // the xy swell pushes it outward at the same time.
        let lo = particle_pos(0.2, 0.7, 0.5, 0.9, blast(0.0));
        let hi = particle_pos(0.2, 0.7, 0.5, 0.9, blast(1.0));
        assert!(
            hi[2] > lo[2],
            "z should rise toward the camera: {lo:?} {hi:?}"
        );
        assert!(hi[0].abs() > lo[0].abs(), "xy should swell: {lo:?} {hi:?}");

        // A dark texel goes the other way — that is the whole trick.
        let dark_lo = particle_pos(0.2, 0.7, 0.5, 0.05, blast(0.0));
        let dark_hi = particle_pos(0.2, 0.7, 0.5, 0.05, blast(1.0));
        assert!(dark_hi[2] < dark_lo[2], "dark pixels sink away");

        // And it shows up on screen: the hit frame differs from the lull.
        let mut a = ImgDust::new(plates(1));
        let mut b = ImgDust::new(plates(1));
        let hit = render(&mut a, 40, 12, &ctx(8.0, 0.001));
        let lull = render(&mut b, 40, 12, &ctx(8.0, 0.5));
        assert_ne!(hit, lull);
    }

    #[test]
    fn keys_queue_the_next_plate() {
        let mut fx = ImgDust::new(plates(3));
        assert_eq!(fx.status().as_deref(), Some("p0"));
        assert!(fx.on_key(KeyCode::Char('n')));
        assert_eq!(fx.status().as_deref(), Some("p0 → p1"));
        assert!(fx.on_key(KeyCode::Char('p')));
        assert_eq!(fx.status().as_deref(), Some("p0 → p0"));
        assert!(!fx.on_key(KeyCode::Char('x')));

        // The queued change lands on the next bar line, not immediately.
        fx.on_key(KeyCode::Char('n'));
        render(&mut fx, 20, 6, &ctx(0.5, 0.01));
        assert_eq!(fx.status().as_deref(), Some("p0 → p1"));
        render(&mut fx, 20, 6, &ctx(4.5, 0.01));
        assert_eq!(fx.status().as_deref(), Some("p1"));
    }
}
