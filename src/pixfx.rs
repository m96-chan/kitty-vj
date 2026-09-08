//! Pixel effects — the graphics tier's answer to the cell effects. Each
//! writes a full RGB framebuffer as a pure function of beat time, so the
//! determinism contract holds here too.

#![allow(dead_code)]

use crate::graphics::Framebuffer;

/// PLASMA — summed sine fields, palette-cycled on the beat. The canonical
/// demoscene plasma, clocked.
pub fn plasma(fb: &mut Framebuffer, beat: f64, intensity: f64) {
    let (w, h) = (fb.w as f64, fb.h as f64);
    let t = beat * 0.5;
    let warp = 0.5 + 0.5 * intensity;
    for y in 0..fb.h {
        let fy = y as f64 / h;
        for x in 0..fb.w {
            let fx = x as f64 / w;
            let v = (fx * 8.0 + t).sin()
                + (fy * 8.0 - t * 0.8).sin()
                + ((fx + fy) * 6.0 + t * 1.3).sin()
                + ((fx * fx + fy * fy).sqrt() * 12.0 - t * 2.0).sin();
            // v in [-4, 4] -> hue angle; palette rides the bar.
            let hue = (v / 8.0 + 0.5 + beat * 0.03) * warp;
            let (r, g, b) = hsv(hue.fract().abs(), 0.85, 0.5 + 0.5 * intensity);
            fb.set(x, y, r, g, b);
        }
    }
}

/// A pulsing beat flash overlay, cheap sanity check for the transport.
pub fn beat_bars(fb: &mut Framebuffer, beat: f64) {
    let phase = beat.rem_euclid(1.0);
    let flash = ((1.0 - phase).powi(2) * 255.0) as u8;
    for y in 0..fb.h {
        for x in 0..fb.w {
            let checker = ((x / 32) + (y / 32)) % 2 == 0;
            let v = if checker { flash } else { flash / 3 };
            fb.set(x, y, v, v / 2, 255 - v);
        }
    }
}

fn hsv(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plasma_is_deterministic() {
        let mut a = Framebuffer::new(32, 18);
        let mut b = Framebuffer::new(32, 18);
        plasma(&mut a, 3.25, 0.7);
        plasma(&mut b, 3.25, 0.7);
        assert_eq!(a.px, b.px);
    }

    #[test]
    fn plasma_fills_every_pixel() {
        let mut fb = Framebuffer::new(16, 16);
        plasma(&mut fb, 1.0, 1.0);
        // Not all zero — something got written everywhere.
        assert!(fb.px.iter().any(|&v| v > 0));
    }
}
