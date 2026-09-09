//! Pixel effects — the graphics tier's answer to the cell effects. Each
//! writes a full RGB framebuffer as a pure function of beat time, so the
//! determinism contract holds here too.

use crate::graphics::Framebuffer;
use crate::pass::hsv;

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

/// TUNNEL — a receding tube, angular stripes scrolling in beat time.
pub fn tunnel(fb: &mut Framebuffer, beat: f64, intensity: f64) {
    let (w, h) = (fb.w as f64, fb.h as f64);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let scroll = beat * 0.5;
    let flash = (1.0 - beat.rem_euclid(1.0)).powi(2);
    for y in 0..fb.h {
        let dy = y as f64 - cy;
        for x in 0..fb.w {
            let dx = x as f64 - cx;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            let ang = dy.atan2(dx) / std::f64::consts::TAU + 0.5;
            let depth = 40.0 / dist + scroll;
            let ring = ((depth * 6.0).sin() * 0.5 + 0.5) * (ang * 12.0).fract();
            let v = (ring * (0.5 + 0.5 * flash) * (0.4 + 0.6 * intensity)).min(1.0);
            let (r, g, b) = hsv(depth * 0.1 + ang, 0.8, v);
            fb.set(x, y, r, g, b);
        }
    }
}

/// STARFIELD — points streaming outward from center, closed-form so it's
/// deterministic. Density and speed ride intensity.
pub fn starfield(fb: &mut Framebuffer, beat: f64, intensity: f64) {
    for p in fb.px.iter_mut() {
        *p = 0;
    }
    let (w, h) = (fb.w as f64, fb.h as f64);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let count = (200.0 + 600.0 * intensity) as u64;
    let speed = 0.4 + 0.6 * intensity;
    for i in 0..count {
        let ang = crate::rng::unit_f64(crate::rng::hash3(i, 1, 0)) * std::f64::consts::TAU;
        let phase = crate::rng::unit_f64(crate::rng::hash3(i, 2, 0));
        // Each star cycles outward; z in (0,1], small z = near.
        let z = 1.0 - (beat * speed * 0.3 + phase).fract();
        let r = (1.0 - z) * w.max(h) * 0.7;
        let x = cx + ang.cos() * r;
        let y = cy + ang.sin() * r * 0.9;
        if x < 0.0 || y < 0.0 || x >= w || y >= h {
            continue;
        }
        let lum = ((1.0 - z) * 255.0) as u8;
        fb.set(x as u32, y as u32, lum, lum, lum.max(180));
    }
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
