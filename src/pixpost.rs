//! Post effects — EasyPngVJ's WebGL uber-shader, ported to CPU passes
//! over the RGB framebuffer.
//!
//! Over there these were branches inside one fragment shader run across
//! the composited frame, which is why they all speak the same language:
//! normalised `uv`, a centre-relative `d = uv - 0.5`, and an `amount`
//! mixed out of the drive signals. That contract is kept — the constants
//! are the original's, so a ported look reads the same on stage — but
//! the GPU's free bilinear sampler and free ping-pong targets are not
//! free on a CPU, and that shapes the code:
//!
//! - **Scratch.** A resampling pass cannot read and write one buffer at
//!   once, so each takes a copy of the frame and samples out of it. One
//!   allocation per call, chosen over a borrowed scratch parameter: next
//!   to the per-pixel resample a full-frame memcpy is noise, and it
//!   keeps the eventual call sites free of buffer plumbing. [`Feedback`]
//!   is the exception — its previous frame has to survive *between*
//!   calls, so it owns one.
//! - **[`bloom`] runs at quarter resolution.** Deliberate, not sloppy:
//!   the target is a fanless MacBook Air, and a bright pass plus two
//!   blur passes at full res is the most expensive thing in the tier. A
//!   blur is low frequency by definition, so the detail it loses at
//!   quarter res is detail nobody could have seen in the output anyway.
//!
//! Sampling is bilinear with clamp-to-edge, as the shader's sampler was;
//! at zero displacement it lands exactly on texel centres, so a pass at
//! `amount = 0` is a true identity rather than a slow smear. Nothing
//! here is random and nothing reads a clock: same frame in, same drive,
//! same frame out.
//!
//! Wiring into the mixer is a separate step, hence the transitional dead
//! code.
#![allow(dead_code)]

use crate::drive::Drive;
use crate::graphics::Framebuffer;

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------- passes

/// WARP — barrel lens plus a ripple crawling out of the centre. The
/// original's opener, and the reason a held frame never sits still: the
/// low end swells the lens, the beat pops it. The `r*r*1.8 - 0.14` term
/// is what makes it barrel *and* pincushion at once — the negative
/// constant sucks the middle in while the corners push out, so the frame
/// breathes instead of merely growing.
///
/// `t` is the visual clock in seconds (the shader's `uTime`), `intensity`
/// is the master fader and doubles as the shader's `uAmount`. The
/// original's `uBass` is [`Drive::thump`] here — the kick measure is the
/// closest thing this rig has to a low-band level.
pub fn warp(fb: &mut Framebuffer, d: &Drive, t: f64, intensity: f64) {
    let amount = intensity;
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let src = fb.px.clone();
    let bass = d.thump;
    let beat = d.gbeat();
    for y in 0..h {
        let v0 = (y as f64 + 0.5) / h as f64;
        for x in 0..w {
            let u0 = (x as f64 + 0.5) / w as f64;
            let (dx, dy) = (u0 - 0.5, v0 - 0.5);
            let r = (dx * dx + dy * dy).sqrt();
            let lens = amount * (0.30 * bass + 0.45 * beat) * (r * r * 1.8 - 0.14);
            let ripple = (r * r * 30.0 - t * 3.2).sin() * amount * 0.025 * (0.3 + bass);
            let k = lens + ripple;
            let c = sample(&src, w, h, u0 + dx * k, v0 + dy * k);
            store(&mut fb.px, w, x, y, c);
        }
    }
}

/// KALEIDO — six-fold polar mirror, folded about the centre and turning
/// slowly under `t`. The mix rides the bar rather than the beat, which is
/// the whole trick: it sits half-open through the bar and snaps to a full
/// mirror on the bar line, so the symmetry arrives as a musical event
/// instead of a permanent gimmick.
///
/// At `t = 0` the fold is about the horizontal axis, so rows `y` and
/// `h-1-y` come out identical once the mix reaches 1.
pub fn kaleido(fb: &mut Framebuffer, d: &Drive, t: f64) {
    let amount = (0.55 + 0.45 * d.gbar()).clamp(0.0, 1.0);
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let src = fb.px.clone();
    let seg = PI / 3.0;
    for y in 0..h {
        let v0 = (y as f64 + 0.5) / h as f64;
        for x in 0..w {
            let u0 = (x as f64 + 0.5) / w as f64;
            let (dx, dy) = (u0 - 0.5, v0 - 0.5);
            let r = (dx * dx + dy * dy).sqrt();
            let a = dy.atan2(dx);
            let k = ((a + t * 0.05).rem_euclid(2.0 * seg) - seg).abs();
            let folded = sample(&src, w, h, k.cos() * r + 0.5, k.sin() * r + 0.5);
            let base = texel(&src, w, x, y);
            store(&mut fb.px, w, x, y, mix(base, folded, amount));
        }
    }
}

/// ZOOMBLUR — eight-tap radial smear dragged out of the centre. This is
/// the kick's lunge: `thump` weighs nearly as much as the grid pulse, so
/// the frame appears to jump at the viewer on every kick and settles
/// between them. Taps are weighted down towards the far end so the
/// undisplaced frame still dominates.
pub fn zoomblur(fb: &mut Framebuffer, d: &Drive, intensity: f64) {
    let amount = (0.25 + 1.5 * d.gbeat() + 1.7 * d.thump).clamp(0.0, 2.4) * intensity;
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let src = fb.px.clone();
    for y in 0..h {
        let v0 = (y as f64 + 0.5) / h as f64;
        for x in 0..w {
            let u0 = (x as f64 + 0.5) / w as f64;
            let dirx = (u0 - 0.5) * amount * 0.055;
            let diry = (v0 - 0.5) * amount * 0.055;
            let mut sum = [0.0f64; 3];
            let mut wsum = 0.0;
            for i in 0..8 {
                let f = i as f64 / 7.0;
                let tw = 1.0 - f * 0.72;
                let c = sample(&src, w, h, u0 - dirx * f, v0 - diry * f);
                sum[0] += c[0] * tw;
                sum[1] += c[1] * tw;
                sum[2] += c[2] * tw;
                wsum += tw;
            }
            let inv = 1.0 / wsum;
            store(
                &mut fb.px,
                w,
                x,
                y,
                [sum[0] * inv, sum[1] * inv, sum[2] * inv],
            );
        }
    }
}

/// RGB SPLIT — chromatic aberration, and note it is *radial*, not the
/// lateral shift most ports settle for: the offset runs along `d`, so the
/// fringing grows towards the corners and vanishes at the centre. That is
/// what makes it read as a lens rather than as a broken cable. The onset
/// flinch is in the amount, so it fires between the beats too.
pub fn rgb_split(fb: &mut Framebuffer, d: &Drive, intensity: f64) {
    let beat = d.gbeat();
    let amount = (0.25 + 1.0 * beat + 0.55 * d.hit).clamp(0.0, 1.6) * intensity;
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let src = fb.px.clone();
    let spread = amount * (0.010 + 0.045 * beat);
    for y in 0..h {
        let v0 = (y as f64 + 0.5) / h as f64;
        for x in 0..w {
            let u0 = (x as f64 + 0.5) / w as f64;
            let ox = (u0 - 0.5) * spread;
            let oy = (v0 - 0.5) * spread;
            let r = sample(&src, w, h, u0 + ox, v0 + oy)[0];
            // Green stays put — a sample at uv is the texel itself.
            let g = texel(&src, w, x, y)[1];
            let b = sample(&src, w, h, u0 - ox, v0 - oy)[2];
            store(&mut fb.px, w, x, y, [r, g, b]);
        }
    }
}

/// EDGE — 3x3 Sobel on luminance, drawn back over a darkened copy of the
/// frame. The original used it as the "outline" look: knock the picture
/// down, then paint the gradient in the scene's two accent colours, which
/// is why the accents are parameters rather than constants. The `pow(g,
/// 1.25)` is a gamma on the gradient — it starves the weak edges so film
/// grain and dithering don't light the whole frame up.
pub fn edge(fb: &mut Framebuffer, d: &Drive, accent_a: (u8, u8, u8), accent_b: (u8, u8, u8)) {
    let amount = (0.55 + 0.45 * d.gbeat()).clamp(0.0, 1.0);
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let src = fb.px.clone();
    let ca = rgb01(accent_a);
    let cb = rgb01(accent_b);
    for y in 0..h {
        for x in 0..w {
            let l = |ox: i64, oy: i64| {
                let sx = (x as i64 + ox).clamp(0, w as i64 - 1) as u32;
                let sy = (y as i64 + oy).clamp(0, h as i64 - 1) as u32;
                luma(texel(&src, w, sx, sy))
            };
            let gx = -l(-1, -1) - 2.0 * l(-1, 0) - l(-1, 1) + l(1, -1) + 2.0 * l(1, 0) + l(1, 1);
            let gy = -l(-1, -1) - 2.0 * l(0, -1) - l(1, -1) + l(-1, 1) + 2.0 * l(0, 1) + l(1, 1);
            let g = ((gx * gx + gy * gy).sqrt() * 2.0).clamp(0.0, 1.0);
            let base = texel(&src, w, x, y);
            let dark = mix(base, scale(base, 0.45), amount * 0.5);
            let acc = mix(ca, cb, g);
            let k = g.powf(1.25) * amount * 1.4;
            let out = [
                dark[0] + acc[0] * k,
                dark[1] + acc[1] * k,
                dark[2] + acc[2] * k,
            ];
            store(&mut fb.px, w, x, y, out);
        }
    }
}

/// FEEDBACK — the frame buffer looking at itself. The one pass that
/// cannot be a pure function of the current frame, so it owns the
/// previous one: on the GPU this was a ping-pong pair of render targets,
/// here it is a second [`Framebuffer`] that outlives the call.
///
/// Two details carry the whole look. The previous frame is sampled
/// *zoomed in* by a fraction of a percent, which is what turns a plain
/// trail into the tunnel of ghosts crawling outward. And the composite is
/// `max`, not `mix` — so trails never dim what is in front of them, they
/// only ever add, and a bright frame wipes its own history clean.
pub struct Feedback {
    prev: Framebuffer,
}

impl Feedback {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            prev: Framebuffer::new(w, h),
        }
    }

    /// Blend the stored frame into `fb`, then keep the result as the new
    /// history. A resize drops the history rather than smearing a stale
    /// one across the new geometry.
    pub fn apply(&mut self, fb: &mut Framebuffer, d: &Drive, intensity: f64) {
        if fb.w == 0 || fb.h == 0 {
            return;
        }
        if self.prev.w != fb.w || self.prev.h != fb.h {
            self.prev.resize(fb.w, fb.h);
            // `resize` leaves contents undefined; a trail must not start
            // out of whatever was in the old buffer.
            self.prev.px.fill(0);
        }
        let amount = (0.45 + 0.55 * d.thump).clamp(0.0, 1.0) * intensity;
        if amount > 0.0 {
            let (w, h) = (fb.w, fb.h);
            let gain = 0.70 + 0.26 * amount;
            let zoom = 1.0 - 0.008 * amount;
            for y in 0..h {
                let v0 = (y as f64 + 0.5) / h as f64;
                for x in 0..w {
                    let u0 = (x as f64 + 0.5) / w as f64;
                    let p = sample(
                        &self.prev.px,
                        w,
                        h,
                        (u0 - 0.5) * zoom + 0.5,
                        (v0 - 0.5) * zoom + 0.5,
                    );
                    let c = texel(&fb.px, w, x, y);
                    let out = [
                        c[0].max(p[0] * gain),
                        c[1].max(p[1] * gain),
                        c[2].max(p[2] * gain),
                    ];
                    store(&mut fb.px, w, x, y, out);
                }
            }
        }
        self.prev.px.copy_from_slice(&fb.px);
    }
}

/// BLOOM — bright pass, separable blur, added back on top. The threshold
/// runs on max-channel luminance rather than a weighted luma so a
/// saturated primary blooms as readily as a white blowout; dividing the
/// excess by the level keeps the bloom the colour of the source instead
/// of pulling it towards white.
///
/// Everything but the final composite happens at quarter resolution —
/// see the module note. Two separable 5-tap passes (horizontal, then
/// vertical) stand in for the original's 25-tap kernel, and the weights
/// are the original's, normalised because they do not sum to 1.
pub fn bloom(fb: &mut Framebuffer, d: &Drive, intensity: f64) {
    let amount =
        ((0.42 + 0.55 * d.gbeat() + 0.30 * d.hit + 0.60 * d.thump) * intensity).clamp(0.0, 1.6);
    if fb.w == 0 || fb.h == 0 || amount <= 0.0 {
        return;
    }
    let (w, h) = (fb.w, fb.h);
    let qw = w.div_ceil(4).max(1);
    let qh = h.div_ceil(4).max(1);

    // Downsample by box average, thresholding as we go.
    let mut a = vec![0.0f64; (qw as usize) * (qh as usize) * 3];
    for qy in 0..qh {
        for qx in 0..qw {
            let mut acc = [0.0f64; 3];
            let mut n = 0.0;
            for sy in (qy * 4)..((qy + 1) * 4).min(h) {
                for sx in (qx * 4)..((qx + 1) * 4).min(w) {
                    let c = texel(&fb.px, w, sx, sy);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                    n += 1.0;
                }
            }
            // At least one source texel always lands in a block: qh is
            // ceil(h/4), so (qh-1)*4 < h.
            let c = [acc[0] / n, acc[1] / n, acc[2] / n];
            let l = c[0].max(c[1]).max(c[2]);
            let k = if l > 0.0 {
                (l - 0.62).max(0.0) / l
            } else {
                0.0
            };
            let i = ((qy * qw + qx) * 3) as usize;
            a[i] = c[0] * k;
            a[i + 1] = c[1] * k;
            a[i + 2] = c[2] * k;
        }
    }

    let mut b = vec![0.0f64; a.len()];
    blur_pass(&a, &mut b, qw, qh, true);
    blur_pass(&b, &mut a, qw, qh, false);

    for y in 0..h {
        let v0 = (y as f64 + 0.5) / h as f64;
        for x in 0..w {
            let u0 = (x as f64 + 0.5) / w as f64;
            let bl = bilinear(qw, qh, u0, v0, |bx, by| {
                let i = ((by * qw + bx) * 3) as usize;
                [a[i], a[i + 1], a[i + 2]]
            });
            let c = texel(&fb.px, w, x, y);
            let out = [
                c[0] + bl[0] * amount,
                c[1] + bl[1] * amount,
                c[2] + bl[2] * amount,
            ];
            store(&mut fb.px, w, x, y, out);
        }
    }
}

/// CRT — scanlines, every second device row knocked down. Trivial, and
/// carrying more weight than it deserves: it is what stops the pixel tier
/// looking like a screenshot next to the cell tier.
pub fn crt(fb: &mut Framebuffer, amount: f64) {
    if fb.w == 0 || fb.h == 0 {
        return;
    }
    let k = 1.0 - 0.38 * amount.clamp(0.0, 1.0);
    let w = fb.w;
    for y in (1..fb.h).step_by(2) {
        let row = (y * w * 3) as usize;
        for p in fb.px[row..row + (w * 3) as usize].iter_mut() {
            *p = (*p as f64 * k + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

// --------------------------------------------------------------- helpers

/// One 5-tap separable Gaussian pass, clamp-to-edge. Weights are the
/// original's; they sum to 0.9108, so normalise or the bloom would fade
/// on every pass.
fn blur_pass(src: &[f64], dst: &mut [f64], w: u32, h: u32, horizontal: bool) {
    const WEIGHTS: [f64; 5] = [0.0703, 0.2270, 0.3162, 0.2270, 0.0703];
    let norm: f64 = 1.0 / WEIGHTS.iter().sum::<f64>();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f64; 3];
            for (i, tw) in WEIGHTS.iter().enumerate() {
                let o = i as i64 - 2;
                let (sx, sy) = if horizontal {
                    ((x as i64 + o).clamp(0, w as i64 - 1) as u32, y)
                } else {
                    (x, (y as i64 + o).clamp(0, h as i64 - 1) as u32)
                };
                let j = ((sy * w + sx) * 3) as usize;
                acc[0] += src[j] * tw;
                acc[1] += src[j + 1] * tw;
                acc[2] += src[j + 2] * tw;
            }
            let i = ((y * w + x) * 3) as usize;
            dst[i] = acc[0] * norm;
            dst[i + 1] = acc[1] * norm;
            dst[i + 2] = acc[2] * norm;
        }
    }
}

/// Bilinear fetch in normalised uv with clamp-to-edge, over any source
/// addressed by texel. Landing exactly on a texel centre returns that
/// texel untouched, which is what makes the zero-amount identities exact.
fn bilinear(w: u32, h: u32, u: f64, v: f64, fetch: impl Fn(u32, u32) -> [f64; 3]) -> [f64; 3] {
    if w == 0 || h == 0 {
        return [0.0; 3];
    }
    let fx = (u * w as f64 - 0.5).clamp(0.0, (w - 1) as f64);
    let fy = (v * h as f64 - 0.5).clamp(0.0, (h - 1) as f64);
    let (bx, by) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - bx, fy - by);
    // Saturating casts: a NaN uv lands on texel 0 rather than panicking.
    let x0 = (bx as u32).min(w - 1);
    let y0 = (by as u32).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (c00, c10) = (fetch(x0, y0), fetch(x1, y0));
    let (c01, c11) = (fetch(x0, y1), fetch(x1, y1));
    let mut out = [0.0f64; 3];
    for k in 0..3 {
        let top = c00[k] + (c10[k] - c00[k]) * tx;
        let bot = c01[k] + (c11[k] - c01[k]) * tx;
        out[k] = top + (bot - top) * ty;
    }
    out
}

/// Bilinear sample of an RGB byte buffer, returning 0..1 components.
fn sample(src: &[u8], w: u32, h: u32, u: f64, v: f64) -> [f64; 3] {
    bilinear(w, h, u, v, |x, y| texel(src, w, x, y))
}

/// One texel as 0..1 components. The GLSL constants are all in that
/// space, so the passes work there and only convert on the way out.
#[inline]
fn texel(src: &[u8], w: u32, x: u32, y: u32) -> [f64; 3] {
    let i = ((y * w + x) * 3) as usize;
    [
        src[i] as f64 / 255.0,
        src[i + 1] as f64 / 255.0,
        src[i + 2] as f64 / 255.0,
    ]
}

#[inline]
fn store(dst: &mut [u8], w: u32, x: u32, y: u32, c: [f64; 3]) {
    let i = ((y * w + x) * 3) as usize;
    dst[i] = to_u8(c[0]);
    dst[i + 1] = to_u8(c[1]);
    dst[i + 2] = to_u8(c[2]);
}

#[inline]
fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn rgb01(c: (u8, u8, u8)) -> [f64; 3] {
    [c.0 as f64 / 255.0, c.1 as f64 / 255.0, c.2 as f64 / 255.0]
}

#[inline]
fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[inline]
fn scale(c: [f64; 3], k: f64) -> [f64; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

#[inline]
fn luma(c: [f64; 3]) -> f64 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, deliberately asymmetric test picture.
    fn noise(w: u32, h: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = |c: u64| {
                    (crate::rng::unit_f64(crate::rng::hash3(x as u64, y as u64, c)) * 255.0) as u8
                };
                fb.set(x, y, v(0), v(1), v(2));
            }
        }
        fb
    }

    fn drive(beat: f64, bar: f64, hit: f64, thump: f64) -> Drive {
        let mut d = Drive::default();
        d.beat = beat;
        d.bar = bar;
        d.hit = hit;
        d.thump = thump;
        d.groove = 1.0;
        d
    }

    #[test]
    fn resampler_hits_texel_centres_exactly() {
        // If this drifts, the zero-amount identity tests below are vacuous.
        let fb = noise(9, 5);
        for y in 0..5 {
            for x in 0..9 {
                let u = (x as f64 + 0.5) / 9.0;
                let v = (y as f64 + 0.5) / 5.0;
                let c = sample(&fb.px, 9, 5, u, v);
                assert_eq!(
                    [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])],
                    {
                        let t = texel(&fb.px, 9, x, y);
                        [to_u8(t[0]), to_u8(t[1]), to_u8(t[2])]
                    },
                    "at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn warp_is_identity_at_zero_intensity() {
        let mut fb = noise(24, 16);
        let before = fb.px.clone();
        warp(&mut fb, &drive(1.0, 1.0, 1.0, 1.0), 3.7, 0.0);
        assert_eq!(fb.px, before);
    }

    #[test]
    fn zoomblur_is_identity_at_zero_intensity() {
        let mut fb = noise(24, 16);
        let before = fb.px.clone();
        zoomblur(&mut fb, &drive(1.0, 1.0, 1.0, 1.0), 0.0);
        assert_eq!(fb.px, before);
    }

    #[test]
    fn rgb_split_is_identity_at_zero_intensity() {
        let mut fb = noise(24, 16);
        let before = fb.px.clone();
        rgb_split(&mut fb, &drive(1.0, 1.0, 1.0, 1.0), 0.0);
        assert_eq!(fb.px, before);
    }

    #[test]
    fn warp_moves_pixels_when_driven() {
        let mut fb = noise(24, 16);
        let before = fb.px.clone();
        warp(&mut fb, &drive(1.0, 0.0, 0.0, 1.0), 0.0, 1.0);
        assert_ne!(fb.px, before);
    }

    #[test]
    fn warp_is_deterministic() {
        let (mut a, mut b) = (noise(20, 12), noise(20, 12));
        let d = drive(0.6, 0.2, 0.3, 0.4);
        warp(&mut a, &d, 2.5, 0.8);
        warp(&mut b, &d, 2.5, 0.8);
        assert_eq!(a.px, b.px);
    }

    #[test]
    fn kaleido_mirrors_about_the_horizontal_axis() {
        // Full mix (bar line, full groove) at t = 0: the six-fold fold is
        // symmetric about y = 0.5, so row y must equal row h-1-y.
        let mut fb = noise(24, 16);
        kaleido(&mut fb, &drive(0.0, 1.0, 0.0, 0.0), 0.0);
        let row = |y: u32| {
            let i = (y * 24 * 3) as usize;
            fb.px[i..i + 24 * 3].to_vec()
        };
        for y in 0..16 {
            assert_eq!(row(y), row(15 - y), "row {y} not mirrored");
        }
    }

    #[test]
    fn kaleido_partial_mix_keeps_some_of_the_original() {
        // Off the bar the mix is 0.55, so the frame must be neither the
        // original nor a perfect mirror.
        let mut fb = noise(24, 16);
        let before = fb.px.clone();
        kaleido(&mut fb, &drive(0.0, 0.0, 0.0, 0.0), 0.0);
        assert_ne!(fb.px, before);
        let mirrored = (0..16).all(|y: u32| {
            let a = ((y * 24) * 3) as usize;
            let b = (((15 - y) * 24) * 3) as usize;
            fb.px[a..a + 72] == fb.px[b..b + 72]
        });
        assert!(!mirrored, "partial mix should not be fully symmetric");
    }

    #[test]
    fn edge_finds_an_edge() {
        // Left half black, right half white; accents both red so the
        // gradient shows up in one channel only.
        let mut fb = Framebuffer::new(16, 16);
        for y in 0..16 {
            for x in 8..16 {
                fb.set(x, y, 255, 255, 255);
            }
        }
        edge(&mut fb, &Drive::default(), (255, 0, 0), (255, 0, 0));
        let at = |x: u32, y: u32| {
            let i = ((y * 16 + x) * 3) as usize;
            (fb.px[i], fb.px[i + 1], fb.px[i + 2])
        };
        // Deep in the black field: no gradient, nothing added.
        assert_eq!(at(2, 8), (0, 0, 0));
        // On the boundary: strong accent.
        assert!(at(7, 8).0 > 150, "boundary red {:?}", at(7, 8));
        assert_eq!(at(7, 8).1, 0, "accent is red only");
        // Deep in the white field: darkened base, no accent, so all three
        // channels stay equal and below 255.
        let w = at(13, 8);
        assert!(w.0 == w.1 && w.1 == w.2 && w.0 < 255, "white field {w:?}");
    }

    #[test]
    fn bloom_leaves_black_black() {
        let mut fb = Framebuffer::new(20, 20);
        bloom(&mut fb, &drive(1.0, 1.0, 1.0, 1.0), 1.0);
        assert!(fb.px.iter().all(|&p| p == 0));
    }

    #[test]
    fn bloom_spreads_a_bright_spot() {
        let mut fb = Framebuffer::new(32, 32);
        for y in 8..16 {
            for x in 8..16 {
                fb.set(x, y, 255, 255, 255);
            }
        }
        let before = fb.px.clone();
        bloom(&mut fb, &Drive::default(), 1.0);
        let at = |px: &[u8], x: u32, y: u32| px[((y * 32 + x) * 3) as usize];
        // Outside the block, where there was nothing, there is now glow.
        assert_eq!(at(&before, 6, 12), 0);
        assert!(at(&fb.px, 6, 12) > 0, "no glow beside the spot");
        // And the frame as a whole gained energy.
        let sum = |px: &[u8]| px.iter().map(|&p| p as u64).sum::<u64>();
        assert!(sum(&fb.px) > sum(&before));
    }

    #[test]
    fn bloom_ignores_a_dim_frame() {
        // Below the 0.62 threshold there is nothing to bloom.
        let mut fb = Framebuffer::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                fb.set(x, y, 100, 100, 100);
            }
        }
        let before = fb.px.clone();
        bloom(&mut fb, &drive(1.0, 1.0, 1.0, 1.0), 1.0);
        assert_eq!(fb.px, before);
    }

    #[test]
    fn feedback_leaves_a_decaying_trail() {
        let mut fx = Feedback::new(8, 8);
        let d = Drive::default();
        let mut fb = Framebuffer::new(8, 8);
        fb.px.fill(255);
        fx.apply(&mut fb, &d, 1.0);
        assert!(fb.px.iter().all(|&p| p == 255), "bright frame survives");

        let peak = |fb: &Framebuffer| *fb.px.iter().max().unwrap();
        let mut trail = Vec::new();
        for _ in 0..4 {
            fb.px.fill(0);
            fx.apply(&mut fb, &d, 1.0);
            trail.push(peak(&fb));
        }
        assert!(trail[0] > 0, "black frame should keep a ghost");
        assert!(trail[0] < 255, "the ghost must be dimmer than the source");
        for w in trail.windows(2) {
            assert!(w[1] < w[0], "trail must decay: {trail:?}");
        }
    }

    #[test]
    fn feedback_forgets_on_resize() {
        let mut fx = Feedback::new(8, 8);
        let mut fb = Framebuffer::new(8, 8);
        fb.px.fill(255);
        fx.apply(&mut fb, &Drive::default(), 1.0);
        let mut small = Framebuffer::new(4, 4);
        fx.apply(&mut small, &Drive::default(), 1.0);
        assert!(small.px.iter().all(|&p| p == 0), "stale history leaked");
    }

    #[test]
    fn crt_darkens_every_second_row() {
        let mut fb = Framebuffer::new(4, 4);
        fb.px.fill(200);
        crt(&mut fb, 1.0);
        let row = |y: u32| fb.px[(y * 4 * 3) as usize];
        assert_eq!(row(0), 200);
        assert_eq!(row(2), 200);
        assert_eq!(row(1), (200.0 * (1.0 - 0.38) + 0.5) as u8);
        assert_eq!(row(3), row(1));
    }

    #[test]
    fn tiny_and_empty_framebuffers_survive() {
        for (w, h) in [(1, 1), (0, 0), (1, 7), (7, 1), (0, 4)] {
            let mut fb = Framebuffer::new(w, h);
            if !fb.px.is_empty() {
                fb.px.fill(180);
            }
            let d = drive(1.0, 1.0, 1.0, 1.0);
            let mut fx = Feedback::new(w, h);
            warp(&mut fb, &d, 1.25, 1.0);
            kaleido(&mut fb, &d, 1.25);
            zoomblur(&mut fb, &d, 1.0);
            rgb_split(&mut fb, &d, 1.0);
            edge(&mut fb, &d, (255, 0, 128), (0, 200, 255));
            fx.apply(&mut fb, &d, 1.0);
            bloom(&mut fb, &d, 1.0);
            crt(&mut fb, 1.0);
            assert_eq!(fb.px.len(), (w * h * 3) as usize);
        }
    }
}
