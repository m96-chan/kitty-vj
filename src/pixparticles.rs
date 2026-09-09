//! Pixel particles — EasyPngVJ's GPU point modes, rasterised on the CPU.
//!
//! Over there each of these was a `THREE.Points` / `LineSegments` scene
//! drawn with `blending: AdditiveBlending`: a dust box wrapped around the
//! camera, a cylinder of points streaming past, a synthwave floor, and a
//! shockwave ring fired on the bar. The vertex shaders read the same drive
//! uniforms the rest of the rig runs on, so the field expands on the bar
//! and the points swell on the kick.
//!
//! Ported the way [`crate::pixfx::starfield`] is: **every position is a
//! pure function of beat time and a stable per-point hash**. No velocity
//! integration, no event pool, no per-frame state at all. That is what
//! keeps a backspin or a jog scrub honest — drag the beat backwards and
//! the field runs backwards with it, instead of a simulation that has
//! already eaten the frames and can't give them back.
//!
//! **These accumulate.** The originals composited `lighter`, and so do
//! these: every point is a saturating add into whatever is already in the
//! framebuffer, never a write. Call [`clear`] first for a single mode, or
//! stack several and get the one additive scene the original drew.

// Wired into the mixer separately; the entry points are dead until then.
#![allow(dead_code)]

use std::f64::consts::TAU;

use crate::drive::Drive;
use crate::graphics::Framebuffer;
use crate::rng::{hash3, unit_f64};

/// Camera sits here on +Z looking down -Z, as in the original scene.
const CAM_Z: f64 = 420.0;
/// Anything nearer than this is behind us; drop it rather than divide by
/// a depth heading for zero.
const NEAR: f64 = 40.0;
/// Focal length, as a fraction of framebuffer height — about an 84°
/// vertical field of view. Deliberately wide: these fields are built at a
/// scale that encloses the camera, and a normal lens on a pane a few
/// hundred pixels across frames a keyhole of them — the dust box lands
/// entirely outside the frame on the bar, when it should be filling it.
/// At this focal the far wall of the box, the tunnel mouth and the floor's
/// far edge all sit just inside the frame, with the horizon a touch below
/// the centre line.
const FOCAL_FRAC: f64 = 0.55;
/// Widest a point splat gets, in pixels. The originals let `gl_PointSize`
/// run; we cap it because a near point is otherwise a full-screen quad
/// rasterised in scalar Rust.
const MAX_POINT: f64 = 6.0;

/// The original clocked on wall seconds. We clock on beats, with 120 BPM
/// as the reference tempo, so drift and swirl stay tempo-locked instead
/// of running away from the music.
const SEC_PER_BEAT: f64 = 0.5;

/// Point budgets. Density scales with framebuffer area — one point per
/// this many pixels — and then hits a hard cap, because these run every
/// frame at 60fps on a fan-less laptop. The caps are what a full-screen
/// 1200x700 frame lands on; below that the count falls with the area, so
/// a small pane is cheap rather than absurdly dense.
const PX_PER_SPARK: u64 = 300;
const SPARK_CAP: u64 = 3000;
const PX_PER_TUNNEL: u64 = 350;
const TUNNEL_CAP: u64 = 2500;
/// Floor on the count: below this a field stops reading as a field, and
/// a hundred-odd points costs nothing at any size.
const MIN_POINTS: u64 = 120;
/// Floor: the original's fixed 61x28 lines, as a maximum.
const GRID_COLS: usize = 61;
const GRID_ROWS: usize = 28;
const CELL: f64 = 190.0;
const FLOOR_Y: f64 = -400.0;
/// Rings alive at once. See [`rings`] for why this is a scan, not a pool.
const RING_POOL: i64 = 5;
/// The original's 1.2 s ring life, in beats at the reference tempo.
const RING_LIFE: f64 = 2.4;
/// One grid cell of floor travel per beat — the floor steps in time.
const GRID_SCROLL_PER_BEAT: f64 = CELL;
/// The tunnel's depth cycle, and one full pass of it per bar.
const TUNNEL_CYCLE: f64 = 1800.0;
const TUNNEL_SCROLL_PER_BEAT: f64 = TUNNEL_CYCLE / 4.0;

/// Wipe to black. These modes add, so the caller decides when a frame
/// starts.
pub fn clear(fb: &mut Framebuffer) {
    fb.px.fill(0);
}

/// SPARKS — a dust box enclosing the camera, swirling about the view
/// axis. Points sit on the surface of a box the camera is inside, so
/// there is always dust in frame and always dust behind you. The bar is
/// the loud one here: `gbar` dominates the burst term, so the whole field
/// shoves outward on the bar line and settles back over the next beat,
/// while the kick swells the point size.
pub fn sparks(
    fb: &mut Framebuffer,
    beat: f64,
    intensity: f64,
    drive: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    let Some(view) = View::of(fb) else { return };
    let t = secs(beat);
    let pulse = drive.gbeat();
    let bass = drive.thump;
    let swell = 1.0 + (130.0 * drive.gbar() + 60.0 * pulse + 190.0 * drive.thump) * 0.0022;
    let gain = 0.35 + 0.65 * intensity;
    let level = gain * (0.45 + 1.9 * pulse + 1.2 * bass);

    for i in 0..budget(fb, PX_PER_SPARK, SPARK_CAP) {
        let (dx, dy, dz) = box_dir(i);
        let seed = unit_f64(hash3(i, 1, 0));
        let arad = unit_f64(hash3(i, 2, 0));

        let (mut x, mut y) = (dx * 1500.0, dy * 950.0);
        let mut z = dz * 1100.0 - 350.0;
        let (s, c) = (t * (0.06 + 0.10 * arad)).sin_cos();
        (x, y) = (x * c - y * s, x * s + y * c);
        y += 30.0 * (t * 0.7 + seed * 17.0).sin();
        x *= swell;
        y *= swell;
        z *= swell;

        let Some((sx, sy, depth)) = view.project(x, y, z) else {
            continue;
        };
        let fade = (1.0 - depth / 2200.0).clamp(0.0, 1.0);
        if fade <= 0.0 {
            continue;
        }
        let size = (0.5 + seed * 1.8) * (1.0 + 1.8 * pulse) * (700.0 / depth);
        let (r, g, b) = mix(ca, cb, seed);
        let k = fade * level;
        splat(fb, sx, sy, size, (r * k, g * k, b * k));
    }
}

/// TUNNEL — a cylinder of points streaming past the camera, radius biased
/// outward so the wall reads as a wall and not a haze. The beat brightens
/// the whole tube; the kick pushes the wall out.
///
/// The original accumulated `scroll` per frame. We derive it from the
/// beat instead — an accumulator is simulation state, and state is
/// exactly what desyncs when the jog wheel drags time backwards. One full
/// depth cycle per bar, so the tube arrives on the bar line.
pub fn tunnel_px(
    fb: &mut Framebuffer,
    beat: f64,
    intensity: f64,
    drive: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    let Some(view) = View::of(fb) else { return };
    let t = secs(beat);
    let pulse = drive.gbeat();
    let scroll = beat * TUNNEL_SCROLL_PER_BEAT;
    let gain = 0.35 + 0.65 * intensity;
    let beat_gain = gain * (0.65 + 1.2 * pulse);

    for i in 0..budget(fb, PX_PER_TUNNEL, TUNNEL_CAP) {
        let radn = unit_f64(hash3(i, 12, 0)).powf(0.4);
        let rad = 130.0 + 300.0 * radn + 70.0 * drive.thump;
        let z0 = unit_f64(hash3(i, 13, 0)) * TUNNEL_CYCLE;
        let z = (z0 + scroll).rem_euclid(TUNNEL_CYCLE) - 1700.0;
        let (s, c) = (unit_f64(hash3(i, 11, 0)) * TAU + t * 0.14).sin_cos();

        let Some((sx, sy, depth)) = view.project(c * rad, s * rad, z) else {
            continue;
        };
        let near = (1.0 - depth / 1700.0).clamp(0.0, 1.0);
        if near <= 0.0 {
            continue;
        }
        let seed = unit_f64(hash3(i, 14, 0));
        let size = (0.6 + 1.6 * seed) * (1.0 + 1.2 * pulse) * (700.0 / depth);
        let (r, g, b) = mix(ca, cb, radn);
        let k = beat_gain * (0.25 + 2.1 * near * near);
        splat(fb, sx, sy, size, (r * k, g * k, b * k));
    }
}

/// GRID — the synthwave floor: a lit lattice under the camera receding to
/// a horizon, scrolling one cell per beat so the lines step in time. Near
/// lines are hot and accent-A coloured, far ones cool into accent B and
/// fade out before the horizon, which is what sells the depth.
///
/// Scroll is derived from the beat for the same reason the tunnel's is.
/// The floor is periodic in one cell, so only the fractional cell matters
/// and the whole thing stays closed-form.
pub fn grid_floor(
    fb: &mut Framebuffer,
    beat: f64,
    intensity: f64,
    drive: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    let Some(view) = View::of(fb) else { return };
    let pulse = drive.gbeat();
    let gain = (0.35 + 0.65 * intensity) * (0.7 + 1.1 * pulse);
    let (cols, rows) = grid_extent(fb);
    let shift = (beat * GRID_SCROLL_PER_BEAT).rem_euclid(CELL);
    let z_at = |r: usize| r as f64 * CELL + shift - CELL * (rows - 1) as f64;
    let half = CELL * ((cols - 1) / 2) as f64;

    // Colour is a pure function of depth, so a whole line at fixed z is
    // one flat segment; the lines running away from us get subdivided per
    // cell so they shade along their length.
    let shade = |z: f64| -> Option<(f64, f64, f64)> {
        let near = (1.0 - (CAM_Z - z) / 3000.0).clamp(0.0, 1.0);
        if near <= 0.0 {
            return None;
        }
        let (r, g, b) = mix(cb, ca, near);
        let k = gain * (0.35 + 2.2 * near);
        Some((r * k, g * k, b * k))
    };

    for c in 0..cols {
        let x = -half + c as f64 * CELL;
        for r in 0..rows.saturating_sub(1) {
            let (z0, z1) = (z_at(r), z_at(r + 1));
            let Some(col) = shade((z0 + z1) * 0.5) else {
                continue;
            };
            add_seg(fb, &view, (x, FLOOR_Y, z0), (x, FLOOR_Y, z1), col);
        }
    }
    for r in 0..rows {
        let z = z_at(r);
        let Some(col) = shade(z) else { continue };
        add_seg(fb, &view, (-half, FLOOR_Y, z), (half, FLOOR_Y, z), col);
    }
}

/// RINGS — an expanding shockwave fired on every bar line, decelerating
/// out and fading as it goes, until it washes past the camera.
///
/// The original kept a pool of five ring objects and fired the next free
/// one on the bar — mutable events, written once and then animated. That
/// cannot survive time running backwards: scrub back over a bar line and
/// the pool has no memory of un-firing. So we invert it. Nothing is
/// stored; we look at the last [`RING_POOL`] bar lines behind the current
/// beat and give each one an age of *now minus that bar line*. Same five
/// slots, same look, but the state lives in the clock where the jog wheel
/// can reach it. At the reference tempo the 1.2 s life is shorter than a
/// bar, so in practice one ring is alive and the other slots cost a
/// compare each.
pub fn rings(
    fb: &mut Framebuffer,
    beat: f64,
    intensity: f64,
    drive: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    let Some(view) = View::of(fb) else { return };
    let gain = (0.35 + 0.65 * intensity) * (0.7 + 0.8 * drive.gbeat());
    let latest = (beat / 4.0).floor() as i64;

    for k in 0..RING_POOL {
        let age = beat - (latest - k) as f64 * 4.0;
        if age < 0.0 {
            continue;
        }
        let q = age / RING_LIFE;
        if q >= 1.0 {
            continue;
        }
        let scale = 40.0 + 620.0 * (1.0 - (1.0 - q).powf(2.4));
        let opacity = (1.0 - q).powf(1.6) * 0.5;
        // The ring lives in the z=0 plane and faces the camera, so its
        // projection is a circle about the centre.
        let radius = scale * view.focal / CAM_Z;
        if radius < 0.5 {
            continue;
        }
        let width = (2.0 + radius * 0.02).min(MAX_POINT);
        let (r, g, b) = mix(ca, cb, q);
        let level = gain * opacity * (1.5 / width);
        let col = (r * level, g * level, b * level);

        let steps = ((TAU * radius).ceil() as usize).clamp(64, 8000);
        for s in 0..steps {
            let (sn, cs) = (TAU * s as f64 / steps as f64).sin_cos();
            splat(fb, view.cx + cs * radius, view.cy + sn * radius, width, col);
        }
    }
}

// ---------------------------------------------------------------------
// projection + rasterisation
// ---------------------------------------------------------------------

/// The camera. Screen y grows downward, world y grows up.
struct View {
    cx: f64,
    cy: f64,
    focal: f64,
}

impl View {
    /// `None` for a zero-sized framebuffer — there is nothing to draw on.
    fn of(fb: &Framebuffer) -> Option<Self> {
        if fb.w == 0 || fb.h == 0 {
            return None;
        }
        Some(Self {
            cx: fb.w as f64 * 0.5,
            cy: fb.h as f64 * 0.5,
            focal: fb.h as f64 * FOCAL_FRAC,
        })
    }

    /// Screen position and view depth, or `None` if it's behind us.
    fn project(&self, x: f64, y: f64, z: f64) -> Option<(f64, f64, f64)> {
        let depth = CAM_Z - z;
        if !depth.is_finite() || depth <= NEAR {
            return None;
        }
        let k = self.focal / depth;
        Some((self.cx + x * k, self.cy - y * k, depth))
    }
}

fn secs(beat: f64) -> f64 {
    beat * SEC_PER_BEAT
}

/// Point count for this framebuffer: density by area, then the cap.
fn budget(fb: &Framebuffer, px_each: u64, cap: u64) -> u64 {
    let area = fb.w as u64 * fb.h as u64;
    (area / px_each).clamp(MIN_POINTS, cap)
}

/// Lines resolvable on this framebuffer, never more than the original's.
fn grid_extent(fb: &Framebuffer) -> (usize, usize) {
    let cols = ((fb.w as usize / 16) | 1).clamp(9, GRID_COLS);
    let rows = (fb.h as usize / 12).clamp(5, GRID_ROWS);
    (cols, rows)
}

/// A direction on the surface of the unit box, per point.
fn box_dir(i: u64) -> (f64, f64, f64) {
    let u = unit_f64(hash3(i, 8, 0)) * 2.0 - 1.0;
    let v = unit_f64(hash3(i, 9, 0)) * 2.0 - 1.0;
    match hash3(i, 7, 0) % 6 {
        0 => (1.0, u, v),
        1 => (-1.0, u, v),
        2 => (u, 1.0, v),
        3 => (u, -1.0, v),
        4 => (u, v, 1.0),
        _ => (u, v, -1.0),
    }
}

fn mix(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| x as f64 + (y as f64 - x as f64) * t;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Additive point splat, `size` in pixels across. Sub-pixel points dim
/// rather than vanish, which is what keeps a distant field looking like a
/// field instead of aliasing into sparkle.
fn splat(fb: &mut Framebuffer, sx: f64, sy: f64, size: f64, col: (f64, f64, f64)) {
    if !sx.is_finite() || !sy.is_finite() || !size.is_finite() {
        return;
    }
    let d = size.clamp(0.0, MAX_POINT);
    if d <= 0.0 {
        return;
    }
    if d <= 1.0 {
        let k = d.max(0.08);
        add_px(fb, sx as i64, sy as i64, (col.0 * k, col.1 * k, col.2 * k));
        return;
    }
    let rad = d * 0.5;
    let (x0, x1) = ((sx - rad).floor() as i64, (sx + rad).ceil() as i64);
    let (y0, y1) = ((sy - rad).floor() as i64, (sy + rad).ceil() as i64);
    for y in y0..=y1 {
        let dy = y as f64 + 0.5 - sy;
        for x in x0..=x1 {
            let dx = x as f64 + 0.5 - sx;
            let w = 1.0 - (dx * dx + dy * dy) / (rad * rad);
            if w <= 0.0 {
                continue;
            }
            add_px(fb, x, y, (col.0 * w, col.1 * w, col.2 * w));
        }
    }
}

/// A world-space segment: clipped at the near plane, projected, drawn.
fn add_seg(
    fb: &mut Framebuffer,
    view: &View,
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    col: (f64, f64, f64),
) {
    let (mut a, mut b) = (a, b);
    let (da, db) = (CAM_Z - a.2, CAM_Z - b.2);
    if !da.is_finite() || !db.is_finite() {
        return;
    }
    if da <= NEAR && db <= NEAR {
        return;
    }
    if da <= NEAR {
        a = lerp3(b, a, (NEAR - db) / (da - db));
    } else if db <= NEAR {
        b = lerp3(a, b, (NEAR - da) / (db - da));
    }
    let (Some(pa), Some(pb)) = (view.project(a.0, a.1, a.2), view.project(b.0, b.1, b.2)) else {
        return;
    };
    add_line(fb, pa.0, pa.1, pb.0, pb.1, col);
}

fn lerp3(from: (f64, f64, f64), to: (f64, f64, f64), t: f64) -> (f64, f64, f64) {
    (
        from.0 + (to.0 - from.0) * t,
        from.1 + (to.1 - from.1) * t,
        from.2 + (to.2 - from.2) * t,
    )
}

/// Additive DDA line. Clipped to the framebuffer *before* stepping, so a
/// segment whose endpoint projects a mile off-screen costs screen pixels,
/// not its own length.
fn add_line(fb: &mut Framebuffer, x0: f64, y0: f64, x1: f64, y1: f64, col: (f64, f64, f64)) {
    if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return;
    }
    let (w, h) = (fb.w as f64, fb.h as f64);
    let (dx, dy) = (x1 - x0, y1 - y0);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    // Liang-Barsky against [0, w) x [0, h).
    const EPS: f64 = 1e-6;
    for (p, q) in [(-dx, x0), (dx, w - EPS - x0), (-dy, y0), (dy, h - EPS - y0)] {
        if p == 0.0 {
            if q < 0.0 {
                return;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return;
            }
            t1 = t1.min(r);
        }
    }
    let (ax, ay) = (x0 + dx * t0, y0 + dy * t0);
    let (bx, by) = (x0 + dx * t1, y0 + dy * t1);
    let n = ((bx - ax).abs().max((by - ay).abs()).ceil() as usize).max(1);
    for s in 0..=n {
        let f = s as f64 / n as f64;
        add_px(
            fb,
            (ax + (bx - ax) * f) as i64,
            (ay + (by - ay) * f) as i64,
            col,
        );
    }
}

/// The `lighter` composite: saturating add, never a write.
#[inline]
fn add_px(fb: &mut Framebuffer, x: i64, y: i64, col: (f64, f64, f64)) {
    if x < 0 || y < 0 || x >= fb.w as i64 || y >= fb.h as i64 {
        return;
    }
    let i = ((y as u32 * fb.w + x as u32) * 3) as usize;
    for (c, v) in [col.0, col.1, col.2].into_iter().enumerate() {
        if !v.is_finite() || v <= 0.0 {
            continue;
        }
        fb.px[i + c] = (fb.px[i + c] as f64 + v).clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: (u8, u8, u8) = (0, 255, 213);
    const B: (u8, u8, u8) = (255, 40, 160);

    /// A drive with something on every input, so the beat-reactive terms
    /// are actually exercised.
    fn hot() -> Drive {
        let mut d = Drive::default();
        d.update(0.01, 4.0, None, None);
        d
    }

    fn draw_all(fb: &mut Framebuffer, beat: f64, d: &Drive) {
        clear(fb);
        sparks(fb, beat, 0.8, d, A, B);
        tunnel_px(fb, beat, 0.8, d, A, B);
        grid_floor(fb, beat, 0.8, d, A, B);
        rings(fb, beat, 0.8, d, A, B);
    }

    #[test]
    fn same_beat_same_pixels() {
        let d = hot();
        let mut a = Framebuffer::new(96, 64);
        let mut b = Framebuffer::new(96, 64);
        draw_all(&mut a, 4.37, &d);
        draw_all(&mut b, 4.37, &d);
        assert_eq!(a.px, b.px);
        assert!(a.px.iter().any(|&v| v > 0), "nothing was drawn");
    }

    #[test]
    fn reverse_time_reproduces_the_earlier_frame() {
        let d = hot();
        // Run forward past a bar line, then scrub back.
        let mut scrubbed = Framebuffer::new(80, 48);
        draw_all(&mut scrubbed, 5.0, &d);
        draw_all(&mut scrubbed, 3.0, &d);
        // Same beat, reached directly.
        let mut direct = Framebuffer::new(80, 48);
        draw_all(&mut direct, 3.0, &d);
        assert_eq!(scrubbed.px, direct.px);
    }

    #[test]
    fn each_mode_survives_a_backwards_scrub() {
        let d = hot();
        for f in [
            sparks as fn(&mut Framebuffer, f64, f64, &Drive, (u8, u8, u8), (u8, u8, u8)),
            tunnel_px,
            grid_floor,
            rings,
        ] {
            let mut back = Framebuffer::new(64, 40);
            clear(&mut back);
            f(&mut back, 9.75, 1.0, &d, A, B);
            clear(&mut back);
            f(&mut back, 2.25, 1.0, &d, A, B);
            let mut fwd = Framebuffer::new(64, 40);
            clear(&mut fwd);
            f(&mut fwd, 2.25, 1.0, &d, A, B);
            assert_eq!(back.px, fwd.px);
        }
    }

    #[test]
    fn accumulation_saturates_and_never_wraps() {
        let d = hot();
        let mut fb = Framebuffer::new(48, 32);
        clear(&mut fb);
        let mut prev = fb.px.clone();
        // Stack the same frame many times over: additive means every
        // channel may only climb, and must stop at 255.
        for _ in 0..200 {
            sparks(&mut fb, 4.1, 1.0, &d, A, B);
            grid_floor(&mut fb, 4.1, 1.0, &d, A, B);
            for (now, was) in fb.px.iter().zip(prev.iter()) {
                assert!(now >= was, "channel went down: {was} -> {now}");
            }
            prev.copy_from_slice(&fb.px);
        }
        assert!(fb.px.contains(&255), "should have saturated");

        // And directly: an absurd add clamps instead of wrapping.
        let mut one = Framebuffer::new(1, 1);
        add_px(&mut one, 0, 0, (1e9, f64::INFINITY, f64::NAN));
        assert_eq!(&one.px[..], &[255, 0, 0]);
    }

    #[test]
    fn tiny_framebuffers_do_not_panic() {
        let d = hot();
        for (w, h) in [(0, 0), (1, 1), (1, 40), (40, 1), (2, 3), (3, 2)] {
            let mut fb = Framebuffer::new(w, h);
            for beat in [-3.5, 0.0, 4.01, 137.9] {
                draw_all(&mut fb, beat, &d);
            }
            assert_eq!(fb.px.len(), (w * h * 3) as usize);
        }
    }

    #[test]
    fn a_bar_line_fires_a_ring() {
        let d = Drive::default();
        let mut fb = Framebuffer::new(96, 96);

        // Late in the bar every ring has expired: nothing on screen.
        clear(&mut fb);
        rings(&mut fb, 3.9, 1.0, &d, A, B);
        assert!(fb.px.iter().all(|&v| v == 0), "a dead ring is still lit");

        // Just past the bar line one is expanding.
        clear(&mut fb);
        rings(&mut fb, 4.05, 1.0, &d, A, B);
        let lit = fb.px.iter().filter(|&&v| v > 0).count();
        assert!(lit > 20, "bar line should fire a visible ring, got {lit}");

        // And it grows: more circumference on screen a moment later, on a
        // frame big enough to still contain the ring.
        let lit_at = |beat: f64| {
            let mut fb = Framebuffer::new(256, 256);
            clear(&mut fb);
            rings(&mut fb, beat, 1.0, &d, A, B);
            fb.px.iter().filter(|&&v| v > 0).count()
        };
        assert!(lit_at(4.3) > lit_at(4.05), "ring should expand");
    }

    #[test]
    fn modes_draw_something_at_every_size() {
        let d = hot();
        for (w, h) in [(64, 48), (200, 120), (400, 260)] {
            for (name, f) in [
                (
                    "sparks",
                    sparks as fn(&mut Framebuffer, f64, f64, &Drive, (u8, u8, u8), (u8, u8, u8)),
                ),
                ("tunnel", tunnel_px),
                ("grid", grid_floor),
                ("rings", rings),
            ] {
                let mut fb = Framebuffer::new(w, h);
                clear(&mut fb);
                f(&mut fb, 4.2, 1.0, &d, A, B);
                assert!(
                    fb.px.iter().any(|&v| v > 0),
                    "{name} drew nothing at {w}x{h}"
                );
            }
        }
    }

    #[test]
    fn point_count_scales_with_area_and_caps() {
        let small = Framebuffer::new(64, 48);
        let big = Framebuffer::new(1920, 1080);
        assert!(budget(&small, PX_PER_SPARK, SPARK_CAP) < SPARK_CAP);
        assert_eq!(budget(&big, PX_PER_SPARK, SPARK_CAP), SPARK_CAP);
        assert_eq!(
            budget(&Framebuffer::new(1, 1), PX_PER_SPARK, SPARK_CAP),
            MIN_POINTS
        );
    }
}
