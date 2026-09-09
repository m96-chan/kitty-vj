//! MESHCUBE — EasyPngVJ's `cube` particle mode, on the rasteriser.
//!
//! Over there the mode was a group of six textured quads, one per plate,
//! each oriented by a quaternion onto ±X/±Y/±Z and closing into a solid
//! cube; a spin that rose with the grid and the onsets tumbled the whole
//! group; an additive edge outline sat 0.4% outside the faces; and on the
//! bar a `burst` term threw the six faces outward and spun each about its
//! own normal, so the cube came apart and was pulled back inside a beat.
//! The plate behind the rig dimmed while it ran — see [`PLATE_DIM`] — so
//! the solid geometry carried the frame instead of competing with a
//! full-brightness backdrop.
//!
//! This is the **pixel tier** port, on [`crate::raster`]. The cell-tier
//! [`crate::effects::cube`] stays as it is: it forward-maps a bilinear
//! walk of each quad's surface and splats the samples into a halfblock
//! grid, which is why it has to oversample against the longest projected
//! edge. Here the rasteriser fills the faces properly, so the UV is
//! perspective-correct and a plate no longer swims across a face as it
//! turns.
//!
//! # Spin: closed form, not integrated
//!
//! The original integrated its orientation —
//! `spin = 0.55 + 2.6·gbar + 1.5·react`, then `rx += dt·spin·0.62`,
//! `ry += dt·spin`, `rz += dt·spin·0.24`. Faithful, and this port does
//! **not** do it, because integration is state and state is what a jog
//! scrub cannot undo: drag the beat backwards and an accumulator has
//! already eaten those frames, so the cube would keep turning forwards
//! through a backspin. Every orientation in this project is a function of
//! beat time for exactly that reason (see [`crate::raster::Rot`], and the
//! contract in `effects/mod.rs`).
//!
//! So the angle is [`spin_angle`]: a linear ramp at the original's `0.55`
//! rad/s floor, plus a monotone staircase that gains one beat pulse's
//! worth of rotation across each beat and one bar pulse's worth across
//! each bar. The step sizes are the *integrals* of the pulses the
//! original was integrating — an exponential pulse of amplitude 1 and
//! time constant `tau` is worth `tau` seconds of its coefficient — so the
//! cube covers the same ground per bar and still lurches on the grid,
//! but the angle at beat 12.5 is the same whether you arrived there
//! forwards, backwards, or by dropping the needle.
//!
//! What that costs, honestly: an off-grid onset (`hit`) no longer speeds
//! the tumble, and a breakdown no longer slows it — `groove` scaling the
//! ramp would move the *whole* accumulated angle every time the groove
//! moved. The beat-reactive feel lives in the terms that are already
//! instantaneous rather than integrated: the size punch, the burst, and
//! the outline's flash, all of which read the drive directly and all of
//! which return to rest when the drive does.
//!
//! # Other deviations
//!
//! - **Euler, not quaternion.** The rasteriser's [`Transform`] is
//!   yaw/pitch/roll and applies `Ry·Rx·Rz` where the original composed
//!   `Rx·Ry·Rz`. For a cube tumbling on three ratioed angles the two are
//!   different tumbles, not a different *kind* of tumble.
//! - **Double-sided faces.** The burst opens gaps between the six plates;
//!   with backface culling those gaps show nothing and the exploded cube
//!   reads as three lonely plates. Both sides draw, and the depth buffer
//!   sorts them, so an exploded cube shows all six.
//! - **Unlit faces.** The plates carry the colour; the only modulation is
//!   the house intensity fader.
//! - **No plates loaded** is not a reason to draw nothing: the cube draws
//!   untextured, its faces shaded by which way they point, so the
//!   geometry still reads. See [`FACE_SHADE`].

// Wired into the mixer with the other mesh modes; dead until then.
#![allow(dead_code)]

use std::sync::OnceLock;

use image::RgbImage;

use crate::assets::Plate;
use crate::drive::{Drive, SEC_PER_BEAT};
use crate::graphics::Framebuffer;
use crate::raster::{
    Camera, Cull, DepthBuffer, DrawOpts, LineSegments, Raster, Transform, Varyings, Vec3, Vertex,
    v3,
};

/// How far the plate behind dims while this mode runs. The caller applies
/// it — this module only draws the cube — but the constant lives here
/// because it is part of the look: at full brightness the backdrop and
/// the six plates on the faces fight, and the cube stops reading as a
/// solid.
pub const PLATE_DIM: f64 = 0.62;

/// Edge of the cube in world units, before the beat punches it.
const SIZE: f64 = 320.0;
/// Where the cube sits down −Z, and how far the kick shoves it away.
const BASE_Z: f64 = -360.0;
const THUMP_Z: f64 = 80.0;

/// The original's spin floor, radians a second.
const SPIN_BASE: f64 = 0.55;
/// What one beat pulse and one bar pulse are worth in radians — the
/// coefficient the original multiplied the pulse by, times the pulse's
/// own time constant, which is its integral. See the module docs.
const BEAT_KICK: f64 = 1.5 * 0.070;
const BAR_KICK: f64 = 2.6 * 0.130;
/// The three axes' share of the spin, straight from the original.
const AXIS_YAW: f64 = 1.0;
const AXIS_PITCH: f64 = 0.62;
const AXIS_ROLL: f64 = 0.24;

/// Burst geometry: where a face sits along its own normal at rest and at
/// full burst, and how far it spins about that normal on the way out.
const FACE_REST: f64 = 0.5;
const FACE_THROW: f64 = 0.65;
const FACE_SPIN: f64 = 1.1;

/// The outline sits just outside the faces so it reads as an edge rather
/// than z-fighting into a dashed mess.
const OUTLINE_SCALE: f64 = 1.004;

/// Beats between plate rotation steps. Matches the cell-tier CUBE, so the
/// two tiers step together when a show runs both.
const PLATE_BEATS: f64 = 32.0;

/// The six faces: outward direction, then the two in-plane axes the plate
/// is laid out on. `u × v = dir`, which is what makes the corner order in
/// [`CORNERS`] come out wound counter-clockwise from outside.
const FACES: [[Vec3; 3]; 6] = [
    [v3(1.0, 0.0, 0.0), v3(0.0, 0.0, -1.0), v3(0.0, 1.0, 0.0)],
    [v3(-1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), v3(0.0, 1.0, 0.0)],
    [v3(0.0, 1.0, 0.0), v3(1.0, 0.0, 0.0), v3(0.0, 0.0, -1.0)],
    [v3(0.0, -1.0, 0.0), v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0)],
    [v3(0.0, 0.0, 1.0), v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)],
    [v3(0.0, 0.0, -1.0), v3(-1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)],
];

/// Untextured brightness per face, in [`FACES`] order. A sky-ish
/// gradient — up is bright, down is dark — because a plateless cube of
/// one flat grey is a silhouette, not a solid.
const FACE_SHADE: [f64; 6] = [0.72, 0.50, 0.92, 0.34, 0.66, 0.44];

/// A face's corners in its own `(u, v)` plane, cyclic, with the plate UV
/// each carries. `v` runs the other way because image rows go down while
/// world +Y goes up.
const CORNERS: [(f64, f64, f64, f64); 4] = [
    (-0.5, -0.5, 0.0, 1.0),
    (0.5, -0.5, 1.0, 1.0),
    (0.5, 0.5, 1.0, 0.0),
    (-0.5, 0.5, 0.0, 0.0),
];

/// CUBE — six plate-textured faces on a cube tumbling in beat time, with
/// an additive outline, thrown apart on the bar and pulled back.
///
/// The depth buffer comes from the caller so several mesh modes can share
/// one full-screen allocation; this mode **clears** it and owns the pass,
/// which is what lets a caller hand the same buffer to each mode in turn.
/// The framebuffer is not cleared: the faces are opaque and land over
/// whatever is behind them, which is the dimmed plate.
pub fn cube(
    fb: &mut Framebuffer,
    depth: &mut DepthBuffer,
    beat: f64,
    intensity: f64,
    d: &Drive,
    plates: &[Plate],
) {
    let cam = Camera::matching(fb);
    let Some(mut ras) = Raster::new(fb, depth, cam) else {
        return;
    };
    ras.clear_depth();
    let beat = if beat.is_finite() { beat } else { 0.0 };
    let s = Shape::of(beat, intensity, d);
    draw_faces(&mut ras, &s, plates, plate_base(beat, plates.len()));
    draw_outline(&mut ras, &s);
}

/// Everything the drive and the clock decide, resolved once so the two
/// passes agree and so a test can state a pose directly.
struct Shape {
    /// Model → world for the closed cube: the tumble, the beat punch, and
    /// the shove down −Z.
    xf: Transform,
    /// `gbar²` — the bar's throw, sharpened so it reads as a hit rather
    /// than a swell.
    burst: f64,
    /// Face brightness from the intensity fader.
    gain: f64,
    /// Outline opacity, already faded out by the burst.
    outline: f64,
}

impl Shape {
    fn of(beat: f64, intensity: f64, d: &Drive) -> Self {
        // The grid pulse or a hard onset, whichever is louder — the
        // original's `react`.
        let react = d.react().clamp(0.0, 1.0);
        let thump = d.thump.clamp(0.0, 1.0);
        let burst = d.gbar().clamp(0.0, 1.0).powi(2);
        let a = spin_angle(beat);
        let xf = Transform::from_euler(a * AXIS_YAW, a * AXIS_PITCH, a * AXIS_ROLL)
            .with_uniform_scale(SIZE * (1.0 + 0.18 * react + 0.10 * thump))
            .with_translation(v3(0.0, 0.0, BASE_Z - THUMP_Z * thump));
        Self {
            xf,
            burst,
            gain: 0.45 + 0.55 * intensity.clamp(0.0, 1.0),
            outline: ((0.45 + 0.9 * react) * (1.0 - burst)).clamp(0.0, 1.0),
        }
    }
}

/// The tumble angle, in radians, as a closed-form function of beat.
///
/// A linear ramp at the original's spin floor plus one beat pulse's and
/// one bar pulse's worth of rotation eased in across each beat and each
/// bar. Monotone in `beat` and evaluated fresh every frame, so scrubbing
/// back to a beat puts the cube back where it was.
fn spin_angle(beat: f64) -> f64 {
    SPIN_BASE * SEC_PER_BEAT * beat + BEAT_KICK * staircase(beat) + BAR_KICK * staircase(beat / 4.0)
}

/// A monotone staircase of mean slope 1: it climbs by one across every
/// unit interval, fast at the start and settling towards the end, so the
/// grid line lands as a lurch rather than a constant drift.
fn staircase(u: f64) -> f64 {
    let n = u.floor();
    let p = u - n;
    n + 1.0 - (1.0 - p).powi(3)
}

/// Which plate the first face takes; the rest follow it round the list.
/// The step wraps with `rem_euclid` rather than clamping the beat at zero,
/// so scrubbing back before the start rotates backwards instead of
/// sticking on plate zero. `pub(crate)`: the mesh plate steps its plate
/// with the same law, so the two rotate together in a show.
pub(crate) fn plate_base(beat: f64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let step = (beat / PLATE_BEATS).floor();
    if !step.is_finite() {
        return 0;
    }
    (step as i64).rem_euclid(len as i64) as usize
}

/// The six plates, opaque, depth-tested, both sides drawn.
fn draw_faces(ras: &mut Raster, s: &Shape, plates: &[Plate], base: usize) {
    // Both sides: see the module docs on the burst opening gaps.
    let opts = DrawOpts::textured().with_cull(Cull::None);
    let off = FACE_REST + FACE_THROW * s.burst;

    for (i, [dir, u, v]) in FACES.iter().enumerate() {
        // Each face turns about its own normal on the way out, alternate
        // faces the other way, so the cube unwinds rather than swirls.
        let (sn, cs) = (s.burst * FACE_SPIN * if i % 2 == 0 { 1.0 } else { -1.0 }).sin_cos();
        let normal = s.xf.direction(*dir);
        let mut quad = [Vertex::default(); 4];
        for (k, (a, b, tu, tv)) in CORNERS.iter().enumerate() {
            // Rotate in the face's own plane, then lift the plane out
            // along the normal. The UV is taken from the *unrotated*
            // corner, so the plate turns with the face.
            let local = *u * (a * cs - b * sn) + *v * (a * sn + b * cs) + *dir * off;
            quad[k] = Vertex::at(s.xf.point(local))
                .with_uv(*tu, *tv)
                .with_normal(normal);
        }

        let plate = plates.get((base + i) % plates.len().max(1));
        match plate.and_then(|p| Crop::of(&p.img)) {
            Some(crop) => draw_quad(ras, &quad, &opts, |vy: &Varyings, _| {
                let t = crop.texel(vy.uv[0], vy.uv[1]);
                Some(tint(t, s.gain))
            }),
            None => {
                let k = s.gain * FACE_SHADE[i];
                let flat = tint([235, 235, 235], k);
                draw_quad(ras, &quad, &opts, |_: &Varyings, _| Some(flat));
            }
        }
    }
}

/// The additive edge outline, just outside the faces.
///
/// Depth-tested but not depth-writing, so the far edges stay hidden
/// behind the near faces without the outline pass deciding what the faces
/// occlude. It rides the *closed* cube's transform: on the bar the faces
/// fly out of it while the outline fades, which is what makes the burst
/// read as the cube coming apart rather than as six planes drifting.
fn draw_outline(ras: &mut Raster, s: &Shape) {
    if s.outline <= 0.0 {
        return;
    }
    let xf = Transform {
        scale: s.xf.scale * OUTLINE_SCALE,
        ..s.xf
    };
    let lum = (255.0 * s.outline) as u8;
    let col = (lum, lum, lum);
    let opts = DrawOpts::wire().with_interp(false, false, false);
    ras.draw_lines(edges(), &xf, &opts, |_: &Varyings, _| Some(col));
}

/// The twelve edges of the unit cube, built once. Fixed geometry — the
/// pose is all in the [`Transform`] — so there is no reason to rebuild
/// it, and none of it is state that a scrub could desync.
fn edges() -> &'static LineSegments {
    static EDGES: OnceLock<LineSegments> = OnceLock::new();
    EDGES.get_or_init(|| {
        let mut ls = LineSegments::new();
        let corner = |i: usize| {
            let f = |bit: usize| if i >> bit & 1 == 1 { 0.5 } else { -0.5 };
            v3(f(0), f(1), f(2))
        };
        for i in 0..8usize {
            for bit in 0..3usize {
                let j = i | 1 << bit;
                if j != i {
                    ls.seg(Vertex::at(corner(i)), Vertex::at(corner(j)));
                }
            }
        }
        ls
    })
}

/// Two triangles, one quad, one shader. The rasteriser has no quad entry
/// point on purpose; this keeps the split in one place.
fn draw_quad<S>(ras: &mut Raster, q: &[Vertex; 4], opts: &DrawOpts, mut shade: S)
where
    S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
{
    ras.draw_tri(&[q[0], q[1], q[2]], opts, &mut shade);
    ras.draw_tri(&[q[0], q[2], q[3]], opts, &mut shade);
}

#[inline]
fn tint(t: [u8; 3], k: f64) -> (u8, u8, u8) {
    let c = |v: u8| (v as f64 * k).clamp(0.0, 255.0) as u8;
    (c(t[0]), c(t[1]), c(t[2]))
}

/// A plate, sampled through a **centre square crop**.
///
/// A face is square and a plate is 16:9 or 2:3, so mapping the whole
/// image onto it would stretch it. Taking the largest centred square
/// instead loses the sides of a wide plate and the top and bottom of a
/// tall one, which is the same trade the cell-tier CUBE makes.
struct Crop<'a> {
    img: &'a RgbImage,
    side: f64,
    offx: f64,
    offy: f64,
    maxx: f64,
    maxy: f64,
}

impl<'a> Crop<'a> {
    /// `None` for an empty image — there is nothing to sample.
    fn of(img: &'a RgbImage) -> Option<Self> {
        let (w, h) = (img.width() as f64, img.height() as f64);
        if w < 1.0 || h < 1.0 {
            return None;
        }
        let side = w.min(h) - 1.0;
        Some(Self {
            img,
            side,
            offx: (w - side) / 2.0,
            offy: (h - side) / 2.0,
            maxx: w - 1.0,
            maxy: h - 1.0,
        })
    }

    /// Nearest texel. The clamps are not decoration: `get_pixel` panics
    /// out of bounds, and a UV can land a hair outside `0..1` on a
    /// fragment at the very edge of a triangle.
    #[inline]
    fn texel(&self, u: f64, v: f64) -> [u8; 3] {
        let f = |t: f64, off: f64, max: f64| {
            let s = off + t.clamp(0.0, 1.0) * self.side;
            if s.is_finite() {
                s.clamp(0.0, max)
            } else {
                0.0
            }
        };
        let x = f(u, self.offx, self.maxx) as u32;
        let y = f(v, self.offy, self.maxy) as u32;
        self.img.get_pixel(x, y).0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::f64::consts::FRAC_PI_4;

    use image::Rgb;

    use super::*;

    /// Six flat plates in well-separated colours, so "which faces are
    /// visible" is answerable by counting distinct pixel values.
    const TINTS: [[u8; 3]; 6] = [
        [250, 10, 10],
        [10, 250, 10],
        [10, 10, 250],
        [250, 250, 10],
        [250, 10, 250],
        [10, 250, 250],
    ];

    fn plates() -> Vec<Plate> {
        TINTS
            .iter()
            .enumerate()
            .map(|(i, t)| Plate {
                name: format!("t{i}"),
                img: RgbImage::from_pixel(4, 4, Rgb(*t)),
            })
            .collect()
    }

    /// A drive with something on every input, as `pixparticles`' tests do.
    fn hot() -> Drive {
        let mut d = Drive::default();
        d.update(0.01, 4.0, None, None);
        d
    }

    /// A drive whose bar pulse is pinned, i.e. `burst = 1`.
    fn bursting() -> Drive {
        let mut d = Drive::default();
        d.groove = 1.0;
        d.bar = 1.0;
        d
    }

    fn quiet() -> Drive {
        let mut d = Drive::default();
        d.groove = 1.0;
        d
    }

    fn draw(w: u32, h: u32, beat: f64, d: &Drive) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        let mut depth = DepthBuffer::new(w, h);
        cube(&mut fb, &mut depth, beat, 0.8, d, &plates());
        fb
    }

    fn lit_bbox(fb: &Framebuffer) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in 0..fb.h {
            for x in 0..fb.w {
                let i = ((y * fb.w + x) * 3) as usize;
                if fb.px[i..i + 3].iter().any(|&v| v > 0) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (x0 <= x1).then_some((x0, y0, x1, y1))
    }

    /// Distinct non-black colours on screen. With flat plates that is one
    /// value per visible face.
    fn face_colours(fb: &Framebuffer) -> HashSet<(u8, u8, u8)> {
        let mut set = HashSet::new();
        for px in fb.px.chunks_exact(3) {
            if px != [0, 0, 0] {
                set.insert((px[0], px[1], px[2]));
            }
        }
        set
    }

    /// A pose stated directly, bypassing the clock, so a test can put the
    /// cube corner-on or face-on and know which faces should show.
    fn posed(yaw: f64, pitch: f64, burst: f64) -> Shape {
        Shape {
            xf: Transform::from_euler(yaw, pitch, 0.0)
                .with_uniform_scale(SIZE)
                .with_translation(v3(0.0, 0.0, BASE_Z)),
            burst,
            gain: 1.0,
            outline: 0.0,
        }
    }

    /// Looking straight down the body diagonal: yaw and pitch that carry
    /// `(1,1,1)` onto the view axis, given the rasteriser's `Ry·Rx·Rz`.
    fn corner_on(burst: f64) -> Shape {
        posed(-(1.0 / 2.0f64.sqrt()).atan(), FRAC_PI_4, burst)
    }

    fn render(s: &Shape, w: u32, h: u32, plates: &[Plate]) -> (Framebuffer, DepthBuffer) {
        let mut fb = Framebuffer::new(w, h);
        let mut depth = DepthBuffer::new(w, h);
        {
            let cam = Camera::matching(&fb);
            let mut ras = Raster::new(&mut fb, &mut depth, cam).unwrap();
            ras.clear_depth();
            draw_faces(&mut ras, s, plates, 0);
        }
        (fb, depth)
    }

    #[test]
    fn same_beat_same_pixels() {
        let d = hot();
        let a = draw(160, 120, 6.37, &d);
        let b = draw(160, 120, 6.37, &d);
        assert_eq!(a.px, b.px);
        assert!(a.px.iter().any(|&v| v > 0), "nothing was drawn");
    }

    #[test]
    fn a_backwards_scrub_reproduces_the_earlier_frame() {
        // The whole reason the spin is closed form: run forward past a
        // bar line, scrub back, and land on the same pixels as arriving
        // at that beat directly.
        let d = hot();
        let mut fb = Framebuffer::new(120, 96);
        let mut depth = DepthBuffer::new(120, 96);
        for beat in [9.75, 2.25] {
            fb.px.fill(0);
            cube(&mut fb, &mut depth, beat, 0.8, &d, &plates());
        }
        let direct = draw(120, 96, 2.25, &d);
        assert_eq!(fb.px, direct.px);
    }

    #[test]
    fn the_spin_is_monotone_and_covers_the_originals_ground() {
        let mut prev = f64::NEG_INFINITY;
        for i in 0..400 {
            let a = spin_angle(i as f64 * 0.05);
            assert!(a > prev, "spin went backwards at beat {}", i as f64 * 0.05);
            prev = a;
        }
        // A bar of the original at full grid: 4 beats of the floor, 4
        // beat pulses and 1 bar pulse.
        let want = SPIN_BASE * SEC_PER_BEAT * 4.0 + 4.0 * BEAT_KICK + BAR_KICK;
        let got = spin_angle(8.0) - spin_angle(4.0);
        assert!(
            (got - want).abs() < 1e-9,
            "a bar is worth {got}, want {want}"
        );
    }

    #[test]
    fn a_bar_burst_throws_the_faces_apart() {
        let closed = lit_bbox(&draw(400, 400, 3.0, &quiet())).expect("closed cube drew nothing");
        let thrown = lit_bbox(&draw(400, 400, 3.0, &bursting())).expect("burst cube drew nothing");
        let width = |b: (u32, u32, u32, u32)| b.2 - b.0;
        let height = |b: (u32, u32, u32, u32)| b.3 - b.1;
        assert!(
            width(thrown) > width(closed) * 13 / 10 && height(thrown) > height(closed) * 13 / 10,
            "burst did not separate the faces: closed {closed:?}, thrown {thrown:?}"
        );
    }

    #[test]
    fn faces_visible_corner_on_and_face_on() {
        let p = plates();
        // Face-on down +Z: a closed cube shows exactly the one face it
        // presents, the four side faces are edge-on and the back face is
        // squarely behind the front one.
        let (fb, _) = render(&posed(0.0, 0.0, 0.0), 400, 400, &p);
        assert_eq!(face_colours(&fb).len(), 1, "face-on should show one face");

        // Corner-on: three. A closed convex cube can never show more —
        // which is why the six-face check below needs the burst.
        let (fb, _) = render(&corner_on(0.0), 400, 400, &p);
        assert_eq!(
            face_colours(&fb).len(),
            3,
            "corner-on should show three faces"
        );

        // Thrown apart, the gaps open and all six plates reach the frame.
        let (fb, _) = render(&corner_on(1.0), 400, 400, &p);
        assert_eq!(
            face_colours(&fb).len(),
            6,
            "a burst corner-on view should show all six faces"
        );
    }

    #[test]
    fn the_outline_is_additive_and_leaves_the_face_depth_alone() {
        let p = plates();
        let d = quiet();
        let s = Shape::of(3.0, 0.8, &d);
        assert!(s.outline > 0.0, "no outline to test");

        let (faces, faces_depth) = render(&s, 200, 160, &p);
        let mut fb = Framebuffer::new(200, 160);
        let mut depth = DepthBuffer::new(200, 160);
        cube(&mut fb, &mut depth, 3.0, 0.8, &d, &p);

        assert!(
            fb.px.iter().zip(faces.px.iter()).all(|(&a, &b)| a >= b),
            "the outline replaced a face pixel instead of adding to it"
        );
        assert!(fb.px != faces.px, "the outline drew nothing");
        for y in 0..160 {
            for x in 0..200 {
                assert_eq!(
                    depth.depth_at(x, y),
                    faces_depth.depth_at(x, y),
                    "the outline wrote depth at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn hostile_input_does_not_panic() {
        let empty: Vec<Plate> = Vec::new();
        let p = plates();
        for (w, h) in [(0, 0), (1, 1), (1, 64), (64, 1), (3, 2), (48, 32)] {
            let mut fb = Framebuffer::new(w, h);
            let mut depth = DepthBuffer::new(w, h);
            for beat in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -7.5, 0.0, 1e12] {
                for d in [Drive::default(), hot(), bursting()] {
                    for plates in [&empty, &p] {
                        cube(&mut fb, &mut depth, beat, 1.3, &d, plates);
                    }
                }
            }
            assert_eq!(fb.px.len(), (w * h * 3) as usize);
        }
    }

    #[test]
    fn without_plates_the_cube_still_draws() {
        let mut fb = Framebuffer::new(200, 160);
        let mut depth = DepthBuffer::new(200, 160);
        cube(&mut fb, &mut depth, 3.0, 1.0, &quiet(), &[]);
        assert!(fb.px.iter().any(|&v| v > 0), "untextured cube drew nothing");
    }

    #[test]
    fn the_crop_is_a_centred_square() {
        // A wide plate: the crop keeps the full height and the middle of
        // the width, so a face never shows a stretched plate.
        let mut img = RgbImage::from_pixel(9, 3, Rgb([0, 0, 0]));
        for x in 0..9u32 {
            for y in 0..3u32 {
                img.put_pixel(x, y, Rgb([x as u8 * 10, y as u8 * 10, 0]));
            }
        }
        let c = Crop::of(&img).unwrap();
        assert_eq!(c.texel(0.0, 0.0), [30, 0, 0], "left edge of the crop");
        assert_eq!(c.texel(1.0, 1.0), [50, 20, 0], "right edge of the crop");
        // Out-of-range UV clamps rather than panicking.
        assert_eq!(c.texel(-9.0, 9.0), c.texel(0.0, 1.0));
        assert!(Crop::of(&RgbImage::new(0, 0)).is_none());
        assert!(Crop::of(&RgbImage::from_pixel(1, 1, Rgb([7, 7, 7]))).is_some());
    }

    #[test]
    fn plate_rotation_steps_and_wraps_both_ways() {
        assert_eq!(plate_base(0.0, 6), 0);
        assert_eq!(plate_base(PLATE_BEATS + 0.5, 6), 1);
        assert_eq!(plate_base(-0.5, 6), 5);
        assert_eq!(plate_base(f64::NAN, 6), 0);
        assert_eq!(plate_base(12.0, 0), 0);
    }
}
