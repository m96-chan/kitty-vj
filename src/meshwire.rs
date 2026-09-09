//! MESHWIRE — EasyPngVJ's `wire` mode on the software rasteriser.
//!
//! Over there this was three tumbling driver shapes drawn as glowing
//! wireframes, with a shockwave ring punched out of one of them on every
//! beat. The wireframe was not `THREE.WireframeGeometry` and not
//! `gl_LineWidth`: the mesh was re-expanded so that every corner carried a
//! **barycentric coordinate**, and the fragment shader lit a pixel by how
//! close its interpolated barycentric was to zero. That is the whole
//! trick, and it is why the mode ports at all — line width becomes a
//! uniform (`uWidth`) that the drive signals can push around per frame,
//! instead of a driver-dependent GL state nobody can animate.
//!
//! [`crate::raster`] was built with that in mind: [`Vertex::bary`] is a
//! stored attribute rather than the rasteriser's own weights, and
//! [`Mesh::quad`] rigs it so a quad's internal triangulation diagonal
//! never lights. This module supplies the geometry, the tumble, and the
//! edge shader.
//!
//! # No `fwidth`
//!
//! The original's edge test is
//!
//! ```text
//! a    = smoothstep(0, fwidth(bary) * uWidth, bary)
//! edge = 1 - min(a.x, a.y, a.z)
//! ```
//!
//! `fwidth` is what makes the line a *screen-space* width: it is the
//! change in `bary` over one pixel, so dividing by it converts a
//! barycentric distance into a pixel distance and the wire stays the same
//! thickness whether a face fills the frame or is a speck. A GPU gets it
//! free from the 2x2 quad it shades in lockstep. A scalar CPU rasteriser
//! shades one fragment at a time and has no neighbour to difference
//! against.
//!
//! So it is computed **analytically, once per triangle**, rather than
//! approximated per pixel. For an attribute interpolated across a
//! triangle the screen-space gradient is constant: the rasteriser's own
//! weight `l_i` has gradient `perp(edge_i) / area2`, and the stored
//! attribute is a fixed linear combination of the three corners' values,
//! so `d(bary_c)/dx` and `/dy` are a handful of multiply-adds off the
//! projected corners. `fwidth = |d/dx| + |d/dy|`, exactly the quantity
//! GLSL's coarse derivatives estimate.
//!
//! The two alternatives were worse. A fixed pixel width converted through
//! the triangle's *area* alone is only right for equilateral triangles —
//! a sliver, which is what every wireframed shape has once it turns
//! edge-on, gets a width off by its aspect ratio. And a per-pixel finite
//! difference would mean shading each fragment's neighbours too, four
//! times the shader cost for a number that does not vary. The one thing
//! given up is perspective: the attribute is interpolated
//! perspective-correctly, so its true gradient varies slightly across a
//! steeply-tilted face, and the analytic value is the affine one. At
//! these depths (shapes ~60 units across, ~450 away) that is well under a
//! tenth of a pixel of width variation, and a wire that breathes by a
//! tenth of a pixel is not a wire anybody can see breathing.
//!
//! Consequences: a degenerate triangle is skipped rather than divided by,
//! and a channel that is *constant* over a triangle — which is exactly
//! what [`Mesh::quad`]'s diagonal suppression produces — has zero gradient
//! and is dropped from the `min` instead of being handed an infinite
//! threshold.
//!
//! # Blast rings are a scan, not a pool
//!
//! The original kept 14 ring objects, fired the next free one on each
//! beat, and animated it from there. [`crate::pixparticles::rings`]
//! already met this problem and inverted it: nothing is stored, the last
//! few grid lines behind the current beat are scanned, and each one is
//! given an age of *now minus that grid line*. Scrub backwards and a ring
//! that has not been fired yet simply is not in the scan — the pool
//! version has no way to un-fire it.
//!
//! Doing that here needs one thing more than the particle rings needed.
//! A blast leaves along its driver's **+Z at the moment it fired**, and
//! then ignores the driver — that lag is what makes it read as a
//! shockwave rather than a streamer. A pool remembers that axis; a scan
//! has to *derive* it. Which is possible precisely because orientation
//! here is a closed-form function of beat, so the driver's orientation at
//! beat `n` can be evaluated at beat `n + 2.7` just as easily as it could
//! at the time. See [`spin_phase`] for what closed form costs us.
//!
//! # Budget
//!
//! ~40 triangles of driver and ~40 per live blast, three blasts alive at
//! the reference tempo: about **160 triangles a frame**, against the
//! ~280 the rasteriser's module doc measured at 1.6 ms. Fill is the real
//! cost, not triangle count, so the rings are thin annuli rather than
//! discs and a ring that swallows the camera dissolves — which also caps
//! its projected radius at the focal length, so no frame can degenerate
//! into full-screen quads.
//!
//! Measured on an M-series Air, release build, 1200x700, framebuffer and
//! depth buffer both cleared per frame: **~1.0 ms a frame**, with the
//! worst frame in a four-thousand-frame sweep at 2.1 ms. About 1% of
//! subpixels end up lit, which is what a wireframe should cost — the
//! interior of every face is discarded before it is shaded.
//!
//! **This adds.** Like every pixel-tier mode it is a saturating add into
//! whatever is already in the framebuffer. It tests depth but does not
//! write it, so it neither occludes itself nor cares about submission
//! order; hand it a cleared depth buffer, or one already holding this
//! frame's opaque geometry, and the wire will sit behind that geometry
//! correctly.

// Wired into the mixer separately; the entry points are dead until then.
#![allow(dead_code)]

use std::f64::consts::TAU;
use std::sync::OnceLock;

use crate::drive::{Drive, SEC_PER_BEAT};
use crate::graphics::Framebuffer;
use crate::raster::{
    Camera, DepthBuffer, DrawOpts, Mesh, NEAR, Raster, Rot, Transform, Varyings, Vec3, Vertex, v3,
};

/// The original's `BLAST_LIFE`, 1.5 s, in beats.
const BLAST_LIFE: f64 = 1.5 / SEC_PER_BEAT;
/// How many beats back the scan looks — the original's ring pool size.
/// At the reference tempo a blast outlives three beats, so three slots
/// are ever alive and the remaining eleven cost one compare each; the
/// depth is kept at the original's number so a slower tempo behaves the
/// way the original did rather than truncating.
const BLAST_POOL: i64 = 14;

/// Segments around a blast ring. Twenty keeps the chord sag under three
/// pixels at any radius the near-fade allows, and 40 triangles is a
/// twentieth of the rasteriser's measured budget.
const RING_SEGS: usize = 20;
/// Inner radius of the ring band. Thin, for two reasons: the band is
/// fill, and fill is what this mode spends; and the band's two arcs both
/// light, so a fat one reads as two rings chasing each other rather than
/// as one shockwave.
const RING_INNER: f64 = 0.88;

/// Everything past this fades out; the drivers sit around 450.
const FADE_DEPTH: f64 = 2600.0;

/// Below this the edge term is under half a code value at full level —
/// discard rather than shade, which is what keeps the interior of a face
/// nearly free.
const EDGE_FLOOR: f64 = 0.004;

// ---------------------------------------------------------------------
// the drivers
// ---------------------------------------------------------------------

/// One tumbling shape: where it sits, how big, and its per-axis rates —
/// the original's `sx, sy, sz`.
struct Driver {
    pos: Vec3,
    scale: f64,
    rate: Vec3,
}

const DRIVERS: [Driver; 3] = [
    Driver {
        pos: v3(-215.0, 55.0, -60.0),
        scale: 62.0,
        rate: v3(0.37, 0.53, 0.21),
    },
    Driver {
        pos: v3(15.0, -70.0, 40.0),
        scale: 48.0,
        rate: v3(-0.61, 0.29, 0.44),
    },
    Driver {
        pos: v3(225.0, 80.0, -140.0),
        scale: 74.0,
        rate: v3(0.24, -0.68, 0.33),
    },
];

/// The original's `spin = 0.45 + 2.4*gbar + 1.6*react`, **integrated**.
///
/// Over there that was a rate: `rotation += dt * sx * spin`, an
/// accumulator, and an accumulator is the one thing a jog wheel can't
/// drag backwards. So what is evaluated here is the angle, not the rate —
/// the closed-form integral of that expression from beat zero.
///
/// Two of the three terms integrate exactly. The constant gives `0.45·t`.
/// The grid pulses are unit impulses on the beat and bar lines decaying
/// with a known tau, so their integral is a count of lines already passed
/// plus the part of the current one that has elapsed — see
/// [`pulse_integral`].
///
/// The third does not, and is dropped on purpose. `react` is
/// `max(gbeat, hit·0.85)`, and `hit` is an audio onset: it fires between
/// the grid lines, at times nothing downstream can reconstruct from a
/// beat number. Any term carrying it would make a driver's orientation
/// unrecoverable at a past beat, which is exactly the value the blast
/// rings need — so `react` drives wire width and level here, which is
/// where the original put its most visible work anyway. Likewise the
/// groove gate: it scales a *growing* term, and a slow drift in groove
/// applied to a term that grows with beat is a rotation jump, not a
/// change of pace.
fn spin_phase(beat: f64) -> f64 {
    if !beat.is_finite() {
        return 0.0;
    }
    // Decay constants from `drive`, expressed in beats.
    // Drive's decay taus, converted from seconds to beats — derived
    // from the shared constants so the angular budget cannot drift.
    const TAU_BEAT: f64 = crate::drive::TAU_BEAT / SEC_PER_BEAT;
    const TAU_BAR: f64 = crate::drive::TAU_BAR / SEC_PER_BEAT;
    let base = 0.45 * beat * SEC_PER_BEAT;
    let bar = 2.4 * SEC_PER_BEAT * pulse_integral(beat, 4.0, TAU_BAR);
    let pulse = 1.6 * SEC_PER_BEAT * pulse_integral(beat, 1.0, TAU_BEAT);
    base + bar + pulse
}

/// Integral, in beats, of a unit pulse that fires every `period` beats
/// and decays with time constant `tau`.
///
/// Each pulse contributes `tau·(1 - e^(-age/tau))`, so a pulse older than
/// a few tau has simply contributed `tau`. Only the live one needs the
/// exponential; the error from treating its predecessor as finished is
/// `tau·e^(-period/tau)`, under a thousandth of a beat at these
/// constants, and it is a constant offset rather than a drift.
fn pulse_integral(beat: f64, period: f64, tau: f64) -> f64 {
    if !beat.is_finite() || beat <= 0.0 {
        return 0.0;
    }
    let n = (beat / period).floor();
    let age = beat - n * period;
    tau * (n + 1.0 - (-age / tau).exp())
}

/// A driver's orientation at a given tumble phase. Euler rather than a
/// quaternion for [`Rot`]'s reason: nothing here integrates, so there is
/// nothing for a quaternion to protect.
fn driver_rot(phase: f64, i: usize) -> Rot {
    let r = DRIVERS[i].rate;
    Rot::from_euler(phase * r.y, phase * r.x, phase * r.z)
}

// ---------------------------------------------------------------------
// public entry point
// ---------------------------------------------------------------------

/// WIRE — three tumbling wireframe solids, and a shockwave ring out of
/// one of them on every beat.
///
/// `depth` is tested but never written and never cleared: clear it
/// yourself, or draw this after the opaque geometry it should hide
/// behind.
pub fn wire(
    fb: &mut Framebuffer,
    depth: &mut DepthBuffer,
    beat: f64,
    intensity: f64,
    d: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    if fb.w == 0 || fb.h == 0 || !beat.is_finite() {
        return;
    }
    let react = d.react();
    let look = Look {
        // The original's `uWidth`, verbatim.
        width_px: 0.8 + 0.8 * react,
        level: (0.35 + 0.65 * intensity.clamp(0.0, 1.0))
            * (0.55 + 1.3 * react + 0.5 * d.thump.clamp(0.0, 1.0)),
        ca,
        cb,
    };
    let cam = Camera::matching(fb);
    let scr = Screen {
        cx: fb.w as f64 * 0.5,
        cy: fb.h as f64 * 0.5,
        focal: cam.focal,
    };
    let Some(mut ras) = Raster::new(fb, depth, cam) else {
        return;
    };

    draw_drivers(&mut ras, scr, beat, &look);

    // The scan. Every beat line behind us within the pool depth fired a
    // blast; `blast_state` decides which are still alive.
    let latest = beat.floor() as i64;
    for k in 0..BLAST_POOL {
        draw_blast(&mut ras, scr, latest.saturating_sub(k), beat, &look);
    }
}

/// The per-frame look, bundled so the internals stay readable and the
/// tests can hold one term still while moving another.
#[derive(Clone, Copy)]
struct Look {
    width_px: f64,
    level: f64,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
}

fn draw_drivers(ras: &mut Raster, scr: Screen, beat: f64, look: &Look) {
    let phase = spin_phase(beat);
    let g = geometry();
    for (i, mesh) in g.drivers.iter().enumerate() {
        let d = &DRIVERS[i];
        let xf = Transform::IDENTITY
            .with_rot(driver_rot(phase, i))
            .with_uniform_scale(d.scale)
            .with_translation(d.pos);
        let col = scaled(mix(look.ca, look.cb, i as f64 * 0.5), look.level);
        draw_wire_mesh(ras, mesh, &xf, scr, look.width_px, col);
    }
}

/// One blast, identified by the beat line that fired it.
struct Blast {
    xf: Transform,
    /// Outer radius in world units — the ring mesh is a unit circle, so
    /// this is just the uniform scale, named for what it is used for.
    radius: f64,
    q: f64,
    opacity: f64,
}

/// Where the blast fired on beat `fire` has got to by `beat`, or `None`
/// if it has not fired yet or has expired.
///
/// Nothing is remembered: the driver index is `fire % 3` and the axis is
/// that driver's +Z re-evaluated at the *fire* phase, which is what lets
/// the ring keep flying in a straight line while the driver behind it
/// carries on turning.
fn blast_state(fire: i64, beat: f64) -> Option<Blast> {
    let age = beat - fire as f64;
    if !age.is_finite() || age < 0.0 {
        return None;
    }
    let q = age / BLAST_LIFE;
    if q >= 1.0 {
        return None;
    }
    let i = fire.rem_euclid(3) as usize;
    let d = &DRIVERS[i];
    let rot = driver_rot(spin_phase(fire as f64), i);
    let axis = rot.apply(v3(0.0, 0.0, 1.0));
    let travel = 1.0 - (1.0 - q).powf(2.2);
    let radius = d.scale * (0.55 + 2.9 * travel);
    Some(Blast {
        xf: Transform::IDENTITY
            .with_rot(rot)
            .with_uniform_scale(radius)
            .with_translation(d.pos + axis * (d.scale * 7.5 * travel)),
        radius,
        q,
        opacity: (1.0 - q).powf(1.3),
    })
}

fn draw_blast(ras: &mut Raster, scr: Screen, fire: i64, beat: f64, look: &Look) {
    let Some(b) = blast_state(fire, beat) else {
        return;
    };
    if !b.radius.is_finite() || b.radius <= 0.0 {
        return;
    }
    // A ring wide enough to swallow the camera would straddle the near
    // plane, and a near-plane quad is a full-screen quad. Dissolve it
    // over its own radius instead — which both looks like a wave washing
    // past and bounds the projected radius at the focal length, since
    // anything still drawn has depth >= NEAR + radius.
    let depth = -ras.view_of(b.xf.translate).z;
    if !depth.is_finite() {
        return;
    }
    let near_fade = ((depth - NEAR - b.radius) / (1.5 * b.radius)).clamp(0.0, 1.0);
    if near_fade <= 0.0 {
        return;
    }
    let col = scaled(
        mix(look.ca, look.cb, b.q),
        look.level * b.opacity * near_fade,
    );
    draw_wire_mesh(ras, &geometry().ring, &b.xf, scr, look.width_px, col);
}

// ---------------------------------------------------------------------
// the edge shader
// ---------------------------------------------------------------------

/// The camera, in the two numbers the derivative needs. Duplicated from
/// the [`Raster`] rather than borrowed out of it because the gradient has
/// to be known *before* the triangle is submitted.
///
/// `pub(crate)` with [`draw_wire_mesh`]: the wire speaker borrows this
/// module's edge look wholesale, and a re-derived copy of the analytic
/// fwidth is exactly the kind of duplicate the audit kept finding.
#[derive(Clone, Copy)]
pub(crate) struct Screen {
    cx: f64,
    cy: f64,
    focal: f64,
}

impl Screen {
    /// Build from the framebuffer the raster will draw into — call it
    /// BEFORE [`Raster::new`] borrows the framebuffer.
    pub(crate) fn of_fb(fb: &Framebuffer) -> Self {
        let cam = Camera::matching(fb);
        Screen {
            cx: fb.w as f64 * 0.5,
            cy: fb.h as f64 * 0.5,
            focal: cam.focal,
        }
    }
    /// Project a view-space point, clamping depth at the near plane
    /// rather than rejecting it. A triangle straddling the near plane
    /// still needs *some* gradient — the rasteriser will clip it properly
    /// a moment later, and a clamped corner gives a finite, bounded wire
    /// width instead of a discarded triangle or an infinity.
    fn of(&self, view: Vec3) -> Option<(f64, f64)> {
        if !view.is_finite() {
            return None;
        }
        let k = self.focal / (-view.z).max(NEAR);
        let p = (self.cx + view.x * k, self.cy - view.y * k);
        (p.0.is_finite() && p.1.is_finite()).then_some(p)
    }
}

/// Draw one mesh as glowing edges: per triangle, work out what one pixel
/// is worth in each barycentric channel, then let the rasteriser fill it
/// with a shader that lights whatever is within `width_px` of an edge.
///
/// The triangles are submitted one at a time rather than through
/// [`Raster::draw_mesh`] because the derivative is a property of the
/// triangle and the shader closure is the only thing that sees the
/// fragment — the closure has to be built around the triangle it will run
/// on.
pub(crate) fn draw_wire_mesh(
    ras: &mut Raster,
    mesh: &Mesh,
    xf: &Transform,
    scr: Screen,
    width_px: f64,
    col: (f64, f64, f64),
) {
    let opts = DrawOpts::wire();
    let width = if width_px.is_finite() {
        width_px.max(0.1)
    } else {
        0.8
    };
    let n = mesh.verts.len() as u32;
    for &[i0, i1, i2] in &mesh.tris {
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let src = [
            &mesh.verts[i0 as usize],
            &mesh.verts[i1 as usize],
            &mesh.verts[i2 as usize],
        ];
        let world = [
            xf.point(src[0].pos),
            xf.point(src[1].pos),
            xf.point(src[2].pos),
        ];
        let (Some(p0), Some(p1), Some(p2)) = (
            scr.of(ras.view_of(world[0])),
            scr.of(ras.view_of(world[1])),
            scr.of(ras.view_of(world[2])),
        ) else {
            continue;
        };
        let bary = [src[0].bary, src[1].bary, src[2].bary];
        let Some(fw) = bary_fwidth([p0, p1, p2], &bary) else {
            continue;
        };
        let tri = [
            Vertex::at(world[0]).with_bary(bary[0]),
            Vertex::at(world[1]).with_bary(bary[1]),
            Vertex::at(world[2]).with_bary(bary[2]),
        ];
        ras.draw_tri(&tri, &opts, |v: &Varyings, depth: f64| {
            edge_shade(v, depth, &fw, width, col)
        });
    }
}

/// `1 - min(smoothstep(0, fwidth·uWidth, bary))`, then a distance fade.
#[inline]
fn edge_shade(
    v: &Varyings,
    depth: f64,
    fw: &[f64; 3],
    width_px: f64,
    col: (f64, f64, f64),
) -> Option<(u8, u8, u8)> {
    let mut a = 1.0f64;
    for (&f, &b) in fw.iter().zip(v.bary.iter()) {
        // A channel that does not vary over this triangle carries no
        // edge — that is precisely how `Mesh::quad` hides its diagonal —
        // so it is dropped rather than given a zero-width threshold.
        if !f.is_finite() || f <= 0.0 {
            continue;
        }
        a = a.min(smoothstep01(b / (f * width_px)));
    }
    let edge = 1.0 - a;
    if !edge.is_finite() || edge <= EDGE_FLOOR {
        return None;
    }
    let fade = (1.0 - depth / FADE_DEPTH).clamp(0.0, 1.0);
    let k = edge * fade;
    if k <= 0.0 {
        return None;
    }
    Some((byte(col.0 * k), byte(col.1 * k), byte(col.2 * k)))
}

/// Screen-space derivative of each barycentric channel over one triangle,
/// as GLSL's `fwidth` defines it: `|d/dx| + |d/dy|`.
///
/// `None` for a degenerate triangle — the rasteriser drops those anyway,
/// and it is the reciprocal of the area that would blow up here.
fn bary_fwidth(p: [(f64, f64); 3], bary: &[[f64; 3]; 3]) -> Option<[f64; 3]> {
    let a2 = area2(p[0], p[1], p[2]);
    if !a2.is_finite() || a2.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / a2;
    // Gradient of the rasteriser's own weight for corner i: the
    // perpendicular of the opposite edge, over twice the area.
    let g = [
        (p[2].1 - p[1].1, -(p[2].0 - p[1].0)),
        (p[0].1 - p[2].1, -(p[0].0 - p[2].0)),
        (p[1].1 - p[0].1, -(p[1].0 - p[0].0)),
    ];
    let mut out = [0.0f64; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let (mut dx, mut dy) = (0.0f64, 0.0f64);
        for (b, gi) in bary.iter().zip(g.iter()) {
            dx += b[c] * gi.0 * inv;
            dy += b[c] * gi.1 * inv;
        }
        *o = dx.abs() + dy.abs();
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// Twice the signed screen area, same convention as the rasteriser's own
/// (which is private to it).
#[inline]
fn area2(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.1 - a.1) * (c.0 - a.0) - (b.0 - a.0) * (c.1 - a.1)
}

/// `smoothstep(0, 1, x)` — forwards to the shared guarded helper.
#[inline]
fn smoothstep01(x: f64) -> f64 {
    crate::pass::smoothstep(0.0, 1.0, x)
}

#[inline]
fn byte(v: f64) -> u8 {
    if v.is_finite() {
        v.clamp(0.0, 255.0) as u8
    } else {
        0
    }
}

fn mix(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (f64, f64, f64) {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let f = |x: u8, y: u8| x as f64 + (y as f64 - x as f64) * t;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

#[inline]
fn scaled(c: (f64, f64, f64), k: f64) -> (f64, f64, f64) {
    let k = if k.is_finite() { k.max(0.0) } else { 0.0 };
    (c.0 * k, c.1 * k, c.2 * k)
}

// ---------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------

/// The three driver solids and the blast ring, built once.
///
/// A `OnceLock` rather than a rebuild per frame: the geometry is
/// constant, and four `Vec` allocations sixty times a second is exactly
/// the kind of thing the rasteriser's budget note is about. It is not
/// simulation state — nothing observable depends on whether it has been
/// initialised — so it does not cost the determinism contract anything.
struct Geometry {
    drivers: [Mesh; 3],
    ring: Mesh,
}

fn geometry() -> &'static Geometry {
    static G: OnceLock<Geometry> = OnceLock::new();
    G.get_or_init(|| Geometry {
        // A ball, a diamond and a box: three silhouettes that stay
        // distinguishable while tumbling, 40 triangles between them.
        drivers: [icosahedron(), Mesh::cube(), octahedron()],
        ring: ring_mesh(),
    })
}

/// Eight faces on six corners. `Mesh::tri` gives each face the canonical
/// `[1,0,0] [0,1,0] [0,0,1]`, so all three of its edges light — correct
/// here, because every edge of a triangulated solid is a real edge.
fn octahedron() -> Mesh {
    const P: [Vec3; 6] = [
        v3(1.0, 0.0, 0.0),
        v3(-1.0, 0.0, 0.0),
        v3(0.0, 1.0, 0.0),
        v3(0.0, -1.0, 0.0),
        v3(0.0, 0.0, 1.0),
        v3(0.0, 0.0, -1.0),
    ];
    const F: [[usize; 3]; 8] = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    let mut m = Mesh::new();
    for [a, b, c] in F {
        m.tri(Vertex::at(P[a]), Vertex::at(P[b]), Vertex::at(P[c]));
    }
    m
}

/// Twenty faces on twelve corners, on the unit sphere.
fn icosahedron() -> Mesh {
    let t = (1.0 + 5.0f64.sqrt()) * 0.5;
    let p = [
        v3(-1.0, t, 0.0),
        v3(1.0, t, 0.0),
        v3(-1.0, -t, 0.0),
        v3(1.0, -t, 0.0),
        v3(0.0, -1.0, t),
        v3(0.0, 1.0, t),
        v3(0.0, -1.0, -t),
        v3(0.0, 1.0, -t),
        v3(t, 0.0, -1.0),
        v3(t, 0.0, 1.0),
        v3(-t, 0.0, -1.0),
        v3(-t, 0.0, 1.0),
    ];
    const F: [[usize; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut m = Mesh::new();
    for [a, b, c] in F {
        m.tri(
            Vertex::at(p[a].normalized()),
            Vertex::at(p[b].normalized()),
            Vertex::at(p[c].normalized()),
        );
    }
    m
}

/// The blast ring: a closed annulus in the local XY plane, so its axis is
/// local +Z and the driver's rotation aims it for free.
///
/// The barycentric attribute is hand-rigged rather than taken from
/// [`Mesh::quad`], because what should light here is not a quad's border
/// but the band's two *arcs*. Outer corners carry `[0,1,1]` and inner
/// ones `[1,0,1]`, so the smallest channel is the distance to the nearer
/// arc and peaks at 0.5 in the middle of the band — the radial seams
/// between segments never reach zero and the ring reads as two clean
/// circles instead of a gear.
fn ring_mesh() -> Mesh {
    const OUTER_B: [f64; 3] = [0.0, 1.0, 1.0];
    const INNER_B: [f64; 3] = [1.0, 0.0, 1.0];
    let mut m = Mesh::new();
    for s in 0..RING_SEGS {
        let (s0, c0) = (TAU * s as f64 / RING_SEGS as f64).sin_cos();
        let (s1, c1) = (TAU * (s + 1) as f64 / RING_SEGS as f64).sin_cos();
        let i = m.verts.len() as u32;
        for (c, s, r, b) in [
            (c0, s0, 1.0, OUTER_B),
            (c1, s1, 1.0, OUTER_B),
            (c1, s1, RING_INNER, INNER_B),
            (c0, s0, RING_INNER, INNER_B),
        ] {
            m.verts.push(Vertex::at(v3(c * r, s * r, 0.0)).with_bary(b));
        }
        m.tris.push([i, i + 1, i + 2]);
        m.tris.push([i, i + 2, i + 3]);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: (u8, u8, u8) = (0, 255, 213);
    const B: (u8, u8, u8) = (255, 40, 160);

    /// A drive with something on every input, so the reactive terms are
    /// actually exercised.
    fn hot() -> Drive {
        let mut d = Drive::default();
        d.update(0.01, 4.0, None, None);
        d
    }

    struct Rig {
        fb: Framebuffer,
        db: DepthBuffer,
    }

    fn rig(w: u32, h: u32) -> Rig {
        Rig {
            fb: Framebuffer::new(w, h),
            db: DepthBuffer::new(w, h),
        }
    }

    impl Rig {
        fn frame(&mut self, beat: f64, d: &Drive) {
            self.fb.px.fill(0);
            self.db.clear();
            wire(&mut self.fb, &mut self.db, beat, 0.8, d, A, B);
        }

        fn lit(&self) -> usize {
            self.fb.px.iter().filter(|&&v| v > 0).count()
        }

        /// Bounding-box diagonal of everything lit, in pixels. Stands in
        /// for "how big is the thing on screen".
        fn extent(&self) -> f64 {
            let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
            for y in 0..self.fb.h {
                for x in 0..self.fb.w {
                    let i = ((y * self.fb.w + x) * 3) as usize;
                    if self.fb.px[i] | self.fb.px[i + 1] | self.fb.px[i + 2] == 0 {
                        continue;
                    }
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
            if x0 == u32::MAX {
                return 0.0;
            }
            let (dx, dy) = ((x1 - x0) as f64, (y1 - y0) as f64);
            (dx * dx + dy * dy).sqrt()
        }

        /// One blast alone, so the always-present drivers do not drown
        /// out what is being measured.
        fn blast_only(&mut self, fire: i64, beat: f64, look: &Look) {
            self.fb.px.fill(0);
            self.db.clear();
            let cam = Camera::matching(&self.fb);
            let scr = Screen {
                cx: self.fb.w as f64 * 0.5,
                cy: self.fb.h as f64 * 0.5,
                focal: cam.focal,
            };
            let mut ras = Raster::new(&mut self.fb, &mut self.db, cam).unwrap();
            draw_blast(&mut ras, scr, fire, beat, look);
        }
    }

    fn look(width_px: f64) -> Look {
        Look {
            width_px,
            level: 1.0,
            ca: A,
            cb: B,
        }
    }

    #[test]
    fn same_beat_same_pixels() {
        let d = hot();
        let mut a = rig(192, 128);
        let mut b = rig(192, 128);
        a.frame(4.37, &d);
        b.frame(4.37, &d);
        assert_eq!(a.fb.px, b.fb.px);
        assert!(a.lit() > 0, "nothing was drawn");
    }

    /// The test the scan exists for. A pool would have fired and animated
    /// rings on the way to beat 5 and could not take them back; a scan
    /// simply does not see them.
    #[test]
    fn reverse_time_reproduces_the_earlier_frame() {
        let d = hot();
        let mut scrubbed = rig(160, 112);
        scrubbed.frame(5.0, &d);
        scrubbed.frame(3.0, &d);

        let mut direct = rig(160, 112);
        direct.frame(3.0, &d);
        assert_eq!(scrubbed.fb.px, direct.fb.px);
        assert!(direct.lit() > 0, "nothing was drawn");
    }

    #[test]
    fn a_beat_fires_a_blast_and_it_expands() {
        // Not yet fired: nothing at all.
        assert!(blast_state(4, 3.99).is_none());
        // Expired: the pool slot is dead again after its life.
        assert!(blast_state(4, 4.0 + BLAST_LIFE).is_none());

        // The model expands and departs, monotonically, over its life.
        let origin = DRIVERS[4usize.rem_euclid(3)].pos;
        let mut prev = (0.0f64, -1.0f64);
        for step in 0..12 {
            let b = blast_state(4, 4.0 + step as f64 * 0.25).expect("blast should be alive");
            let travelled = (b.xf.translate - origin).length();
            assert!(b.radius > prev.0, "radius shrank at step {step}");
            assert!(travelled > prev.1, "blast came back at step {step}");
            prev = (b.radius, travelled);
        }

        // And on screen. Every driver's blast grows in projected size,
        // whichever way its axis happens to be pointing when it fires.
        for fire in 3..6i64 {
            let mut r = rig(320, 320);
            r.blast_only(fire, fire as f64 + 0.08, &look(1.2));
            let (early_lit, early) = (r.lit(), r.extent());
            r.blast_only(fire, fire as f64 + 0.75, &look(1.2));
            let late = r.extent();
            assert!(early_lit > 30, "blast {fire} barely drew: {early_lit}");
            assert!(
                late > early,
                "blast {fire} did not expand: {early} -> {late}"
            );
        }

        // Nothing at all once the slot has expired.
        let mut r = rig(320, 320);
        r.blast_only(3, 3.0 + BLAST_LIFE + 0.01, &look(1.2));
        assert_eq!(r.lit(), 0, "a dead blast is still lit");
    }

    #[test]
    fn wire_width_responds_to_react() {
        // `uWidth` is the original's, verbatim.
        let width = |react: f64| 0.8 + 0.8 * react;
        assert!((width(0.0) - 0.8).abs() < 1e-12);
        assert!((width(1.0) - 1.6).abs() < 1e-12);

        // And it reaches the pixels: same geometry, same level, wider
        // wire lights strictly more of the frame.
        let lit_at = |w: f64| {
            let mut r = rig(320, 240);
            r.fb.px.fill(0);
            r.db.clear();
            let cam = Camera::matching(&r.fb);
            let scr = Screen {
                cx: r.fb.w as f64 * 0.5,
                cy: r.fb.h as f64 * 0.5,
                focal: cam.focal,
            };
            {
                let mut ras = Raster::new(&mut r.fb, &mut r.db, cam).unwrap();
                draw_drivers(&mut ras, scr, 4.2, &look(w));
            }
            r.lit()
        };
        let thin = lit_at(width(0.0));
        let fat = lit_at(width(1.0));
        assert!(thin > 0, "the thin wire drew nothing");
        assert!(fat > thin, "wire did not thicken: {thin} -> {fat}");

        // The whole path agrees: a live drive with an onset on it is
        // wider than the same drive settled, at the same beat.
        let mut cold = Drive::default();
        cold.update(0.01, 4.6, None, None);
        cold.update(0.4, 4.9, None, None); // let the beat pulse decay away
        let mut warm = cold;
        warm.hit = 1.0;
        assert!(cold.gbeat() < 0.01 && warm.hit * 0.85 > 0.8, "rig is wrong");
        let mut a = rig(256, 192);
        let mut b = rig(256, 192);
        a.frame(4.9, &cold);
        b.frame(4.9, &warm);
        assert!(b.lit() > a.lit(), "react did not widen the wire");
    }

    #[test]
    fn accumulation_saturates_and_never_wraps() {
        let d = hot();
        let mut r = rig(160, 120);
        r.fb.px.fill(0);
        r.db.clear();
        let mut prev = r.fb.px.clone();
        for _ in 0..200 {
            wire(&mut r.fb, &mut r.db, 4.1, 1.0, &d, A, B);
            for (now, was) in r.fb.px.iter().zip(prev.iter()) {
                assert!(now >= was, "channel went down: {was} -> {now}");
            }
            prev.copy_from_slice(&r.fb.px);
        }
        assert!(r.fb.px.contains(&255), "should have saturated");
        // A shader cannot smuggle a wrap through either: an absurd colour
        // clamps at the byte conversion.
        assert_eq!(byte(1e9), 255);
        assert_eq!(byte(f64::NAN), 0);
        assert_eq!(byte(-4.0), 0);
    }

    #[test]
    fn nothing_panics_on_degenerate_input() {
        let d = hot();
        for (w, h) in [(0, 0), (1, 1), (1, 64), (64, 1), (2, 3), (7, 5)] {
            let mut r = rig(w, h);
            for beat in [
                -9.5,
                0.0,
                4.01,
                1e12,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                r.frame(beat, &d);
            }
            assert_eq!(r.fb.px.len(), (w * h * 3) as usize);
        }
        // The scan's arithmetic cannot overflow at the ends of the range.
        for beat in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, 1e300] {
            let latest = beat.floor() as i64;
            for k in 0..BLAST_POOL {
                assert!(blast_state(latest.saturating_sub(k), beat).is_none() || beat.is_finite());
            }
        }
        // A degenerate triangle has no derivative and is skipped, not
        // divided by.
        assert!(bary_fwidth([(0.0, 0.0); 3], &[[1.0, 0.0, 0.0]; 3]).is_none());
        assert!(
            bary_fwidth(
                [(0.0, 0.0), (f64::NAN, 1.0), (1.0, 0.0)],
                &[[1.0, 0.0, 0.0]; 3]
            )
            .is_none()
        );
    }

    #[test]
    fn a_constant_channel_carries_no_edge() {
        // `Mesh::quad`'s diagonal suppression works by making a channel
        // constant over a triangle. That channel must drop out of the
        // min rather than being handed a zero-width threshold.
        let p = [(0.0, 0.0), (32.0, 0.0), (0.0, 32.0)];
        let quad_bary = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let fw = bary_fwidth(p, &quad_bary).expect("non-degenerate");
        assert!(fw[1].abs() < 1e-12, "constant channel had a gradient");
        assert!(fw[0] > 0.0 && fw[2] > 0.0);

        // A pixel deep inside, with the constant channel at 1: it must
        // not light.
        let v = Varyings {
            x: 8,
            y: 8,
            uv: [0.0; 2],
            normal: v3(0.0, 0.0, 1.0),
            bary: [0.5, 1.0, 0.5],
            pos_view: v3(0.0, 0.0, -420.0),
        };
        assert!(edge_shade(&v, 420.0, &fw, 1.0, (255.0, 255.0, 255.0)).is_none());
    }

    #[test]
    fn the_tumble_is_closed_form_and_monotone() {
        // The property the blast rings depend on: an orientation at a
        // past beat is recoverable at any later beat.
        assert_eq!(spin_phase(3.0), spin_phase(3.0));
        assert_eq!(
            driver_rot(spin_phase(3.0), 0),
            driver_rot(spin_phase(3.0), 0)
        );
        // And it only ever advances.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..400 {
            let p = spin_phase(i as f64 * 0.25);
            assert!(p > prev, "phase went backwards at {i}");
            prev = p;
        }
        assert_eq!(spin_phase(f64::NAN), 0.0);
        assert_eq!(pulse_integral(-3.0, 1.0, 0.14), 0.0);
    }

    #[test]
    fn geometry_stays_inside_the_budget() {
        let g = geometry();
        let driver_tris: usize = g.drivers.iter().map(|m| m.tris.len()).sum();
        assert_eq!(driver_tris, 40, "driver triangle count moved");
        assert_eq!(g.ring.tris.len(), RING_SEGS * 2);
        // Worst case a frame can reach: every pool slot alive at once.
        let worst = driver_tris + BLAST_POOL as usize * g.ring.tris.len();
        assert!(worst < 620, "worst-case triangle count is {worst}");
        // What it actually reaches at the reference tempo.
        let alive = (0..BLAST_POOL)
            .filter(|k| blast_state(20 - k, 20.4).is_some())
            .count();
        assert_eq!(alive, 3, "expected three live blasts, got {alive}");
    }
}
