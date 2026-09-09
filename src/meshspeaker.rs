//! MESH SPEAKER — EasyPngVJ's `speaker` mode, on the software rasteriser.
//!
//! A wall of 23 woofers hanging in the dark, and a kick that **rolls
//! outward through it**: the hero cone punches first, the inner ring of
//! eight follows a moment later, the outer ring of fourteen later still.
//! That delay is the whole effect. Fire all 23 on the same frame and it
//! reads as one flat throb; stagger them and the room has a direction —
//! the kick starts at your face and leaves through the back wall. It is
//! the best-looking of the three mesh modes for exactly that reason, and
//! every other decision here is in service of it.
//!
//! Over there the mode decoded a baked mesh payload and drew it as 23
//! `THREE.Mesh` instances of one lit material. The rig, unchanged: one
//! hero at `z = -420`, a ring of 8 at radius 660 on `z = -820`, a ring of
//! 14 at radius 1140 on `z = -1180`, the whole group rolling slowly about
//! the view axis.
//!
//! # The woofer
//!
//! **Generated, not decoded.** A baked payload is a few hundred kilobytes
//! of vertices nobody can edit, and the shading the original ran on it
//! only ever asks two questions — *how far out from the axis is this
//! fragment* and *which way does it face* — so the mesh only has to be a
//! believable driver, not a specific one. [`woofer`] sweeps a 12-knot
//! cross-section around the axis: a domed dust cap, a flared cone, a
//! half-roll surround, a basket flange, and a closed cabinet behind. The
//! sweep is per-rank — 28 segments for the hero, 16 for the inner ring,
//! 10 for the outer — because the outer ring is fifty pixels across and
//! paying hero tessellation for it is paying for nothing. That comes to
//! 5,920 triangles for the whole rig, and every one of them costs a
//! transform of three vertices in [`Raster::draw_mesh`], so the count
//! is worth caring about at this instance count.
//!
//! Closing the cabinet costs a back plate that is always culled, and buys
//! a **watertight** solid: with backface culling on, every pixel inside a
//! woofer's silhouette is covered by exactly one front face, so the depth
//! test cannot open a hole in it. There is a test for that.
//!
//! # The delay: beats, not frames
//!
//! The original kept a 48-slot ring buffer of past drive values and each
//! ring read `history[head - delay]`, `delay` being 5 frames for the
//! inner ring and 10 for the outer. That is genuine state, and state is
//! what this project does not keep: scrub the jog wheel backwards and a
//! history buffer has already eaten those frames and cannot give them
//! back, so the rig would go dead for 10 frames on every reverse and
//! desync from a scene the rest of the pixel tier reproduces exactly.
//! `pixparticles::rings` had the same problem with its event pool and
//! solved it by scanning the clock instead.
//!
//! So the delay is **expressed in beats**: the inner ring samples the
//! drive at `beat - 1/6`, the outer at `beat - 1/3`. At the reference
//! tempo those are 83 ms and 167 ms — 5 and 10 frames at 60 fps, the
//! original's numbers exactly — and off it they scale, so the roll-out
//! stays a sixteenth and an eighth behind the kick instead of drifting
//! into it at 170 BPM and lagging a bar behind at 60. Arguably the more
//! musical reading of what the original was reaching for.
//!
//! **What it costs.** A past drive value cannot be recovered exactly
//! without storing it, so [`ring_drive`] reconstructs it: the grid-locked
//! envelope has a closed form (a unit pulse on each beat line decaying at
//! the [`Drive`]'s own tau), and the live drive scalar supplies its
//! amplitude. Grid-locked content — which is all of it when running dry,
//! and the kick itself when running on audio — replays exactly. An onset
//! that lands *off* the grid replays with the right amplitude and the
//! wrong shape: the rings still fire, still late, but the far ring's
//! flinch is a beat-shaped decay rather than that onset's own. Given the
//! alternative is a rig that stutters on every scrub, that is the trade
//! worth taking.
//!
//! # Determinism
//!
//! Every position, punch and shake is a closed-form function of `beat`
//! and a stable index. The cabinet shake seeds [`crate::rng::hash3`] with
//! the instance's ring, slot, axis and a 1/32-beat tick — groove-locked
//! flicker, the same trick the cell tier uses, and it re-derives the same
//! jitter when time runs backwards over it.
//!
//! # Cost
//!
//! Measured, release build, M-series Air, 1200x700, the full 23-instance
//! rig on a live kick: **2.9 ms a frame**. Of that, the same geometry
//! with a trivial shader is 1.3 ms — inside the 1.6 ms the rasteriser's
//! own 23-instance probe recorded — and the other 1.6 ms is the four-term
//! fragment shading the probe did not have, over about 165k covered
//! fragments. The mode is fill-bound, so it is only that expensive on a
//! full-screen pane; the cost falls with the area.
//!
//! What that bought, and what it cost:
//!
//! - The **hero's size is fixed by the original** (300 at `z = -420`, so
//!   272 pixels across at this pane), and it alone is a third of the
//!   fill. The ring sizes are ours and are set where the ranks still read
//!   as three ranks of a rig rather than a wallpaper — 135 and 105, not
//!   the hero's 300, which would have cost four times as much.
//! - The shader **does not divide**, computes the depth fade per rank
//!   rather than per fragment, gates the two radial terms on a compare,
//!   and leaves the normal unnormalised. See [`shade`].
//!
//! Opaque, so this **writes** colour rather than adding: call it before
//! whatever else wants the frame, and let the plate behind show through
//! where nothing was drawn. It clears the depth buffer itself and leaves
//! the framebuffer alone.

// Wired into the mixer with the other mesh modes; dead until then.
#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::drive::Drive;
use crate::graphics::Framebuffer;
use crate::raster::{
    CAM_Z, Camera, DepthBuffer, DrawOpts, Mesh, Raster, Transform, Varyings, Vec3, Vertex, v3,
};
use crate::rng::{hash3, unit_f64};

/// How far the plate behind dims while this mode is up. The rig is
/// opaque and lit, and a plate at full brightness behind it reads as a
/// second, brighter scene rather than as the room the speakers are in.
/// The caller applies it; it lives here because it belongs to the look.
pub const PLATE_DIM: f64 = 0.40;

/// Reference tempo, as in `pixparticles`: the rig is clocked on beats,
/// and this is what turns them back into the seconds the original's
/// decay constants and roll rate were written in.
const SEC_PER_BEAT: f64 = 0.5;

/// Mirrors `drive::TAU_BEAT`, which is private. The delayed rings replay
/// that envelope, so if one moves, move both.
const TAU_BEAT: f64 = 0.070;

/// The drive scalar's ceiling, straight from the original.
const DRIVE_MAX: f64 = 1.6;

/// Cone excursion per unit drive, in model radii. At full drive the cone
/// travels a third of its own radius — absurd for a real driver, correct
/// for this one.
const PUNCH: f64 = 0.20;

/// Cabinet shake, forward surge and swell per unit drive, in world units
/// and as a scale fraction.
const SHAKE: f64 = 11.0;
const SURGE: f64 = 34.0;
const SWELL: f64 = 0.09;

/// Shake re-rolls this often. 32 steps a beat is ~64 Hz at the reference
/// tempo — a fresh jitter about every frame, which is what the original
/// got for free by rolling per frame, but keyed to the clock so a scrub
/// reproduces it.
const JITTER_PER_BEAT: f64 = 32.0;

/// The whole rig rolls about the view axis: `sin(t * RATE) * AMOUNT`.
const RIG_ROLL_RATE: f64 = 0.31;
const RIG_ROLL: f64 = 0.22;

/// Depth fade. The rig is three ranks deep and reads flat without one —
/// the far ring competes with the hero instead of sitting behind it.
/// Not in the original, which had scene fog doing the same job.
const FOG_NEAR: f64 = 500.0;
const FOG_SPAN: f64 = 2600.0;
const FOG_FLOOR: f64 = 0.25;

/// Key and fill directions, unit to within the rounding. Key is up and
/// to the left of the camera, fill is low and opposite, which is what
/// gives the cones a readable inside.
const KEY: Vec3 = v3(-0.420, 0.681, 0.600);
const FILL: Vec3 = v3(0.620, -0.300, 0.724);

/// One rank of the rig.
struct Rank {
    count: u32,
    /// Ring radius in world units; the hero's is zero.
    radius: f64,
    z: f64,
    /// Model scale — the driver's outer radius is ~0.97, so this is very
    /// nearly the cabinet's half-width in world units.
    size: f64,
    /// How far behind the kick this rank runs, in beats.
    delay: f64,
    /// Sweep segments. Level of detail: the outer rank is a few dozen
    /// pixels across.
    seg: usize,
}

/// The rig, near to far. Drawn in this order so the near ranks fill the
/// depth buffer first and the far ones reject against it.
///
/// The original fixed the hero at size 300; the ring sizes are ours, and
/// chosen so each rank reads as a step further back rather than as a
/// wallpaper of identical cabinets.
const RIG: [Rank; 3] = [
    Rank {
        count: 1,
        radius: 0.0,
        z: -420.0,
        size: 300.0,
        delay: 0.0,
        seg: 28,
    },
    Rank {
        count: 8,
        radius: 660.0,
        z: -820.0,
        size: 135.0,
        delay: 1.0 / 6.0,
        seg: 16,
    },
    Rank {
        count: 14,
        radius: 1140.0,
        z: -1180.0,
        size: 105.0,
        delay: 1.0 / 3.0,
        seg: 10,
    },
];

/// SPEAKER — the wall of woofers, and the kick rolling out through it.
///
/// Writes opaque pixels and clears `depth` itself; leaves `fb` untouched
/// where nothing was drawn, so the dimmed plate behind shows through.
pub fn speaker(
    fb: &mut Framebuffer,
    depth: &mut DepthBuffer,
    beat: f64,
    intensity: f64,
    d: &Drive,
    ca: (u8, u8, u8),
    cb: (u8, u8, u8),
) {
    let beat = if beat.is_finite() { beat } else { 0.0 };
    let intensity = if intensity.is_finite() {
        intensity.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let cam = Camera::matching(fb);
    let Some(mut ras) = Raster::new(fb, depth, cam) else {
        return;
    };
    ras.clear_depth();

    let roll = (beat * SEC_PER_BEAT * RIG_ROLL_RATE).sin() * RIG_ROLL;
    let tick = (beat * JITTER_PER_BEAT).floor() as i64 as u64;
    let level = (0.35 + 0.65 * intensity) * (0.85 + 0.35 * d.gbeat());
    // UV carries the normalised radius every shading term is a function
    // of, so it has to interpolate; the wireframe attribute does not.
    let opts = DrawOpts::lit().with_interp(true, true, false);
    let (caf, cbf) = (rgbf(ca), rgbf(cb));

    for (k, rank) in RIG.iter().enumerate() {
        // One mesh per rank, not per instance: the cone displacement is a
        // function of the rank's delayed drive, and every cabinet in a
        // rank shares it. Twenty-three draws, three builds.
        let v = ring_drive(d, beat, rank.delay);
        let punch = PUNCH * v;
        let mesh = woofer(punch, rank.seg);
        // The depth fade is per *rank*, not per fragment: a rank is one
        // plane a cabinet's depth deep, and a divide on every one of a
        // hundred thousand fragments to resolve that is a divide wasted.
        let fog = (1.0 - (CAM_Z - rank.z - FOG_NEAR) * (1.0 / FOG_SPAN)).clamp(FOG_FLOOR, 1.0);
        let look = Look {
            ca: caf,
            cb: cbf,
            body: (caf.0 - cbf.0, caf.1 - cbf.1, caf.2 - cbf.2),
            hi: mix(caf, (255.0, 255.0, 255.0), 0.45),
            level: level * fog,
            punch,
        };
        for i in 0..rank.count {
            let slot = (i as f64 + 0.5 * k as f64) / rank.count.max(1) as f64;
            let (s, c) = (TAU * slot + roll).sin_cos();
            let pos = v3(
                rank.radius * c + jitter(k, i, 0, tick) * SHAKE * v,
                rank.radius * s + jitter(k, i, 1, tick) * SHAKE * v,
                rank.z + SURGE * v,
            );
            let xf = Transform::from_euler(0.0, 0.0, roll)
                .with_uniform_scale(rank.size * (1.0 + SWELL * v))
                .with_translation(pos);
            ras.draw_mesh(&mesh, &xf, &opts, |vy, _| shade(vy, &look));
        }
    }
}

// ---------------------------------------------------------------------
// drive
// ---------------------------------------------------------------------

/// The drive scalar this rank runs on, `delay` beats behind the kick.
///
/// `delay == 0` is the hero and is the original's expression verbatim.
/// Behind it, see the module docs: the beat grid supplies the envelope's
/// *shape* at the delayed instant and the live drive supplies its
/// *amplitude*, which is exact for everything on the grid and an
/// approximation for anything that is not.
fn ring_drive(d: &Drive, beat: f64, delay: f64) -> f64 {
    let react = d.gbeat().max(d.hit * 0.85);
    let live = (1.15 * d.thump + 0.85 * react + 0.5 * d.hit).clamp(0.0, DRIVE_MAX);
    if !delay.is_finite() || delay <= 0.0 || !beat.is_finite() {
        return live;
    }
    let now = pulse_env(beat);
    if now <= 0.0 {
        return 0.0;
    }
    // `live / now` is the amplitude the current pulse was fired at. Both
    // terms shrink together through the beat, so the ratio stays bounded
    // even deep into it; the cap catches an off-grid onset inflating it.
    let peak = (live / now).min(DRIVE_MAX);
    (peak * pulse_env(beat - delay)).clamp(0.0, DRIVE_MAX)
}

/// A unit pulse fired on every beat line, decaying at the [`Drive`]'s
/// beat tau. The shape the delayed ranks replay.
fn pulse_env(beat: f64) -> f64 {
    if !beat.is_finite() {
        return 0.0;
    }
    (-beat.rem_euclid(1.0) * SEC_PER_BEAT / TAU_BEAT).exp()
}

/// Cabinet shake, in `[-1, 1]`. Ring, slot, axis and clock tick — no
/// wall time, so a scrub back over a frame re-rolls the same jitter.
fn jitter(rank: usize, slot: u32, axis: u64, tick: u64) -> f64 {
    let id = ((rank as u64) << 8) | slot as u64;
    unit_f64(hash3(id, axis, tick)) * 2.0 - 1.0
}

// ---------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------

/// Knots in the cross-section, and therefore bands in the sweep.
const KNOTS: usize = 12;
const BANDS: usize = KNOTS - 1;
/// Sweep segments the stack ring buffer is sized for.
const MAX_SEG: usize = 32;

/// Cross-section constants, model units. The driver's outer radius is
/// ~0.97 so the whole cabinet sits inside the unit box the instance
/// transform scales, the way [`Mesh::cube`] does.
const CAP_R: f64 = 0.26;
const CAP_H: f64 = 0.15;
const CAP_Z: f64 = -0.12;
const CONE_R: f64 = 0.60;
const CONE_RISE: f64 = 0.26;
const CONE_Z: f64 = CAP_Z + CONE_RISE;
const SUR_C: f64 = 0.73;
const SUR_R: f64 = 0.13;
const SUR_H: f64 = 0.075;
const SUR_END_R: f64 = SUR_C + SUR_R;
const SUR_END_Z: f64 = 0.10;
const BASKET_R: f64 = 0.97;
const FLANGE_Z: f64 = 0.06;
const BACK_Z: f64 = -0.60;

/// Cosine of the smoothing angle. Steeper than this between two bands and
/// the knot between them keeps a crease — the cone and the surround want
/// to be smooth, the basket's corners very much do not.
const COS_CREASE: f64 = 0.64;

/// How much of `punch` reaches a point on the cross-section.
///
/// The original's two smoothsteps, unchanged. The radial one is 1 out to
/// `r = 0.58` and gone by `0.80`, so the **cone travels as a body** and
/// the **surround stretches across the falloff**, dying at the bead where
/// it is glued to the basket — which is what a surround is for. The
/// axial one keeps the *back* of the cabinet still: without it the closed
/// back plate, which is inside `r = 0.58` for most of its span, would
/// pump along with the cone.
fn cone_weight(r: f64, z: f64) -> f64 {
    smoothstep(0.80, 0.58, r) * smoothstep(-0.52, -0.30, z)
}

/// The woofer in cross-section, `punch` already applied.
///
/// Traversed from the dust cap's apex outward over the cone, surround and
/// basket, back along the cabinet wall and in to the centre of the back
/// plate — one continuous sweep, so the winding and the normals come out
/// consistent without a table of face orientations.
fn profile(punch: f64) -> [(f64, f64); KNOTS] {
    let mut p = [(0.0, 0.0); KNOTS];
    // Dust cap: a dome over the voice coil, flat-topped on the axis so it
    // does not come to a point.
    for (j, knot) in p.iter_mut().enumerate().take(3) {
        let u = j as f64 / 2.0;
        *knot = (CAP_R * u, CAP_Z + CAP_H * (FRAC_PI_2 * u).cos());
    }
    // Cone: the coil neck out to the rim, flaring forward.
    for (j, knot) in p[3..5].iter_mut().enumerate() {
        let u = (j + 1) as f64 / 2.0;
        *knot = (
            CAP_R + (CONE_R - CAP_R) * u,
            CAP_Z + CONE_RISE * u.powf(1.35),
        );
    }
    // Surround: a half-roll bulging forward of the cone rim.
    for (j, knot) in p[5..8].iter_mut().enumerate() {
        let t = (j + 1) as f64 / 3.0;
        let (s, c) = (PI * t).sin_cos();
        *knot = (
            SUR_C - SUR_R * c,
            CONE_Z + (SUR_END_Z - CONE_Z) * t + SUR_H * s,
        );
    }
    p[8] = (BASKET_R, FLANGE_Z); // basket flange, the cabinet's face
    p[9] = (BASKET_R, BACK_Z); // cabinet wall
    p[10] = (SUR_END_R, BACK_Z - 0.06); // back chamfer
    p[11] = (0.0, BACK_Z - 0.08); // back plate, on the axis
    for knot in p.iter_mut() {
        knot.1 += cone_weight(knot.0, knot.1) * punch;
    }
    p
}

/// Cross-section normals, per band and per end, in the `(radial, z)`
/// half-plane.
///
/// Per band rather than per knot so a hard edge can keep two normals.
/// A band's own face normal is `(-dz, dr)` — outward, because the sweep
/// runs front to back — and a knot smooths across only when its two
/// bands agree to within [`COS_CREASE`]. Anywhere they do not, the knot
/// keeps a crease and the cabinet gets its corners back.
fn band_normals(p: &[(f64, f64); KNOTS]) -> [((f64, f64), (f64, f64)); BANDS] {
    let mut face = [(0.0, 0.0); BANDS];
    for (b, f) in face.iter_mut().enumerate() {
        *f = norm2(p[b].1 - p[b + 1].1, p[b + 1].0 - p[b].0);
    }
    let mut out = [((0.0, 0.0), (0.0, 0.0)); BANDS];
    for (b, o) in out.iter_mut().enumerate() {
        let lo = match b.checked_sub(1) {
            Some(prev) if smooths(face[prev], face[b]) => mean2(face[prev], face[b]),
            _ => face[b],
        };
        let hi = match face.get(b + 1) {
            Some(&next) if smooths(face[b], next) => mean2(face[b], next),
            _ => face[b],
        };
        *o = (lo, hi);
    }
    out
}

/// Do two neighbouring bands lie flat enough to share a normal?
fn smooths(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 * b.0 + a.1 * b.1 > COS_CREASE
}

/// The woofer, swept. `seg` segments around, [`KNOTS`] knots along.
///
/// `uv.x` carries the knot's radius — the one quantity every shading term
/// in this mode is a function of — and `uv.y` the position along the
/// cross-section, which the shading does not read but which comes free
/// once the pair is interpolated.
fn woofer(punch: f64, seg: usize) -> Mesh {
    let seg = seg.clamp(6, MAX_SEG);
    let p = profile(punch);
    let n = band_normals(&p);

    // Half a segment of phase, and it is not cosmetic. A seam that runs
    // exactly through pixel centres — a vertical, horizontal or 45°
    // one, all of which a sweep starting at angle zero puts through the
    // middle of a centred instance — lands on the rasteriser's fill-rule
    // tie-break with an edge function of zero on both sides, and one bit
    // of rounding then rejects the pixel from *both* triangles. The
    // result is a one-pixel crack straight down the hero's face. Offset
    // the sweep and no seam is axis- or diagonal-aligned.
    let mut ring = [(0.0, 0.0); MAX_SEG];
    for (k, slot) in ring.iter_mut().enumerate().take(seg) {
        let (s, c) = (TAU * (k as f64 + 0.5) / seg as f64).sin_cos();
        *slot = (c, s);
    }

    let mut m = Mesh::new();
    for (b, &(nlo, nhi)) in n.iter().enumerate() {
        let ((r0, z0), (r1, z1)) = (p[b], p[b + 1]);
        let (t0, t1) = (b as f64 / BANDS as f64, (b + 1) as f64 / BANDS as f64);
        for k in 0..seg {
            let (c0, s0) = ring[k];
            let (c1, s1) = ring[(k + 1) % seg];
            let lo = |c: f64, s: f64| sweep(r0, z0, nlo, c, s, t0);
            let hi = |c: f64, s: f64| sweep(r1, z1, nhi, c, s, t1);
            // `a b c d` cyclic, counter-clockwise seen from outside: out
            // along the cross-section, one segment around, back in.
            if r0 <= 0.0 {
                m.tri(lo(c0, s0), hi(c0, s0), hi(c1, s1));
            } else if r1 <= 0.0 {
                m.tri(lo(c0, s0), hi(c0, s0), lo(c1, s1));
            } else {
                m.quad(lo(c0, s0), hi(c0, s0), hi(c1, s1), lo(c1, s1));
            }
        }
    }
    m
}

/// One swept vertex. On the axis the radial part of the normal is
/// meaningless — every segment would claim a different one and the apex
/// would shade as a flower — so it points straight along z there.
fn sweep(r: f64, z: f64, n: (f64, f64), c: f64, s: f64, t: f64) -> Vertex {
    let normal = if r <= 0.0 {
        v3(0.0, 0.0, if n.1 < 0.0 { -1.0 } else { 1.0 })
    } else {
        v3(n.0 * c, n.0 * s, n.1)
    };
    Vertex::at(v3(r * c, r * s, z))
        .with_normal(normal)
        .with_uv(r, t)
}

// ---------------------------------------------------------------------
// shading
// ---------------------------------------------------------------------

/// Everything the fragment shader needs that is constant over a rank.
///
/// The colour work that does not depend on the fragment is done once
/// here: the body mix as a difference so it is a multiply-add per
/// channel, the surround's whitened highlight outright, and the depth
/// fade folded into `level`.
struct Look {
    ca: (f64, f64, f64),
    cb: (f64, f64, f64),
    /// `ca - cb`, the body mix's slope.
    body: (f64, f64, f64),
    /// Accent A pulled toward white — the surround's highlight.
    hi: (f64, f64, f64),
    level: f64,
    punch: f64,
}

/// Key/fill, rim, the surround's highlight and the voice coil's glow.
///
/// The original's terms, with one substitution: its `pow(1 - n.z, 2.6)`
/// rim is `x²·√x` here. A `powf` is tens of nanoseconds and this runs on
/// every fragment of 23 instances; the exponent is 2.5 instead of 2.6 and
/// nobody will ever see the difference.
///
/// The normal is used **unnormalised**. Interpolated across a facet of a
/// mesh this smooth it is within a fraction of a percent of unit length,
/// and that is a square root a frame's worth of fragments does not need.
///
/// Nothing in here divides. Every threshold is a literal, so
/// [`smoothstep`] folds its reciprocal at compile time; a division is a
/// dozen-odd cycles that cannot be pipelined and there were four of them
/// in the obvious spelling of this function. The final cast leans on
/// Rust's saturating float-to-int conversion instead of clamping: out of
/// range goes to 0 or 255, and `NaN` goes to 0, which is what a clamp
/// would have had to spell out three times.
#[inline]
fn shade(v: &Varyings, k: &Look) -> Option<(u8, u8, u8)> {
    let n = v.normal;
    let key = n.dot(KEY).max(0.0);
    let fill = n.dot(FILL).max(0.0);
    let edge = (1.0 - n.z).clamp(0.0, 1.0);
    let rim = edge * edge * edge.sqrt() * 0.55;

    // The surround's ring highlight, and the coil glowing through the
    // dust cap on the punch — the tell that the cone is moving even when
    // the excursion itself is head-on and hard to read. Both are radial
    // bands, and both are gated on the radius first: a fragment outside
    // one pays a compare rather than a pair of Hermite steps, and the
    // branch is per-band and so nearly always predicted.
    let r = v.uv[0];
    let band = if r > 0.60 && r < 0.86 {
        smoothstep(0.60, 0.71, r) * smoothstep(0.86, 0.73, r)
    } else {
        0.0
    };
    let coil = if r < 0.34 && k.punch != 0.0 {
        smoothstep(0.34, 0.13, r) * k.punch.abs() * 3.2
    } else {
        0.0
    };

    // Fold the rank's brightness into every weight, so the three channels
    // below are four multiply-adds each and no final scale.
    let s = k.level;
    let lit = (0.12 + 0.92 * key + 0.38 * fill) * s;
    let tint = (0.18 + 0.62 * key).min(1.0) * lit;
    let spec = band * (0.22 + 0.55 * key) * s;
    let rim = rim * s;
    // The body is accent B pulled toward A by the key light, and the
    // coil's glow rides accent B with it — so a punch reads as the driver
    // lighting up from inside rather than as a white spot stuck on it.
    let base = lit + coil * s;

    let chan = |cb: f64, slope: f64, ca: f64, hi: f64| -> u8 {
        (cb * base + slope * tint + ca * rim + hi * spec) as u8
    };
    Some((
        chan(k.cb.0, k.body.0, k.ca.0, k.hi.0),
        chan(k.cb.1, k.body.1, k.ca.1, k.hi.1),
        chan(k.cb.2, k.body.2, k.ca.2, k.hi.2),
    ))
}

// ---------------------------------------------------------------------
// small maths
// ---------------------------------------------------------------------

/// Hermite step, and reversed edges give a reversed step — which is how
/// the original writes its falloffs.
///
/// Spelled as a reciprocal and a multiply rather than a divide: every
/// call site passes literal edges, so the reciprocal is folded at compile
/// time and the fragment shader does no division at all.
#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    if e0 == e1 {
        return if x < e0 { 0.0 } else { 1.0 };
    }
    let inv = 1.0 / (e1 - e0);
    let t = ((x - e0) * inv).clamp(0.0, 1.0);
    if !t.is_finite() {
        return 0.0;
    }
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn norm2(x: f64, y: f64) -> (f64, f64) {
    let len = (x * x + y * y).sqrt();
    if len.is_finite() && len > 0.0 {
        (x / len, y / len)
    } else {
        (0.0, 1.0)
    }
}

#[inline]
fn mean2(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    norm2(a.0 + b.0, a.1 + b.1)
}

fn rgbf(c: (u8, u8, u8)) -> (f64, f64, f64) {
    (c.0 as f64, c.1 as f64, c.2 as f64)
}

fn mix(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);
    let f = |x: f64, y: f64| x + (y - x) * t;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: (u8, u8, u8) = (0, 255, 213);
    const B: (u8, u8, u8) = (255, 40, 160);

    /// A drive with a kick on it, as of `beat`.
    fn hot(beat: f64) -> Drive {
        let mut d = Drive::default();
        d.update(0.01, beat, None, None);
        d
    }

    fn frame(w: u32, h: u32, beat: f64, d: &Drive) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        let mut db = DepthBuffer::new(w, h);
        speaker(&mut fb, &mut db, beat, 0.9, d, A, B);
        fb
    }

    #[test]
    fn same_beat_same_pixels() {
        let d = hot(4.0);
        let a = frame(160, 120, 4.05, &d);
        let b = frame(160, 120, 4.05, &d);
        assert_eq!(a.px, b.px);
        assert!(a.px.iter().any(|&v| v > 0), "nothing was drawn");
        // And a different beat is a different frame — the rig rolls, the
        // cones move, the shake re-rolls.
        let c = frame(160, 120, 4.30, &d);
        assert_ne!(a.px, c.px, "the rig is frozen");
    }

    #[test]
    fn a_scrub_backwards_reproduces_the_earlier_frame() {
        // The whole reason the delay is a beat offset and not a history
        // buffer: reaching a beat from the future must look identical to
        // reaching it from the past.
        let d = hot(4.0);
        let mut fb = Framebuffer::new(96, 72);
        let mut db = DepthBuffer::new(96, 72);
        // The mode is opaque and does not clear colour — the caller paints
        // the plate first, so the test has to as well or the comparison is
        // against last frame's leftovers.
        fb.px.fill(9);
        speaker(&mut fb, &mut db, 9.75, 0.9, &d, A, B);
        fb.px.fill(9);
        speaker(&mut fb, &mut db, 4.12, 0.9, &d, A, B);

        let mut direct = Framebuffer::new(96, 72);
        let mut ddb = DepthBuffer::new(96, 72);
        direct.px.fill(9);
        speaker(&mut direct, &mut ddb, 4.12, 0.9, &d, A, B);
        assert_eq!(fb.px, direct.px);
    }

    /// The signature move. Sweep a real drive across a beat line and find
    /// when each rank peaks: the hero on the kick, the rings after it, in
    /// rank order.
    #[test]
    fn the_kick_reaches_the_outer_ring_last() {
        const STEP: f64 = 1.0 / 240.0;
        let mut d = Drive::default();
        // Warm up: the first update from a default `Drive` crosses every
        // beat line at once and fires a pulse, which would land a false
        // peak on the first sample.
        for i in 0..=120 {
            d.update(STEP * SEC_PER_BEAT, 3.0 + i as f64 * STEP, None, None);
        }
        let mut peak = [(f64::NEG_INFINITY, 0.0f64); 3]; // (value, beat)
        for i in 0..=360 {
            let beat = 3.5 + i as f64 * STEP;
            d.update(STEP * SEC_PER_BEAT, beat, None, None);
            for (rank, slot) in RIG.iter().zip(peak.iter_mut()) {
                let v = ring_drive(&d, beat, rank.delay);
                if v > slot.0 {
                    *slot = (v, beat);
                }
            }
        }
        for slot in &peak {
            assert!(slot.0 > 0.5, "a rank never fired: {slot:?}");
        }
        // Hero on the beat line, inner a sixteenth later, outer an eighth.
        assert!(
            (peak[0].1 - 4.0).abs() < 0.02,
            "hero should peak on the beat: {}",
            peak[0].1
        );
        for (k, want) in [(1usize, 1.0 / 6.0), (2, 1.0 / 3.0)] {
            let lag = peak[k].1 - peak[0].1;
            assert!(
                (lag - want).abs() < 0.03,
                "rank {k} lagged by {lag} beats, wanted {want}"
            );
        }
        assert!(peak[2].1 > peak[1].1, "the rings fired out of order");

        // And on the instant the hero punches, the outer ring has not
        // heard it yet.
        let d = hot(4.0);
        let hero = ring_drive(&d, 4.0, RIG[0].delay);
        let outer = ring_drive(&d, 4.0, RIG[2].delay);
        assert!(hero > 0.8, "hero should be lit on the kick: {hero}");
        assert!(outer < 0.1, "outer ring heard it early: {outer}");
    }

    #[test]
    fn the_cone_pumps_and_the_surround_and_cabinet_do_not() {
        // Straight at the weight function: cone, surround bead, basket,
        // and the back plate — which is inside the cone's radius and is
        // held still by the axial term alone.
        assert!((cone_weight(0.30, -0.05) - 1.0).abs() < 1e-12, "mid-cone");
        assert!((cone_weight(0.00, 0.03) - 1.0).abs() < 1e-12, "cap apex");
        assert_eq!(cone_weight(SUR_END_R, SUR_END_Z), 0.0, "surround bead");
        assert_eq!(cone_weight(BASKET_R, FLANGE_Z), 0.0, "basket flange");
        assert_eq!(cone_weight(0.10, BACK_Z - 0.08), 0.0, "back plate");

        // And on the cross-section it actually builds: the cone knots all
        // move by the full punch, nothing outboard of the surround moves
        // at all, and nothing moves radially.
        const PUNCHED: f64 = 0.30;
        let rest = profile(0.0);
        let hot = profile(PUNCHED);
        for (i, (a, b)) in rest.iter().zip(hot.iter()).enumerate() {
            assert!((a.0 - b.0).abs() < 1e-12, "knot {i} moved radially");
            let dz = b.1 - a.1;
            if a.0 <= 0.58 && a.1 >= -0.30 {
                assert!((dz - PUNCHED).abs() < 1e-12, "knot {i} did not travel");
            }
            if a.0 >= 0.80 || a.1 <= -0.52 {
                assert!(dz.abs() < 1e-12, "knot {i} should be still, moved {dz}");
            }
            assert!((-1e-12..=PUNCHED + 1e-12).contains(&dz), "knot {i}: {dz}");
        }
    }

    /// One woofer, head on, filling most of the frame. Every pixel
    /// strictly between the leftmost and rightmost covered pixel of a row
    /// must be covered too: the sweep is a closed solid, so backface
    /// culling can leave a gap only if the mesh has a seam in it.
    #[test]
    fn the_sweep_is_watertight() {
        for punch in [0.0, PUNCH * DRIVE_MAX] {
            let (w, h) = (300u32, 300u32);
            let mut fb = Framebuffer::new(w, h);
            let mut db = DepthBuffer::new(w, h);
            let cam = Camera::matching(&fb);
            {
                let mut ras = Raster::new(&mut fb, &mut db, cam).unwrap();
                ras.clear_depth();
                let m = woofer(punch, 32);
                let xf = Transform::IDENTITY
                    .with_uniform_scale(300.0)
                    .with_translation(v3(0.0, 0.0, -420.0));
                let opts = DrawOpts::lit().with_interp(true, true, false);
                ras.draw_mesh(&m, &xf, &opts, |_, _| Some((255, 255, 255)));
            }
            let mut rows = 0;
            for y in 0..h {
                let lo = (0..w).find(|&x| db.covered(x, y));
                let Some(lo) = lo else { continue };
                let hi = (0..w).rev().find(|&x| db.covered(x, y)).unwrap();
                if hi <= lo {
                    continue;
                }
                rows += 1;
                for x in lo..=hi {
                    assert!(db.covered(x, y), "hole at ({x}, {y}), punch {punch}");
                }
            }
            assert!(rows > 80, "the woofer barely covered anything: {rows} rows");
        }
    }

    #[test]
    fn nothing_panics_on_bad_input() {
        for d in [Drive::default(), hot(4.0)] {
            for (w, h) in [(0, 0), (1, 1), (1, 90), (90, 1), (3, 2), (64, 48)] {
                let mut fb = Framebuffer::new(w, h);
                // Deliberately mismatched: the rasteriser resizes it.
                let mut db = DepthBuffer::new(1, 1);
                for beat in [f64::NAN, f64::INFINITY, -37.5, 0.0, 4.01, 1.0e9] {
                    for intensity in [0.0, 1.0, f64::NAN] {
                        speaker(&mut fb, &mut db, beat, intensity, &d, A, B);
                    }
                }
                assert_eq!(fb.px.len(), (w * h * 3) as usize);
            }
        }
    }

    #[test]
    fn a_dead_drive_still_draws_the_rig() {
        let d = Drive::default();
        let fb = frame(160, 120, 0.0, &d);
        assert!(fb.px.iter().any(|&v| v > 0), "the rig went dark at rest");
        for rank in &RIG {
            assert_eq!(ring_drive(&d, 0.0, rank.delay), 0.0);
        }
    }

    #[test]
    fn every_rank_lands_on_screen() {
        // Each rank drawn alone must put something in frame, or the rig
        // is quietly a hero and two invisible rings.
        for (k, rank) in RIG.iter().enumerate() {
            let (w, h) = (400u32, 300u32);
            let mut fb = Framebuffer::new(w, h);
            let mut db = DepthBuffer::new(w, h);
            let cam = Camera::matching(&fb);
            let mut hits = 0usize;
            {
                let mut ras = Raster::new(&mut fb, &mut db, cam).unwrap();
                ras.clear_depth();
                let m = woofer(0.0, rank.seg);
                let opts = DrawOpts::lit().with_interp(true, true, false);
                for i in 0..rank.count {
                    let (s, c) = (TAU * i as f64 / rank.count.max(1) as f64).sin_cos();
                    let xf = Transform::IDENTITY
                        .with_uniform_scale(rank.size)
                        .with_translation(v3(rank.radius * c, rank.radius * s, rank.z));
                    ras.draw_mesh(&m, &xf, &opts, |_, _| {
                        hits += 1;
                        Some((255, 255, 255))
                    });
                }
            }
            assert!(hits > 200, "rank {k} covered {hits} pixels");
        }
    }

    #[test]
    fn the_mesh_stays_inside_its_triangle_budget() {
        let total: usize = RIG
            .iter()
            .map(|r| woofer(0.0, r.seg).tris.len() * r.count as usize)
            .sum();
        assert!(total < 6_500, "the rig grew to {total} triangles");
    }

    /// Not a correctness check — the budget probe the module docs quote.
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn budget_probe() {
        let (w, h) = (1200u32, 700u32);
        let mut fb = Framebuffer::new(w, h);
        let mut db = DepthBuffer::new(w, h);
        let d = hot(4.0);
        for i in 0..20 {
            speaker(&mut fb, &mut db, 4.0 + i as f64 * 0.01, 1.0, &d, A, B);
        }
        const N: u32 = 120;
        let t0 = std::time::Instant::now();
        for i in 0..N {
            speaker(&mut fb, &mut db, 4.0 + i as f64 * 0.017, 1.0, &d, A, B);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / N as f64;
        println!("speaker: {ms:.3} ms/frame at {w}x{h}");
    }
}
