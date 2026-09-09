//! Software rasteriser — the depth-buffered triangle engine the mesh
//! modes share.
//!
//! In [`crate::pass`] terms this is **not** one of the four shapes. It is
//! infrastructure *under* Sources: `raster` never decides what a frame
//! looks like, it only turns geometry a Source hands it into pixels in a
//! [`Framebuffer`]. Three ports are waiting on it and they want quite
//! different things out of the same engine:
//!
//! - **cube** — six textured quads, backface culled, opaque. Needs
//!   perspective-correct **UV**, or the plate visibly swims across a face
//!   as it turns. This is the one thing `effects::cube` gets wrong today:
//!   it forward-maps a bilinear walk of the quad *surface* and splats the
//!   samples, which is why it has to oversample against the longest
//!   projected edge and still can't guarantee it filled every pixel.
//! - **speaker** — 23 instances of one lit mesh. Needs an interpolated
//!   **normal** and, above all, needs the per-instance cost to be a
//!   transform and a draw, not a rebuild.
//! - **wire** — additive edges over the same geometry. Needs
//!   **barycentric** varyings for the edge-width test, culling *off* (an
//!   additive wireframe wants both sides), and a depth bias so its edges
//!   win against the faces they sit on.
//!
//! So the varyings are the union of those three, the shading hook is a
//! closure that may discard, and culling and blending are flags rather
//! than assumptions.
//!
//! # Space
//!
//! The camera convention is [`crate::pixparticles`]', deliberately: a
//! mesh and a particle field drawn into the same framebuffer have to
//! agree about where the world is or they read as two scenes. World +Y is
//! up, screen +Y is down, the camera sits at [`CAM_Z`] on +Z looking down
//! −Z and does not rotate — *the scene turns, not the lens*, which is how
//! the originals were built. [`Camera::matching`] reproduces its focal
//! length exactly.
//!
//! **Aspect**: [`Camera::focal`] is in pixels and is applied to both axes,
//! so pixels are square and the horizontal field of view simply follows
//! from the framebuffer's width. There is no aspect divisor to get wrong.
//! (The cell tier is a different story — a character cell is about 1:2 —
//! but this module only ever draws into the pixel tier.)
//!
//! **Winding**: front faces are wound counter-clockwise seen from
//! *outside*, in the y-up world frame. That is the winding
//! [`Mesh::cube`] emits, and it is what [`Cull::Back`] keeps.
//!
//! # Budget
//!
//! A fanless MacBook Air, 60 fps, ~16 ms for the whole frame — of which
//! the pixel tier already spends a good share on particles and the kitty
//! transport. The speaker mode alone draws 23 instances. So:
//!
//! - **No allocation anywhere in a draw call.** Vertices are transformed
//!   three at a time on the stack rather than into a scratch buffer; the
//!   near-plane clip works in fixed arrays. A shared cube vertex is
//!   therefore transformed ~6 times — 9 multiplies each, against
//!   thousands of pixels of fill, a trade worth taking at these triangle
//!   counts and worth revisiting if a mesh ever reaches thousands of
//!   triangles.
//! - **The shader is a generic**, not a `dyn` — it monomorphises and
//!   inlines into the span loop.
//! - **Varyings are opt-in** ([`DrawOpts::interp_uv`] and friends).
//!   Interpolating all eight floats costs about ten cycles a pixel, which
//!   is milliseconds over a full-screen mesh; a mode pays only for what
//!   it reads.
//! - The depth buffer is `f32` inverse depth and is [reused across
//!   frames](DepthBuffer::clear), the way `effects::cube` reuses its own.
//!
//! Measured, release build, M-series Air: the speaker's shape — 23 cube
//! instances into a 1200x700 framebuffer, lit shading, both buffers
//! cleared each frame — runs about **1.6 ms a frame**, a tenth of the
//! budget. That is the headroom the ports should assume, not more.
//!
//! # Determinism
//!
//! No wall clock, no RNG, no interior mutability. Everything
//! time-dependent arrives as a [`Transform`] the caller built from beat
//! time, so a jog scrub drags the geometry backwards with it.
//!
//! # Robustness
//!
//! Nothing here panics on bad input. Zero-area triangles, NaN
//! coordinates, a 1x1 framebuffer, geometry entirely behind the camera:
//! all draw nothing and return.

// Wired into the mixer by the mesh modes that sit on it; the entry points
// are dead until they land.
#![allow(dead_code)]

use std::ops::{Add, Mul, Neg, Sub};

use crate::graphics::Framebuffer;

/// Camera height on +Z. This is the definition — `pixparticles` and the
/// mesh modes import it rather than keeping their own, because they all
/// composite into the same framebuffer and a camera that differed
/// between them would frame two different worlds in one picture.
pub const CAM_Z: f64 = 420.0;

/// Nothing nearer than this is drawn: past it the projection divides by
/// a depth heading for zero and a vertex smears off to infinity.
pub const NEAR: f64 = 40.0;

/// Focal length as a fraction of framebuffer height — the wide lens the
/// particle fields are framed for, about an 84° vertical field of view.
pub const FOCAL_FRAC: f64 = 0.55;

/// Below this many square pixels a triangle is treated as degenerate.
/// Guards the reciprocal that turns edge functions into barycentrics.
const MIN_AREA: f64 = 1e-9;

/// Fill-rule bias for an edge that is neither top nor left: a pixel
/// centre landing exactly on it belongs to the neighbouring triangle.
const EDGE_BIAS: f64 = 1e-12;

// ---------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------

/// A point or direction in 3-space.
///
/// A named struct rather than `[f64; 3]`, which is what `effects::cube`
/// uses. That module pays for the array in `for k in 0..3` loops around
/// every interpolation and a hand-inlined dot product — arithmetic
/// narration, and it would be copied into three more effects. Normals,
/// cross products and instance transforms read as geometry here instead,
/// and `Copy` + `#[inline]` means the codegen is identical.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Terse constructor — geometry tables are long and `Vec3 { x: .., .. }`
/// buries them.
#[inline]
pub const fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = v3(1.0, 1.0, 1.0);

    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit vector, or [`Vec3::ZERO`] for a zero-length or non-finite
    /// input. Never `NaN`, so a degenerate normal shades to nothing
    /// rather than poisoning a whole face.
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let len = self.length();
        if len.is_finite() && len > 0.0 {
            self * (1.0 / len)
        } else {
            Vec3::ZERO
        }
    }

    /// Component-wise product — scale factors, not a geometric operation.
    #[inline]
    pub fn mul_each(self, o: Vec3) -> Vec3 {
        v3(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    #[inline]
    pub fn lerp(self, o: Vec3, t: f64) -> Vec3 {
        self + (o - self) * t
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, k: f64) -> Vec3 {
        v3(self.x * k, self.y * k, self.z * k)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}

// ---------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------

/// A mesh vertex: a position plus the union of what the three mesh modes
/// interpolate.
///
/// `bary` is a *stored attribute*, not the rasteriser's own barycentric
/// weights, and that is deliberate. The wire mode wants
/// `min(bary) < width` to light an edge; if the weights came from the
/// triangle being filled, every quad would show its internal diagonal and
/// the near-plane clip would rewrite the meaning of the attribute
/// mid-triangle. As an attribute the caller controls it — see
/// [`Mesh::quad`], which suppresses exactly the diagonal.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    /// Model space in a [`Mesh`]; view space once the rasteriser has it.
    pub pos: Vec3,
    pub uv: [f64; 2],
    /// Model space in a [`Mesh`]. The camera does not rotate, so by the
    /// time it reaches a shader it is equally a world- and a view-space
    /// direction.
    pub normal: Vec3,
    /// Wireframe edge attribute; `[1,0,0] [0,1,0] [0,0,1]` at a
    /// triangle's corners draws all three edges.
    pub bary: [f64; 3],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            uv: [0.0, 0.0],
            normal: v3(0.0, 0.0, 1.0),
            bary: [1.0, 0.0, 0.0],
        }
    }
}

impl Vertex {
    #[inline]
    pub fn at(pos: Vec3) -> Self {
        Self {
            pos,
            ..Self::default()
        }
    }

    #[inline]
    pub fn with_uv(mut self, u: f64, v: f64) -> Self {
        self.uv = [u, v];
        self
    }

    #[inline]
    pub fn with_normal(mut self, n: Vec3) -> Self {
        self.normal = n;
        self
    }

    #[inline]
    pub fn with_bary(mut self, b: [f64; 3]) -> Self {
        self.bary = b;
        self
    }

    /// Linear blend of every field. Used only by the near-plane clip,
    /// where the vertices are in view space and every attribute is still
    /// linear in it — which is the whole reason clipping happens before
    /// projection rather than after.
    #[inline]
    fn lerp(&self, o: &Vertex, t: f64) -> Vertex {
        let f = |a: f64, b: f64| a + (b - a) * t;
        Vertex {
            pos: self.pos.lerp(o.pos, t),
            uv: [f(self.uv[0], o.uv[0]), f(self.uv[1], o.uv[1])],
            normal: self.normal.lerp(o.normal, t),
            bary: [
                f(self.bary[0], o.bary[0]),
                f(self.bary[1], o.bary[1]),
                f(self.bary[2], o.bary[2]),
            ],
        }
    }
}

/// An indexed triangle mesh in model space.
///
/// Indexed rather than a flat triangle soup because the speaker draws one
/// mesh 23 times: the vertices are built once and each instance is a
/// [`Transform`] and a [`Raster::draw_mesh`] call.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub tris: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one triangle, giving its corners the canonical wireframe
    /// attribute so all three edges draw.
    pub fn tri(&mut self, a: Vertex, b: Vertex, c: Vertex) {
        let i = self.verts.len() as u32;
        self.verts.push(a.with_bary([1.0, 0.0, 0.0]));
        self.verts.push(b.with_bary([0.0, 1.0, 0.0]));
        self.verts.push(c.with_bary([0.0, 0.0, 1.0]));
        self.tris.push([i, i + 1, i + 2]);
    }

    /// Append a quad `a b c d` in cyclic order as two triangles, with the
    /// wireframe attribute rigged so the **internal diagonal `a–c` never
    /// lights**.
    ///
    /// The trick is to give the two vertices on the diagonal an extra 1
    /// in the third vertex's channel, so along that diagonal the smallest
    /// component peaks at 0.5 instead of falling to 0. Without it every
    /// wireframed quad in the scene shows a triangulation artefact that
    /// no amount of edge-width tuning removes.
    pub fn quad(&mut self, a: Vertex, b: Vertex, c: Vertex, d: Vertex) {
        let i = self.verts.len() as u32;
        self.verts.push(a.with_bary([1.0, 1.0, 0.0]));
        self.verts.push(b.with_bary([0.0, 1.0, 0.0]));
        self.verts.push(c.with_bary([0.0, 1.0, 1.0]));
        self.verts.push(a.with_bary([1.0, 0.0, 1.0]));
        self.verts.push(c.with_bary([0.0, 1.0, 1.0]));
        self.verts.push(d.with_bary([0.0, 0.0, 1.0]));
        self.tris.push([i, i + 1, i + 2]);
        self.tris.push([i + 3, i + 4, i + 5]);
    }

    /// The unit cube spanning ±1, six quads, outward normals, each face
    /// carrying a full `0..1` UV square. Used by the wire mode and as
    /// the winding reference for [`Cull::Back`].
    ///
    /// Not the cube *mode's* geometry: `meshcube` builds its own faces
    /// at ±0.5 so it can throw them apart individually, so the two are
    /// half a scale factor apart and cannot be swapped.
    pub fn cube() -> Self {
        // Three consecutive corners and the outward normal, per face,
        // wound counter-clockwise seen from outside in the y-up world
        // frame. The fourth corner closes the parallelogram.
        const FACES: [[Vec3; 4]; 6] = [
            [
                v3(-1., -1., 1.),
                v3(1., -1., 1.),
                v3(1., 1., 1.),
                v3(0., 0., 1.),
            ],
            [
                v3(1., -1., -1.),
                v3(-1., -1., -1.),
                v3(-1., 1., -1.),
                v3(0., 0., -1.),
            ],
            [
                v3(1., -1., 1.),
                v3(1., -1., -1.),
                v3(1., 1., -1.),
                v3(1., 0., 0.),
            ],
            [
                v3(-1., -1., -1.),
                v3(-1., -1., 1.),
                v3(-1., 1., 1.),
                v3(-1., 0., 0.),
            ],
            [
                v3(-1., 1., 1.),
                v3(1., 1., 1.),
                v3(1., 1., -1.),
                v3(0., 1., 0.),
            ],
            [
                v3(-1., -1., -1.),
                v3(1., -1., -1.),
                v3(1., -1., 1.),
                v3(0., -1., 0.),
            ],
        ];
        let mut m = Mesh::new();
        for [a, b, c, n] in FACES {
            let d = a + (c - b);
            m.quad(
                Vertex::at(a).with_uv(0.0, 0.0).with_normal(n),
                Vertex::at(b).with_uv(1.0, 0.0).with_normal(n),
                Vertex::at(c).with_uv(1.0, 1.0).with_normal(n),
                Vertex::at(d).with_uv(0.0, 1.0).with_normal(n),
            );
        }
        m
    }
}

/// An indexed set of line segments — three.js `LineSegments`, which is
/// what the wire mode's edges and the grid floor both were over there.
///
/// Separate from [`Mesh`] because a line has no area and so no winding,
/// no culling and no fill rule; sharing the triangle path would mean
/// degenerate triangles and a pile of special cases.
#[derive(Clone, Debug, Default)]
pub struct LineSegments {
    pub verts: Vec<Vertex>,
    pub pairs: Vec<[u32; 2]>,
}

impl LineSegments {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seg(&mut self, a: Vertex, b: Vertex) {
        let i = self.verts.len() as u32;
        self.verts.push(a);
        self.verts.push(b);
        self.pairs.push([i, i + 1]);
    }
}

// ---------------------------------------------------------------------
// transform
// ---------------------------------------------------------------------

/// A rotation, stored as the 3x3 matrix it will be applied as.
///
/// **Euler yaw/pitch/roll, not a quaternion.** A quaternion earns its
/// keep when orientations are *integrated* (drift-free) or *interpolated*
/// (slerp). Neither happens here: every orientation in this project is a
/// closed-form function of beat time, because that is the property that
/// survives a jog scrub. Gimbal lock is a hazard for an incremental
/// integrator, not for `yaw = beat * 0.7` evaluated fresh each frame. So
/// a quaternion would buy nothing and cost a conversion to a matrix per
/// instance anyway — and the speaker has 23 of those.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rot {
    rows: [Vec3; 3],
}

impl Default for Rot {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Rot {
    pub const IDENTITY: Rot = Rot {
        rows: [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)],
    };

    /// `R = Ry(yaw) · Rx(pitch) · Rz(roll)` — roll about the model's own
    /// +Z first, then pitch about +X, then yaw about +Y. The aircraft
    /// order, and with `roll = 0` it is the Y-then-X spin
    /// `effects::cube` already turns on.
    pub fn from_euler(yaw: f64, pitch: f64, roll: f64) -> Self {
        let (sa, ca) = yaw.sin_cos();
        let (sb, cb) = pitch.sin_cos();
        let (sc, cc) = roll.sin_cos();
        Rot {
            rows: [
                v3(ca * cc + sa * sb * sc, -ca * sc + sa * sb * cc, sa * cb),
                v3(cb * sc, cb * cc, -sb),
                v3(-sa * cc + ca * sb * sc, sa * sc + ca * sb * cc, ca * cb),
            ],
        }
    }

    #[inline]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        v3(
            self.rows[0].dot(p),
            self.rows[1].dot(p),
            self.rows[2].dot(p),
        )
    }
}

/// Model → world: scale, then rotate, then translate.
///
/// One of these per draw call is the whole of instancing. The speaker's
/// 23 cabinets are 23 of these over one [`Mesh`].
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub rot: Rot,
    pub scale: Vec3,
    pub translate: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        rot: Rot::IDENTITY,
        scale: Vec3::ONE,
        translate: Vec3::ZERO,
    };

    pub fn from_euler(yaw: f64, pitch: f64, roll: f64) -> Self {
        Self {
            rot: Rot::from_euler(yaw, pitch, roll),
            ..Self::IDENTITY
        }
    }

    pub fn with_rot(mut self, rot: Rot) -> Self {
        self.rot = rot;
        self
    }

    /// Uniform scale — the common case, and the one that leaves normals
    /// alone.
    pub fn with_uniform_scale(mut self, s: f64) -> Self {
        self.scale = v3(s, s, s);
        self
    }

    pub fn with_scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }

    pub fn with_translation(mut self, t: Vec3) -> Self {
        self.translate = t;
        self
    }

    #[inline]
    pub fn point(&self, p: Vec3) -> Vec3 {
        self.rot.apply(p.mul_each(self.scale)) + self.translate
    }

    /// Transform a normal. Divides by the scale before rotating — the
    /// inverse-transpose, which for a diagonal scale is just that — so a
    /// squashed speaker cabinet still lights correctly instead of
    /// shading as if it were a cube. A zero scale component yields
    /// [`Vec3::ZERO`] rather than `NaN`.
    #[inline]
    pub fn direction(&self, n: Vec3) -> Vec3 {
        let inv = v3(
            recip_or_zero(self.scale.x),
            recip_or_zero(self.scale.y),
            recip_or_zero(self.scale.z),
        );
        self.rot.apply(n.mul_each(inv)).normalized()
    }
}

#[inline]
fn recip_or_zero(v: f64) -> f64 {
    if v.is_finite() && v != 0.0 {
        1.0 / v
    } else {
        0.0
    }
}

/// Where the eye is and how long the lens is.
///
/// No orientation, on purpose: `pixparticles` has none either, and the
/// ported scenes all turn the world rather than the camera. Adding a view
/// rotation later means a [`Rot`] here and one more `apply` in
/// [`Raster::view_of`] — but until a mode needs it, it would be a matrix
/// multiply per vertex bought for nothing.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: Vec3,
    /// Focal length **in pixels**, applied to both axes.
    pub focal: f64,
}

impl Camera {
    /// The camera `pixparticles` draws with, for this framebuffer. Use
    /// this and a mesh will sit in the same space as the dust box, the
    /// tunnel and the synthwave floor.
    pub fn matching(fb: &Framebuffer) -> Self {
        Self {
            pos: v3(0.0, 0.0, CAM_Z),
            focal: fb.h as f64 * FOCAL_FRAC,
        }
    }
}

// ---------------------------------------------------------------------
// depth buffer
// ---------------------------------------------------------------------

/// Per-pixel depth, held as **inverse** view depth: bigger is nearer, and
/// `0.0` means nothing has been drawn there.
///
/// Inverse depth because that is what interpolates linearly in screen
/// space — the rasteriser needs `1/z` per pixel anyway to do
/// perspective-correct varyings, so storing it costs nothing and the test
/// is one compare. `f32` halves the memory traffic against an `f64`
/// buffer and still leaves ~7 digits on values around `1/420`.
///
/// Owned by the effect and cleared per frame, the way `effects::cube`
/// keeps its `zbuf` — a full-screen buffer is megabytes and reallocating
/// it 60 times a second is exactly the kind of thing that shows up as a
/// fan.
#[derive(Clone, Debug, Default)]
pub struct DepthBuffer {
    w: u32,
    h: u32,
    inv: Vec<f32>,
}

impl DepthBuffer {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            inv: vec![0.0; (w as usize) * (h as usize)],
        }
    }

    pub fn w(&self) -> u32 {
        self.w
    }

    pub fn h(&self) -> u32 {
        self.h
    }

    /// Match a new framebuffer size. Only touches the allocation when the
    /// size actually changed; contents become cleared either way.
    pub fn resize(&mut self, w: u32, h: u32) {
        let n = (w as usize) * (h as usize);
        if w != self.w || h != self.h || self.inv.len() != n {
            self.w = w;
            self.h = h;
            self.inv.clear();
            self.inv.resize(n, 0.0);
        }
    }

    /// Start a frame. Keeps the allocation.
    pub fn clear(&mut self) {
        self.inv.fill(0.0);
    }

    /// View depth written at this pixel, or `None` if nothing covered it.
    /// For a caller compositing a mesh against something else — fog, a
    /// depth-keyed transition — and for tests.
    pub fn depth_at(&self, x: u32, y: u32) -> Option<f64> {
        if x >= self.w || y >= self.h {
            return None;
        }
        let inv = self.inv[(y * self.w + x) as usize];
        if inv > 0.0 {
            Some(1.0 / inv as f64)
        } else {
            None
        }
    }

    pub fn covered(&self, x: u32, y: u32) -> bool {
        self.depth_at(x, y).is_some()
    }
}

// ---------------------------------------------------------------------
// draw options
// ---------------------------------------------------------------------

/// Which side of a triangle survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    /// Keep front faces — counter-clockwise from outside, y-up world.
    /// What an opaque mesh wants.
    Back,
    /// Keep back faces. Useful for drawing the inside of a shape, or the
    /// far half of a transparent one before the near half.
    Front,
    /// Keep everything. What an **additive wireframe** wants: the far
    /// edges are half of what makes it read as a solid.
    None,
}

/// How an accepted fragment reaches the framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// Overwrite. Opaque geometry.
    Replace,
    /// Saturating add — the `lighter` composite the ported scenes were
    /// built on, and the same one `pixparticles` uses, so a mesh stacks
    /// with a particle field instead of punching a hole in it.
    Add,
}

/// Everything about a draw that is not geometry.
///
/// The `interp_*` flags are a performance knob, not a correctness one:
/// a varying that is switched off is held **flat at the triangle's first
/// vertex** rather than zeroed, so turning off `interp_normal` gives
/// honest flat shading on a faceted mesh. Turning off a varying the
/// shader actually reads will look wrong, not crash.
#[derive(Clone, Copy, Debug)]
pub struct DrawOpts {
    pub cull: Cull,
    pub blend: Blend,
    /// Whether an accepted fragment updates the depth buffer. Additive
    /// passes usually test but do not write, so they neither occlude each
    /// other nor depend on submission order.
    pub depth_write: bool,
    /// Added to a fragment's inverse depth before the test. A small
    /// positive value pulls it toward the camera; that is how the wire
    /// mode's edges beat the faces they are coincident with, instead of
    /// z-fighting into a dashed mess. Inverse depth around `1/420`, so
    /// useful values are ~`1e-5`.
    pub depth_bias: f64,
    pub interp_uv: bool,
    pub interp_normal: bool,
    pub interp_bary: bool,
}

impl Default for DrawOpts {
    /// Safe rather than fast: everything interpolated, opaque, culled.
    fn default() -> Self {
        Self {
            cull: Cull::Back,
            blend: Blend::Replace,
            depth_write: true,
            depth_bias: 0.0,
            interp_uv: true,
            interp_normal: true,
            interp_bary: true,
        }
    }
}

impl DrawOpts {
    /// Opaque textured faces — the cube mode. UV only.
    pub fn textured() -> Self {
        Self {
            interp_normal: false,
            interp_bary: false,
            ..Self::default()
        }
    }

    /// Opaque lit faces — the speaker mode. Normal only; per-face flat
    /// shading is cheaper still with `interp_normal` off.
    pub fn lit() -> Self {
        Self {
            interp_uv: false,
            interp_bary: false,
            ..Self::default()
        }
    }

    /// Additive wireframe — the wire mode. Both sides, no depth write,
    /// biased toward the camera, barycentric only.
    pub fn wire() -> Self {
        Self {
            cull: Cull::None,
            blend: Blend::Add,
            depth_write: false,
            depth_bias: 1e-5,
            interp_uv: false,
            interp_normal: false,
            interp_bary: true,
        }
    }

    pub fn with_cull(mut self, cull: Cull) -> Self {
        self.cull = cull;
        self
    }

    pub fn with_blend(mut self, blend: Blend) -> Self {
        self.blend = blend;
        self
    }

    pub fn with_depth_write(mut self, on: bool) -> Self {
        self.depth_write = on;
        self
    }

    pub fn with_depth_bias(mut self, bias: f64) -> Self {
        self.depth_bias = bias;
        self
    }

    pub fn with_interp(mut self, uv: bool, normal: bool, bary: bool) -> Self {
        self.interp_uv = uv;
        self.interp_normal = normal;
        self.interp_bary = bary;
        self
    }
}

/// What a shader sees at one fragment.
///
/// Everything except `x`/`y`/`pos_view` is a perspective-correct blend of
/// the triangle's vertex attributes. `pos_view` and the screen
/// coordinates come straight from the pixel, so they are exact rather
/// than interpolated.
#[derive(Clone, Copy, Debug)]
pub struct Varyings {
    /// Pixel column and row. For screen-space work — dither, scanlines,
    /// and for tests that want to sample one fragment.
    pub x: u32,
    pub y: u32,
    pub uv: [f64; 2],
    /// **Not renormalised.** Call [`Vec3::normalized`] if the shading
    /// model needs a unit normal; a mode doing a cheap `dot` against a
    /// fixed light often does not.
    pub normal: Vec3,
    /// The wireframe attribute. `bary.iter().fold(f64::MAX, f64::min)`
    /// is the distance-to-nearest-edge the wire mode tests.
    pub bary: [f64; 3],
    /// Fragment position relative to the camera, which — since the camera
    /// does not rotate — is the world position minus [`Camera::pos`].
    /// `z` is negative in front of the eye; `depth` is its magnitude.
    pub pos_view: Vec3,
}

// ---------------------------------------------------------------------
// the rasteriser
// ---------------------------------------------------------------------

/// A framebuffer, a depth buffer and a camera, bound together for the
/// span of a frame.
///
/// Borrowing both targets for the lifetime of the `Raster` is what keeps
/// the inner loop free of bounds bookkeeping: the two buffers are known
/// to be the same size for as long as this exists. Clear the depth buffer
/// yourself — a mode may want several passes against one depth pass.
pub struct Raster<'a> {
    fb: &'a mut Framebuffer,
    depth: &'a mut DepthBuffer,
    cam: Camera,
    cx: f64,
    cy: f64,
    focal: f64,
    w: u32,
    h: u32,
}

impl<'a> Raster<'a> {
    /// `None` for a zero-sized framebuffer — there is nothing to draw on.
    /// Resizes `depth` to match `fb`, which also clears it if the pane
    /// changed size.
    pub fn new(fb: &'a mut Framebuffer, depth: &'a mut DepthBuffer, cam: Camera) -> Option<Self> {
        if fb.w == 0 || fb.h == 0 || !cam.focal.is_finite() || cam.focal <= 0.0 {
            return None;
        }
        depth.resize(fb.w, fb.h);
        let (w, h) = (fb.w, fb.h);
        Some(Self {
            fb,
            depth,
            cam,
            cx: w as f64 * 0.5,
            cy: h as f64 * 0.5,
            focal: cam.focal,
            w,
            h,
        })
    }

    pub fn framebuffer(&mut self) -> &mut Framebuffer {
        self.fb
    }

    pub fn depth_buffer(&mut self) -> &mut DepthBuffer {
        self.depth
    }

    /// Start a frame's depth pass.
    pub fn clear_depth(&mut self) {
        self.depth.clear();
    }

    /// World → view. A translation, because the camera does not rotate.
    #[inline]
    pub fn view_of(&self, world: Vec3) -> Vec3 {
        world - self.cam.pos
    }

    /// View position → screen position and view depth, or `None` if it is
    /// at or behind the near plane. Exposed so a mode can place a 2D
    /// annotation — a label, a HUD tick — where a 3D point landed.
    #[inline]
    pub fn project(&self, view: Vec3) -> Option<(f64, f64, f64)> {
        let depth = -view.z;
        if !depth.is_finite() || depth <= NEAR || !view.is_finite() {
            return None;
        }
        let k = self.focal / depth;
        Some((self.cx + view.x * k, self.cy - view.y * k, depth))
    }

    /// Draw an indexed mesh under one transform.
    ///
    /// `shade` runs once per surviving fragment and returns `None` to
    /// discard it — no colour written, and no depth written either, which
    /// is what makes a discard behave like a hole rather than a black
    /// patch that still occludes.
    pub fn draw_mesh<S>(&mut self, mesh: &Mesh, xf: &Transform, opts: &DrawOpts, mut shade: S)
    where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        let n = mesh.verts.len() as u32;
        for &[i0, i1, i2] in &mesh.tris {
            if i0 >= n || i1 >= n || i2 >= n {
                continue;
            }
            let tri = [
                self.to_view(xf, &mesh.verts[i0 as usize]),
                self.to_view(xf, &mesh.verts[i1 as usize]),
                self.to_view(xf, &mesh.verts[i2 as usize]),
            ];
            self.clip_and_fill(&tri, opts, &mut shade);
        }
    }

    /// Draw one triangle given directly in **world** space, for geometry
    /// that is generated per frame and never worth storing in a [`Mesh`].
    pub fn draw_tri<S>(&mut self, tri: &[Vertex; 3], opts: &DrawOpts, mut shade: S)
    where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        let id = Transform::IDENTITY;
        let tri = [
            self.to_view(&id, &tri[0]),
            self.to_view(&id, &tri[1]),
            self.to_view(&id, &tri[2]),
        ];
        self.clip_and_fill(&tri, opts, &mut shade);
    }

    /// Draw line segments with the same depth test as the triangles.
    ///
    /// Lines have no area, so `cull` is ignored; everything else applies.
    /// The wire mode draws these over its faces with
    /// [`DrawOpts::depth_bias`] set, and the grid floor draws them alone.
    ///
    /// A segment crossing the near plane is **clamped** to it rather than
    /// dropped, so a line you fly along shortens instead of blinking out
    /// — the same treatment `pixparticles::add_seg` gives the floor.
    pub fn draw_lines<S>(
        &mut self,
        lines: &LineSegments,
        xf: &Transform,
        opts: &DrawOpts,
        mut shade: S,
    ) where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        let n = lines.verts.len() as u32;
        for &[i0, i1] in &lines.pairs {
            if i0 >= n || i1 >= n {
                continue;
            }
            let a = self.to_view(xf, &lines.verts[i0 as usize]);
            let b = self.to_view(xf, &lines.verts[i1 as usize]);
            self.draw_seg(&a, &b, opts, &mut shade);
        }
    }

    /// A camera-facing quad of `size_px` pixels across, centred on a
    /// world point and depth-tested like anything else.
    ///
    /// `size_px` is a *screen* size because the point sprites this
    /// replaces are capped in screen space — a near point is otherwise a
    /// full-screen quad rasterised in scalar Rust. UV runs `0..1` across
    /// the quad, so the circular discard every one of these wants is
    /// `((u-0.5)^2 + (v-0.5)^2 > 0.25).then_some(None)` in the shader.
    pub fn draw_sprite<S>(
        &mut self,
        center_world: Vec3,
        size_px: f64,
        opts: &DrawOpts,
        mut shade: S,
    ) where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        if !size_px.is_finite() || size_px <= 0.0 {
            return;
        }
        let c = self.view_of(center_world);
        let Some((_, _, depth)) = self.project(c) else {
            return;
        };
        // Half-extent in view units that lands on `size_px` pixels here.
        let r = 0.5 * size_px * depth / self.focal;
        if !r.is_finite() || r <= 0.0 {
            return;
        }
        let n = v3(0.0, 0.0, 1.0);
        let corner = |dx: f64, dy: f64, u: f64, v: f64| {
            Vertex::at(c + v3(dx * r, dy * r, 0.0))
                .with_uv(u, v)
                .with_normal(n)
        };
        // Counter-clockwise from outside in the y-up frame, so a sprite
        // survives Cull::Back like any other front face.
        let quad = [
            corner(-1.0, -1.0, 0.0, 1.0),
            corner(1.0, -1.0, 1.0, 1.0),
            corner(1.0, 1.0, 1.0, 0.0),
            corner(-1.0, 1.0, 0.0, 0.0),
        ];
        for [i, j, k] in [[0usize, 1, 2], [0, 2, 3]] {
            let tri = [quad[i], quad[j], quad[k]];
            self.clip_and_fill(&tri, opts, &mut shade);
        }
    }

    // -- internals ----------------------------------------------------

    #[inline]
    fn to_view(&self, xf: &Transform, v: &Vertex) -> Vertex {
        Vertex {
            pos: self.view_of(xf.point(v.pos)),
            uv: v.uv,
            normal: xf.direction(v.normal),
            bary: v.bary,
        }
    }

    /// Near-plane **clip**, then fill each resulting triangle.
    ///
    /// Clipping rather than clamping, for triangles. A clamp — pushing an
    /// offending vertex forward to the near plane — is cheaper but bends
    /// the surface: a cube face you fly into visibly shears as one corner
    /// slides along the plane. Clipping splits it instead, so the face
    /// stretches past the frame edges the way it should. The cost is that
    /// one triangle may become two, which is why the fill is a separate
    /// function. Lines *do* clamp — a segment has no interior to shear.
    fn clip_and_fill<S>(&mut self, tri: &[Vertex; 3], opts: &DrawOpts, shade: &mut S)
    where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        if !tri[0].pos.is_finite() || !tri[1].pos.is_finite() || !tri[2].pos.is_finite() {
            return;
        }
        let inside = |v: &Vertex| v.pos.z <= -NEAR;
        if inside(&tri[0]) && inside(&tri[1]) && inside(&tri[2]) {
            self.fill_tri(tri, opts, shade);
            return;
        }
        // Sutherland-Hodgman against z <= -NEAR. A triangle clipped by a
        // single plane yields at most four vertices, so this is a fixed
        // stack array and the whole path stays allocation-free.
        let mut poly = [Vertex::default(); 4];
        let mut n = 0usize;
        for i in 0..3 {
            let (cur, nxt) = (&tri[i], &tri[(i + 1) % 3]);
            let (ci, ni) = (inside(cur), inside(nxt));
            if ci && n < 4 {
                poly[n] = *cur;
                n += 1;
            }
            if ci != ni && n < 4 {
                let den = nxt.pos.z - cur.pos.z;
                if den == 0.0 || !den.is_finite() {
                    continue;
                }
                let t = (-NEAR - cur.pos.z) / den;
                if !t.is_finite() {
                    continue;
                }
                poly[n] = cur.lerp(nxt, t.clamp(0.0, 1.0));
                n += 1;
            }
        }
        if n < 3 {
            return;
        }
        for i in 1..n - 1 {
            self.fill_tri(&[poly[0], poly[i], poly[i + 1]], opts, shade);
        }
    }

    /// Fill one view-space triangle, entirely in front of the near plane.
    ///
    /// **Bounding box plus edge functions**, not a scanline walk. Two
    /// reasons. The edge functions *are* the barycentric weights, so
    /// perspective-correct varyings and the wireframe attribute fall out
    /// of the coverage test instead of needing a second interpolator down
    /// each span. And the box is clipped against the framebuffer once, so
    /// a triangle whose vertex projects a mile off-screen costs screen
    /// pixels rather than its own extent. The price is wasted coverage
    /// tests on a long thin triangle — cheap ones, three adds each, and
    /// these meshes are not made of slivers.
    fn fill_tri<S>(&mut self, tri: &[Vertex; 3], opts: &DrawOpts, shade: &mut S)
    where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        // Project. Depth is positive going away from the eye; `iw` is its
        // reciprocal, the quantity that interpolates linearly on screen.
        let mut sx = [0.0f64; 3];
        let mut sy = [0.0f64; 3];
        let mut iw = [0.0f64; 3];
        for i in 0..3 {
            let depth = -tri[i].pos.z;
            if !depth.is_finite() || depth <= 0.0 {
                return;
            }
            let k = self.focal / depth;
            sx[i] = self.cx + tri[i].pos.x * k;
            sy[i] = self.cy - tri[i].pos.y * k;
            iw[i] = 1.0 / depth;
            if !sx[i].is_finite() || !sy[i].is_finite() {
                return;
            }
        }

        let area = area2((sx[0], sy[0]), (sx[1], sy[1]), (sx[2], sy[2]));
        if !area.is_finite() || area.abs() < MIN_AREA {
            return; // degenerate: zero area, or collapsed to a line
        }
        match opts.cull {
            Cull::Back if area <= 0.0 => return,
            Cull::Front if area >= 0.0 => return,
            _ => {}
        }

        // Work only in the positive-area orientation: swapping two
        // vertices costs one branch out here and saves a sign test per
        // pixel in there.
        let mut v = [&tri[0], &tri[1], &tri[2]];
        if area < 0.0 {
            v.swap(1, 2);
            sx.swap(1, 2);
            sy.swap(1, 2);
            iw.swap(1, 2);
        }
        let area = area.abs();
        let inv_area = 1.0 / area;

        // Bounding box, clipped to the framebuffer. `as i64` saturates on
        // non-finite input in Rust, and the clamp catches the rest.
        let (fw, fh) = (self.w as f64, self.h as f64);
        let x0 = sx[0].min(sx[1]).min(sx[2]).floor().clamp(0.0, fw) as i64;
        let x1 = sx[0].max(sx[1]).max(sx[2]).ceil().clamp(0.0, fw) as i64;
        let y0 = sy[0].min(sy[1]).min(sy[2]).floor().clamp(0.0, fh) as i64;
        let y1 = sy[0].max(sy[1]).max(sy[2]).ceil().clamp(0.0, fh) as i64;
        let (x0, x1) = (x0 as u32, (x1 as u32).min(self.w));
        let (y0, y1) = (y0 as u32, (y1 as u32).min(self.h));
        if x0 >= x1 || y0 >= y1 {
            return;
        }

        // Top-left fill rule. Without it, two triangles sharing an edge
        // both claim the pixels exactly on it: invisible under Replace,
        // a bright seam under Add. An edge is "left" when the interior
        // lies to its right, "top" when it is horizontal with the
        // interior below.
        let e = [
            (sx[2] - sx[1], sy[2] - sy[1]),
            (sx[0] - sx[2], sy[0] - sy[2]),
            (sx[1] - sx[0], sy[1] - sy[0]),
        ];
        let mut bias = [0.0f64; 3];
        for i in 0..3 {
            let top_left = e[i].1 > 0.0 || (e[i].1 == 0.0 && e[i].0 < 0.0);
            bias[i] = if top_left { 0.0 } else { EDGE_BIAS };
        }

        // Attribute-over-w, the numerators of the perspective-correct
        // blend. Hoisted so the span loop is three multiply-adds a
        // varying and nothing else.
        let (interp_uv, interp_n, interp_b) =
            (opts.interp_uv, opts.interp_normal, opts.interp_bary);
        let flat = v[0];

        let px_start = x0 as f64 + 0.5;
        let py_start = y0 as f64 + 0.5;
        let mut w_row = [
            area2((sx[1], sy[1]), (sx[2], sy[2]), (px_start, py_start)),
            area2((sx[2], sy[2]), (sx[0], sy[0]), (px_start, py_start)),
            area2((sx[0], sy[0]), (sx[1], sy[1]), (px_start, py_start)),
        ];
        let dwdx = [e[0].1, e[1].1, e[2].1];
        let dwdy = [-e[0].0, -e[1].0, -e[2].0];

        let fw_px = self.w;
        let focal = self.focal;
        let (cx, cy) = (self.cx, self.cy);
        let bias_z = opts.depth_bias;
        let add = matches!(opts.blend, Blend::Add);
        let write_depth = opts.depth_write;
        // Disjoint fields, so both stay borrowed across the loop.
        let colour = &mut self.fb.px;
        let zbuf = &mut self.depth.inv;

        for y in y0..y1 {
            let mut w = w_row;
            let row = (y * fw_px) as usize;
            for x in x0..x1 {
                let (w0, w1, w2) = (w[0], w[1], w[2]);
                w[0] += dwdx[0];
                w[1] += dwdx[1];
                w[2] += dwdx[2];
                if w0 < bias[0] || w1 < bias[1] || w2 < bias[2] {
                    continue;
                }
                let (l0, l1, l2) = (w0 * inv_area, w1 * inv_area, w2 * inv_area);
                let iw_p = l0 * iw[0] + l1 * iw[1] + l2 * iw[2];
                if !iw_p.is_finite() || iw_p <= 0.0 {
                    continue;
                }
                let idx = row + x as usize;
                let tested = (iw_p + bias_z) as f32;
                if tested <= zbuf[idx] {
                    continue;
                }
                let depth = 1.0 / iw_p;

                // Perspective-correct weights: the screen-linear weights
                // reweighted by each vertex's 1/z. They sum to 1.
                let p0 = l0 * iw[0] * depth;
                let p1 = l1 * iw[1] * depth;
                let p2 = l2 * iw[2] * depth;

                let uv = if interp_uv {
                    [
                        p0 * v[0].uv[0] + p1 * v[1].uv[0] + p2 * v[2].uv[0],
                        p0 * v[0].uv[1] + p1 * v[1].uv[1] + p2 * v[2].uv[1],
                    ]
                } else {
                    flat.uv
                };
                let normal = if interp_n {
                    v[0].normal * p0 + v[1].normal * p1 + v[2].normal * p2
                } else {
                    flat.normal
                };
                let bary = if interp_b {
                    [
                        p0 * v[0].bary[0] + p1 * v[1].bary[0] + p2 * v[2].bary[0],
                        p0 * v[0].bary[1] + p1 * v[1].bary[1] + p2 * v[2].bary[1],
                        p0 * v[0].bary[2] + p1 * v[1].bary[2] + p2 * v[2].bary[2],
                    ]
                } else {
                    flat.bary
                };

                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;
                let vary = Varyings {
                    x,
                    y,
                    uv,
                    normal,
                    bary,
                    pos_view: v3(
                        (px - cx) * depth / focal,
                        -(py - cy) * depth / focal,
                        -depth,
                    ),
                };
                let Some((r, g, b)) = shade(&vary, depth) else {
                    continue; // discarded: no colour, and no depth either
                };
                let ci = idx * 3;
                if add {
                    colour[ci] = colour[ci].saturating_add(r);
                    colour[ci + 1] = colour[ci + 1].saturating_add(g);
                    colour[ci + 2] = colour[ci + 2].saturating_add(b);
                } else {
                    colour[ci] = r;
                    colour[ci + 1] = g;
                    colour[ci + 2] = b;
                }
                if write_depth {
                    zbuf[idx] = tested;
                }
            }
            w_row[0] += dwdy[0];
            w_row[1] += dwdy[1];
            w_row[2] += dwdy[2];
        }
    }

    /// One view-space segment: near-clamped, projected, stepped.
    fn draw_seg<S>(&mut self, a: &Vertex, b: &Vertex, opts: &DrawOpts, shade: &mut S)
    where
        S: FnMut(&Varyings, f64) -> Option<(u8, u8, u8)>,
    {
        if !a.pos.is_finite() || !b.pos.is_finite() {
            return;
        }
        let (mut a, mut b) = (*a, *b);
        let inside = |v: &Vertex| v.pos.z <= -NEAR;
        match (inside(&a), inside(&b)) {
            (false, false) => return,
            (true, true) => {}
            (true, false) => {
                let den = b.pos.z - a.pos.z;
                if den == 0.0 || !den.is_finite() {
                    return;
                }
                b = a.lerp(&b, ((-NEAR - a.pos.z) / den).clamp(0.0, 1.0));
            }
            (false, true) => {
                let den = a.pos.z - b.pos.z;
                if den == 0.0 || !den.is_finite() {
                    return;
                }
                a = b.lerp(&a, ((-NEAR - b.pos.z) / den).clamp(0.0, 1.0));
            }
        }
        let (Some((ax, ay, da)), Some((bx, by, db))) = (self.project(a.pos), self.project(b.pos))
        else {
            return;
        };
        let (ia, ib) = (1.0 / da, 1.0 / db);

        // Step along the longer screen axis, one sample per pixel, and
        // let the framebuffer bound reject what falls outside. A segment
        // whose endpoint lands far off-screen is bounded by the clamp
        // below rather than by its own projected length.
        let span = (bx - ax).abs().max((by - ay).abs());
        if !span.is_finite() {
            return;
        }
        let steps = (span.ceil() as i64).clamp(1, 8192) as usize;

        let (cx, cy, focal) = (self.cx, self.cy, self.focal);
        let (w, h) = (self.w, self.h);
        let bias_z = opts.depth_bias;
        let add = matches!(opts.blend, Blend::Add);
        let write_depth = opts.depth_write;
        let colour = &mut self.fb.px;
        let zbuf = &mut self.depth.inv;

        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let sxp = ax + (bx - ax) * t;
            let syp = ay + (by - ay) * t;
            if !sxp.is_finite() || !syp.is_finite() {
                continue;
            }
            if sxp < 0.0 || syp < 0.0 || sxp >= w as f64 || syp >= h as f64 {
                continue;
            }
            let (x, y) = (sxp as u32, syp as u32);
            if x >= w || y >= h {
                continue;
            }
            let iw_p = ia + (ib - ia) * t;
            if !iw_p.is_finite() || iw_p <= 0.0 {
                continue;
            }
            let idx = (y * w + x) as usize;
            let tested = (iw_p + bias_z) as f32;
            if tested <= zbuf[idx] {
                continue;
            }
            let depth = 1.0 / iw_p;
            // Perspective-correct parameter along the segment.
            let q = (t * ib) / iw_p;
            let (p0, p1) = (1.0 - q, q);
            let uv = if opts.interp_uv {
                [p0 * a.uv[0] + p1 * b.uv[0], p0 * a.uv[1] + p1 * b.uv[1]]
            } else {
                a.uv
            };
            let normal = if opts.interp_normal {
                a.normal * p0 + b.normal * p1
            } else {
                a.normal
            };
            let bary = if opts.interp_bary {
                [
                    p0 * a.bary[0] + p1 * b.bary[0],
                    p0 * a.bary[1] + p1 * b.bary[1],
                    p0 * a.bary[2] + p1 * b.bary[2],
                ]
            } else {
                a.bary
            };
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let vary = Varyings {
                x,
                y,
                uv,
                normal,
                bary,
                pos_view: v3(
                    (px - cx) * depth / focal,
                    -(py - cy) * depth / focal,
                    -depth,
                ),
            };
            let Some((r, g, bl)) = shade(&vary, depth) else {
                continue;
            };
            let ci = idx * 3;
            if add {
                colour[ci] = colour[ci].saturating_add(r);
                colour[ci + 1] = colour[ci + 1].saturating_add(g);
                colour[ci + 2] = colour[ci + 2].saturating_add(bl);
            } else {
                colour[ci] = r;
                colour[ci + 1] = g;
                colour[ci + 2] = bl;
            }
            if write_depth {
                zbuf[idx] = tested;
            }
        }
    }
}

/// Twice the signed screen-space area of `a b c`, **positive for a front
/// face**.
///
/// Screen y grows downward while world y grows up, so the sign flip lives
/// here rather than at every call site. Used both as the coverage test
/// (all three sub-areas non-negative) and as the barycentric numerator,
/// which is the reason this rasteriser walks a bounding box.
#[inline]
fn area2(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.1 - a.1) * (c.0 - a.0) - (b.0 - a.0) * (c.1 - a.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 32;
    const H: u32 = 32;
    /// A camera whose focal length equals the test depth, so one world
    /// unit is one pixel and the expected coverage can be written down.
    const D: f64 = 100.0;

    struct Rig {
        fb: Framebuffer,
        db: DepthBuffer,
        cam: Camera,
    }

    fn rig(w: u32, h: u32) -> Rig {
        Rig {
            fb: Framebuffer::new(w, h),
            db: DepthBuffer::new(w, h),
            cam: Camera {
                pos: v3(0.0, 0.0, D),
                focal: D,
            },
        }
    }

    impl Rig {
        fn raster(&mut self) -> Raster<'_> {
            Raster::new(&mut self.fb, &mut self.db, self.cam).unwrap()
        }

        /// The world point that lands on screen `(sx, sy)` at `depth`.
        /// Inverse of the projection, so tests can state geometry in the
        /// coordinates they actually assert on.
        fn world_at(&self, sx: f64, sy: f64, depth: f64) -> Vec3 {
            let (cx, cy) = (self.fb.w as f64 * 0.5, self.fb.h as f64 * 0.5);
            let k = depth / self.cam.focal;
            v3((sx - cx) * k, -(sy - cy) * k, self.cam.pos.z - depth)
        }

        fn px(&self, x: u32, y: u32) -> (u8, u8, u8) {
            let i = ((y * self.fb.w + x) * 3) as usize;
            (self.fb.px[i], self.fb.px[i + 1], self.fb.px[i + 2])
        }

        fn lit(&self) -> Vec<(u32, u32)> {
            let mut out = Vec::new();
            for y in 0..self.fb.h {
                for x in 0..self.fb.w {
                    if self.px(x, y) != (0, 0, 0) {
                        out.push((x, y));
                    }
                }
            }
            out
        }
    }

    fn white(_: &Varyings, _: f64) -> Option<(u8, u8, u8)> {
        Some((255, 255, 255))
    }

    /// A front-facing triangle at `depth` through the three screen points.
    fn screen_tri(r: &Rig, p: [(f64, f64); 3], depth: f64) -> [Vertex; 3] {
        [
            Vertex::at(r.world_at(p[0].0, p[0].1, depth)),
            Vertex::at(r.world_at(p[1].0, p[1].1, depth)),
            Vertex::at(r.world_at(p[2].0, p[2].1, depth)),
        ]
    }

    #[test]
    fn a_triangle_fills_exactly_the_pixels_it_covers() {
        let mut r = rig(8, 8);
        // A right triangle with the square corner at the top left. The
        // hypotenuse runs x+y = 7.8 in screen units, so it never passes
        // through a pixel centre and the expected set is exact rather
        // than at the mercy of a tie-break.
        let tri = screen_tri(&r, [(0.2, 0.2), (0.2, 7.6), (7.6, 0.2)], D);
        r.raster().draw_tri(&tri, &DrawOpts::default(), white);

        let mut want: Vec<(u32, u32)> = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                if x + y <= 6 {
                    want.push((x, y));
                }
            }
        }
        want.sort();
        let mut got = r.lit();
        got.sort();
        assert_eq!(got, want, "coverage differs from the analytic set");
        assert_eq!(got.len(), 28);
    }

    #[test]
    fn the_nearer_triangle_wins_whichever_order_it_arrives_in() {
        let near = (255, 0, 0);
        let far = (0, 0, 255);
        let corners = [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)];

        for near_first in [true, false] {
            let mut r = rig(W, H);
            let near_tri = screen_tri(&r, corners, 100.0);
            let far_tri = screen_tri(&r, corners, 400.0);
            {
                let mut ras = r.raster();
                let submitted = [(&near_tri, near), (&far_tri, far)];
                let order = if near_first { [0, 1] } else { [1, 0] };
                for i in order {
                    let (tri, col) = submitted[i];
                    ras.draw_tri(tri, &DrawOpts::default(), |_, _| Some(col));
                }
            }
            assert_eq!(
                r.px(8, 8),
                near,
                "near_first = {near_first}: the far triangle won"
            );
            // The buffer holds f32 inverse depth, so the round trip is
            // good to a few parts per million, not exact.
            let d = r.db.depth_at(8, 8).unwrap();
            assert!((d - 100.0).abs() < 1e-3, "depth buffer holds {d}");
        }
    }

    #[test]
    fn backface_culling_drops_reversed_winding_and_keeps_it_when_off() {
        let corners = [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)];
        let reversed = [corners[0], corners[2], corners[1]];

        let mut r = rig(W, H);
        let tri = screen_tri(&r, reversed, D);
        r.raster().draw_tri(&tri, &DrawOpts::default(), white);
        assert!(r.lit().is_empty(), "a back face survived Cull::Back");

        let mut r = rig(W, H);
        let tri = screen_tri(&r, reversed, D);
        r.raster()
            .draw_tri(&tri, &DrawOpts::default().with_cull(Cull::None), white);
        assert!(!r.lit().is_empty(), "Cull::None dropped a back face");

        // And Cull::Front is the mirror image.
        let mut r = rig(W, H);
        let front = screen_tri(&r, corners, D);
        r.raster()
            .draw_tri(&front, &DrawOpts::default().with_cull(Cull::Front), white);
        assert!(r.lit().is_empty(), "a front face survived Cull::Front");
    }

    #[test]
    fn uv_interpolation_is_perspective_correct_not_linear() {
        // The same screen triangle twice: once with one vertex pushed
        // eight times further away, once flat. A flat triangle's
        // perspective-correct interpolation *is* the linear one, so the
        // flat pass is an honest reference for what a naive rasteriser
        // would produce, and the difference between them is exactly the
        // perspective correction.
        let screen = [(4.0, 4.0), (4.0, 28.0), (28.0, 4.0)];
        let probe = (14u32, 8u32);

        let sample = |depths: [f64; 3]| -> f64 {
            let mut r = rig(W, H);
            let tri = [
                Vertex::at(r.world_at(screen[0].0, screen[0].1, depths[0])).with_uv(0.0, 0.0),
                Vertex::at(r.world_at(screen[1].0, screen[1].1, depths[1])).with_uv(1.0, 0.0),
                Vertex::at(r.world_at(screen[2].0, screen[2].1, depths[2])).with_uv(1.0, 0.0),
            ];
            let mut u = f64::NAN;
            r.raster().draw_tri(&tri, &DrawOpts::textured(), |v, _| {
                if (v.x, v.y) == probe {
                    u = v.uv[0];
                }
                Some((255, 255, 255))
            });
            u
        };

        let linear = sample([D, D, D]);
        let persp = sample([D, D * 8.0, D * 8.0]);
        assert!(linear.is_finite() && persp.is_finite(), "probe never ran");
        // The far vertices carry u = 1; foreshortening must pull the
        // fragment's u back toward the near vertex's 0.
        assert!(
            persp < linear - 0.15,
            "no perspective correction: linear {linear}, perspective {persp}"
        );
        // Sanity: the linear reference is the screen-space barycentric.
        assert!(linear > 0.3 && linear < 0.9, "reference off: {linear}");
    }

    #[test]
    fn a_discarding_shader_leaves_both_buffers_untouched() {
        let mut r = rig(W, H);
        let tri = screen_tri(&r, [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)], D);
        r.raster().draw_tri(&tri, &DrawOpts::default(), |v, _| {
            // Keep only even columns.
            (v.x % 2 == 0).then_some((255, 255, 255))
        });
        let lit = r.lit();
        assert!(!lit.is_empty(), "everything was discarded");
        assert!(
            lit.iter().all(|&(x, _)| x % 2 == 0),
            "a discarded fragment was written"
        );
        // A discard must not occlude either, or the wire mode's holes
        // would still hide what is behind them.
        for &(x, y) in &lit {
            assert!(r.db.covered(x, y));
            assert!(!r.db.covered(x + 1, y), "discard wrote depth at {x},{y}");
        }
    }

    #[test]
    fn lines_are_drawn_and_depth_tested() {
        // A line across the middle, then an occluding quad in front of
        // its right half. The left half survives, the right does not.
        let mut r = rig(W, H);
        {
            let mut lines = LineSegments::new();
            lines.seg(
                Vertex::at(r.world_at(1.0, 16.0, D)),
                Vertex::at(r.world_at(30.0, 16.0, D)),
            );
            let mut ras = r.raster();
            ras.draw_lines(&lines, &Transform::IDENTITY, &DrawOpts::default(), white);
        }
        assert!(r.px(4, 16) != (0, 0, 0), "line did not draw");
        assert!(r.px(24, 16) != (0, 0, 0), "line did not draw");
        let d = r.db.depth_at(4, 16).expect("line wrote no depth");
        assert!((d - D).abs() < 1e-3, "line depth is {d}");

        // Fresh frame: a nearer quad over the right half, then the same
        // line. The left half draws, the right half z-fails.
        let mut r = rig(W, H);
        {
            let a = r.world_at(16.0, 0.0, D / 2.0);
            let b = r.world_at(16.0, 32.0, D / 2.0);
            let c = r.world_at(32.0, 32.0, D / 2.0);
            let d = r.world_at(32.0, 0.0, D / 2.0);
            let mut mesh = Mesh::new();
            mesh.quad(Vertex::at(a), Vertex::at(b), Vertex::at(c), Vertex::at(d));
            let mut ras = r.raster();
            ras.draw_mesh(&mesh, &Transform::IDENTITY, &DrawOpts::default(), |_, _| {
                Some((0, 0, 255))
            });
        }
        {
            let mut lines = LineSegments::new();
            lines.seg(
                Vertex::at(r.world_at(1.0, 16.0, D)),
                Vertex::at(r.world_at(30.0, 16.0, D)),
            );
            let mut ras = r.raster();
            ras.draw_lines(
                &lines,
                &Transform::IDENTITY,
                &DrawOpts::default(),
                |_, _| Some((255, 0, 0)),
            );
        }
        assert_eq!(r.px(4, 16), (255, 0, 0), "left half should have drawn");
        assert_eq!(r.px(24, 16), (0, 0, 255), "line drew through the occluder");
    }

    #[test]
    fn a_depth_bias_lets_an_edge_win_against_its_own_surface() {
        let mut r = rig(W, H);
        let corners = [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)];
        let tri = screen_tri(&r, corners, D);
        r.raster()
            .draw_tri(&tri, &DrawOpts::default(), |_, _| Some((10, 10, 10)));

        // A line lying exactly in the surface loses without a bias...
        let seg = |r: &Rig| {
            let mut l = LineSegments::new();
            l.seg(
                Vertex::at(r.world_at(3.0, 6.0, D)),
                Vertex::at(r.world_at(20.0, 6.0, D)),
            );
            l
        };
        let l = seg(&r);
        r.raster().draw_lines(
            &l,
            &Transform::IDENTITY,
            &DrawOpts::default().with_depth_bias(0.0),
            |_, _| Some((0, 255, 0)),
        );
        assert_eq!(r.px(10, 6), (10, 10, 10), "coincident line should z-fail");

        let l = seg(&r);
        r.raster().draw_lines(
            &l,
            &Transform::IDENTITY,
            &DrawOpts::default().with_depth_bias(1e-5),
            |_, _| Some((0, 255, 0)),
        );
        assert_eq!(
            r.px(10, 6),
            (0, 255, 0),
            "bias did not pull the line forward"
        );
    }

    #[test]
    fn additive_blending_stacks_and_saturates() {
        let mut r = rig(W, H);
        let tri = screen_tri(&r, [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)], D);
        let opts = DrawOpts::default()
            .with_blend(Blend::Add)
            .with_depth_write(false);
        for _ in 0..3 {
            r.raster().draw_tri(&tri, &opts, |_, _| Some((100, 0, 0)));
        }
        assert_eq!(r.px(6, 6).0, 255, "three adds of 100 should saturate");

        // And a shared edge is claimed exactly once, or a wireframed
        // quad shows a bright seam down its diagonal.
        let mut r = rig(W, H);
        let mut mesh = Mesh::new();
        mesh.quad(
            Vertex::at(r.world_at(4.0, 4.0, D)),
            Vertex::at(r.world_at(4.0, 24.0, D)),
            Vertex::at(r.world_at(24.0, 24.0, D)),
            Vertex::at(r.world_at(24.0, 4.0, D)),
        );
        r.raster().draw_mesh(
            &mesh,
            &Transform::IDENTITY,
            &DrawOpts::default()
                .with_blend(Blend::Add)
                .with_depth_write(false),
            |_, _| Some((40, 0, 0)),
        );
        for y in 4..24u32 {
            for x in 4..24u32 {
                assert_eq!(r.px(x, y).0, 40, "double-covered pixel at {x},{y}");
            }
        }
    }

    #[test]
    fn a_quad_hides_its_own_diagonal_from_the_wireframe_test() {
        let mut r = rig(64, 64);
        let mut mesh = Mesh::new();
        mesh.quad(
            Vertex::at(r.world_at(8.0, 8.0, D)),
            Vertex::at(r.world_at(8.0, 56.0, D)),
            Vertex::at(r.world_at(56.0, 56.0, D)),
            Vertex::at(r.world_at(56.0, 8.0, D)),
        );
        r.raster()
            .draw_mesh(&mesh, &Transform::IDENTITY, &DrawOpts::wire(), |v, _| {
                let m = v.bary[0].min(v.bary[1]).min(v.bary[2]);
                (m < 0.02).then_some((255, 255, 255))
            });
        // The border lights up...
        assert!(r.px(8, 30) != (0, 0, 0), "quad border missing");
        // ...and the diagonal from (8,8) to (56,56) does not.
        assert_eq!(r.px(32, 32), (0, 0, 0), "internal diagonal lit up");
    }

    #[test]
    fn a_reused_depth_buffer_clears_between_frames() {
        let mut r = rig(W, H);
        let corners = [(2.0, 2.0), (2.0, 28.0), (28.0, 2.0)];
        let near = screen_tri(&r, corners, 100.0);
        let far = screen_tri(&r, corners, 400.0);

        r.raster()
            .draw_tri(&near, &DrawOpts::default(), |_, _| Some((255, 0, 0)));
        assert_eq!(r.px(6, 6), (255, 0, 0));

        // Without a clear the far triangle loses.
        r.raster()
            .draw_tri(&far, &DrawOpts::default(), |_, _| Some((0, 0, 255)));
        assert_eq!(r.px(6, 6), (255, 0, 0));

        // With one, it wins — and the allocation is the same one.
        let cap = r.db.inv.capacity();
        r.db.clear();
        assert!(r.db.depth_at(6, 6).is_none(), "clear left depth behind");
        r.raster()
            .draw_tri(&far, &DrawOpts::default(), |_, _| Some((0, 0, 255)));
        assert_eq!(r.px(6, 6), (0, 0, 255));
        assert_eq!(r.db.inv.capacity(), cap, "clear reallocated");
    }

    #[test]
    fn geometry_straddling_the_near_plane_is_clipped_not_dropped() {
        let mut r = rig(W, H);
        // One vertex well in front of the eye, two behind it.
        let front = r.world_at(16.0, 16.0, 300.0);
        let behind_a = v3(-200.0, -200.0, CAM_Z + 50.0);
        let behind_b = v3(200.0, -200.0, CAM_Z + 50.0);
        let tri = [
            Vertex::at(front),
            Vertex::at(behind_a),
            Vertex::at(behind_b),
        ];
        r.raster()
            .draw_tri(&tri, &DrawOpts::default().with_cull(Cull::None), white);
        assert!(!r.lit().is_empty(), "a straddling triangle vanished");

        // Entirely behind the camera draws nothing at all.
        let mut r = rig(W, H);
        let tri = [
            Vertex::at(v3(0.0, 0.0, CAM_Z + 10.0)),
            Vertex::at(v3(50.0, 0.0, CAM_Z + 10.0)),
            Vertex::at(v3(0.0, 50.0, CAM_Z + 10.0)),
        ];
        r.raster()
            .draw_tri(&tri, &DrawOpts::default().with_cull(Cull::None), white);
        assert!(r.lit().is_empty(), "geometry behind the eye drew");
    }

    #[test]
    fn sprites_are_screen_sized_and_can_discard_to_a_circle() {
        let mut r = rig(W, H);
        let centre = r.world_at(16.0, 16.0, 200.0);
        r.raster()
            .draw_sprite(centre, 12.0, &DrawOpts::default(), |v, _| {
                let (du, dv) = (v.uv[0] - 0.5, v.uv[1] - 0.5);
                (du * du + dv * dv <= 0.25).then_some((255, 255, 255))
            });
        let lit = r.lit();
        assert!(!lit.is_empty(), "sprite drew nothing");
        // A 12px disc: everything inside a 12px box, nothing in a corner.
        for &(x, y) in &lit {
            let (dx, dy) = (x as f64 + 0.5 - 16.0, y as f64 + 0.5 - 16.0);
            assert!(
                dx.abs() <= 6.5 && dy.abs() <= 6.5,
                "sprite leaked to {x},{y}"
            );
            assert!(dx * dx + dy * dy <= 40.0, "corner of the quad survived");
        }
        assert!(lit.len() > 60, "disc looks too small: {}", lit.len());
    }

    #[test]
    fn the_cube_is_wound_so_back_culling_shows_the_near_faces() {
        let mut r = rig(64, 64);
        let cube = Mesh::cube();
        // Far enough back that the whole cube is inside the frustum;
        // parked at the origin the camera would sit inside it and the
        // near faces would all be clipped away.
        let xf = Transform::from_euler(0.6, 0.4, 0.0)
            .with_uniform_scale(60.0)
            .with_translation(v3(0.0, 0.0, -400.0));

        r.raster()
            .draw_mesh(&cube, &xf, &DrawOpts::textured(), white);
        let front_depth = r.db.depth_at(32, 32).expect("cube missed the centre");

        let mut r = rig(64, 64);
        r.raster().draw_mesh(
            &cube,
            &xf,
            &DrawOpts::textured().with_cull(Cull::Front),
            white,
        );
        let back_depth = r.db.depth_at(32, 32).expect("far faces missed the centre");

        assert!(
            front_depth < back_depth,
            "Cull::Back should keep the near faces: {front_depth} vs {back_depth}"
        );
        // Outward normals: every cube normal is a unit axis vector.
        for v in &cube.verts {
            assert!((v.normal.length() - 1.0).abs() < 1e-12);
            assert!(v.normal.dot(v.pos) > 0.0, "normal points inward");
        }
    }

    #[test]
    fn instancing_one_mesh_costs_only_a_transform() {
        // What the speaker mode does 23 times: same Mesh, different
        // Transform, and the results land in different places.
        let mut r = rig(64, 64);
        let cube = Mesh::cube();
        {
            let mut ras = r.raster();
            for i in 0..23 {
                let x = -300.0 + 30.0 * i as f64;
                let xf = Transform::from_euler(0.3 * i as f64, 0.2, 0.1)
                    .with_uniform_scale(12.0)
                    .with_translation(v3(x, 0.0, 0.0));
                ras.draw_mesh(&cube, &xf, &DrawOpts::lit(), |v, _| {
                    let l = v.normal.normalized().dot(v3(0.0, 0.0, 1.0)).max(0.0);
                    let c = (40.0 + 200.0 * l) as u8;
                    Some((c, c, c))
                });
            }
        }
        assert!(!r.lit().is_empty(), "no instance landed on screen");
    }

    #[test]
    fn degenerate_input_never_panics() {
        // Zero-area, collinear, NaN, infinite, behind the eye, and a
        // 1x1 target.
        let bad: Vec<[Vertex; 3]> = vec![
            [Vertex::at(Vec3::ZERO); 3],
            [
                Vertex::at(v3(0.0, 0.0, 0.0)),
                Vertex::at(v3(1.0, 1.0, 0.0)),
                Vertex::at(v3(2.0, 2.0, 0.0)),
            ],
            [
                Vertex::at(v3(f64::NAN, 0.0, 0.0)),
                Vertex::at(v3(0.0, f64::NAN, 0.0)),
                Vertex::at(v3(0.0, 0.0, f64::NAN)),
            ],
            [
                Vertex::at(v3(f64::INFINITY, 0.0, 0.0)),
                Vertex::at(v3(0.0, f64::NEG_INFINITY, 0.0)),
                Vertex::at(v3(0.0, 0.0, 1e300)),
            ],
            [
                Vertex::at(v3(0.0, 0.0, CAM_Z)),
                Vertex::at(v3(1.0, 0.0, CAM_Z)),
                Vertex::at(v3(0.0, 1.0, CAM_Z)),
            ],
            [
                Vertex::at(v3(-1e9, -1e9, -1e9)),
                Vertex::at(v3(1e9, -1e9, -1e9)),
                Vertex::at(v3(0.0, 1e9, -1e9)),
            ],
        ];

        for (w, h) in [(1u32, 1u32), (1, 40), (40, 1), (3, 2), (32, 32)] {
            let mut r = rig(w, h);
            for cull in [Cull::Back, Cull::Front, Cull::None] {
                let opts = DrawOpts::default().with_cull(cull);
                for tri in &bad {
                    r.raster().draw_tri(tri, &opts, white);
                }
                let mut lines = LineSegments::new();
                for tri in &bad {
                    lines.seg(tri[0], tri[1]);
                    lines.seg(tri[1], tri[2]);
                }
                r.raster()
                    .draw_lines(&lines, &Transform::IDENTITY, &opts, white);
                for s in [f64::NAN, 0.0, -3.0, 1e9] {
                    r.raster().draw_sprite(Vec3::ZERO, s, &opts, white);
                }
            }
            assert_eq!(r.fb.px.len(), (w * h * 3) as usize);
        }

        // A zero-sized target has no rasteriser at all.
        let mut fb = Framebuffer::new(0, 0);
        let mut db = DepthBuffer::new(0, 0);
        let cam = Camera::matching(&fb);
        assert!(Raster::new(&mut fb, &mut db, cam).is_none());

        // Degenerate transforms and cameras, too.
        let mut r = rig(16, 16);
        let cube = Mesh::cube();
        let nasty = [
            Transform::IDENTITY.with_scale(Vec3::ZERO),
            Transform::IDENTITY.with_uniform_scale(f64::NAN),
            Transform::from_euler(f64::NAN, 0.0, 0.0),
            Transform::IDENTITY.with_translation(v3(f64::INFINITY, 0.0, 0.0)),
        ];
        for xf in nasty {
            r.raster()
                .draw_mesh(&cube, &xf, &DrawOpts::default(), white);
        }
        for focal in [0.0, -1.0, f64::NAN] {
            let cam = Camera {
                pos: v3(0.0, 0.0, CAM_Z),
                focal,
            };
            assert!(Raster::new(&mut r.fb, &mut r.db, cam).is_none());
        }
    }

    #[test]
    fn out_of_range_indices_are_ignored() {
        let mut r = rig(16, 16);
        let mesh = Mesh {
            verts: vec![Vertex::at(Vec3::ZERO)],
            tris: vec![[0, 5, 9], [7, 7, 7]],
        };
        let lines = LineSegments {
            verts: vec![Vertex::at(Vec3::ZERO)],
            pairs: vec![[0, 4], [9, 0]],
        };
        let mut ras = r.raster();
        ras.draw_mesh(&mesh, &Transform::IDENTITY, &DrawOpts::default(), white);
        ras.draw_lines(&lines, &Transform::IDENTITY, &DrawOpts::default(), white);
    }

    #[test]
    fn the_camera_matches_the_particle_tier() {
        // Same framebuffer, same world point, same pixel: a mesh and a
        // particle field have to agree or they read as two scenes.
        let fb = Framebuffer::new(200, 120);
        let cam = Camera::matching(&fb);
        assert_eq!(cam.pos.z, CAM_Z);
        assert!((cam.focal - 120.0 * FOCAL_FRAC).abs() < 1e-12);

        let mut fb = fb;
        let mut db = DepthBuffer::new(0, 0);
        let ras = Raster::new(&mut fb, &mut db, cam).unwrap();
        let p = v3(100.0, 50.0, -300.0);
        let (sx, sy, depth) = ras.project(ras.view_of(p)).unwrap();
        // The projection pixparticles::View::project computes.
        let want_depth = CAM_Z - p.z;
        let k = cam.focal / want_depth;
        assert!((depth - want_depth).abs() < 1e-9);
        assert!((sx - (100.0 + p.x * k)).abs() < 1e-9);
        assert!((sy - (60.0 - p.y * k)).abs() < 1e-9);
        // And the near plane rejects the same things.
        assert!(ras.project(v3(0.0, 0.0, -NEAR + 1.0)).is_none());
        assert!(ras.project(v3(0.0, 0.0, -NEAR - 1.0)).is_some());
    }

    #[test]
    fn rotation_composes_the_way_it_says_it_does() {
        let p = v3(1.0, 0.0, 0.0);
        // Yaw alone spins about +Y: +X goes to -Z at a quarter turn.
        let q = Rot::from_euler(std::f64::consts::FRAC_PI_2, 0.0, 0.0).apply(p);
        assert!((q.x).abs() < 1e-12 && (q.z + 1.0).abs() < 1e-12, "{q:?}");
        // Roll alone spins about +Z: +X goes to +Y.
        let q = Rot::from_euler(0.0, 0.0, std::f64::consts::FRAC_PI_2).apply(p);
        assert!((q.y - 1.0).abs() < 1e-12 && q.x.abs() < 1e-12, "{q:?}");
        // Pitch alone spins about +X: +Y goes to +Z.
        let q = Rot::from_euler(0.0, std::f64::consts::FRAC_PI_2, 0.0).apply(v3(0.0, 1.0, 0.0));
        assert!((q.z - 1.0).abs() < 1e-12 && q.y.abs() < 1e-12, "{q:?}");
        // Rotation is rigid: lengths survive an arbitrary one.
        let r = Rot::from_euler(0.7, -1.3, 2.1);
        let v = v3(3.0, -1.0, 2.0);
        assert!((r.apply(v).length() - v.length()).abs() < 1e-12);
        assert_eq!(Rot::default().apply(v), v);
    }

    #[test]
    fn a_non_uniform_scale_still_gives_a_usable_normal() {
        // Squash in y: the normal of a 45-degree face must tilt the other
        // way, not follow the surface.
        let xf = Transform::IDENTITY.with_scale(v3(1.0, 0.25, 1.0));
        let n = xf.direction(v3(0.0, 1.0, 0.0).normalized());
        assert!((n.length() - 1.0).abs() < 1e-12);
        let n = xf.direction(v3(1.0, 1.0, 0.0).normalized());
        assert!(n.y > n.x, "inverse-transpose not applied: {n:?}");
        // Degenerate scale degrades to zero rather than NaN.
        let n = Transform::IDENTITY
            .with_scale(Vec3::ZERO)
            .direction(v3(0.0, 1.0, 0.0));
        assert_eq!(n, Vec3::ZERO);
    }

    #[test]
    fn the_same_draw_twice_gives_the_same_pixels() {
        let draw = || {
            let mut r = rig(64, 48);
            let cube = Mesh::cube();
            {
                let mut ras = r.raster();
                for i in 0..4 {
                    let xf = Transform::from_euler(0.9 * i as f64, 0.3, 0.15)
                        .with_uniform_scale(30.0)
                        .with_translation(v3(-60.0 + 40.0 * i as f64, 0.0, -40.0));
                    ras.draw_mesh(&cube, &xf, &DrawOpts::textured(), |v, d| {
                        let s = (1.0 - d / 900.0).clamp(0.0, 1.0);
                        Some((
                            (v.uv[0] * 255.0 * s) as u8,
                            (v.uv[1] * 255.0 * s) as u8,
                            (200.0 * s) as u8,
                        ))
                    });
                }
            }
            r.fb.px
        };
        assert_eq!(draw(), draw());
    }
}
