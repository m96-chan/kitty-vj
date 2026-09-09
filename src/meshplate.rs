//! MPLATE — the artwork on a panel in space, framed in glowing wire.
//!
//! The full-resolution plate the pixel tier already had (`PXPLATE`)
//! fills the screen; this one *floats*. One aspect-true quad textured
//! with the current plate, tumbling gently in closed form, with a wire
//! frame around it — the "立体フレーム" reading of a plate. It draws
//! straight into the framebuffer with a depth pass of its own and
//! touches nothing outside its silhouette, so the field behind it stays
//! the room it hangs in. That is the whole reason it exists: the
//! full-screen plate replaces a scene, the panel joins one.
//!
//! Determinism as everywhere: the tumble is a closed-form function of
//! beat, the plate index steps with `meshcube::plate_base` (so the two
//! plate-fed meshes rotate together in a show), and nothing accumulates.
//!
//! With no plates loaded it still draws the empty tumbling frame —
//! silence was how `PXPLATE` got reported missing.

use crate::assets::Plate;
use crate::drive::{Drive, SEC_PER_BEAT};
use crate::graphics::Framebuffer;
use crate::raster::{
    Camera, Cull, DepthBuffer, DrawOpts, LineSegments, Raster, Transform, Varyings, Vertex, v3,
};

/// Where the panel hangs and how big its half-height is, world units.
/// Nearer than the cube's -360 so the two can share a frame with the
/// panel in front.
const BASE_Z: f64 = -300.0;
const HALF_H: f64 = 150.0;
/// The kick shoves it away, gently — it is a picture, not a woofer.
const THUMP_Z: f64 = 40.0;

/// Aspect (width / height) is clamped so a banner plate cannot become a
/// sliver and a tall one cannot leave the frustum.
const ASPECT_MIN: f64 = 0.55;
const ASPECT_MAX: f64 = 1.9;
/// The frame with nothing to show keeps a poster-ish stance.
const EMPTY_ASPECT: f64 = 1.5;

/// The wire frame sits just outside the panel, and an inner line sits
/// well inside it, which is what makes it read as a frame rather than
/// an outline.
const FRAME_SCALE: f64 = 1.015;
const INNER_SCALE: f64 = 0.93;

/// MPLATE — the tumbling framed panel.
///
/// Clears `depth` and owns the pass; leaves `fb` untouched where
/// nothing was drawn, so the field behind shows through.
// The signature is the mesh modes' shared calling convention plus the
// plate list; folding arguments into a struct here would make this the
// one mode called differently.
#[allow(clippy::too_many_arguments)]
pub fn plate(
    fb: &mut Framebuffer,
    depth: &mut DepthBuffer,
    beat: f64,
    intensity: f64,
    d: &Drive,
    plates: &[Plate],
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

    let img = plates
        .get(crate::meshcube::plate_base(beat, plates.len()))
        .map(|p| &p.img);
    let aspect = img
        .map(|im| im.width().max(1) as f64 / im.height().max(1) as f64)
        .unwrap_or(EMPTY_ASPECT)
        .clamp(ASPECT_MIN, ASPECT_MAX);
    let (hx, hy) = (HALF_H * aspect, HALF_H);

    // The tumble: three incommensurate swings, so the pose never quite
    // repeats, plus the kick's shove. All closed-form in beat time.
    let t = beat * SEC_PER_BEAT;
    let thump = d.thump.clamp(0.0, 1.0);
    let xf = Transform::from_euler(
        0.55 * (t * 0.27).sin(),
        0.38 * (t * 0.201 + 1.4).sin(),
        0.08 * (t * 0.157 + 0.6).sin(),
    )
    .with_uniform_scale(1.0 + 0.07 * d.gbeat() + 0.05 * thump)
    .with_translation(v3(0.0, 0.0, BASE_Z - THUMP_Z * thump));

    let level = 0.45 + 0.55 * intensity;
    if let Some(im) = img {
        // Both sides drawn: the back shows the plate mirrored, which is
        // what a print hanging in space would do.
        let opts = DrawOpts::textured().with_cull(Cull::None);
        let (w, h) = (im.width() as f64, im.height() as f64);
        let quad = [
            Vertex::at(xf.point(v3(-hx, -hy, 0.0))).with_uv(0.0, 1.0),
            Vertex::at(xf.point(v3(hx, -hy, 0.0))).with_uv(1.0, 1.0),
            Vertex::at(xf.point(v3(hx, hy, 0.0))).with_uv(1.0, 0.0),
            Vertex::at(xf.point(v3(-hx, hy, 0.0))).with_uv(0.0, 0.0),
        ];
        let mut shade = |vy: &Varyings, _: f64| {
            let px = (vy.uv[0].clamp(0.0, 1.0) * (w - 1.0)) as u32;
            let py = (vy.uv[1].clamp(0.0, 1.0) * (h - 1.0)) as u32;
            let p = im.get_pixel(px.min(im.width() - 1), py.min(im.height() - 1));
            let c = |v: u8| (v as f64 * level).clamp(0.0, 255.0) as u8;
            Some((c(p.0[0]), c(p.0[1]), c(p.0[2])))
        };
        ras.draw_tri(&[quad[0], quad[1], quad[2]], &opts, &mut shade);
        ras.draw_tri(&[quad[0], quad[2], quad[3]], &opts, &mut shade);
    }

    // The frame: accent outside, its second voice inset and quieter.
    // Depth-tested against the panel but drawn just outside it, so the
    // near half of the frame reads over the picture edge.
    let react = d.react().clamp(0.0, 1.0);
    let opts = DrawOpts::wire().with_interp(false, false, false);
    let glow = (0.55 + 0.45 * react) * (0.35 + 0.65 * intensity);
    for (scale, col, k) in [(FRAME_SCALE, ca, 1.0), (INNER_SCALE, cb, 0.55)] {
        let g = glow * k;
        let c = (
            (col.0 as f64 * g).clamp(0.0, 255.0) as u8,
            (col.1 as f64 * g).clamp(0.0, 255.0) as u8,
            (col.2 as f64 * g).clamp(0.0, 255.0) as u8,
        );
        ras.draw_lines(&rect(hx * scale, hy * scale), &xf, &opts, |_: &Varyings, _| {
            Some(c)
        });
    }
}

/// The four edges of a `±hx × ±hy` rectangle in the panel's plane.
fn rect(hx: f64, hy: f64) -> LineSegments {
    let mut ls = LineSegments::new();
    let p = [
        v3(-hx, -hy, 0.0),
        v3(hx, -hy, 0.0),
        v3(hx, hy, 0.0),
        v3(-hx, hy, 0.0),
    ];
    for i in 0..4 {
        ls.seg(Vertex::at(p[i]), Vertex::at(p[(i + 1) % 4]));
    }
    ls
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn plate_list() -> Vec<Plate> {
        let mut img = RgbImage::new(8, 4);
        for p in img.pixels_mut() {
            p.0 = [200, 120, 40];
        }
        vec![Plate {
            name: "T".into(),
            img,
        }]
    }

    fn render(plates: &[Plate], beat: f64) -> Vec<u8> {
        let mut fb = Framebuffer::new(96, 54);
        let mut depth = DepthBuffer::new(96, 54);
        let mut d = Drive::default();
        d.update(0.01, beat, None, None);
        plate(&mut fb, &mut depth, beat, 0.8, &d, plates, (0, 255, 213), (255, 0, 200));
        fb.px
    }

    #[test]
    fn the_panel_is_deterministic() {
        let p = plate_list();
        assert_eq!(render(&p, 5.25), render(&p, 5.25));
    }

    #[test]
    fn the_panel_floats_instead_of_filling() {
        // Corners stay untouched: the field behind must survive. And
        // the middle carries the plate.
        let p = plate_list();
        let px = render(&p, 1.0);
        assert_eq!(&px[0..3], &[0, 0, 0], "corner painted");
        let mid = 3 * (27 * 96 + 48);
        assert!(
            px[mid] > 0 || px[mid + 1] > 0 || px[mid + 2] > 0,
            "nothing drawn at centre"
        );
    }

    #[test]
    fn no_plates_still_shows_the_frame() {
        // Silence was how PXPLATE got reported missing; the empty frame
        // is the honest failure mode.
        let px = render(&[], 1.0);
        assert!(px.iter().any(|&v| v > 0), "empty frame drew nothing");
    }
}
