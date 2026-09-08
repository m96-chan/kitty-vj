//! CUBE — a cube spinning in beat time, scale punching on the beat.
//! With plates loaded, the faces are textured: forward-mapped samples
//! into a halfblock pixel grid with a z-test, backface culling, and
//! lambert-ish shading, one plate per face rotating by phrase. Without
//! assets it stays a wireframe (plus a counter-rotating inner cube).

use std::rc::Rc;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};
use crate::assets::Plate;

const VERTS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Quads in cyclic order, so bilinear interpolation walks the surface.
const FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // z = -1
    [5, 4, 7, 6], // z = +1
    [1, 5, 6, 2], // x = +1
    [4, 0, 3, 7], // x = -1
    [4, 5, 1, 0], // y = -1
    [3, 2, 6, 7], // y = +1
];

/// Depth-graded glyphs for the wireframe fallback, far to near.
const DEPTH_GLYPHS: &[char] = &['.', ':', '+', '*', '#', '@'];

const DIST: f64 = 3.2;

/// Width/height of one halfblock pixel as it lands on screen. 1.0 for
/// the common 1:2 cell font; raise it if cubes render tall, lower if
/// they render wide.
const PX_ASPECT: f64 = 1.0;

pub struct Cube {
    plates: Rc<Vec<Plate>>,
    /// Halfblock z/color buffer, reused across frames.
    zbuf: Vec<Option<(f64, [u8; 3])>>,
}

fn rotate(p: [f64; 3], rx: f64, ry: f64) -> [f64; 3] {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    // Around Y, then X.
    let x1 = p[0] * cy + p[2] * sy;
    let z1 = -p[0] * sy + p[2] * cy;
    let y2 = p[1] * cx - z1 * sx;
    let z2 = p[1] * sx + z1 * cx;
    [x1, y2, z2]
}

impl Cube {
    pub fn new(plates: Rc<Vec<Plate>>) -> Self {
        Self {
            plates,
            zbuf: Vec::new(),
        }
    }

    fn render_textured(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx, scale: f64) {
        let (w, h) = (area.width as usize, area.height as usize);
        let (pw, ph) = (w, h * 2); // halfblock pixel grid
        self.zbuf.clear();
        self.zbuf.resize(pw * ph, None);

        let (cx, cy) = (pw as f64 / 2.0, ph as f64 / 2.0);
        // A halfblock pixel (cell width x half a cell height) is close to
        // square on a ~1:2 cell font, so both axes share one unit. Tweak
        // PX_ASPECT if a font renders the cube visibly non-cubic.
        let unit = (pw as f64).min(ph as f64) * scale * 0.32;

        let rx = ctx.beat * 0.45;
        let ry = ctx.beat * 0.70;
        let pts: Vec<[f64; 3]> = VERTS.iter().map(|v| rotate(*v, rx, ry)).collect();

        let flash = 0.7 + 0.3 * (1.0 - ctx.phase).powi(2);
        // One plate per face, rotating one step per phrase.
        let base = (ctx.beat.max(0.0) / 32.0) as usize;

        for (fi, face) in FACES.iter().enumerate() {
            let quad: Vec<[f64; 3]> = face.iter().map(|&i| pts[i]).collect();
            let center = [
                (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]) / 4.0,
                (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]) / 4.0,
                (quad[0][2] + quad[1][2] + quad[2][2] + quad[3][2]) / 4.0,
            ];
            // Outward normal of a centered cube is the face center itself.
            // Camera sits at (0, 0, -DIST): cull faces pointing away.
            let to_cam = [-center[0], -center[1], -DIST - center[2]];
            let ndc = center[0] * to_cam[0] + center[1] * to_cam[1] + center[2] * to_cam[2];
            if ndc <= 0.0 {
                continue;
            }
            // Lambert-ish: light from the camera.
            let nl = (ndc
                / (1.0 * (to_cam[0].powi(2) + to_cam[1].powi(2) + to_cam[2].powi(2)).sqrt()))
            .clamp(0.0, 1.0);
            let shade = (0.45 + 0.55 * nl) * flash;

            let img = &self.plates[(base + fi) % self.plates.len()].img;
            let (sw, sh) = (img.width() as f64, img.height() as f64);
            // Center square crop so the face isn't stretched.
            let side = sw.min(sh) - 1.0;
            let (offx, offy) = ((sw - side) / 2.0, (sh - side) / 2.0);

            let project = |p: &[f64; 3]| -> (f64, f64) {
                let persp = DIST / (DIST + p[2]);
                (
                    cx + p[0] * unit * PX_ASPECT * persp,
                    cy + p[1] * unit * persp,
                )
            };

            // Oversample against the longest projected edge to avoid holes.
            let corners: Vec<(f64, f64)> = quad.iter().map(&project).collect();
            let mut longest: f64 = 0.0;
            for i in 0..4 {
                let (ax, ay) = corners[i];
                let (bx, by) = corners[(i + 1) % 4];
                longest = longest.max(((bx - ax).powi(2) + (by - ay).powi(2)).sqrt());
            }
            let n = (longest * 1.5).clamp(8.0, 400.0) as usize;

            for a in 0..=n {
                let u = a as f64 / n as f64;
                for b in 0..=n {
                    let v = b as f64 / n as f64;
                    // Bilinear over the quad surface.
                    let mut p = [0.0; 3];
                    for k in 0..3 {
                        p[k] = quad[0][k] * (1.0 - u) * (1.0 - v)
                            + quad[1][k] * u * (1.0 - v)
                            + quad[2][k] * u * v
                            + quad[3][k] * (1.0 - u) * v;
                    }
                    let (px, py) = project(&p);
                    if px < 0.0 || py < 0.0 || px >= pw as f64 || py >= ph as f64 {
                        continue;
                    }
                    let idx = py as usize * pw + px as usize;
                    if let Some((z, _)) = self.zbuf[idx]
                        && z <= p[2]
                    {
                        continue;
                    }
                    let t = img
                        .get_pixel((offx + u * side) as u32, (offy + v * side) as u32)
                        .0;
                    let c = [
                        (t[0] as f64 * shade).min(255.0) as u8,
                        (t[1] as f64 * shade).min(255.0) as u8,
                        (t[2] as f64 * shade).min(255.0) as u8,
                    ];
                    self.zbuf[idx] = Some((p[2], c));
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

    fn draw_wire(
        buf: &mut Buffer,
        area: Rect,
        ctx: &FrameCtx,
        scale: f64,
        rx: f64,
        ry: f64,
        hue_shift: f64,
    ) {
        let (w, h) = (area.width as f64, area.height as f64);
        let (cx, cy) = (w / 2.0, h / 2.0);
        let unit = (w / 2.0).min(h) * scale;

        let pts: Vec<[f64; 3]> = VERTS.iter().map(|v| rotate(*v, rx, ry)).collect();

        let samples = (unit * 2.0).max(8.0) as usize;
        for &(a, b) in &EDGES {
            let (pa, pb) = (pts[a], pts[b]);
            for s in 0..=samples {
                let t = s as f64 / samples as f64;
                let x = pa[0] + (pb[0] - pa[0]) * t;
                let y = pa[1] + (pb[1] - pa[1]) * t;
                let z = pa[2] + (pb[2] - pa[2]) * t;

                let persp = DIST / (DIST + z);
                let px = cx + x * unit * persp;
                let py = cy + y * unit * persp * 0.5; // cell aspect
                if px < 0.0 || py < 0.0 || px >= w || py >= h {
                    continue;
                }

                let depth = ((-z + 1.8) / 3.6).clamp(0.0, 1.0);
                let g = DEPTH_GLYPHS[(depth * (DEPTH_GLYPHS.len() - 1) as f64).round() as usize];
                let flash = (1.0 - ctx.phase).powi(2);
                let v = (0.35 + 0.65 * depth) * (0.6 + 0.4 * flash);

                let cell = &mut buf[(area.x + px as u16, area.y + py as u16)];
                cell.set_char(g);
                let lum = (v * 255.0) as u8;
                let warm = (hue_shift * 255.0) as u8;
                cell.set_fg(Color::Rgb(
                    lum.saturating_sub(warm / 3),
                    lum,
                    lum.saturating_sub(warm),
                ));
            }
        }
    }
}

impl Effect for Cube {
    fn name(&self) -> &'static str {
        "CUBE"
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        if area.width < 8 || area.height < 4 {
            return;
        }
        let punch = 1.0 + 0.25 * (1.0 - ctx.phase).powi(2) * (0.3 + 0.7 * ctx.intensity);

        if !self.plates.is_empty() {
            self.render_textured(buf, area, ctx, 0.52 * punch);
            return;
        }

        // Wireframe fallback.
        let rx = ctx.beat * 0.45;
        let ry = ctx.beat * 0.70;
        let base = 0.52 * punch;
        Self::draw_wire(buf, area, ctx, base, rx, ry, 0.0);
        if ctx.intensity > 0.5 {
            Self::draw_wire(buf, area, ctx, base * 0.45, -rx * 1.3, -ry * 0.8, 0.8);
        }
    }
}
