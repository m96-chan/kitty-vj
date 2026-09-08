//! CUBE — wireframe cube spinning in beat time, scale punching on the
//! beat, a counter-rotating inner cube past half intensity. Edges are
//! plotted as sampled points, glyph and brightness by depth.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx};

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

/// Depth-graded glyphs, far to near.
const DEPTH_GLYPHS: &[char] = &['.', ':', '+', '*', '#', '@'];

pub struct Cube;

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
    #[allow(clippy::too_many_arguments)]
    fn draw_cube(
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
        // Cells are ~1:2, so x gets doubled to look square.
        let unit = (w / 2.0).min(h * 2.0 / 2.0) * scale;
        let dist = 3.2;

        let pts: Vec<[f64; 3]> = VERTS.iter().map(|v| rotate(*v, rx, ry)).collect();

        let samples = (unit * 2.0).max(8.0) as usize;
        for &(a, b) in &EDGES {
            let (pa, pb) = (pts[a], pts[b]);
            for s in 0..=samples {
                let t = s as f64 / samples as f64;
                let x = pa[0] + (pb[0] - pa[0]) * t;
                let y = pa[1] + (pb[1] - pa[1]) * t;
                let z = pa[2] + (pb[2] - pa[2]) * t;

                let persp = dist / (dist + z);
                let px = cx + x * unit * persp;
                let py = cy + y * unit * persp * 0.5; // cell aspect
                if px < 0.0 || py < 0.0 || px >= w || py >= h {
                    continue;
                }

                // Near = bright. z runs roughly -sqrt3..sqrt3.
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
        // Spin lives in beat time; the punch snaps scale on each beat.
        let rx = ctx.beat * 0.45;
        let ry = ctx.beat * 0.70;
        let punch = 1.0 + 0.25 * (1.0 - ctx.phase).powi(2) * (0.3 + 0.7 * ctx.intensity);
        let base = 0.52 * punch;

        Self::draw_cube(buf, area, ctx, base, rx, ry, 0.0);
        if ctx.intensity > 0.5 {
            // Inner cube counter-rotates, warm-tinted.
            Self::draw_cube(buf, area, ctx, base * 0.45, -rx * 1.3, -ry * 0.8, 0.8);
        }
    }
}
