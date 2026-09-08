//! Post effects — EasyPngVJ's row-indexed and geometric passes, ported to
//! the cell grid. These run over the finished composite, after the channel
//! mix and the pad triggers, because every one of them is a function of
//! *where a cell is*, not of what drew it.
//!
//! Two things travel badly from a 1920x1080 canvas to a ~200x50 grid and
//! are handled once, here, rather than fudged per effect:
//!
//! - **Units.** Over there every displacement is in pixels of [`SRC_W`] x
//!   [`SRC_H`]. A 70 px slice is 3.6% of the width, and that fraction —
//!   not the number 70 — is what gets ported. Anything quoted in pixels
//!   below is divided through by the source canvas first.
//! - **Cell aspect.** A terminal cell is about twice as tall as it is
//!   wide, so a shape that is square in pixels is square in cells only if
//!   its row count is halved. That correction is applied where a pass has
//!   an opinion about shape ([`pixelate`]'s blocks) and deliberately not
//!   where it does not (a uniform zoom scales both axes alike, so
//!   [`zoom_punch_cells`] needs no fixup).
//!
//! Determinism contract, same as everywhere: the per-frame randomness
//! these passes want comes from hashing beat time and coordinates. Nothing
//! here reads the wall clock, so a fixed seed and a fixed timestep replay
//! frame-identically — which is what makes a recorded set reproducible and
//! what lets the tests below assert on exact cells.

use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::Rect;

use crate::drive::Drive;
use crate::rng::{hash3, unit_f64};

/// The canvas the original ran on. Pixel constants lifted from EasyPngVJ
/// are meaningless without it.
const SRC_W: f64 = 1920.0;
const SRC_H: f64 = 1080.0;

/// Pixels per cell for [`shake`]. A 200x50 grid over 1920x1080 is ~9.6 px
/// per column and ~21.6 px per row; one number splitting the difference
/// keeps the jolt square-ish without pretending we know the font metrics.
const SHAKE_PX_PER_CELL: f64 = 20.0;

/// A shake bigger than this stops reading as a camera knock and starts
/// reading as a cut — the whole frame lands somewhere else. Two cells is
/// the most a grid this coarse can take.
const MAX_SHAKE_CELLS: f64 = 2.0;

/// Below this a drive value is off, not quiet. Saves a buffer clone per
/// frame on every pass that is idle, which is most of them most of the
/// time.
const EPS: f64 = 1e-3;

/// Fetch a cell, clamping the coordinates to the area. Used by the
/// geometric passes so that sampling off the edge smears the border
/// instead of punching a hole in the frame.
fn clamped(src: &Buffer, area: Rect, x: i32, y: i32) -> Cell {
    let cx = x.clamp(0, area.width as i32 - 1) as u16;
    let cy = y.clamp(0, area.height as i32 - 1) as u16;
    src[(area.x + cx, area.y + cy)].clone()
}

/// Fetch a cell, wrapping in x. The row-displacing passes wrap rather than
/// clamp: over there they run on a `fract`ed uv, and a wrapped band reads
/// as tape damage where a clamped one reads as a smear.
fn wrapped_row(src: &Buffer, area: Rect, x: i32, y: u16) -> Cell {
    let cx = x.rem_euclid(area.width as i32) as u16;
    src[(area.x + cx, area.y + y)].clone()
}

/// SLICE — 5 to 11 horizontal bands yanked sideways. The original picks
/// band position and height fresh every frame; at 60 fps that is a boil,
/// at terminal frame rates it would be a stutter, so the roll is keyed to
/// the 16th note instead. Bands then hold for a musical unit and change on
/// one, which is what the effect looked like anyway.
///
/// `gslice` is the trigger amount (the pad's own envelope, not a grid
/// pulse) and gates the whole pass: at zero this is a no-op and costs
/// nothing. Band height is 12-90 px and the shear ±70 px on the source
/// canvas, both carried across as fractions.
pub fn slice(buf: &mut Buffer, area: Rect, beat: f64, gslice: f64, intensity: f64) {
    if gslice <= EPS || intensity <= EPS || area.width < 2 || area.height == 0 {
        return;
    }
    let tick = (beat.max(0.0) * 16.0) as u64;
    let bands = 5 + (unit_f64(hash3(tick, 1, 0)) * 7.0) as u64;
    let src = buf.clone();
    for i in 0..bands {
        let y0 = (unit_f64(hash3(tick, 2, i)) * area.height as f64) as u16;
        // 12..90 px tall over there; at least one row here, because a band
        // that rounds to zero rows is a band the audience never sees.
        let px_h = 12.0 + 78.0 * unit_f64(hash3(tick, 3, i));
        let rows = ((px_h / SRC_H * area.height as f64).round() as u16).max(1);
        let px_dx = (unit_f64(hash3(tick, 4, i)) - 0.5) * 140.0;
        let dx = (px_dx / SRC_W * area.width as f64 * gslice * intensity).round() as i32;
        if dx == 0 {
            continue;
        }
        for y in y0..(y0 + rows).min(area.height) {
            for x in 0..area.width {
                buf[(area.x + x, area.y + y)] = wrapped_row(&src, area, x as i32 - dx, y);
            }
        }
    }
}

/// How many bands the glitch pass cuts the frame into. Fixed at 26 in the
/// original regardless of resolution, so it stays 26 here — the band count
/// is the rhythm of the effect, and scaling it with the terminal size
/// would make the same track look different on a different screen.
const GLITCH_BANDS: u64 = 26;

/// GLITCH ROWS — digital dropout. Each of [`GLITCH_BANDS`] bands rolls a
/// hash against the threshold `1 - amount`; the winners shift sideways by
/// up to 8% of the width. Both hashes are per-band, so a band that is
/// displaced this frame is displaced by the *same* distance every frame
/// until the time term rolls it out — that stickiness is the difference
/// between a broken signal and mere noise.
///
/// The time term is `floor(beat*15)`, straight from the original's
/// `floor(t*15)`, with beat time standing in for wall time so the tear
/// rate follows the tempo instead of the machine.
///
/// Amount is `0.18·gbeat + 0.55·gbar + 0.30·hit` clamped to 0..1 — bar
/// weighted hardest, so the frame comes apart on the downbeat.
pub fn glitch_rows(buf: &mut Buffer, area: Rect, d: &Drive, beat: f64, intensity: f64) {
    let amount = (0.18 * d.gbeat() + 0.55 * d.gbar() + 0.30 * d.hit).clamp(0.0, 1.0) * intensity;
    if amount <= EPS || area.width < 2 || area.height == 0 {
        return;
    }
    let tick = (beat.max(0.0) * 15.0) as u64;
    let src = buf.clone();
    for y in 0..area.height {
        let band = (y as u64 * GLITCH_BANDS) / area.height as u64;
        if unit_f64(hash3(band, tick, 0)) <= 1.0 - amount {
            continue;
        }
        // The original's `hash(row, 7.3)` — a second, time-independent
        // draw, so the direction is a property of the band.
        let h = unit_f64(hash3(band, 73, 0));
        let dx = ((h - 0.5) * 0.16 * amount * area.width as f64).round() as i32;
        if dx == 0 {
            continue;
        }
        for x in 0..area.width {
            // `uv.x += dx`: the sample point moves right, the picture left.
            buf[(area.x + x, area.y + y)] = wrapped_row(&src, area, x as i32 + dx, y);
        }
    }
}

/// MIRROR (2-way) — left half reflected into the right. Unlike the colour
/// passes this one is destructive of half the frame, so it belongs late in
/// the chain: mirroring a slice looks intentional, slicing a mirror does
/// not.
///
/// On an odd width the centre column maps to itself and survives.
pub fn mirror_v(buf: &mut Buffer, area: Rect) {
    if area.width < 2 || area.height == 0 {
        return;
    }
    let src = buf.clone();
    let w = area.width;
    for y in 0..area.height {
        for x in w / 2..w {
            buf[(area.x + x, area.y + y)] = src[(area.x + (w - 1 - x), area.y + y)].clone();
        }
    }
}

/// MIRROR (4-way) — the top-left quadrant reflected into the other three.
/// Cheap kaleidoscope: any effect underneath, however shapeless, comes out
/// as symmetry, which is why it is the panic button when a channel is
/// looking like mush.
///
/// Odd dimensions leave a centre row/column that maps to itself.
pub fn mirror_quad(buf: &mut Buffer, area: Rect) {
    if area.width < 2 || area.height < 2 {
        return;
    }
    let src = buf.clone();
    let (w, h) = (area.width, area.height);
    for y in 0..h {
        let sy = if y < h / 2 { y } else { h - 1 - y };
        for x in 0..w {
            let sx = if x < w / 2 { x } else { w - 1 - x };
            if sx == x && sy == y {
                continue;
            }
            buf[(area.x + x, area.y + y)] = src[(area.x + sx, area.y + sy)].clone();
        }
    }
}

/// PIXELATE — `mix(220, 26, amount)` blocks across, each taking the glyph
/// and colours of its top-left cell.
///
/// Over there this is a downsample-and-magnify and costs a whole extra
/// pass; here the grid is already the low-resolution buffer, so blocking
/// it further is a handful of clones. That asymmetry is the point — on a
/// terminal this is the cheapest big-looking move available.
///
/// Blocks are half as tall in rows as they are wide in columns so they
/// read square on screen. At rest `n` is 220, wider than any real
/// terminal, so the block is one cell and the pass returns untouched.
///
/// Driven by `gbar·1.1` — clamped, so the top of a downbeat holds at full
/// chunk for a moment instead of only touching it.
pub fn pixelate(buf: &mut Buffer, area: Rect, d: &Drive, intensity: f64) {
    let amount = (d.gbar() * 1.1).clamp(0.0, 1.0) * intensity;
    if amount <= EPS || area.width < 2 || area.height < 2 {
        return;
    }
    let across = 220.0 + (26.0 - 220.0) * amount;
    let bw = ((area.width as f64 / across).round() as u16).max(1);
    let bh = (bw / 2).max(1);
    if bw == 1 && bh == 1 {
        return;
    }
    let mut by = 0;
    while by < area.height {
        let mut bx = 0;
        while bx < area.width {
            let seed = buf[(area.x + bx, area.y + by)].clone();
            for y in by..(by + bh).min(area.height) {
                for x in bx..(bx + bw).min(area.width) {
                    buf[(area.x + x, area.y + y)] = seed.clone();
                }
            }
            bx += bw;
        }
        by += bh;
    }
}

/// ZOOM PUNCH — the frame lunges at the camera on the kick.
/// `1 + 0.085·gbeat + 0.135·thump + 0.030·hit`, the exact weights from the
/// original: `thump` leads because the punch should follow the kick's
/// *attack*, `hit` only garnishes, and `gbeat` keeps it moving when the
/// analyser has nothing to say.
///
/// Nearest-neighbour about the centre, which on a grid this coarse means
/// the zoom is visible as duplicated columns rather than as smooth scale —
/// accept it; interpolating glyphs is not a thing.
pub fn zoom_punch_cells(buf: &mut Buffer, area: Rect, d: &Drive, intensity: f64) {
    let zoom = 1.0
        + 0.085 * d.gbeat() * intensity
        + 0.135 * d.thump * intensity
        + 0.030 * d.hit * intensity;
    if zoom <= 1.0 + EPS || area.width < 2 || area.height < 2 {
        return;
    }
    let src = buf.clone();
    let cx = (area.width as f64 - 1.0) / 2.0;
    let cy = (area.height as f64 - 1.0) / 2.0;
    for y in 0..area.height {
        let sy = (cy + (y as f64 - cy) / zoom).round() as i32;
        for x in 0..area.width {
            let sx = (cx + (x as f64 - cx) / zoom).round() as i32;
            buf[(area.x + x, area.y + y)] = clamped(&src, area, sx, sy);
        }
    }
}

/// SHAKE — the whole frame knocked off its mark by up to `m` cells, where
/// `m = (6 + 30·thump)·gbeat + 9·hit + 7·thump` pixels. The `30·thump`
/// inside the beat term is what makes a hard kick shake harder than a soft
/// one on the same grid position.
///
/// Vacated edges sample clamped, so the border column smears rather than
/// leaving a black seam. See [`SHAKE_PX_PER_CELL`] and
/// [`MAX_SHAKE_CELLS`]: a cell is worth roughly twenty source pixels and
/// more than two cells of travel stops reading as a knock.
///
/// The offset is re-rolled 30 times per beat rather than per frame — fast
/// enough to look like vibration, slow enough that it is the same
/// vibration on every replay.
pub fn shake(buf: &mut Buffer, area: Rect, d: &Drive, beat: f64, intensity: f64) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let px = (6.0 + 30.0 * d.thump) * d.gbeat() * intensity
        + 9.0 * d.hit * intensity
        + 7.0 * d.thump * intensity;
    let m = (px / SHAKE_PX_PER_CELL).min(MAX_SHAKE_CELLS);
    if m <= EPS {
        return;
    }
    let tick = (beat.max(0.0) * 30.0) as u64;
    let dx = ((unit_f64(hash3(tick, 11, 0)) * 2.0 - 1.0) * m).round() as i32;
    let dy = ((unit_f64(hash3(tick, 12, 0)) * 2.0 - 1.0) * m).round() as i32;
    if dx == 0 && dy == 0 {
        return;
    }
    let src = buf.clone();
    for y in 0..area.height {
        for x in 0..area.width {
            buf[(area.x + x, area.y + y)] = clamped(&src, area, x as i32 - dx, y as i32 - dy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    /// A frame where every cell is distinguishable from every other, so a
    /// displacement of any size in any direction shows up as inequality.
    fn grid(w: u16, h: u16) -> Buffer {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        for y in 0..h {
            for x in 0..w {
                let cell = &mut buf[(x, y)];
                cell.set_char((b'a' + (x % 26) as u8) as char);
                cell.set_fg(Color::Rgb(x as u8, y as u8, 7));
            }
        }
        buf
    }

    fn same(a: &Buffer, b: &Buffer, x: u16, y: u16) -> bool {
        a[(x, y)] == b[(x, y)]
    }

    /// A drive with beat presence, so `gv()` is 1 and the grid terms are
    /// not damped down to their breakdown floor.
    fn driven(beat: f64, bar: f64, hit: f64, thump: f64) -> Drive {
        let mut d = Drive::default();
        d.beat = beat;
        d.bar = bar;
        d.hit = hit;
        d.thump = thump;
        d.groove = 1.0;
        d
    }

    #[test]
    fn mirror_v_is_symmetric_and_keeps_the_left() {
        let src = grid(41, 9);
        let mut buf = src.clone();
        let area = buf.area;
        mirror_v(&mut buf, area);
        for y in 0..9 {
            for x in 0..41 {
                if x < 41 / 2 {
                    assert!(same(&buf, &src, x, y), "left half touched at {x},{y}");
                }
                assert_eq!(buf[(x, y)], buf[(40 - x, y)], "not mirrored at {x},{y}");
            }
        }
    }

    #[test]
    fn mirror_quad_is_symmetric_in_both_axes() {
        let src = grid(20, 12);
        let mut buf = src.clone();
        let area = buf.area;
        mirror_quad(&mut buf, area);
        for y in 0..12 {
            for x in 0..20 {
                if x < 10 && y < 6 {
                    assert!(same(&buf, &src, x, y), "source quadrant touched {x},{y}");
                }
                assert_eq!(buf[(x, y)], buf[(19 - x, y)], "x mirror at {x},{y}");
                assert_eq!(buf[(x, y)], buf[(x, 11 - y)], "y mirror at {x},{y}");
            }
        }
    }

    #[test]
    fn pixelate_produces_uniform_blocks() {
        let mut buf = grid(200, 50);
        let area = buf.area;
        // Full downbeat: n = 26 across, so 8x4 cell blocks.
        pixelate(&mut buf, area, &driven(0.0, 1.0, 0.0, 0.0), 1.0);
        let (bw, bh) = (8u16, 4u16);
        let mut by = 0;
        while by < 50 {
            let mut bx = 0;
            while bx < 200 {
                let seed = buf[(bx, by)].clone();
                for y in by..(by + bh).min(50) {
                    for x in bx..(bx + bw).min(200) {
                        assert_eq!(buf[(x, y)], seed, "block at {bx},{by} not flat at {x},{y}");
                    }
                }
                bx += bw;
            }
            by += bh;
        }
        // And it really did something: neighbouring blocks still differ.
        assert_ne!(buf[(0, 0)], buf[(8, 0)]);
    }

    #[test]
    fn pixelate_at_rest_is_identity() {
        let src = grid(200, 50);
        let mut buf = src.clone();
        let area = buf.area;
        // n = 220 across is finer than the grid: nothing to block down.
        pixelate(&mut buf, area, &Drive::default(), 1.0);
        assert_eq!(buf, src);
    }

    #[test]
    fn slice_at_zero_amount_is_a_no_op() {
        let src = grid(200, 50);
        let mut buf = src.clone();
        let area = buf.area;
        slice(&mut buf, area, 3.25, 0.0, 1.0);
        assert_eq!(buf, src, "untriggered slice must not touch the frame");
        slice(&mut buf, area, 3.25, 1.0, 0.0);
        assert_eq!(buf, src, "zero intensity must not touch the frame");
    }

    #[test]
    fn slice_displaces_bands_and_leaves_the_rest() {
        let src = grid(200, 50);
        let mut buf = src.clone();
        let area = buf.area;
        slice(&mut buf, area, 3.25, 1.0, 1.0);
        let changed = |b: &Buffer, y: u16| (0..200).any(|x| !same(b, &src, x, y));
        let moved: Vec<u16> = (0..50).filter(|&y| changed(&buf, y)).collect();
        assert!(!moved.is_empty(), "a triggered slice should displace bands");
        assert!(moved.len() < 50, "slice should not move the whole frame");
        // Displaced rows are permutations, not new content.
        for &y in &moved {
            let mut a: Vec<String> = (0..200).map(|x| buf[(x, y)].symbol().into()).collect();
            let mut b: Vec<String> = (0..200).map(|x| src[(x, y)].symbol().into()).collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "row {y} gained or lost content");
        }
    }

    #[test]
    fn slice_is_deterministic() {
        let area = Rect::new(0, 0, 200, 50);
        let mut a = grid(200, 50);
        let mut b = grid(200, 50);
        slice(&mut a, area, 7.125, 0.8, 1.0);
        slice(&mut b, area, 7.125, 0.8, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn glitch_rows_displaces_only_some_rows() {
        let src = grid(200, 52);
        let mut buf = src.clone();
        let area = buf.area;
        // Bar-driven: amount = 0.55, so roughly half the 26 bands tear.
        glitch_rows(&mut buf, area, &driven(0.0, 1.0, 0.0, 0.0), 4.0, 1.0);
        let torn = (0..52u16)
            .filter(|&y| (0..200).any(|x| !same(&buf, &src, x, y)))
            .count();
        assert!(torn > 0, "some rows should tear");
        assert!(torn < 52, "not every row should tear");
    }

    #[test]
    fn glitch_rows_at_rest_is_identity() {
        let src = grid(80, 30);
        let mut buf = src.clone();
        let area = buf.area;
        glitch_rows(&mut buf, area, &Drive::default(), 4.0, 1.0);
        assert_eq!(buf, src);
        // Intensity gates it too, even with the drive lit up.
        glitch_rows(&mut buf, area, &driven(1.0, 1.0, 1.0, 1.0), 4.0, 0.0);
        assert_eq!(buf, src);
    }

    #[test]
    fn zoom_at_amount_zero_is_identity() {
        let src = grid(80, 30);
        let mut buf = src.clone();
        let area = buf.area;
        zoom_punch_cells(&mut buf, area, &Drive::default(), 1.0);
        assert_eq!(buf, src, "no drive, no zoom");
        zoom_punch_cells(&mut buf, area, &driven(1.0, 1.0, 1.0, 1.0), 0.0);
        assert_eq!(buf, src, "zero intensity, no zoom");
    }

    #[test]
    fn zoom_punch_scales_about_the_centre() {
        let src = grid(80, 30);
        let mut buf = src.clone();
        let area = buf.area;
        zoom_punch_cells(&mut buf, area, &driven(1.0, 0.0, 1.0, 1.0), 1.0);
        assert_ne!(buf, src, "a lit drive should zoom");
        // The centre is the fixed point of the resample.
        assert_eq!(buf[(39, 14)], src[(39, 14)]);
        // Zooming in only ever pulls content inward, so the frame's own
        // edge columns must come from further in than they were.
        assert_ne!(buf[(0, 15)], src[(0, 15)]);
    }

    #[test]
    fn shake_offsets_the_frame_and_is_deterministic() {
        let src = grid(80, 30);
        let mut a = src.clone();
        let mut b = src.clone();
        let area = src.area;
        let d = driven(1.0, 0.0, 1.0, 1.0);
        shake(&mut a, area, &d, 2.5, 1.0);
        shake(&mut b, area, &d, 2.5, 1.0);
        assert_eq!(a, b, "shake must replay identically");
        assert_ne!(a, src, "a lit drive should knock the frame");
        // Never further than the cap, whatever the drive claims.
        let hot = driven(1.0, 1.0, 1.0, 9.0);
        let mut c = src.clone();
        shake(&mut c, area, &hot, 2.5, 1.0);
        let matches_offset = |dx: i32, dy: i32| {
            (0..30).all(|y| {
                (0..80).all(|x| c[(x, y)] == clamped(&src, area, x as i32 - dx, y as i32 - dy))
            })
        };
        let bounded = (-2..=2).any(|dy| (-2..=2).any(|dx| matches_offset(dx, dy)));
        assert!(bounded, "shake exceeded the two-cell cap");
    }

    #[test]
    fn shake_at_rest_is_identity() {
        let src = grid(80, 30);
        let mut buf = src.clone();
        let area = buf.area;
        shake(&mut buf, area, &Drive::default(), 2.5, 1.0);
        assert_eq!(buf, src);
    }

    #[test]
    fn everything_survives_a_one_cell_area() {
        let d = driven(1.0, 1.0, 1.0, 1.0);
        for (w, h) in [(1u16, 1u16), (1, 8), (8, 1), (2, 1), (1, 2)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = grid(w, h);
            slice(&mut buf, area, 3.25, 1.0, 1.0);
            glitch_rows(&mut buf, area, &d, 3.25, 1.0);
            mirror_v(&mut buf, area);
            mirror_quad(&mut buf, area);
            pixelate(&mut buf, area, &d, 1.0);
            zoom_punch_cells(&mut buf, area, &d, 1.0);
            shake(&mut buf, area, &d, 3.25, 1.0);
        }
    }

    #[test]
    fn offset_areas_stay_inside_their_rect() {
        // The stage is not always at the buffer origin; nothing may write
        // outside it.
        let full = Rect::new(0, 0, 40, 20);
        let stage = Rect::new(4, 3, 30, 12);
        let mut buf = Buffer::empty(full);
        for y in stage.y..stage.y + stage.height {
            for x in stage.x..stage.x + stage.width {
                buf[(x, y)].set_char('#');
            }
        }
        let before = buf.clone();
        let d = driven(1.0, 1.0, 1.0, 1.0);
        slice(&mut buf, stage, 3.25, 1.0, 1.0);
        glitch_rows(&mut buf, stage, &d, 3.25, 1.0);
        mirror_quad(&mut buf, stage);
        pixelate(&mut buf, stage, &d, 1.0);
        zoom_punch_cells(&mut buf, stage, &d, 1.0);
        shake(&mut buf, stage, &d, 3.25, 1.0);
        for y in 0..20 {
            for x in 0..40 {
                let inside = x >= stage.x
                    && x < stage.x + stage.width
                    && y >= stage.y
                    && y < stage.y + stage.height;
                if !inside {
                    assert!(same(&buf, &before, x, y), "wrote outside the stage {x},{y}");
                }
            }
        }
    }
}
