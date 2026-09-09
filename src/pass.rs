//! The four dataflow shapes every visual unit in this project fits.
//!
//! The porting work turned up ~88 effects, and the temptation was one
//! `Effect` trait over all of them. That breaks, because "effect" is a
//! description of what something looks like, not of how data moves
//! through it. Sorted by arity instead, they fall into four shapes that
//! stay stable however many effects get added:
//!
//! | shape | reads | writes | examples |
//! |---|---|---|---|
//! | **Source** | nothing (or a plate) | one frame | `pixfx`, `pixparticles`, the cell effects, `imgdust` |
//! | **Transform** | one frame | that frame | `looks`, `scfx`, `postfx`, `pixpost`, the beat hits |
//! | **Mixer** | **two** frames + a position | one frame | transitions |
//! | **Modulator** | the clock | parameters, never pixels | `drive`, show sequences, camera |
//! | **Orchestration** | nothing | *which units are active* | `scene` |
//!
//! Only Mixer takes two inputs, which is why transitions could not have
//! been retrofitted onto a one-frame trait later. Modulators produce no
//! pixels at all and must never be folded in with Transforms.
//!
//! Three things the ports taught us, after this table was first written:
//!
//! - **A Modulator is stateful by default.** The canonical signature is
//!   `update(&mut self, dt, input) -> Exports`, as `drive` and `show`
//!   both have: a state machine's position is not recoverable from a
//!   clock reading. `camera::Modulator`'s pure `cam(&self, t)` is the
//!   special case — a modulator whose constants were rolled once — not
//!   the rule.
//! - **A Mixer needed more channels than a mask.** Luma keys off the
//!   incoming frame and Flash wants an additive bloom, neither of which
//!   "which side wins at this cell" can express. Both arrived as
//!   defaulted methods rather than a signature change, so the shape
//!   stretched instead of breaking.
//! - **Orchestration is a fifth shape, above the four rather than
//!   inside them.** A scene reads no frame and writes no pixel, but it
//!   is not a Modulator either: a modulator exports numbers every
//!   frame, while a scene speaks only when it changes, and what it
//!   hands over is a cast list rather than a parameter.
//!
//! This module defines the narrowest of those shapes — a per-cell colour
//! transform — because three ports independently arrived at it:
//! `looks::apply`, `scfx::apply` and the beat hits were the same
//! function wearing three names.

use ratatui::style::Color;

use crate::drive::Drive;

/// Everything a per-cell colour transform is allowed to read.
///
/// One context for all of them: a look wants the row (scanlines), an
/// SCFX pass wants the column (the dub-echo wave) and the scene hue,
/// and every one of them wants the drive signals. Passing a single
/// struct keeps the trait honest — a pass cannot quietly reach for wall
/// time, which is what the determinism contract forbids.
#[derive(Clone, Copy)]
pub struct CellCtx<'a> {
    pub drive: &'a Drive,
    /// Visual clock in seconds — wall time, frozen with the show.
    pub t: f64,
    /// Beat time. Grid-locked passes (a strobe on eighths) read this;
    /// clock-locked ones (a hue cycle per second) read `t`. Both are
    /// carried because confusing them is exactly the units bug that
    /// made the hue cycle tempo-dependent.
    pub beat: f64,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    /// Global fader, [0,1].
    pub intensity: f64,
    /// Per-scene random hue, degrees.
    pub hue_base: f64,
    pub accent: (u8, u8, u8),
}

/// A colour in, a colour out. The smallest shape, and the one three
/// separate ports converged on.
pub trait ColorPass {
    fn name(&self) -> &'static str;

    /// How hard this pass is biting, [0,1]-ish. Kept separate from
    /// `map` so callers can skip the work — and so the HUD can show it
    /// without running the transform.
    fn amount(&self, ctx: &CellCtx) -> f64;

    fn map(&self, c: Color, ctx: &CellCtx) -> Color;

    /// Apply to both halves of a halfblock cell.
    fn apply(&self, cell: &mut ratatui::buffer::Cell, ctx: &CellCtx) {
        cell.fg = self.map(cell.fg, ctx);
        cell.bg = self.map(cell.bg, ctx);
    }
}

/// Clamp and pack a float triple back into a terminal colour.
///
/// This is the one packer. Five modules had five conventions — round,
/// truncate, clamp one end, clamp both, rely on the saturating cast —
/// which means the same computed colour landed a unit apart depending on
/// which module packed it. The decision here: round half-up, clamp both
/// ends.
pub fn rgb(r: f64, g: f64, b: f64) -> Color {
    Color::Rgb(
        (r + 0.5).clamp(0.0, 255.0) as u8,
        (g + 0.5).clamp(0.0, 255.0) as u8,
        (b + 0.5).clamp(0.0, 255.0) as u8,
    )
}

/// Rec.601 luminance, the weighting every ported effect uses.
pub fn luma(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

// ---------------------------------------------------------------------
// The shared maths. Each of these existed as two to six private copies
// across the ported modules, drifting in small ways (an unguarded
// divide, a missing clamp, a different rounding). One definition each;
// the ports' "the numbers match the original" claim is only checkable
// when the arithmetic under the numbers is shared.

/// HSV to RGB bytes. Two identical copies existed (looks, pixfx).
pub fn hsv(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f64;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Luma-preserving hue rotation (Rec.601 matrix). Two identical
/// nine-coefficient copies existed (looks, scfx).
pub fn hue_rotate((r, g, b): (f64, f64, f64), deg: f64) -> (f64, f64, f64) {
    let (s, co) = deg.to_radians().sin_cos();
    (
        r * (0.213 + co * 0.787 - s * 0.213)
            + g * (0.715 - co * 0.715 - s * 0.715)
            + b * (0.072 - co * 0.072 + s * 0.928),
        r * (0.213 - co * 0.213 + s * 0.143)
            + g * (0.715 + co * 0.285 + s * 0.140)
            + b * (0.072 - co * 0.072 - s * 0.283),
        r * (0.213 - co * 0.213 - s * 0.787)
            + g * (0.715 - co * 0.715 + s * 0.715)
            + b * (0.072 + co * 0.928 + s * 0.072),
    )
}

/// Saturation about the luma axis.
pub fn saturate((r, g, b): (f64, f64, f64), k: f64) -> (f64, f64, f64) {
    let l = luma(r, g, b);
    (l + (r - l) * k, l + (g - l) * k, l + (b - l) * k)
}

/// Contrast about mid-grey, in the 0-255 domain.
pub fn contrast(v: f64, k: f64) -> f64 {
    (v - 128.0) * k + 128.0
}

/// GLSL-style smoothstep, guarded: a degenerate edge pair steps rather
/// than dividing by zero (one prior copy divided unguarded).
pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    if e0 == e1 {
        return if x < e0 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Beat time quantized to 16ths, for groove-locked flicker. Four copies
/// existed; this is the definition FrameCtx::tick16 forwards to.
pub fn tick16(beat: f64) -> u64 {
    (beat.max(0.0) * 16.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Half;
    impl ColorPass for Half {
        fn name(&self) -> &'static str {
            "HALF"
        }
        fn amount(&self, _: &CellCtx) -> f64 {
            1.0
        }
        fn map(&self, c: Color, _: &CellCtx) -> Color {
            match c {
                Color::Rgb(r, g, b) => Color::Rgb(r / 2, g / 2, b / 2),
                other => other,
            }
        }
    }

    fn ctx<'a>(d: &'a Drive) -> CellCtx<'a> {
        CellCtx {
            drive: d,
            t: 0.0,
            beat: 0.0,
            x: 0,
            y: 0,
            w: 80,
            intensity: 1.0,
            hue_base: 0.0,
            accent: (0, 255, 213),
        }
    }

    #[test]
    fn apply_touches_both_halves_of_a_cell() {
        let d = Drive::default();
        let mut cell = ratatui::buffer::Cell::default();
        cell.set_fg(Color::Rgb(200, 100, 50));
        cell.set_bg(Color::Rgb(80, 40, 20));
        Half.apply(&mut cell, &ctx(&d));
        assert_eq!(cell.fg, Color::Rgb(100, 50, 25));
        assert_eq!(cell.bg, Color::Rgb(40, 20, 10));
    }

    #[test]
    fn non_rgb_colours_pass_through() {
        let d = Drive::default();
        assert_eq!(Half.map(Color::Reset, &ctx(&d)), Color::Reset);
    }

    #[test]
    fn helpers_clamp_and_weight() {
        assert_eq!(rgb(-5.0, 300.0, 12.4), Color::Rgb(0, 255, 12));
        // Green carries most of perceived brightness.
        assert!(luma(0.0, 255.0, 0.0) > luma(255.0, 0.0, 0.0));
    }
}
