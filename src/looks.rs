//! Looks — persistent per-scene colour treatments, ported from
//! EasyPngVJ's `looks` pool. Each is a per-cell colour transform applied
//! after the channel mix, driven by the grid pulses rather than raw
//! phase, so the numbers match the original.

use ratatui::style::Color;

use crate::drive::Drive;
use crate::pass::{CellCtx, ColorPass, contrast, hsv, hue_rotate, luma, rgb, saturate};

#[derive(Clone, Copy, PartialEq)]
pub enum Look {
    Plain,
    /// grayscale + contrast/brightness hardening on the beat.
    HardMono,
    /// Luminance mapped onto one per-scene hue.
    Duotone,
    /// Continuous 22°/s hue rotation, kicked on a phrase.
    HueCycle,
    /// Saturation breathes with the kick.
    SatPump,
    /// Every other row darkened.
    Scan,
    /// Luminance ramped into the scene palette — what a terminal wants.
    Lut,
}

pub const LOOKS: [Look; 7] = [
    Look::Plain,
    Look::HardMono,
    Look::Duotone,
    Look::HueCycle,
    Look::SatPump,
    Look::Scan,
    Look::Lut,
];

impl Look {
    pub fn name(&self) -> &'static str {
        match self {
            Look::Plain => "PLAIN",
            Look::HardMono => "MONO",
            Look::Duotone => "DUO",
            Look::HueCycle => "HUE",
            Look::SatPump => "SATP",
            Look::Scan => "SCAN",
            Look::Lut => "LUT",
        }
    }

    pub fn next(&self) -> Look {
        let i = LOOKS.iter().position(|l| l == self).unwrap_or(0);
        LOOKS[(i + 1) % LOOKS.len()]
    }
}

/// Apply a look to one colour. `hue_base` is the scene's random hue,
/// `t` the visual clock in seconds, `y` the cell row.
pub fn apply(
    c: Color,
    look: Look,
    d: &Drive,
    t: f64,
    y: u16,
    hue_base: f64,
    accent: (u8, u8, u8),
) -> Color {
    let Color::Rgb(r0, g0, b0) = c else { return c };
    let (mut r, mut g, mut b) = (r0 as f64, g0 as f64, b0 as f64);
    let beat = d.gbeat();

    match look {
        Look::Plain => {
            // Even without a look, every plate pulses over there.
            let k = 0.92 + 0.28 * beat;
            r *= k;
            g *= k;
            b *= k;
        }
        Look::HardMono => {
            let l = luma(r, g, b);
            let v = contrast(l, 1.55 + 0.55 * beat) * (0.95 + 0.3 * beat);
            r = v;
            g = v;
            b = v;
        }
        Look::Duotone => {
            // Luminance onto one per-scene hue, contrast-hardened on the
            // beat: the whole frame agrees on a single colour.
            let l = (contrast(luma(r, g, b), 1.35 + 0.3 * beat) / 255.0).clamp(0.0, 1.0);
            let (hr, hg, hb) = hsv(hue_base / 360.0, 0.85, 1.0);
            let bright = 0.72 + 0.3 * beat;
            r = hr as f64 * l * bright;
            g = hg as f64 * l * bright;
            b = hb as f64 * l * bright;
        }
        Look::HueCycle => {
            let deg = (t * 22.0 + d.gphrase() * 40.0).rem_euclid(360.0);
            let rot = hue_rotate((r, g, b), deg);
            let sat = 1.3 + 0.6 * d.thump;
            let (rr, gg, bb) = saturate(rot, sat);
            r = rr;
            g = gg;
            b = bb;
        }
        Look::SatPump => {
            let (rr, gg, bb) = saturate((r, g, b), 1.0 + 2.1 * d.thump);
            r = contrast(rr, 1.05 + 0.25 * beat);
            g = contrast(gg, 1.05 + 0.25 * beat);
            b = contrast(bb, 1.05 + 0.25 * beat);
        }
        Look::Scan => {
            // Row parity: free in a cell grid, where the original needed
            // a 4x4 pattern fill.
            if y.is_multiple_of(2) {
                let k = 0.45;
                r *= k;
                g *= k;
                b *= k;
            }
        }
        Look::Lut => {
            // Luminance ramp into the palette: black → accent → white.
            let l = (luma(r, g, b) / 255.0).clamp(0.0, 1.0);
            let (ar, ag, ab) = (accent.0 as f64, accent.1 as f64, accent.2 as f64);
            let (rr, gg, bb) = if l < 0.5 {
                let k = l * 2.0;
                (2.0 + ar * k, 0.0 + ag * k, 11.0 + ab * k)
            } else {
                let k = l * 2.0 - 1.0;
                (
                    ar + (255.0 - ar) * k,
                    ag + (255.0 - ag) * k,
                    ab + (255.0 - ab) * k,
                )
            };
            let scale = 0.5 + 0.9 * l;
            r = rr * scale;
            g = gg * scale;
            b = bb * scale;
        }
    }

    rgb(r, g, b)
}

/// The Transform shape: a look is a per-cell colour transform, the same
/// as an SCFX pass or a beat hit. Three ports arrived at this
/// independently; the trait is where they finally agree.
impl ColorPass for Look {
    fn name(&self) -> &'static str {
        Look::name(self)
    }

    fn amount(&self, _ctx: &CellCtx) -> f64 {
        if *self == Look::Plain { 0.0 } else { 1.0 }
    }

    fn map(&self, c: Color, ctx: &CellCtx) -> Color {
        apply(c, *self, ctx.drive, ctx.t, ctx.y, ctx.hue_base, ctx.accent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCENT: (u8, u8, u8) = (0, 255, 213);

    fn d() -> Drive {
        Drive::default()
    }

    #[test]
    fn hardmono_is_colourless() {
        let c = apply(
            Color::Rgb(200, 40, 90),
            Look::HardMono,
            &d(),
            0.0,
            0,
            0.0,
            ACCENT,
        );
        if let Color::Rgb(r, g, b) = c {
            assert_eq!(r, g);
            assert_eq!(g, b);
        } else {
            panic!("expected rgb");
        }
    }

    #[test]
    fn scan_darkens_alternate_rows_only() {
        let src = Color::Rgb(200, 200, 200);
        let even = apply(src, Look::Scan, &d(), 0.0, 0, 0.0, ACCENT);
        let odd = apply(src, Look::Scan, &d(), 0.0, 1, 0.0, ACCENT);
        assert_ne!(even, odd, "scanlines need row parity");
        assert_eq!(odd, src, "odd rows pass through");
    }

    #[test]
    fn huecycle_moves_with_time() {
        let src = Color::Rgb(220, 30, 30);
        let a = apply(src, Look::HueCycle, &d(), 0.0, 0, 0.0, ACCENT);
        let b = apply(src, Look::HueCycle, &d(), 4.0, 0, 0.0, ACCENT);
        assert_ne!(a, b, "hue should rotate over time");
    }

    #[test]
    fn lut_maps_gray_into_the_palette() {
        // Mid gray should land near the accent, not stay gray.
        let c = apply(
            Color::Rgb(128, 128, 128),
            Look::Lut,
            &d(),
            0.0,
            0,
            0.0,
            ACCENT,
        );
        if let Color::Rgb(r, g, b) = c {
            assert!(g > r && g > b, "expected the accent hue, got {r},{g},{b}");
        }
    }

    #[test]
    fn every_look_stays_in_range() {
        for look in LOOKS {
            for v in [0u8, 1, 127, 254, 255] {
                let c = apply(
                    Color::Rgb(v, v / 2, 255 - v),
                    look,
                    &d(),
                    3.3,
                    v as u16,
                    210.0,
                    ACCENT,
                );
                assert!(matches!(c, Color::Rgb(..)), "{} broke", look.name());
            }
        }
    }
}
