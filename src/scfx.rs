//! Sound Color FX — the visual side of the mixer's SCFX section. One
//! global type (like the hardware's radio buttons), one COLOR knob per
//! channel: center is neutral, twisting applies the transform to that
//! channel's cells as they land in the composite.

use ratatui::style::Color;

#[derive(Clone, Copy, PartialEq)]
pub enum ScfxType {
    Space,
    DubEcho,
    Sweep,
    Noise,
    Crush,
    Filter,
}

/// Hardware order, left to right on the FLX10's SCFX row.
pub const TYPES: [ScfxType; 6] = [
    ScfxType::Space,
    ScfxType::DubEcho,
    ScfxType::Sweep,
    ScfxType::Noise,
    ScfxType::Crush,
    ScfxType::Filter,
];

impl ScfxType {
    pub fn name(&self) -> &'static str {
        match self {
            ScfxType::Space => "SPACE",
            ScfxType::DubEcho => "DUBECHO",
            ScfxType::Sweep => "SWEEP",
            ScfxType::Noise => "NOISE",
            ScfxType::Crush => "CRUSH",
            ScfxType::Filter => "FILTER",
        }
    }

    pub fn next(&self) -> ScfxType {
        let i = TYPES.iter().position(|t| t == self).unwrap_or(0);
        TYPES[(i + 1) % TYPES.len()]
    }
}

fn map3(c: Color, f: impl Fn(f64) -> f64) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            f(r as f64).clamp(0.0, 255.0) as u8,
            f(g as f64).clamp(0.0, 255.0) as u8,
            f(b as f64).clamp(0.0, 255.0) as u8,
        ),
        other => other,
    }
}

fn transform(c: Color, t: ScfxType, knob: f64, beat: f64, x: u16, y: u16, w: u16) -> Color {
    // knob 0..1, center neutral; a in [-1, 1].
    let a = (knob - 0.5) * 2.0;
    let amt = a.abs();
    if amt < 0.04 {
        return c;
    }
    match t {
        ScfxType::Filter => {
            if a < 0.0 {
                // LPF: sink into the dark, blue surviving longest.
                match c {
                    Color::Rgb(r, g, b) => Color::Rgb(
                        (r as f64 * (1.0 - amt * 0.85)) as u8,
                        (g as f64 * (1.0 - amt * 0.7)) as u8,
                        (b as f64 * (1.0 - amt * 0.4)) as u8,
                    ),
                    other => other,
                }
            } else {
                // HPF: wash toward white.
                map3(c, |v| v + (255.0 - v) * amt * 0.7)
            }
        }
        ScfxType::Crush => {
            let levels = (7.0 - amt * 5.0).max(2.0);
            map3(c, |v| (v / 255.0 * levels).round() / levels * 255.0)
        }
        ScfxType::Space => {
            // Saturation push + a slow hue roll that deepens with the knob.
            match c {
                Color::Rgb(r, g, b) => {
                    let (r, g, b) = (r as f64, g as f64, b as f64);
                    let mean = (r + g + b) / 3.0;
                    let sat = 1.0 + amt * 0.8;
                    let (r, g, b) = (
                        mean + (r - mean) * sat,
                        mean + (g - mean) * sat,
                        mean + (b - mean) * sat,
                    );
                    // Cheap hue roll: rotate the channels toward each other.
                    let roll = (beat * 0.25).fract() * amt;
                    let (r2, g2, b2) = (
                        r * (1.0 - roll) + b * roll,
                        g * (1.0 - roll) + r * roll,
                        b * (1.0 - roll) + g * roll,
                    );
                    Color::Rgb(
                        r2.clamp(0.0, 255.0) as u8,
                        g2.clamp(0.0, 255.0) as u8,
                        b2.clamp(0.0, 255.0) as u8,
                    )
                }
                other => other,
            }
        }
        ScfxType::Sweep => {
            // A curtain of darkness drawn in from the knob's side.
            let pos = x as f64 / w.max(1) as f64;
            let g = if a > 0.0 { pos } else { 1.0 - pos };
            let k = 1.0 - amt * ((g * 1.5) - 0.25).clamp(0.0, 1.0);
            map3(c, |v| v * k)
        }
        ScfxType::Noise => {
            // Static: per-cell brightness jitter, gray speckle at the top.
            let tick = (beat.max(0.0) * 16.0) as u64;
            let n = crate::rng::unit_f64(crate::rng::hash3(x as u64, y as u64, tick ^ 0xcafe));
            if n > 1.0 - amt * 0.25 {
                Color::Rgb(200, 200, 200)
            } else {
                let k = 1.0 + (n - 0.5) * 1.8 * amt;
                map3(c, |v| v * k)
            }
        }
        ScfxType::DubEcho => {
            // A brightness wave travels across the frame in beat time —
            // the repeats of the delay, spatialized. Direction follows
            // the knob's side of center.
            let dir = a.signum();
            let phase = (x as f64 * 0.035 * dir - beat).rem_euclid(1.0);
            let wave = 0.45 + 0.55 * (1.0 - phase).powi(2);
            let k = 1.0 * (1.0 - amt) + wave * amt * 1.3;
            map3(c, |v| v * k)
        }
    }
}

/// Apply to a cell's colors in place.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    cell: &mut ratatui::buffer::Cell,
    t: ScfxType,
    knob: f64,
    beat: f64,
    x: u16,
    y: u16,
    w: u16,
) {
    cell.fg = transform(cell.fg, t, knob, beat, x, y, w);
    cell.bg = transform(cell.bg, t, knob, beat, x, y, w);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_is_neutral() {
        for t in TYPES {
            let c = transform(Color::Rgb(120, 80, 200), t, 0.5, 3.7, 10, 4, 80);
            assert_eq!(c, Color::Rgb(120, 80, 200), "{} not neutral", t.name());
        }
    }

    #[test]
    fn filter_ends_dark_and_bright() {
        let dark = transform(
            Color::Rgb(200, 200, 200),
            ScfxType::Filter,
            0.0,
            0.0,
            0,
            0,
            80,
        );
        let bright = transform(Color::Rgb(50, 50, 50), ScfxType::Filter, 1.0, 0.0, 0, 0, 80);
        if let (Color::Rgb(r1, ..), Color::Rgb(r2, ..)) = (dark, bright) {
            assert!(r1 < 60, "lpf should darken, got {r1}");
            assert!(r2 > 150, "hpf should brighten, got {r2}");
        } else {
            panic!("expected rgb");
        }
    }
}
