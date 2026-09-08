//! A/B crossfade masks. The terminal can't alpha-blend two glyphs, so a
//! transition is a per-cell decision: show deck A or deck B. Masks are
//! monotonic in the fader position and stable per cell (seed-hashed, no
//! per-frame reshuffle), so pulling the fader accumulates cells instead
//! of boiling them.

use crate::rng::{hash3, unit_f64};

#[derive(Clone, Copy, PartialEq)]
pub enum XfadeStyle {
    /// Per-cell hash threshold — the terminal-native crossfade.
    Dissolve,
    /// Left-to-right sweep with a hashed ragged edge.
    Wipe,
    /// Vertical slats, alternating directions (EasyPngVJ's wipe).
    Blinds,
}

pub const STYLES: [XfadeStyle; 3] = [XfadeStyle::Dissolve, XfadeStyle::Wipe, XfadeStyle::Blinds];

const BLINDS: f64 = 14.0;

impl XfadeStyle {
    pub fn name(&self) -> &'static str {
        match self {
            XfadeStyle::Dissolve => "DSLV",
            XfadeStyle::Wipe => "WIPE",
            XfadeStyle::Blinds => "BLND",
        }
    }

    pub fn next(&self) -> XfadeStyle {
        let i = STYLES.iter().position(|s| s == self).unwrap_or(0);
        STYLES[(i + 1) % STYLES.len()]
    }

    /// true = this cell shows deck B. `t` is the fader position [0, 1].
    pub fn shows_b(&self, x: u16, y: u16, w: u16, h: u16, t: f64) -> bool {
        if t <= 0.0 {
            return false;
        }
        if t >= 1.0 {
            return true;
        }
        let _ = h;
        match self {
            XfadeStyle::Dissolve => unit_f64(hash3(x as u64, y as u64, 3)) < t,
            XfadeStyle::Wipe => {
                let edge = unit_f64(hash3(y as u64, 4, 0)) * 0.08;
                (x as f64 + 0.5) / (w as f64) < t * 1.08 - edge
            }
            XfadeStyle::Blinds => {
                let slat = (x as f64 / w as f64 * BLINDS) as u64;
                let within = (x as f64 / w as f64 * BLINDS).fract();
                // Alternate slats sweep in opposite directions.
                let p = if slat.is_multiple_of(2) { within } else { 1.0 - within };
                p < t
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_pure() {
        for s in STYLES {
            for x in 0..40 {
                for y in 0..20 {
                    assert!(!s.shows_b(x, y, 40, 20, 0.0));
                    assert!(s.shows_b(x, y, 40, 20, 1.0));
                }
            }
        }
    }

    #[test]
    fn dissolve_is_monotonic() {
        // A cell shown at t stays shown for every larger t.
        for x in 0..40 {
            for y in 0..20 {
                let mut shown = false;
                for i in 0..=20 {
                    let now = XfadeStyle::Dissolve.shows_b(x, y, 40, 20, i as f64 / 20.0);
                    assert!(!shown || now, "cell flickered back at t={}", i);
                    shown = now;
                }
            }
        }
    }
}
