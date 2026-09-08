//! Camera — the Modulator shape: reads the clock, writes parameters,
//! never touches a pixel.
//!
//! Keeping this separate from the Transform passes is the point. A
//! modulator has no frame to hand back, so folding it into an
//! effect trait would mean either a fake return value or a trait that
//! sometimes draws and sometimes doesn't. Over there the camera, the
//! show sequences and the drive signals all worked this way: they
//! export numbers, and whatever draws reads them.

use crate::drive::Drive;
use crate::rng::{hash3, unit_f64};

/// What a modulator hands to whatever is sampling artwork.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    /// Scale about the focus point; 1.0 = cover fit.
    pub zoom: f64,
    /// Focus point in normalised image space, 0..1.
    pub fx: f64,
    pub fy: f64,
}

impl Default for Cam {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            fx: 0.5,
            fy: 0.5,
        }
    }
}

pub trait Modulator {
    /// For the HUD, and so a modulator can be named in a config the way
    /// an effect can.
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
    /// Produce this frame's parameters. `t` is the visual clock in
    /// seconds; `d` carries the grid pulses and the kick.
    fn cam(&self, t: f64, d: &Drive) -> Cam;
}

/// Ken Burns — a slow zoom and pan across the plate, re-rolled per
/// scene. Deliberately **not** groove-gated: it keeps breathing through
/// a breakdown, which is what stops a quiet passage reading as a freeze.
pub struct KenBurns {
    z0: f64,
    z1: f64,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    dur: f64,
    /// Visual-clock time the scene started.
    t0: f64,
}

impl KenBurns {
    /// Roll a fresh move. `seed` distinguishes scenes; `t0` anchors the
    /// ramp so a scene change restarts the drift rather than jumping
    /// into the middle of one.
    pub fn roll(seed: u64, t0: f64) -> Self {
        let r = |k: u64| unit_f64(hash3(seed, k, 0));
        Self {
            z0: 1.02 + r(1) * 0.14,
            z1: 1.02 + r(2) * 0.18,
            x0: (r(3) - 0.5) * 0.12,
            x1: (r(4) - 0.5) * 0.12,
            y0: (r(5) - 0.5) * 0.10,
            y1: (r(6) - 0.5) * 0.10,
            dur: 14.0 + r(7) * 16.0,
            t0,
        }
    }
}

impl Modulator for KenBurns {
    fn name(&self) -> &'static str {
        "KENBURNS"
    }

    fn cam(&self, t: f64, d: &Drive) -> Cam {
        let p = ((t - self.t0) / self.dur).clamp(0.0, 1.0);
        let lerp = |a: f64, b: f64| a + (b - a) * p;
        // The kick adds a lateral swing on top of the drift — the plate
        // leans into the beat without leaving the framing.
        let swing_x = 0.022 * (t * 0.9).sin() * d.thump;
        let swing_y = 0.016 * (t * 0.7).cos() * d.thump;
        Cam {
            zoom: lerp(self.z0, self.z1),
            fx: (0.5 + lerp(self.x0, self.x1) + swing_x).clamp(0.0, 1.0),
            fy: (0.5 + lerp(self.y0, self.y1) + swing_y).clamp(0.0, 1.0),
        }
    }
}

/// Punch-in — on a phrase the camera dives into a random region and
/// eases back out. Over there this fired 80% of the time on a phrase;
/// here it rides the phrase pulse directly, so it needs no state.
pub struct PunchIn {
    seed: u64,
}

impl PunchIn {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Modulator for PunchIn {
    fn name(&self) -> &'static str {
        "PUNCHIN"
    }

    fn cam(&self, t: f64, d: &Drive) -> Cam {
        // Which region depends on which phrase we are in, so the dive
        // lands somewhere new each time without storing anything.
        let phrase = (t / 8.0).floor() as u64;
        let r = |k: u64| unit_f64(hash3(self.seed, phrase, k));
        let depth = d.gphrase();
        Cam {
            zoom: 1.0 + 1.2 * depth,
            fx: 0.18 + r(1) * 0.64,
            fy: 0.18 + r(2) * 0.64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ken_burns_drifts_over_its_duration() {
        let k = KenBurns::roll(7, 0.0);
        let a = k.cam(0.0, &Drive::default());
        let b = k.cam(k.dur, &Drive::default());
        assert_ne!(a, b, "the camera should have moved");
        // And it stops at the end rather than running away.
        let past = k.cam(k.dur * 3.0, &Drive::default());
        assert_eq!(b, past, "drift should clamp at the end of its ramp");
    }

    #[test]
    fn ken_burns_keeps_moving_without_groove() {
        // A breakdown must not freeze the frame: with every drive signal
        // at zero the drift still advances.
        let k = KenBurns::roll(3, 0.0);
        let dead = Drive::default();
        assert_ne!(k.cam(0.0, &dead), k.cam(k.dur * 0.5, &dead));
    }

    #[test]
    fn focus_stays_inside_the_image() {
        let k = KenBurns::roll(11, 0.0);
        let mut d = Drive::default();
        d.thump = 1.0;
        for i in 0..200 {
            let c = k.cam(i as f64 * 0.37, &d);
            assert!((0.0..=1.0).contains(&c.fx), "fx {}", c.fx);
            assert!((0.0..=1.0).contains(&c.fy), "fy {}", c.fy);
            assert!(c.zoom >= 1.0, "zoom {}", c.zoom);
        }
    }

    #[test]
    fn punchin_dives_only_on_a_phrase() {
        let p = PunchIn::new(5);
        let quiet = Drive::default();
        assert!((p.cam(3.0, &quiet).zoom - 1.0).abs() < 1e-9);
        let mut hot = Drive::default();
        hot.phrase = 1.0;
        hot.groove = 1.0;
        assert!(p.cam(3.0, &hot).zoom > 1.5, "phrase should punch in");
    }

    #[test]
    fn punchin_picks_a_new_region_each_phrase() {
        let p = PunchIn::new(5);
        let d = Drive::default();
        let a = p.cam(1.0, &d);
        let b = p.cam(9.0, &d); // next phrase
        assert!(a.fx != b.fx || a.fy != b.fy);
    }

    #[test]
    fn rolls_are_deterministic() {
        let a = KenBurns::roll(42, 1.5);
        let b = KenBurns::roll(42, 1.5);
        let d = Drive::default();
        assert_eq!(a.cam(4.0, &d), b.cam(4.0, &d));
    }
}
