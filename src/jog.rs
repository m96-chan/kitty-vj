//! Jog wheels. The FLX10's platters send relative CC around 64 — values
//! above are forward, below are back — so a turn arrives as a stream of
//! small deltas. We integrate them into a velocity that coasts and
//! decays, and expose a scrub offset the effects add to beat time: the
//! continuous cousin of the backspin pad.

/// A turn of this many ticks equals one beat of scrub.
const TICKS_PER_BEAT: f64 = 90.0;
/// Velocity decay per second when the hand is off (coast to a stop).
const DECAY: f64 = 6.0;
/// Below this the wheel is considered still.
const STILL: f64 = 0.02;

#[derive(Default)]
pub struct Jog {
    /// Scrub offset in beats, added to the clock by the renderer.
    offset: f64,
    /// Beats per second of scrub, from the last deltas.
    vel: f64,
    /// Hand on the platter (touch note held): grabbed = no decay.
    grabbed: bool,
}

impl Jog {
    /// Feed a relative CC value (FLX10 sends 64 ± delta).
    pub fn cc(&mut self, val: u8) {
        // Two's-complement-ish around 64; ignore the no-op center.
        let d = val as i32 - 64;
        if d == 0 {
            return;
        }
        let beats = d as f64 / TICKS_PER_BEAT;
        self.offset += beats;
        // Deltas arrive in bursts; treat each as an impulse on velocity.
        self.vel = self.vel * 0.6 + beats * 40.0 * 0.4;
    }

    pub fn touch(&mut self, down: bool) {
        self.grabbed = down;
        if down {
            self.vel = 0.0; // a hand on the platter kills the coast
        }
    }

    /// Advance by real time: coast while spinning, ease back to zero
    /// once still, so the picture returns to the clock on its own.
    pub fn tick(&mut self, dt: f64) {
        if self.grabbed {
            return;
        }
        if self.vel.abs() > STILL {
            self.offset += self.vel * dt;
            self.vel *= (1.0 - DECAY * dt).max(0.0);
        } else {
            self.vel = 0.0;
            // Recentre — a scrub is a departure, not a new home. Fast
            // enough to be back on the grid within a beat or so.
            self.offset *= (1.0 - 5.0 * dt).max(0.0);
            if self.offset.abs() < 1e-4 {
                self.offset = 0.0;
            }
        }
    }

    /// Beats to add to clock time.
    pub fn offset(&self) -> f64 {
        self.offset
    }

    pub fn active(&self) -> bool {
        self.grabbed || self.offset.abs() > 1e-3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_back_cancel() {
        let mut j = Jog::new_grabbed();
        j.cc(74); // +10
        j.cc(54); // -10
        assert!(j.offset().abs() < 1e-9, "offset {}", j.offset());
    }

    #[test]
    fn direction_matches_sign() {
        let mut j = Jog::new_grabbed();
        j.cc(70);
        assert!(j.offset() > 0.0);
        let mut k = Jog::new_grabbed();
        k.cc(58);
        assert!(k.offset() < 0.0);
    }

    #[test]
    fn coasts_then_returns_to_clock() {
        let mut j = Jog::default();
        for _ in 0..10 {
            j.cc(70); // spin it up
        }
        assert!(j.offset() > 0.0);
        // Let go: a couple of seconds of ticks and it's back on the grid.
        for _ in 0..200 {
            j.tick(1.0 / 60.0);
        }
        assert!(j.offset().abs() < 1e-3, "did not recentre: {}", j.offset());
        assert!(!j.active());
    }

    #[test]
    fn grabbed_holds_position() {
        let mut j = Jog::new_grabbed();
        j.cc(80);
        let held = j.offset();
        for _ in 0..120 {
            j.tick(1.0 / 60.0);
        }
        assert_eq!(j.offset(), held, "grabbed wheel should not drift back");
    }

    impl Jog {
        fn new_grabbed() -> Self {
            let mut j = Jog::default();
            j.touch(true);
            j
        }
    }
}
