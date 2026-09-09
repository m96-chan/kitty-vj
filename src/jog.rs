//! Jog wheels. The FLX10's platters send relative CC around 64 — values
//! above are forward, below are back — so a turn arrives as a stream of
//! small deltas, integrated into a scrub offset the effects add to beat
//! time.
//!
//! Two gestures come out of the same wheel, and telling them apart is
//! the whole point:
//!
//! - **Nudge** — a hand on the platter, moved a little. Beat time
//!   follows the hand and eases back when it lets go. The continuous
//!   cousin of the backspin pad.
//! - **Spin** — a flick released above [`FLICK_VEL`]. The platter is now
//!   free and carries on turning, so the picture keeps travelling long
//!   after the hand is gone: **backspin** running time backwards,
//!   **frontspin** running it forwards. This is the gesture, not the
//!   scrub.
//!
//! A released platter is a flywheel against friction, so a spin's
//! displacement is `v0 * tau * (1 - e^(-t/tau))` — it covers most of its
//! ground early and asymptotes. The gesture is therefore *how far it
//! travels*, `v0 * tau`, and a spin gets a long tau precisely so a flick
//! throws the picture a bar or more rather than a fraction of a beat.
//!
//! Unlike everything else in this project, a jog is real-time state: it
//! is a hand moving now, so it cannot be a function of beat time. The
//! determinism contract applies to what the effects do with the offset,
//! not to the offset itself.

/// A turn of this many ticks equals one beat of scrub.
const TICKS_PER_BEAT: f64 = 90.0;
/// Coast time constant for a nudge — back on the grid inside a beat.
const NUDGE_TAU: f64 = 1.0 / 6.0;
/// Coast time constant for a released spin. Long, because a platter
/// let go at speed keeps going; this is what makes the gesture read.
const SPIN_TAU: f64 = 0.55;
/// Release speed, in beats per second, above which a turn is a spin
/// rather than a nudge. About a beat of travel in the first tenth of a
/// second — faster than anyone nudges by accident.
const FLICK_VEL: f64 = 2.5;
/// Below this the wheel is considered still.
const STILL: f64 = 0.02;
/// How hard the offset is pulled back to the grid once the wheel stops.
const RECENTRE: f64 = 5.0;

/// Which gesture is running, for the HUD and for effects that want to
/// react to a spin rather than to the offset it produces.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Spin {
    #[default]
    None,
    /// Flicked backwards: time runs the wrong way, accelerating away.
    Back,
    /// Flicked forwards: the record spun up and ran ahead.
    Front,
}

impl Spin {
    pub fn name(&self) -> &'static str {
        match self {
            Spin::None => "",
            Spin::Back => "BACKSPIN",
            Spin::Front => "FRONTSPIN",
        }
    }
}

#[derive(Default)]
pub struct Jog {
    /// Scrub offset in beats, added to the clock by the renderer.
    offset: f64,
    /// Beats per second of scrub, from the last deltas.
    vel: f64,
    /// Hand on the platter (touch note held): grabbed = no decay.
    grabbed: bool,
    /// The gesture currently gliding, if any.
    spin: Spin,
    /// Speed the current spin was released at, for `spin_amount`.
    launch: f64,
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
        // The outer ring sends no touch note, so a flick there has to be
        // caught here or it would only ever read as a nudge.
        if !self.grabbed {
            self.arm_release();
        }
    }

    pub fn touch(&mut self, down: bool) {
        if down {
            // A hand on the platter kills the coast and any gesture: you
            // caught the record.
            self.grabbed = true;
            self.vel = 0.0;
            self.spin = Spin::None;
            self.launch = 0.0;
        } else {
            self.grabbed = false;
            self.arm_release();
        }
    }

    /// Decide, at the moment the wheel is let go, whether this was a
    /// nudge or a spin. Speed at release is the whole difference.
    fn arm_release(&mut self) {
        if self.vel.abs() >= FLICK_VEL {
            self.spin = if self.vel < 0.0 {
                Spin::Back
            } else {
                Spin::Front
            };
            self.launch = self.vel.abs();
        } else if self.spin == Spin::None {
            self.launch = 0.0;
        }
    }

    /// The gesture gliding right now.
    pub fn spin(&self) -> Spin {
        self.spin
    }

    /// How much of the launch speed is left, [0,1] — for effects that
    /// want to react to the gesture and not just to where it moved time.
    pub fn spin_amount(&self) -> f64 {
        if self.spin == Spin::None || self.launch <= 0.0 {
            0.0
        } else {
            (self.vel.abs() / self.launch).clamp(0.0, 1.0)
        }
    }

    /// Advance by real time: glide while the wheel is still turning,
    /// then ease back to the grid, so the picture returns on its own.
    pub fn tick(&mut self, dt: f64) {
        if self.grabbed {
            return;
        }
        if self.vel.abs() > STILL {
            // A spin coasts far because its tau is long; a nudge dies
            // almost at once. Same integration, different flywheel.
            let tau = if self.spin == Spin::None {
                NUDGE_TAU
            } else {
                SPIN_TAU
            };
            self.offset += self.vel * dt;
            self.vel *= (-dt / tau).exp();
        } else {
            self.vel = 0.0;
            self.spin = Spin::None;
            self.launch = 0.0;
            // Recentre — a scrub is a departure, not a new home.
            self.offset *= (1.0 - RECENTRE * dt).max(0.0);
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

    /// Turn the wheel gently: one delta, well under the flick speed.
    fn nudge(j: &mut Jog) {
        j.touch(true);
        j.cc(66);
        j.touch(false);
    }

    /// Throw it: enough deltas in a row to pass FLICK_VEL on release.
    fn flick(j: &mut Jog, val: u8) {
        j.touch(true);
        for _ in 0..10 {
            j.cc(val);
        }
        j.touch(false);
    }

    #[test]
    fn a_nudge_is_not_a_spin() {
        let mut j = Jog::default();
        nudge(&mut j);
        assert_eq!(j.spin(), Spin::None, "a gentle turn must stay a nudge");
        assert!(j.offset() > 0.0, "but it should still move time");
    }

    #[test]
    fn a_flick_becomes_a_spin_either_way() {
        let mut back = Jog::default();
        flick(&mut back, 58);
        assert_eq!(back.spin(), Spin::Back);
        let mut front = Jog::default();
        flick(&mut front, 70);
        assert_eq!(front.spin(), Spin::Front);
    }

    #[test]
    fn a_spin_travels_much_further_than_a_nudge() {
        // The gesture is how far it carries after the hand is gone,
        // which is what makes a backspin read as a backspin.
        let glide = |j: &mut Jog| {
            let before = j.offset();
            for _ in 0..60 {
                j.tick(1.0 / 60.0);
            }
            j.offset() - before
        };
        let mut n = Jog::default();
        nudge(&mut n);
        let nudged = glide(&mut n).abs();
        let mut f = Jog::default();
        flick(&mut f, 70);
        let flicked = glide(&mut f).abs();
        assert!(
            flicked > nudged * 5.0,
            "flick {flicked} should dwarf nudge {nudged}"
        );
    }

    #[test]
    fn a_backspin_runs_time_backwards() {
        let mut j = Jog::default();
        flick(&mut j, 58);
        for _ in 0..30 {
            j.tick(1.0 / 60.0);
        }
        assert!(j.offset() < 0.0, "backspin should be behind the clock");
    }

    #[test]
    fn spin_amount_falls_from_one_to_zero() {
        let mut j = Jog::default();
        flick(&mut j, 70);
        let start = j.spin_amount();
        assert!(start > 0.9, "just released, should be near full: {start}");
        for _ in 0..30 {
            j.tick(1.0 / 60.0);
        }
        let mid = j.spin_amount();
        assert!(mid < start && mid > 0.0, "should be decaying: {mid}");
    }

    #[test]
    fn every_gesture_ends_back_on_the_clock() {
        // However hard it was thrown, the picture returns by itself —
        // a scrub is a departure, not a new home.
        for val in [58u8, 70, 66] {
            let mut j = Jog::default();
            flick(&mut j, val);
            for _ in 0..900 {
                j.tick(1.0 / 60.0);
            }
            assert!(
                j.offset().abs() < 1e-3,
                "val {val} did not recentre: {}",
                j.offset()
            );
            assert_eq!(j.spin(), Spin::None);
            assert!(!j.active());
        }
    }

    #[test]
    fn catching_the_platter_kills_the_gesture() {
        let mut j = Jog::default();
        flick(&mut j, 70);
        assert_eq!(j.spin(), Spin::Front);
        j.touch(true); // hand back on the record
        assert_eq!(j.spin(), Spin::None);
        assert_eq!(j.spin_amount(), 0.0);
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
