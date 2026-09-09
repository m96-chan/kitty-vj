//! Drive signals — the inputs EasyPngVJ's effects are functions of.
//!
//! Porting an effect without these produces a different effect: over
//! there `zoompunch` is `1 + 0.085·gbeat + 0.135·thump + 0.030·hit`, not
//! "something that pulses". So the grid pulses, the kick measure and the
//! onset flinch live here, derived from the clock when running dry and
//! from the audio analyser when A-mode is on.

/// Decay constants, seconds. Straight from the original.
const TAU_BEAT: f64 = 0.070;
const TAU_BAR: f64 = 0.130;
const TAU_PHRASE: f64 = 0.060;
const TAU_HIT: f64 = 0.048;

#[derive(Clone, Copy, Default)]
pub struct Drive {
    /// Grid pulses: 1 on the beat / bar / phrase, decaying exponentially.
    pub beat: f64,
    pub bar: f64,
    pub phrase: f64,
    /// Onset flinch — fires between the beats too, and deliberately is
    /// NOT groove-gated, so a breakdown swell still shimmers.
    pub hit: f64,
    /// Kick-ness: the low band's *rise* measured against its own running
    /// mean, not its level. A sustained sub reads as 0.
    pub thump: f64,
    /// Beat presence, [0,1]. Scales every grid-driven effect at once, so
    /// a breakdown calms the whole rig instead of hammering through it.
    pub groove: f64,
    /// Last integer beat/bar/phrase seen, for edge detection.
    last_beat: i64,
}

impl Drive {
    /// Advance by real time, given the current beat position. `groove`
    /// and `thump`/`hit` come from audio when there is any; without it
    /// the clock alone drives the grid and groove sits at 1.
    ///
    /// `motion` is a live camera's frame difference, when one is routed
    /// in. It lifts the kick measure and fires the flinch, so an effect
    /// written against these signals reacts to the room without knowing
    /// a camera exists at all.
    pub fn update(
        &mut self,
        dt: f64,
        beat: f64,
        audio: Option<AudioDrive>,
        motion: Option<MotionDrive>,
    ) {
        // Grid edges.
        let b = beat.floor() as i64;
        if b > self.last_beat {
            let crossed = b - self.last_beat;
            self.beat = 1.0;
            // A bar/phrase edge may be crossed in the same step.
            for k in (self.last_beat + 1)..=b {
                if k.rem_euclid(4) == 0 {
                    self.bar = 1.0;
                }
                if k.rem_euclid(16) == 0 {
                    self.phrase = 1.0;
                }
            }
            let _ = crossed;
            self.last_beat = b;
        } else if b < self.last_beat {
            // Time ran backwards (backspin, scrub): follow it without
            // firing pulses, and resync the edge detector.
            self.last_beat = b;
        }

        self.beat *= (-dt / TAU_BEAT).exp();
        self.bar *= (-dt / TAU_BAR).exp();
        self.phrase *= (-dt / TAU_PHRASE).exp();
        self.hit *= (-dt / TAU_HIT).exp();

        match audio {
            Some(a) => {
                self.thump = a.thump;
                self.groove = a.groove;
                if a.hit {
                    self.hit = 1.0;
                }
            }
            None => {
                // Dry: fake the kick off the beat pulse so effects that
                // read thump still breathe, and never gate the grid.
                self.thump = self.beat * 0.8;
                self.groove = 1.0;
            }
        }

        if let Some(m) = motion {
            let e = m.energy.clamp(0.0, 1.0);
            // Movement adds to the kick rather than replacing it, so a
            // still room still follows the music.
            self.thump = (self.thump + e * 0.8).min(1.0);
            // A sudden surge is a flinch — the same channel an onset
            // uses, and for the same reason: it fires off the grid.
            if e > 0.55 {
                self.hit = self.hit.max(e);
            }
        }
    }

    /// Grid pulses scaled by beat presence — what effects should read.
    pub fn gbeat(&self) -> f64 {
        self.beat * self.gv()
    }
    pub fn gbar(&self) -> f64 {
        self.bar * self.gv()
    }
    pub fn gphrase(&self) -> f64 {
        self.phrase * self.gv()
    }

    /// Groove mapped so a dead breakdown still leaves a floor of motion.
    pub fn gv(&self) -> f64 {
        0.22 + 0.78 * self.groove.clamp(0.0, 1.0)
    }
}

/// Motion from a live camera, when it is routed in. Unlike everything
/// else here this comes from the room rather than the record, which is
/// the point: it lets the crowd push the visuals.
#[derive(Clone, Copy, Default)]
pub struct MotionDrive {
    pub energy: f64,
}

/// What the audio analyser contributes, when A-mode is running.
#[derive(Clone, Copy)]
pub struct AudioDrive {
    pub thump: f64,
    pub groove: f64,
    /// An onset crossed the adaptive threshold this frame.
    pub hit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_pulse_fires_and_decays() {
        let mut d = Drive::default();
        d.update(0.01, 0.99, None, None);
        assert!(d.beat < 0.5, "no pulse before the beat");
        d.update(0.01, 1.00, None, None);
        assert!(d.beat > 0.85, "pulse should fire on the beat: {}", d.beat);
        // One tau later it should be down to ~1/e.
        d.update(TAU_BEAT, 1.10, None, None);
        assert!(d.beat > 0.3 && d.beat < 0.45, "decay off: {}", d.beat);
    }

    #[test]
    fn bar_and_phrase_fire_on_their_multiples() {
        let mut d = Drive::default();
        d.update(0.01, 4.0, None, None); // beat 4 = bar line
        assert!(d.bar > 0.85, "bar {}", d.bar);
        assert!(d.phrase < 0.1, "phrase should not fire on beat 4");
        let mut e = Drive::default();
        e.update(0.01, 16.0, None, None);
        // Phrase has the fastest tau (0.06 s), so it is already off 1.0.
        assert!(e.phrase > 0.8, "phrase {}", e.phrase);
    }

    #[test]
    fn groove_scales_the_grid() {
        let mut d = Drive::default();
        // Beat presence gone: the grid pulse still exists but is damped.
        d.update(
            0.01,
            1.0,
            Some(AudioDrive {
                thump: 0.0,
                groove: 0.0,
                hit: false,
            }),
            None,
        );
        assert!(d.beat > 0.85);
        assert!(d.gbeat() < 0.3, "gbeat should be damped: {}", d.gbeat());
        assert!((d.gv() - 0.22).abs() < 1e-9);
    }

    #[test]
    fn hit_is_not_groove_gated() {
        let mut d = Drive::default();
        d.update(
            0.01,
            0.5,
            Some(AudioDrive {
                thump: 0.0,
                groove: 0.0,
                hit: true,
            }),
            None,
        );
        // The flinch fires between beats and keeps full amplitude.
        assert!(d.hit > 0.9, "hit {}", d.hit);
    }

    #[test]
    fn motion_lifts_the_kick_without_replacing_it() {
        // A still room must still follow the music, so motion adds.
        let mut quiet = Drive::default();
        quiet.update(0.01, 1.0, None, None);
        let mut moving = Drive::default();
        moving.update(0.01, 1.0, None, Some(MotionDrive { energy: 0.5 }));
        assert!(
            moving.thump > quiet.thump,
            "motion should lift thump: {} vs {}",
            moving.thump,
            quiet.thump
        );
        assert!(moving.thump <= 1.0);
    }

    #[test]
    fn a_surge_of_movement_fires_the_flinch() {
        // The room jumping should reach the same channel an onset does,
        // and off the grid, since that is when a crowd actually moves.
        let mut d = Drive::default();
        d.update(0.01, 0.5, None, Some(MotionDrive { energy: 0.9 }));
        assert!(d.hit > 0.8, "hit {}", d.hit);
        let mut calm = Drive::default();
        calm.update(0.01, 0.5, None, Some(MotionDrive { energy: 0.2 }));
        assert!(calm.hit < 0.1, "gentle movement is not a flinch");
    }

    #[test]
    fn motion_energy_is_clamped() {
        let mut d = Drive::default();
        d.update(0.01, 1.0, None, Some(MotionDrive { energy: 99.0 }));
        assert!(d.thump <= 1.0 && d.hit <= 1.0);
    }

    #[test]
    fn reverse_time_does_not_fire_pulses() {
        let mut d = Drive::default();
        d.update(0.01, 8.0, None, None);
        d.beat = 0.0;
        d.bar = 0.0;
        // Backspin drags us back past bar lines — no machine-gun pulses.
        d.update(0.01, 5.0, None, None);
        assert!(d.beat < 0.01 && d.bar < 0.01);
    }
}
