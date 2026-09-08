//! Show sequences — the opening and closing of a set, as a state machine
//! that exports parameters. The Modulator shape from `pass.rs`: it reads
//! a clock and writes numbers, and never touches a pixel.
//!
//! Over there this was the piece that made the rig feel like a show
//! rather than a screensaver. Nothing draws the intro; the intro tells
//! everything else how to behave for five seconds, and the plate loader,
//! the particle system, the camera and the post chain all read the same
//! eight numbers. That is why this is one struct with eight exports
//! instead of an "intro effect" — an intro is not a layer, it is a
//! global bias on every layer at once.
//!
//! **This clock is wall time, not beat time.** Every other module here
//! is a function of the beat, deliberately, so the picture stays locked
//! to the music. The show sequences are the exception: a five-second
//! opening is five seconds whether the track is 90 or 174 BPM, and an
//! outro that stretched because the DJ pitched down would be an outro
//! that missed the house lights. So `update` takes `dt` in seconds and
//! the sequences are written in absolute durations.
//!
//! Two behaviours worth stating up front, because both are the result of
//! something going wrong at a gig:
//!
//! - **Standby arms on sustained loudness, never on a transient.** A
//!   cable pop, a mic bump, someone's laptop chiming into the mixer —
//!   any of those is a single loud frame, and any of them would have
//!   opened the set into an empty room. Loudness has to hold for
//!   `ARM_HOLD` seconds of accumulated time before the intro fires, and
//!   the accumulator bleeds back down faster than it fills, so an
//!   intermittent rattle never sums its way over the line.
//! - **The master fade waits for the logo.** In the outro the artwork,
//!   the particles and the colour have all been taken away by t≈5 s, and
//!   what is left on screen is the mark. Fading the master any earlier
//!   would dim the mark while it is still being read; it only starts at
//!   7.2 s, once the logo bell has crested and is on its way down.
//!
//! The other rule the original enforced from here: **the silence gate
//! and automatic scene changes only run during `live`.** An intro must
//! not be interrupted by a scene change three seconds in, and the
//! silence gate must not black out a standby screen that is silent by
//! definition. Callers ask `is_live()` before doing either.
//!
//! Nothing in main.rs reads this yet — the wiring is a separate change —
//! hence the transitional dead code.
#![allow(dead_code)]

use std::f64::consts::PI;

/// Loudness above this counts towards arming. Well over a room's noise
/// floor, well under anything anyone would call "playing".
const ARM_LOUDNESS: f64 = 0.10;
/// Accumulated loud time needed to open the set, seconds.
const ARM_HOLD: f64 = 0.20;
/// The accumulator bleeds down at this rate while it is quiet — faster
/// than it fills, so a rattle cannot ratchet its way to the threshold.
const ARM_DECAY: f64 = 2.5;

/// Intro length, seconds.
const INTRO_LEN: f64 = 5.2;
/// The intro is two hits, not one: the second lands here, after the
/// plate has arrived, and reads as the set actually starting.
const INTRO_SECOND_HIT: f64 = 2.55;
const INTRO_SECOND_HIT_LEVEL: f64 = 0.55;

/// Outro length, seconds.
const OUTRO_LEN: f64 = 9.5;
/// How long the pipeline stays frozen at the top of the outro — about a
/// beat at club tempo. The frame stopping dead is the punctuation.
const OUTRO_HOLD: f64 = 0.85;

/// White hit decay, units per second. Linear, from the original.
const FLASH_DECAY: f64 = 4.0;

/// Where the show is. Public so the HUD can name it and so callers can
/// branch on more than `is_live()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Idle, waiting for sound. A breathing mark over a whisper of dust.
    Standby,
    /// The opening sequence.
    Intro,
    /// The set. Every export neutral; this is the only stage in which
    /// the rig is allowed to change scenes or gate on silence.
    Live,
    /// The closing sequence.
    Outro,
    /// Finished. Black, and it stays black.
    Ended,
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Standby => "STANDBY",
            Stage::Intro => "INTRO",
            Stage::Live => "LIVE",
            Stage::Outro => "OUTRO",
            Stage::Ended => "ENDED",
        }
    }
}

/// The eight numbers the rest of the rig reads. One struct, produced
/// fresh every frame, so no subsystem can end up a frame out of step
/// with another during a sequence.
///
/// Documented ranges — the tests hold these:
///
/// | export | range | neutral |
/// |---|---|---|
/// | `fade` | 0..=1 | 1 |
/// | `plate` | 0..=1 | 1 |
/// | `particles` | 0..=1.5 | 1 |
/// | `zoom` | 0..=2.5 | 1 |
/// | `mono` | 0..=1 | 0 |
/// | `logo` | 0..=0.95 | 0 |
/// | `flash` | 0..=1 | 0 |
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShowState {
    /// Master brightness. Everything is multiplied by this last.
    pub fade: f64,
    /// Artwork opacity — how much of the plate is present.
    pub plate: f64,
    /// Particle gain, a multiplier on the normal count/brightness. Goes
    /// above 1 for the outro burst.
    pub particles: f64,
    /// Extra camera push, multiplied onto whatever the camera modulator
    /// is already doing. 1 = leave the framing alone.
    pub zoom: f64,
    /// Grayscale amount: 0 full colour, 1 fully desaturated.
    pub mono: f64,
    /// Opacity of the centred mark.
    pub logo: f64,
    /// White hit over the whole frame, decaying.
    pub flash: f64,
    /// Freeze the pipeline — hold the last frame, advance nothing.
    pub hold: bool,
}

impl ShowState {
    /// Every export neutral: what `live` emits, and what a caller should
    /// assume if it is running without a show at all.
    pub fn neutral() -> Self {
        Self {
            fade: 1.0,
            plate: 1.0,
            particles: 1.0,
            zoom: 1.0,
            mono: 0.0,
            logo: 0.0,
            flash: 0.0,
            hold: false,
        }
    }

    /// Every export at zero. Black, nothing drawn, no framing to keep.
    fn black() -> Self {
        Self {
            fade: 0.0,
            plate: 0.0,
            particles: 0.0,
            zoom: 0.0,
            mono: 0.0,
            logo: 0.0,
            flash: 0.0,
            hold: false,
        }
    }
}

impl Default for ShowState {
    fn default() -> Self {
        Self::neutral()
    }
}

/// The show sequencer. Unlike the camera modulators this one is
/// stateful: a sequence has a beginning, and "which stage, how far in"
/// cannot be recovered from the visual clock. It is still deterministic
/// — no wall-time reads, no RNG, everything a pure function of the
/// accumulated `dt` and the stage.
pub struct Show {
    stage: Stage,
    /// Seconds elapsed *in the current stage*, wall time.
    t: f64,
    /// Accumulated loud time in standby, seconds.
    armed: f64,
    /// Current white-hit level, carried across stages so a hit fired at
    /// a transition decays through the frames after it.
    flash: f64,
}

impl Default for Show {
    fn default() -> Self {
        Self::new()
    }
}

impl Show {
    /// A show that has not started. Begins in `standby`.
    pub fn new() -> Self {
        Self {
            stage: Stage::Standby,
            t: 0.0,
            armed: 0.0,
            flash: 0.0,
        }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// True only during the set proper. Callers must consult this before
    /// running the silence gate or an automatic scene change — both are
    /// suspended for the duration of a sequence.
    pub fn is_live(&self) -> bool {
        self.stage == Stage::Live
    }

    /// Back to standby, waiting for sound. Also the way to reset a show
    /// that has ended so the rig can open a second set.
    pub fn arm(&mut self) {
        self.enter(Stage::Standby);
        self.armed = 0.0;
    }

    /// Play the set out. Valid from anywhere — if the DJ hits it during
    /// the intro, the closing sequence takes over from where the picture
    /// currently is.
    pub fn play_out(&mut self) {
        if self.stage != Stage::Outro && self.stage != Stage::Ended {
            self.enter(Stage::Outro);
        }
    }

    /// Advance by `dt` **seconds of wall time** and produce this frame's
    /// exports. `loudness` is whatever the caller measures on the input,
    /// roughly 0..1; pass 0.0 when there is no audio, in which case
    /// standby waits forever and the set is opened by hand.
    pub fn update(&mut self, dt: f64, loudness: f64) -> ShowState {
        let dt = dt.max(0.0);
        self.flash = (self.flash - FLASH_DECAY * dt).max(0.0);
        self.t += dt;

        match self.stage {
            Stage::Standby => {
                if loudness > ARM_LOUDNESS {
                    self.armed += dt;
                } else {
                    self.armed = (self.armed - ARM_DECAY * dt).max(0.0);
                }
                if self.armed > ARM_HOLD {
                    self.enter(Stage::Intro);
                }
            }
            Stage::Intro => {
                // Edge-detected so it fires once however long the frame.
                if self.t - dt < INTRO_SECOND_HIT && self.t >= INTRO_SECOND_HIT {
                    self.flash = self.flash.max(INTRO_SECOND_HIT_LEVEL);
                }
                if self.t >= INTRO_LEN {
                    self.enter(Stage::Live);
                }
            }
            Stage::Live => {}
            Stage::Outro => {
                if self.t >= OUTRO_LEN {
                    self.enter(Stage::Ended);
                }
            }
            Stage::Ended => {}
        }

        self.export()
    }

    /// Switch stage, restart the stage clock, and fire whatever the
    /// entry hit is. Keeping the hits here rather than in `export` is
    /// what makes them one-shot instead of a value held all frame.
    fn enter(&mut self, stage: Stage) {
        self.stage = stage;
        self.t = 0.0;
        match stage {
            Stage::Intro => self.flash = 1.0,
            Stage::Outro => self.flash = self.flash.max(0.45),
            _ => {}
        }
    }

    fn export(&self) -> ShowState {
        let t = self.t;
        match self.stage {
            // A breathing mark over a whisper of dust: enough motion to
            // prove the rig is alive, little enough that it reads as
            // "not started" from the back of the room.
            Stage::Standby => ShowState {
                fade: 1.0,
                plate: 0.0,
                particles: 0.10,
                zoom: 1.6,
                mono: 0.7,
                logo: 0.20 + 0.07 * (t * 1.3).sin(),
                flash: self.flash,
                hold: false,
            },
            // Hit, then the plate arrives out of a 2.5x push that pulls
            // back to normal framing while the colour returns and the
            // mark crosses the middle of the sequence and leaves.
            Stage::Intro => ShowState {
                fade: 1.0,
                plate: ease((t - 0.55) / 1.9),
                particles: clamp01((t - 0.9) / 2.1),
                zoom: 1.0 + 1.5 * (1.0 - ease(t / 3.4)),
                mono: clamp01(1.0 - t / 2.4),
                logo: bell(t / 2.2) * 0.85,
                flash: self.flash,
                hold: false,
            },
            Stage::Live => ShowState {
                flash: self.flash,
                ..ShowState::neutral()
            },
            // Freeze, burst, then take it all away in order: artwork,
            // then colour, then the mark, then the light.
            Stage::Outro => ShowState {
                fade: 1.0 - clamp01((t - 7.2) / 2.0),
                plate: 1.0 - clamp01((t - 1.2) / 3.4),
                particles: if t < 2.0 {
                    1.0 + 0.5 * bell(t / 2.0)
                } else {
                    clamp01(1.0 - (t - 2.0) / 2.8)
                },
                zoom: 1.0 + 0.35 * clamp01((t - 0.8) / 5.0),
                mono: clamp01((t - 1.6) / 3.0),
                logo: bell((t - 4.4) / 3.0) * 0.95,
                flash: self.flash,
                hold: t < OUTRO_HOLD,
            },
            Stage::Ended => ShowState::black(),
        }
    }
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Cubic ease-out — fast off the mark, settling into the hold. The
/// original's curve for anything that arrives.
fn ease(x: f64) -> f64 {
    let x = clamp01(x);
    1.0 - (1.0 - x).powi(3)
}

/// Half a sine: up and back down over 0..1, zero outside. Everything
/// that appears and then leaves — the logo, the particle burst — is
/// shaped by this.
fn bell(x: f64) -> f64 {
    (PI * clamp01(x)).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 240.0;

    /// Run the standby arm to completion and stop on the first frame of
    /// the intro (stage clock at 0).
    fn into_intro() -> Show {
        let mut s = Show::new();
        s.arm();
        for _ in 0..1000 {
            s.update(DT, 1.0);
            if s.stage() == Stage::Intro {
                return s;
            }
        }
        panic!("never armed");
    }

    fn into_live() -> Show {
        let mut s = into_intro();
        while s.stage() == Stage::Intro {
            s.update(DT, 1.0);
        }
        assert_eq!(s.stage(), Stage::Live);
        s
    }

    fn into_outro() -> Show {
        let mut s = into_live();
        s.play_out();
        s
    }

    #[test]
    fn a_transient_does_not_open_the_set() {
        let mut s = Show::new();
        s.arm();
        // A cable pop: a tenth of a second of noise, then the room.
        for _ in 0..24 {
            s.update(DT, 1.0);
        }
        assert_eq!(s.stage(), Stage::Standby, "opened on a transient");
        for _ in 0..480 {
            s.update(DT, 0.0);
            assert_eq!(s.stage(), Stage::Standby);
        }
        assert!(!s.is_live());
    }

    #[test]
    fn a_rattle_cannot_ratchet_its_way_in() {
        // Alternating loud/quiet frames: the accumulator must lose more
        // than it gains, so this never reaches the threshold.
        let mut s = Show::new();
        s.arm();
        for i in 0..2000 {
            s.update(DT, if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        assert_eq!(s.stage(), Stage::Standby);
    }

    #[test]
    fn sustained_loudness_opens_the_set() {
        let mut s = Show::new();
        s.arm();
        let mut t = 0.0;
        while s.stage() == Stage::Standby && t < 2.0 {
            s.update(DT, 0.6);
            t += DT;
        }
        assert_eq!(s.stage(), Stage::Intro, "sustained sound should arm");
        assert!(
            (t - ARM_HOLD).abs() < 0.02,
            "armed at {t}s, expected ~{ARM_HOLD}s"
        );
        // And the opening hit went out with it.
        assert!(s.export().flash > 0.9);
    }

    #[test]
    fn intro_hands_over_after_5_2_seconds() {
        let mut s = into_intro();
        let mut t = 0.0;
        while s.stage() == Stage::Intro {
            s.update(DT, 0.0);
            t += DT;
            assert!(t < 6.0, "intro never ended");
        }
        assert_eq!(s.stage(), Stage::Live);
        assert!(
            (t - INTRO_LEN).abs() < DT * 1.5,
            "intro ran {t}s, want {INTRO_LEN}s"
        );
        // And exactly at the boundary, in one step.
        let mut one = into_intro();
        one.update(INTRO_LEN - 0.001, 0.0);
        assert_eq!(one.stage(), Stage::Intro);
        one.update(0.002, 0.0);
        assert_eq!(one.stage(), Stage::Live);
    }

    #[test]
    fn the_second_intro_hit_fires_at_2_55() {
        let mut s = into_intro();
        while s.t + DT < INTRO_SECOND_HIT {
            s.update(DT, 0.0);
        }
        // The opening hit is long gone by here.
        assert!(s.export().flash < 0.01, "flash {}", s.export().flash);
        let st = s.update(DT, 0.0);
        assert!(
            st.flash >= INTRO_SECOND_HIT_LEVEL - 1e-9,
            "second hit missing: {}",
            st.flash
        );
        // One hit, not a held level: it decays away again.
        for _ in 0..60 {
            s.update(DT, 0.0);
        }
        assert!(s.export().flash < 0.01);
    }

    #[test]
    fn outro_freezes_the_frame_for_a_beat() {
        let mut s = into_outro();
        while s.t + DT < OUTRO_HOLD {
            let at = s.t;
            assert!(s.update(DT, 0.0).hold, "should still be frozen at {at}s");
        }
        while s.t < OUTRO_HOLD + 0.1 {
            s.update(DT, 0.0);
        }
        assert!(!s.export().hold, "should have released the freeze");
    }

    #[test]
    fn the_master_fade_waits_for_the_logo() {
        let mut s = into_outro();
        let mut t = 0.0;
        let mut peak_logo: f64 = 0.0;
        let mut logo_at_fade_start = 0.0;
        loop {
            let st = s.update(DT, 0.0);
            let at = s.t; // the time this frame was rendered at
            t += DT;
            if s.stage() != Stage::Outro {
                break;
            }
            peak_logo = peak_logo.max(st.logo);
            if at < 7.2 {
                assert!(
                    (st.fade - 1.0).abs() < 1e-12,
                    "master fade started early, at {at}s: {}",
                    st.fade
                );
            } else if logo_at_fade_start == 0.0 {
                logo_at_fade_start = peak_logo;
            }
        }
        // The mark crested before the light started going, which is the
        // whole reason the fade is held back this long.
        assert!(peak_logo > 0.9, "logo never crested: {peak_logo}");
        assert!(
            logo_at_fade_start > 0.9,
            "fade started before the mark was seen: {logo_at_fade_start}"
        );
        assert!((t - OUTRO_LEN).abs() < 0.05, "outro ran {t}s");
        assert_eq!(s.stage(), Stage::Ended);
        assert!(s.export().fade == 0.0);
    }

    #[test]
    fn ended_holds_on_black() {
        let mut s = into_outro();
        while s.stage() == Stage::Outro {
            s.update(DT, 0.0);
        }
        for _ in 0..600 {
            let st = s.update(DT, 1.0);
            assert_eq!(st, ShowState::black(), "ended should stay black");
        }
        assert_eq!(s.stage(), Stage::Ended);
    }

    #[test]
    fn is_live_only_during_the_set() {
        let mut s = Show::new();
        s.arm();
        assert!(!s.is_live(), "standby is not live");
        let mut s = into_intro();
        assert!(!s.is_live(), "intro is not live");
        while s.stage() == Stage::Intro {
            s.update(DT, 0.0);
        }
        assert!(s.is_live());
        s.play_out();
        assert!(!s.is_live(), "outro is not live");
        while s.stage() == Stage::Outro {
            s.update(DT, 0.0);
        }
        assert!(!s.is_live(), "ended is not live");
    }

    #[test]
    fn every_export_stays_in_range_over_a_whole_show() {
        let check = |st: ShowState, where_: &str| {
            let in_range = |v: f64, lo: f64, hi: f64, name: &str| {
                assert!(
                    v >= lo - 1e-9 && v <= hi + 1e-9,
                    "{name} out of range in {where_}: {v}"
                );
            };
            in_range(st.fade, 0.0, 1.0, "fade");
            in_range(st.plate, 0.0, 1.0, "plate");
            in_range(st.particles, 0.0, 1.5, "particles");
            in_range(st.zoom, 0.0, 2.5, "zoom");
            in_range(st.mono, 0.0, 1.0, "mono");
            in_range(st.logo, 0.0, 0.95, "logo");
            in_range(st.flash, 0.0, 1.0, "flash");
        };

        let mut s = Show::new();
        s.arm();
        // standby, long enough for the breathing mark to go round
        for _ in 0..2400 {
            check(s.update(DT, 0.0), "standby");
        }
        // arm, intro, and on into live
        while s.stage() != Stage::Live {
            check(s.update(DT, 1.0), s.stage().name());
        }
        for _ in 0..1200 {
            check(s.update(DT, 1.0), "live");
        }
        s.play_out();
        while s.stage() == Stage::Outro {
            check(s.update(DT, 0.0), "outro");
        }
        for _ in 0..240 {
            check(s.update(DT, 0.0), "ended");
        }
    }

    #[test]
    fn live_is_neutral() {
        let mut s = into_live();
        let st = s.update(DT, 1.0);
        assert_eq!(st, ShowState::neutral());
        assert!(!st.hold);
    }

    #[test]
    fn the_intro_pulls_back_from_a_push() {
        let mut s = into_intro();
        let start = s.export();
        assert!((start.zoom - 2.5).abs() < 1e-9, "zoom {}", start.zoom);
        assert_eq!(start.plate, 0.0, "plate arrives late, not at t=0");
        for _ in 0..(3.4 * 240.0) as usize {
            s.update(DT, 0.0);
        }
        let settled = s.export();
        assert!(
            settled.zoom < 1.02,
            "should have pulled back: {}",
            settled.zoom
        );
        assert!(
            settled.plate > 0.99,
            "plate should be up: {}",
            settled.plate
        );
        assert!(settled.mono < 1e-9, "colour should be back");
    }

    #[test]
    fn the_outro_bursts_then_collapses() {
        let mut s = into_outro();
        let mut peak: f64 = 0.0;
        for _ in 0..480 {
            peak = peak.max(s.update(DT, 0.0).particles);
        }
        assert!(peak > 1.4, "no burst: {peak}");
        while s.stage() == Stage::Outro {
            s.update(DT, 0.0);
        }
        // Everything is gone before the stage ends, not cut off by it.
        let mut s = into_outro();
        for _ in 0..(6.0 * 240.0) as usize {
            s.update(DT, 0.0);
        }
        let late = s.export();
        assert!(late.particles < 0.01 && late.plate < 0.01);
        assert!(late.mono > 0.99, "should be grey by now: {}", late.mono);
    }

    #[test]
    fn deterministic_for_the_same_dt_sequence() {
        let run = || {
            let mut s = Show::new();
            s.arm();
            let mut out = Vec::new();
            for i in 0..3000 {
                out.push(s.update(DT, if i > 100 { 1.0 } else { 0.0 }));
            }
            s.play_out();
            for _ in 0..2400 {
                out.push(s.update(DT, 0.0));
            }
            out
        };
        assert_eq!(run(), run());
    }
}
