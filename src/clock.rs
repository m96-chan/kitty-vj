//! ClockSource — everything sits on top of this.
//!
//! The trait signature matches Ableton Link's session state so `LinkClock`
//! is one more impl, not a redesign.

/// Beat-time provider. `beat()` is cumulative; the fractional part is phase.
pub trait ClockSource {
    fn beat(&self) -> f64;
    fn tempo(&self) -> f64;
    fn phase(&self, quantum: f64) -> f64;
}

/// Internal clock: `beat = elapsed * tempo / 60.0`, advanced by fixed
/// timestep only. Fixed timestep + fixed seed must reproduce a frame
/// exactly — time may run backwards while the clock is internal.
pub struct InternalClock {
    beats: f64,
    tempo: f64,
}

pub const TEMPO_MIN: f64 = 20.0;
pub const TEMPO_MAX: f64 = 300.0;

impl InternalClock {
    pub fn new(tempo: f64) -> Self {
        Self { beats: 0.0, tempo }
    }

    /// Advance by one fixed timestep (seconds). Negative dt runs time backwards.
    pub fn advance(&mut self, dt: f64) {
        self.beats += dt * self.tempo / 60.0;
    }

    pub fn set_tempo(&mut self, tempo: f64) {
        self.tempo = tempo.clamp(TEMPO_MIN, TEMPO_MAX);
    }

    pub fn nudge_tempo(&mut self, delta: f64) {
        self.set_tempo(self.tempo + delta);
    }

    /// Snap phase to the nearest beat boundary (resync feel on tap).
    pub fn resync(&mut self) {
        self.beats = self.beats.round();
    }
}

impl ClockSource for InternalClock {
    fn beat(&self) -> f64 {
        self.beats
    }

    fn tempo(&self) -> f64 {
        self.tempo
    }

    fn phase(&self, quantum: f64) -> f64 {
        self.beats.rem_euclid(quantum)
    }
}

/// Tap-tempo detector: median of recent inter-tap intervals.
pub struct TapTempo {
    taps: Vec<std::time::Instant>,
}

const TAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const TAP_WINDOW: usize = 8;

impl TapTempo {
    pub fn new() -> Self {
        Self { taps: Vec::new() }
    }

    /// Register a tap; returns a new tempo once two or more taps are in.
    pub fn tap(&mut self, now: std::time::Instant) -> Option<f64> {
        if let Some(&last) = self.taps.last()
            && now.duration_since(last) > TAP_TIMEOUT
        {
            self.taps.clear();
        }
        self.taps.push(now);
        if self.taps.len() > TAP_WINDOW {
            self.taps.remove(0);
        }
        if self.taps.len() < 2 {
            return None;
        }
        let mut intervals: Vec<f64> = self
            .taps
            .windows(2)
            .map(|w| w[1].duration_since(w[0]).as_secs_f64())
            .collect();
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = intervals[intervals.len() / 2];
        Some((60.0 / median).clamp(TEMPO_MIN, TEMPO_MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_advances_by_tempo() {
        let mut c = InternalClock::new(120.0);
        for _ in 0..120 {
            c.advance(1.0 / 60.0); // 2 seconds
        }
        assert!((c.beat() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_timestep_is_deterministic() {
        let run = || {
            let mut c = InternalClock::new(133.0);
            for _ in 0..10_000 {
                c.advance(1.0 / 120.0);
            }
            c.beat()
        };
        assert_eq!(run().to_bits(), run().to_bits());
    }

    #[test]
    fn time_can_run_backwards() {
        let mut c = InternalClock::new(120.0);
        c.advance(1.0);
        c.advance(-1.0);
        assert!(c.beat().abs() < 1e-9);
    }

    #[test]
    fn phase_wraps_on_quantum() {
        let mut c = InternalClock::new(60.0);
        c.advance(5.5); // 5.5 beats
        assert!((c.phase(4.0) - 1.5).abs() < 1e-9);
        assert!((c.phase(1.0) - 0.5).abs() < 1e-9);
    }
}
