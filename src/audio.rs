//! Audio beat detection — the debug clock mode. Port of EasyPngVJ's
//! detector, simplified: onset envelope built on the audio thread,
//! autocorrelation tempo with octave folding, pulse-train phase.
//!
//! The estimate never drives the clock directly; main slews the
//! internal clock toward it (a soft PLL), gated on confidence.
//! Determinism is knowingly forfeited while this is on.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Onset envelope rate (frames per second).
const ENV_RATE: f64 = 60.0;
/// Envelope window: 512 frames ≈ 8.5 s.
const ENV_LEN: usize = 512;
/// One-octave tempo fold window.
const BPM_MIN: f64 = 85.0;
const BPM_MAX: f64 = 170.0;

#[derive(Clone, Copy)]
pub struct BeatEstimate {
    pub bpm: f64,
    /// Instant of a detected beat; beats also fall at anchor ± n periods.
    pub anchor: Instant,
    /// Peak-to-mean ratio of the folded autocorrelation. >1.3 is usable.
    pub confidence: f64,
}

pub struct AudioBeat {
    estimate: Arc<Mutex<Option<BeatEstimate>>>,
    pub device: String,
    _stream: cpal::Stream,
}

impl AudioBeat {
    /// Open the default input device and start detecting.
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "no input device".to_string())?;
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "?".into());
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let sr = config.sample_rate() as f64;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        let estimate = Arc::new(Mutex::new(None));
        let mut an = Analyzer::new(sr, channels, estimate.clone());
        let err_fn = |_e| {};

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                stream_config,
                move |data: &[f32], _: &_| {
                    for &s in data {
                        an.push_raw(s);
                    }
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                stream_config,
                move |data: &[i16], _: &_| {
                    for &s in data {
                        an.push_raw(s as f32 / 32768.0);
                    }
                },
                err_fn,
                None,
            ),
            f => return Err(format!("unsupported sample format {f:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;

        Ok(Self {
            estimate,
            device: name,
            _stream: stream,
        })
    }

    pub fn estimate(&self) -> Option<BeatEstimate> {
        *self.estimate.lock().unwrap()
    }
}

/// Runs entirely on the audio callback thread. The EasyPngVJ lesson:
/// an envelope derived from UI-thread poll timing degrades exactly when
/// the show is busiest.
struct Analyzer {
    channels: usize,
    frame_acc: f32,
    frame_n: usize,
    hop: usize,
    lp: f32,
    lp_k: f32,
    acc_full: f64,
    acc_low: f64,
    n: usize,
    prev_full: f64,
    prev_low: f64,
    env: VecDeque<f32>,
    hops: usize,
    shared: Arc<Mutex<Option<BeatEstimate>>>,
}

impl Analyzer {
    fn new(sr: f64, channels: usize, shared: Arc<Mutex<Option<BeatEstimate>>>) -> Self {
        Self {
            channels,
            frame_acc: 0.0,
            frame_n: 0,
            hop: (sr / ENV_RATE) as usize,
            lp: 0.0,
            // One-pole low-pass at 150 Hz: kicks without the hats.
            lp_k: (std::f64::consts::TAU * 150.0 / sr) as f32,
            acc_full: 0.0,
            acc_low: 0.0,
            n: 0,
            prev_full: 1e-9,
            prev_low: 1e-9,
            env: VecDeque::with_capacity(ENV_LEN + 1),
            hops: 0,
            shared,
        }
    }

    fn push_raw(&mut self, s: f32) {
        self.frame_acc += s;
        self.frame_n += 1;
        if self.frame_n < self.channels {
            return;
        }
        let x = self.frame_acc / self.channels as f32;
        self.frame_acc = 0.0;
        self.frame_n = 0;

        self.lp += self.lp_k * (x - self.lp);
        self.acc_full += (x * x) as f64;
        self.acc_low += (self.lp * self.lp) as f64;
        self.n += 1;
        if self.n >= self.hop {
            self.flush_hop();
        }
    }

    /// Half-wave rectified rise in log energy, full band + low band.
    fn flush_hop(&mut self) {
        let ef = self.acc_full / self.n as f64 + 1e-12;
        let el = self.acc_low / self.n as f64 + 1e-12;
        self.acc_full = 0.0;
        self.acc_low = 0.0;
        self.n = 0;

        let onset =
            (ef.ln() - self.prev_full.ln()).max(0.0) + (el.ln() - self.prev_low.ln()).max(0.0);
        self.prev_full = ef;
        self.prev_low = el;

        self.env.push_back(onset as f32);
        if self.env.len() > ENV_LEN {
            self.env.pop_front();
        }
        self.hops += 1;
        // Re-estimate twice a second once the window has substance.
        if self.hops % 30 == 0 && self.env.len() >= 300 {
            self.analyze();
        }
    }

    fn analyze(&mut self) {
        let now = Instant::now();
        let n = self.env.len();
        let mean = self.env.iter().sum::<f32>() as f64 / n as f64;
        let v: Vec<f64> = self.env.iter().map(|&e| e as f64 - mean).collect();

        let ac = |lag: usize| -> f64 {
            if lag == 0 || lag >= n {
                return 0.0;
            }
            let mut s = 0.0;
            for i in lag..n {
                s += v[i] * v[i - lag];
            }
            s / (n - lag) as f64
        };

        // Candidate beat periods inside the fold window, in envelope
        // frames: lag = ENV_RATE * 60 / bpm.
        let lag_lo = (ENV_RATE * 60.0 / BPM_MAX).floor() as usize; // 21
        let lag_hi = (ENV_RATE * 60.0 / BPM_MIN).ceil() as usize; // 43
        // Fold octave aliases into the candidate BEFORE picking the
        // peak — choosing the raw winner lets noise flip octaves.
        let score = |l: usize| -> f64 { ac(l) + 0.5 * ac(2 * l) + 0.5 * ac(l.div_ceil(2)) };

        let mut best = lag_lo;
        let mut best_s = f64::MIN;
        let mut sum_s = 0.0;
        for l in lag_lo..=lag_hi {
            let s = score(l);
            sum_s += s;
            if s > best_s {
                best_s = s;
                best = l;
            }
        }
        let mean_s = sum_s / (lag_hi - lag_lo + 1) as f64;
        let confidence = if mean_s.abs() > 1e-12 {
            (best_s / mean_s.abs()).max(0.0)
        } else {
            0.0
        };

        // Parabolic refinement around the integer peak.
        let (s0, s1, s2) = (score(best - 1), best_s, score(best + 1));
        let denom = s0 - 2.0 * s1 + s2;
        let frac = if denom.abs() > 1e-12 {
            (0.5 * (s0 - s2) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let period = best as f64 + frac;
        let bpm = ENV_RATE * 60.0 / period;

        // Phase: the pulse-train offset that collects the most onset.
        let k_max = ((n - 1) as f64 / period) as usize;
        let k_use = k_max.clamp(2, 16);
        let mut best_o = 0usize;
        let mut best_os = f64::MIN;
        for o in 0..period as usize {
            let mut s = 0.0;
            for k in 0..k_use {
                let idx = n as f64 - 1.0 - o as f64 - k as f64 * period;
                if idx < 0.0 {
                    break;
                }
                s += self.env[idx as usize] as f64;
            }
            if s > best_os {
                best_os = s;
                best_o = o;
            }
        }
        let anchor = now - std::time::Duration::from_secs_f64(best_o as f64 / ENV_RATE);

        *self.shared.lock().unwrap() = Some(BeatEstimate {
            bpm,
            anchor,
            confidence,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{hash, unit_f64};

    /// Feed a synthetic click track and check the detector locks on.
    fn detect(bpm: f64) -> BeatEstimate {
        let sr = 48_000.0;
        let shared = Arc::new(Mutex::new(None));
        let mut an = Analyzer::new(sr, 1, shared.clone());

        let spb = (sr * 60.0 / bpm) as usize; // samples per beat
        let click = 1200; // click length in samples
        for i in 0..(sr as usize * 12) {
            let in_click = i % spb < click;
            let s = if in_click {
                // Deterministic noise burst.
                (unit_f64(hash(i as u64)) as f32 - 0.5) * 0.8
            } else {
                0.0
            };
            an.push_raw(s);
        }
        shared.lock().unwrap().expect("no estimate produced")
    }

    #[test]
    fn locks_on_128() {
        let est = detect(128.0);
        assert!((est.bpm - 128.0).abs() < 2.0, "got {}", est.bpm);
        assert!(est.confidence > 1.3, "confidence {}", est.confidence);
    }

    #[test]
    fn folds_170_plus_into_window() {
        // 200 BPM folds to 100 inside the 85-170 window.
        let est = detect(200.0);
        assert!((est.bpm - 100.0).abs() < 2.0, "got {}", est.bpm);
    }
}
