//! VIDEO — a file as a source, with a backspin window (#21).
//!
//! The spec that makes this cheap is the user's: backspin only ever
//! drags the picture two or three seconds into the past, so a video
//! source does not need reverse decoding or random seeks — it needs a
//! forward decoder and a **ring of the frames it just showed**. ffmpeg
//! decodes the file forward forever (`-stream_loop -1`), a reader
//! thread keeps the ring stocked a few frames ahead of the playhead,
//! and the last [`BEHIND_SECS`] seconds stay behind it. A jog or a
//! BACKSPIN pad bends visual time backwards and the frame is already
//! in memory; deeper than the window, the picture freezes on the
//! oldest frame it still holds — a graceful floor, not a glitch.
//!
//! At 480×270 RGB24 and 30 fps the window costs ~12 MB a second, ~47 MB
//! for the four-second ring. The one place the determinism contract
//! bends: a streaming decoder is stateful, so "same beat, same frame"
//! holds only within the window — which is exactly the deal the spec
//! struck.
//!
//! The decoder paces itself against the playhead (a show hold freezes
//! the playhead, which parks the decoder), so a paused set does not
//! decode the whole file into a buffer that only holds four seconds.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const VID_W: u32 = 480;
pub const VID_H: u32 = 270;
pub const FPS: f64 = 30.0;
/// The backspin window, seconds — sized past "2, 3 seconds" so the
/// deepest pad backspin still lands inside it at club tempo.
pub const BEHIND_SECS: f64 = 4.0;
const BEHIND_FRAMES: u64 = (BEHIND_SECS * FPS) as u64;
/// Prebuffer past the playhead, so playback never starves on a slow
/// decode burst. Small: everything ahead is memory the window loses.
const AHEAD_FRAMES: u64 = 12;

/// The decoded window: frames indexed from the start of the stream.
struct Ring {
    frames: std::collections::VecDeque<(u64, Arc<Vec<u8>>)>,
    /// Index the decoder writes next.
    next: u64,
}

impl Ring {
    fn new() -> Self {
        Self {
            frames: std::collections::VecDeque::new(),
            next: 0,
        }
    }

    /// The frame for `target`, clamped into what the window still
    /// holds: the newest frame at or before the target, else the
    /// oldest one there is (the backspin floor).
    fn pick(&self, target: u64) -> Option<(u64, Arc<Vec<u8>>)> {
        let mut best: Option<&(u64, Arc<Vec<u8>>)> = None;
        for f in &self.frames {
            if f.0 <= target {
                best = Some(f);
            } else {
                break;
            }
        }
        best.or_else(|| self.frames.front())
            .map(|(i, b)| (*i, b.clone()))
    }

    /// Drop everything older than the window behind `playhead`.
    fn trim(&mut self, playhead: u64) {
        let floor = playhead.saturating_sub(BEHIND_FRAMES);
        while self
            .frames
            .front()
            .is_some_and(|(i, _)| *i < floor)
        {
            self.frames.pop_front();
        }
    }
}

pub struct Video {
    ring: Arc<Mutex<Ring>>,
    /// Frame index the app last asked for — the decoder paces against
    /// it and the trim runs relative to it.
    playhead: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    child: Child,
    /// File stem, for the HUD.
    pub name: String,
}

impl Drop for Video {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Video {
    /// Spawn the decoder on a file. Loops forever; the playhead decides
    /// what is shown.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-stream_loop",
                "-1",
                "-i",
                &path.to_string_lossy(),
                "-vf",
                &format!("scale={VID_W}:{VID_H}"),
                "-r",
                "30",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        let mut out = child.stdout.take().ok_or("no stdout")?;

        let ring = Arc::new(Mutex::new(Ring::new()));
        let playhead = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        {
            let (ring, playhead, stop) = (ring.clone(), playhead.clone(), stop.clone());
            std::thread::spawn(move || {
                let len = (VID_W * VID_H * 3) as usize;
                let mut buf = vec![0u8; len];
                while !stop.load(Ordering::Relaxed) {
                    // Pace: never run further ahead of the playhead
                    // than the prebuffer. A held show parks us here.
                    loop {
                        let ahead = {
                            let r = ring.lock().unwrap();
                            r.next.saturating_sub(playhead.load(Ordering::Relaxed))
                        };
                        if ahead <= AHEAD_FRAMES || stop.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    if out.read_exact(&mut buf).is_err() {
                        break; // decoder died; the floor frame remains
                    }
                    let mut r = ring.lock().unwrap();
                    let idx = r.next;
                    r.frames.push_back((idx, Arc::new(buf.clone())));
                    r.next += 1;
                    let ph = playhead.load(Ordering::Relaxed);
                    r.trim(ph);
                }
            });
        }

        Ok(Self {
            ring,
            playhead,
            stop,
            child,
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "video".into()),
        })
    }

    /// The frame for `t` seconds of playhead time, with its index so
    /// the caller can skip re-converting a frame it already drew.
    /// Clamped into the window at both ends.
    pub fn frame_at(&self, t: f64) -> Option<(u64, Arc<Vec<u8>>)> {
        let target = if t.is_finite() && t > 0.0 {
            (t * FPS) as u64
        } else {
            0
        };
        self.playhead.store(target, Ordering::Relaxed);
        self.ring.lock().unwrap().pick(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_with(range: std::ops::Range<u64>) -> Ring {
        let mut r = Ring::new();
        for i in range {
            r.frames.push_back((i, Arc::new(vec![i as u8])));
            r.next = i + 1;
        }
        r
    }

    #[test]
    fn pick_serves_the_present_and_the_recent_past() {
        let r = ring_with(100..160);
        assert_eq!(r.pick(150).unwrap().0, 150, "exact frame");
        assert_eq!(r.pick(159).unwrap().0, 159, "newest");
        // Backspin inside the window: the exact past frame.
        assert_eq!(r.pick(120).unwrap().0, 120);
        // Deeper than the window holds: freeze on the oldest — the
        // graceful floor the spec asked for.
        assert_eq!(r.pick(40).unwrap().0, 100);
        // Ahead of what is decoded: the newest there is.
        assert_eq!(r.pick(500).unwrap().0, 159);
        assert!(Ring::new().pick(0).is_none());
    }

    #[test]
    fn trim_keeps_exactly_the_backspin_window() {
        let mut r = ring_with(0..200);
        r.trim(180);
        let oldest = r.frames.front().unwrap().0;
        assert_eq!(oldest, 180 - BEHIND_FRAMES, "window depth moved");
        // A playhead near zero must not underflow the floor.
        let mut r = ring_with(0..10);
        r.trim(3);
        assert_eq!(r.frames.front().unwrap().0, 0);
    }

    #[test]
    fn the_window_is_the_size_the_spec_paid_for() {
        // ~4 s at 30 fps; the memory budget in the module docs is
        // derived from these numbers, so pin them together.
        assert_eq!(BEHIND_FRAMES, 120);
        let bytes = (VID_W * VID_H * 3) as u64 * (BEHIND_FRAMES + AHEAD_FRAMES);
        assert!(bytes < 60_000_000, "window grew past ~60 MB: {bytes}");
    }
}
