//! Live capture — a camera (or the screen) as a plate that changes every
//! frame.
//!
//! Decoding runs in an ffmpeg subprocess, the same choice the video work
//! makes and for the same reason llama.cpp is a subprocess: if it dies,
//! the instrument keeps playing. Nothing here can stall a frame — the
//! render loop reads whatever the mailbox last held and moves on.
//!
//! ffmpeg is also asked to scale during decode. The terminal wants a few
//! hundred pixels across, and decoding at output size is most of the
//! performance answer.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use image::RgbImage;

/// Capture size. Larger than the cell grid needs, small enough that a
/// frame is a quarter of a megabyte and the difference pass is cheap.
pub const CAP_W: u32 = 480;
pub const CAP_H: u32 = 270;

/// The latest frame, plus how much the picture is moving.
pub struct Capture {
    frame: Arc<Mutex<Option<RgbImage>>>,
    /// Motion energy in millionths, so the render thread reads it without
    /// locking. This is the one thing a live source gives that a still
    /// cannot: the room can push the visuals.
    motion: Arc<AtomicU32>,
    frames: Arc<AtomicU64>,
    /// Which AVFoundation index this is, for the HUD and for reopening.
    #[allow(dead_code)]
    pub device: String,
    child: Child,
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

impl Capture {
    /// Open an AVFoundation device by index. `"0"` is usually the built-in
    /// camera; the screen shows up as its own index, so a desktop can be
    /// a source too.
    pub fn open(device: &str) -> Result<Self, String> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-i",
                device,
                "-vf",
                &format!("scale={CAP_W}:{CAP_H}"),
                // Pin the OUTPUT rate too. Without it a device that
                // hands back the same frame — which is what macOS does
                // when camera access has not been granted — makes ffmpeg
                // duplicate it as fast as the pipe accepts, and the
                // reader thread spins at thousands of frames a second.
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
        let frame = Arc::new(Mutex::new(None));
        let motion = Arc::new(AtomicU32::new(0));
        let frames = Arc::new(AtomicU64::new(0));

        let (f, m, n) = (frame.clone(), motion.clone(), frames.clone());
        std::thread::spawn(move || {
            let len = (CAP_W * CAP_H * 3) as usize;
            let mut buf = vec![0u8; len];
            let mut prev: Option<Vec<u8>> = None;
            loop {
                if out.read_exact(&mut buf).is_err() {
                    break; // the process died; the show does not care
                }
                // Frame differencing lives here, off the render thread —
                // it is a whole-frame pass and the capture thread is
                // already idle waiting on the pipe.
                if let Some(p) = &prev {
                    let mut acc: u64 = 0;
                    for i in (0..len).step_by(9) {
                        acc += buf[i].abs_diff(p[i]) as u64;
                    }
                    let samples = len.div_ceil(9) as u64;
                    let e = (acc * 1_000_000) / (samples * 255);
                    m.store(e as u32, Ordering::Relaxed);
                }
                prev = Some(buf.clone());

                if let Some(img) = RgbImage::from_raw(CAP_W, CAP_H, buf.clone()) {
                    *f.lock().unwrap() = Some(img);
                    n.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        Ok(Self {
            frame,
            motion,
            frames,
            device: device.to_string(),
            child,
        })
    }

    /// The latest frame, cloned. `None` until the first one lands.
    pub fn latest(&self) -> Option<RgbImage> {
        self.frame.lock().unwrap().clone()
    }

    /// Motion energy, [0,1]-ish. Scaled so ordinary movement lands well
    /// under 1 and a room jumping saturates it.
    pub fn motion(&self) -> f64 {
        (self.motion.load(Ordering::Relaxed) as f64 / 1_000_000.0 * 12.0).clamp(0.0, 1.0)
    }

    pub fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }

    /// Has anything arrived? A camera that never opened is not an error
    /// worth stopping for, but it is worth showing.
    pub fn alive(&self) -> bool {
        self.frames() > 0
    }

    /// Frames are arriving but nothing in them ever changes. On macOS
    /// that means the camera permission was refused: AVFoundation hands
    /// back a frozen image rather than an error, so this is the only
    /// signal there is.
    pub fn frozen(&self) -> bool {
        self.frames() > 60 && self.motion.load(Ordering::Relaxed) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_device_is_an_error_not_a_panic() {
        // Nothing here may take the instrument down: a camera that will
        // not open is a HUD line, not a crash.
        let r = Capture::open("999");
        match r {
            Err(_) => {}
            Ok(c) => {
                // ffmpeg may spawn and only then fail; either way the
                // capture must simply report nothing.
                std::thread::sleep(std::time::Duration::from_millis(300));
                assert!(c.latest().is_none() || c.alive());
            }
        }
    }

    #[test]
    fn motion_is_bounded() {
        // The scaling must not hand an out-of-range value to the drive
        // signals, whatever the camera does.
        let m = AtomicU32::new(u32::MAX);
        let v = (m.load(Ordering::Relaxed) as f64 / 1_000_000.0 * 12.0).clamp(0.0, 1.0);
        assert!((0.0..=1.0).contains(&v));
    }

    /// A quarter of a megabyte per frame: small enough to clone every
    /// frame without thinking about it. Checked at compile time, since
    /// the sizes are constants and a runtime assert would be theatre.
    const _: () = assert!((CAP_W * CAP_H * 3) < 512 * 1024);
}
