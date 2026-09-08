//! kitty graphics protocol — the pixel transport. Full RGB frames go to
//! the terminal as images, not cells. This is the reason the project is
//! kitty-only.
//!
//! Two transmission media:
//! - **direct** (`t=d`): the pixels ride the escape stream as base64.
//!   Simple and universal, but it's the bandwidth bottleneck the README
//!   warns about — fine to prove the pipe, not for 60fps full-screen.
//! - **shared memory** (`t=s`): pixels go through a POSIX shm object,
//!   only its name rides the escape stream. The performant path. See
//!   `transmit_shm` (unix only).
//!
//! Frame animation: one image id, deleted and re-shown each frame inside
//! a synchronized-output pair so the swap is atomic (no tearing).
//!
//! Proven end to end by `examples/gfx_demo`; wiring into the mixer as a
//! channel effect is the next step (#3), hence the transitional dead code.
#![allow(dead_code)]

use std::io::{self, Write};

/// RGB pixel framebuffer, row-major, 3 bytes per pixel.
pub struct Framebuffer {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

impl Framebuffer {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            px: vec![0; (w * h * 3) as usize],
        }
    }

    /// Resize if the target changed; contents become undefined.
    pub fn resize(&mut self, w: u32, h: u32) {
        if w != self.w || h != self.h {
            self.w = w;
            self.h = h;
            self.px.resize((w * h * 3) as usize, 0);
        }
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8) {
        let i = ((y * self.w + x) * 3) as usize;
        self.px[i] = r;
        self.px[i + 1] = g;
        self.px[i + 2] = b;
    }
}

/// Are we talking to kitty? The protocol is kitty-only, so gate on it.
pub fn supported() -> bool {
    std::env::var_os("KITTY_WINDOW_ID").is_some()
        || std::env::var("TERM").is_ok_and(|t| t.contains("kitty"))
}

const CHUNK: usize = 4096; // base64 bytes per escape, per the spec

/// Transmit + display a frame over the direct (escape) medium, replacing
/// the previous frame with the same image id. Wrapped in synchronized
/// output so the swap is atomic. Full-screen at the cursor.
pub fn transmit_direct(out: &mut impl Write, fb: &Framebuffer, id: u32) -> io::Result<()> {
    transmit_placed(out, fb, id, 0, 0, 0)
}

/// As `transmit_direct`, but scaled into a `cols`x`rows` cell box at the
/// current cursor position (0 = the image's native cell size), at
/// z-index `z`. A negative z puts the image **below the text layer**, so
/// cells keeping their default background show it through — that's how
/// the pixel tier composites under the ASCII/halfblock cell tier in one
/// pass, no readback.
pub fn transmit_placed(
    out: &mut impl Write,
    fb: &Framebuffer,
    id: u32,
    cols: u16,
    rows: u16,
    z: i32,
) -> io::Result<()> {
    let b64 = base64(&fb.px);
    // Home the cursor so the image lands top-left, inside a synchronized
    // update so the frame swaps atomically.
    out.write_all(b"\x1b[?2026h\x1b[H")?;

    // Reuse one image id and one placement id (p=1): transmitting new
    // data to the same id replaces the image, and displaying to the same
    // placement id replaces the placement in place. No per-frame delete
    // — deleting then re-adding leaves a visible gap (the image doesn't
    // stay put); replacing keeps it stable.
    let mut placement = String::new();
    if cols > 0 && rows > 0 {
        placement.push_str(&format!(",c={cols},r={rows}"));
    }
    if z != 0 {
        placement.push_str(&format!(",z={z}"));
    }

    let bytes = b64.as_bytes();
    let mut off = 0;
    let mut first = true;
    while off < bytes.len() {
        let end = (off + CHUNK).min(bytes.len());
        let more = if end < bytes.len() { 1 } else { 0 };
        if first {
            write!(
                out,
                "\x1b_Ga=T,f=24,s={},v={},i={},p=1,q=2{placement},m={};",
                fb.w, fb.h, id, more
            )?;
            first = false;
        } else {
            write!(out, "\x1b_Gm={more};")?;
        }
        out.write_all(&bytes[off..end])?;
        out.write_all(b"\x1b\\")?;
        off = end;
    }
    out.write_all(b"\x1b[?2026l")?;
    out.flush()
}

/// Standard base64 (no line wrapping), for the escape payload.
pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    s
}

/// Remove any images we placed, on shutdown.
pub fn clear_all(out: &mut impl Write) -> io::Result<()> {
    out.write_all(b"\x1b_Ga=d,q=2\x1b\\")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn frame_wire_format() {
        let mut fb = Framebuffer::new(2, 1);
        fb.set(0, 0, 255, 0, 0);
        fb.set(1, 0, 0, 255, 0);
        let mut out = Vec::new();
        transmit_direct(&mut out, &fb, 1).unwrap();
        let s = String::from_utf8(out).unwrap();
        // synchronized update, home, transmit-and-display with fixed
        // image + placement ids so frames replace in place (no delete)
        assert!(s.starts_with("\x1b[?2026h\x1b[H"));
        assert!(!s.contains("a=d"), "must not delete per frame");
        assert!(s.contains("a=T,f=24,s=2,v=1,i=1,p=1"));
        assert!(s.contains("m=0;")); // single chunk, final
        assert!(s.ends_with("\x1b[?2026l"));
        // payload is the 6 RGB bytes, base64'd
        assert!(s.contains(&base64(&fb.px)));
    }

    #[test]
    fn large_frame_chunks() {
        // 64x64 RGB = 12288 bytes -> 16384 base64 chars -> 5 chunks.
        let fb = Framebuffer::new(64, 64);
        let mut out = Vec::new();
        transmit_direct(&mut out, &fb, 7).unwrap();
        let s = String::from_utf8(out).unwrap();
        let more = s.matches("m=1;").count();
        let last = s.matches("m=0;").count();
        assert_eq!(last, 1, "exactly one final chunk");
        assert!(
            more >= 3,
            "expected several continuation chunks, got {more}"
        );
    }
}
