//! Standalone graphics-tier probe. Run this on the real kitty to verify
//! the pixel transport end to end, decoupled from the main app:
//!
//!     cargo run --release --example gfx_demo
//!
//! Renders a beat-clocked plasma to a full-RGB framebuffer and ships each
//! frame over the kitty graphics protocol for ~15 s. If you see smooth
//! color plasma filling the window, the transport works and we can wire
//! it into the mixer as a channel effect.

#[path = "../src/graphics.rs"]
mod graphics;
#[path = "../src/pixfx.rs"]
mod pixfx;

use std::io::Write;
use std::time::Instant;

fn main() {
    if !graphics::supported() {
        eprintln!("not running under kitty (no KITTY_WINDOW_ID / TERM). Aborting.");
        return;
    }

    // Modest resolution first — prove the pipe before pushing pixels.
    let mut fb = graphics::Framebuffer::new(480, 270);
    let mut out = std::io::stdout().lock();

    let _ = write!(out, "\x1b[2J\x1b[?25l"); // clear, hide cursor
    let _ = out.flush();

    let start = Instant::now();
    let tempo = 128.0;
    let mut frames = 0u32;
    while start.elapsed().as_secs_f64() < 15.0 {
        let beat = start.elapsed().as_secs_f64() * tempo / 60.0;
        pixfx::plasma(&mut fb, beat, 0.8);
        if graphics::transmit_direct(&mut out, &fb, 1).is_err() {
            break;
        }
        frames += 1;
        // ~30 fps pacing for the direct medium.
        std::thread::sleep(std::time::Duration::from_millis(33));
    }

    let _ = graphics::clear_all(&mut out);
    let _ = write!(out, "\x1b[?25h\x1b[2J\x1b[H"); // show cursor, clear
    let _ = out.flush();
    eprintln!(
        "{frames} frames in {:.1}s ({:.1} fps) at {}x{}",
        start.elapsed().as_secs_f64(),
        frames as f64 / start.elapsed().as_secs_f64(),
        fb.w,
        fb.h
    );
}
