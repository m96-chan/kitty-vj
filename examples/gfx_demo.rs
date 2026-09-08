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
#[allow(dead_code)]
#[path = "../src/pixfx.rs"]
mod pixfx;
#[allow(dead_code)]
#[path = "../src/rng.rs"]
mod rng;

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

    // Uncapped: measure the true ceiling of the direct medium. A frame
    // budget cap belongs in the real app, not this probe.
    let start = Instant::now();
    let tempo = 128.0;
    let mut frames = 0u32;
    let mut worst = 0.0f64;
    while start.elapsed().as_secs_f64() < 10.0 {
        let t0 = Instant::now();
        let beat = start.elapsed().as_secs_f64() * tempo / 60.0;
        pixfx::plasma(&mut fb, beat, 0.8);
        if graphics::transmit_direct(&mut out, &fb, 1).is_err() {
            break;
        }
        frames += 1;
        worst = worst.max(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let _ = graphics::clear_all(&mut out);
    let _ = write!(out, "\x1b[?25h\x1b[2J\x1b[H"); // show cursor, clear
    let _ = out.flush();
    eprintln!(
        "{frames} frames in {:.1}s ({:.1} fps, worst {:.1}ms) at {}x{}",
        start.elapsed().as_secs_f64(),
        frames as f64 / start.elapsed().as_secs_f64(),
        worst,
        fb.w,
        fb.h
    );
}
