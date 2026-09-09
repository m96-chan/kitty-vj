//! CAM — the live capture as a plate.
//!
//! Almost nothing here is camera-specific, and that is the point: the
//! ported set already has the treatments (edge, lut, mono, feedback,
//! imgdust, the mesh cube's faces), so a live source only has to arrive
//! as pixels and everything else applies for free.
//!
//! The one thing worth adding is the mode below. PLATE draws in
//! halfblock, so the ` .:-=+*#%@` luminance ramp — the look that reads
//! as *terminal* rather than as a small image — had nowhere to live. On
//! a live camera it is the most on-brand thing the project can do.

use std::cell::RefCell;
use std::rc::Rc;

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::{Effect, FrameCtx, ramp_glyph};
use crate::capture::Capture;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    /// Full colour, two pixels per cell.
    Halfblock,
    /// Luminance to glyph density, the terminal-native reading.
    Ascii,
    /// Glyph density from luminance, colour kept — the two at once.
    AsciiColour,
}

impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Halfblock => "HALF",
            Mode::Ascii => "ASCII",
            Mode::AsciiColour => "ASCII+C",
        }
    }

    fn next(&self) -> Mode {
        match self {
            Mode::Halfblock => Mode::Ascii,
            Mode::Ascii => Mode::AsciiColour,
            Mode::AsciiColour => Mode::Halfblock,
        }
    }
}

pub struct CamFx {
    cap: Rc<RefCell<Option<Capture>>>,
    mode: Mode,
    /// Mirror the picture. A camera pointed at the operator reads wrong
    /// unless it is flipped, and a projected VJ feed almost always wants
    /// the mirror.
    mirror: bool,
}

impl CamFx {
    pub fn new(cap: Rc<RefCell<Option<Capture>>>) -> Self {
        Self {
            cap,
            mode: Mode::Halfblock,
            mirror: true,
        }
    }
}

impl Effect for CamFx {
    fn name(&self) -> &'static str {
        "CAM"
    }

    fn on_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('v') => {
                self.mode = self.mode.next();
                true
            }
            KeyCode::Char('V') => {
                self.mirror = !self.mirror;
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        let cap = self.cap.borrow();
        let c = cap.as_ref()?;
        Some(format!(
            "{} {}{}",
            self.mode.name(),
            if c.alive() { "live" } else { "..." },
            if self.mirror { " mir" } else { "" }
        ))
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        if area.width < 2 || area.height < 2 {
            return;
        }
        let cap = self.cap.borrow();
        let Some(img) = cap.as_ref().and_then(|c| c.latest()) else {
            return;
        };
        let (sw, sh) = (img.width() as f64, img.height() as f64);

        // Cover fit, with a beat punch — the same camera language the
        // rest of the plates speak.
        let (tw, th) = (area.width as f64, area.height as f64 * 2.0);
        let punch = 1.0 + 0.08 * ctx.drive.gbeat() * (0.3 + 0.7 * ctx.intensity);
        let zoom = (tw / sw).max(th / sh) * punch;

        let sample = |px: f64, py: f64| -> (u8, u8, u8) {
            let x = if self.mirror { tw - px } else { px };
            let sx = ((x - tw / 2.0) / zoom + sw / 2.0).clamp(0.0, sw - 1.0) as u32;
            let sy = ((py - th / 2.0) / zoom + sh / 2.0).clamp(0.0, sh - 1.0) as u32;
            let p = img.get_pixel(sx, sy).0;
            (p[0], p[1], p[2])
        };

        for cy in 0..area.height {
            for cx in 0..area.width {
                let cell = &mut buf[(area.x + cx, area.y + cy)];
                match self.mode {
                    Mode::Halfblock => {
                        let t = sample(cx as f64 + 0.5, cy as f64 * 2.0 + 0.5);
                        let b = sample(cx as f64 + 0.5, cy as f64 * 2.0 + 1.5);
                        cell.set_char('▀');
                        cell.set_fg(Color::Rgb(t.0, t.1, t.2));
                        cell.set_bg(Color::Rgb(b.0, b.1, b.2));
                    }
                    Mode::Ascii | Mode::AsciiColour => {
                        // One glyph per cell, so sample the cell's middle
                        // rather than either half.
                        let (r, g, b) = sample(cx as f64 + 0.5, cy as f64 * 2.0 + 1.0);
                        let l = crate::pass::luma(r as f64, g as f64, b as f64) / 255.0;
                        cell.set_char(ramp_glyph(l));
                        if self.mode == Mode::Ascii {
                            let v = (l * 255.0) as u8;
                            cell.set_fg(Color::Rgb(v, v, v));
                        } else {
                            cell.set_fg(Color::Rgb(r, g, b));
                        }
                        cell.set_bg(Color::Reset);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::Drive;

    fn ctx() -> FrameCtx {
        FrameCtx {
            beat: 1.0,
            phase: 0.0,
            bar_phase: 1.0,
            intensity: 0.5,
            drive: Drive::default(),
        }
    }

    #[test]
    fn without_a_camera_it_draws_nothing() {
        // A camera that never opened must not take the instrument down,
        // and must not paint over whatever else is on the channel.
        let mut fx = CamFx::new(Rc::new(RefCell::new(None)));
        let area = Rect::new(0, 0, 20, 10);
        let mut buf = Buffer::empty(area);
        fx.render(&mut buf, area, &ctx());
        assert!(buf.content().iter().all(|c| c.symbol() == " "));
        assert!(fx.status().is_none());
    }

    #[test]
    fn tiny_areas_do_not_panic() {
        let mut fx = CamFx::new(Rc::new(RefCell::new(None)));
        for (w, h) in [(0u16, 0u16), (1, 1), (1, 9), (9, 1), (3, 2)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            fx.render(&mut buf, area, &ctx());
        }
    }

    #[test]
    fn keys_cycle_the_mode_and_the_mirror() {
        let mut fx = CamFx::new(Rc::new(RefCell::new(None)));
        assert_eq!(fx.mode, Mode::Halfblock);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, Mode::Ascii);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, Mode::AsciiColour);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, Mode::Halfblock);

        assert!(fx.mirror, "a camera pointed at you reads wrong unmirrored");
        assert!(fx.on_key(KeyCode::Char('V')));
        assert!(!fx.mirror);

        assert!(
            !fx.on_key(KeyCode::Char('z')),
            "unclaimed keys pass through"
        );
    }
}
