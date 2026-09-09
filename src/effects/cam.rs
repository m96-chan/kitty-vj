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

use super::{Effect, FrameCtx};
use crate::capture::Capture;
use crate::source::{CellStyle, draw_cells};

fn style_name(s: CellStyle) -> &'static str {
    match s {
        CellStyle::Halfblock => "HALF",
        CellStyle::Ascii => "ASCII",
        CellStyle::AsciiColour => "ASCII+C",
    }
}

fn next_style(s: CellStyle) -> CellStyle {
    match s {
        CellStyle::Halfblock => CellStyle::Ascii,
        CellStyle::Ascii => CellStyle::AsciiColour,
        CellStyle::AsciiColour => CellStyle::Halfblock,
    }
}

pub struct CamFx {
    cap: Rc<RefCell<Option<Capture>>>,
    mode: CellStyle,
    /// Mirror the picture. A camera pointed at the operator reads wrong
    /// unless it is flipped, and a projected VJ feed almost always wants
    /// the mirror.
    mirror: bool,
}

impl CamFx {
    pub fn new(cap: Rc<RefCell<Option<Capture>>>) -> Self {
        Self {
            cap,
            mode: CellStyle::Halfblock,
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
                self.mode = next_style(self.mode);
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
            style_name(self.mode),
            if c.alive() { "live" } else { "..." },
            if self.mirror { " mir" } else { "" }
        ))
    }

    fn render(&mut self, buf: &mut Buffer, area: Rect, ctx: &FrameCtx) {
        let cap = self.cap.borrow();
        let Some(img) = cap.as_ref().and_then(|c| c.latest()) else {
            return;
        };
        // Both tiers draw an image the same way, so this is the same
        // call the pixel tier makes — see source.rs for why.
        draw_cells(
            buf,
            area,
            &img,
            self.mode,
            ctx.drive.gbeat(),
            ctx.intensity,
            self.mirror,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::Drive;

    fn ctx() -> FrameCtx {
        FrameCtx {
            beat: 1.0,
            dt: 1.0 / 60.0,
            vt: 0.0,
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
        assert_eq!(fx.mode, CellStyle::Halfblock);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, CellStyle::Ascii);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, CellStyle::AsciiColour);
        assert!(fx.on_key(KeyCode::Char('v')));
        assert_eq!(fx.mode, CellStyle::Halfblock);

        assert!(fx.mirror, "a camera pointed at you reads wrong unmirrored");
        assert!(fx.on_key(KeyCode::Char('V')));
        assert!(!fx.mirror);

        assert!(
            !fx.on_key(KeyCode::Char('z')),
            "unclaimed keys pass through"
        );
    }
}
