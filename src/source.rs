//! Image sources, drawn once for both tiers.
//!
//! A camera and a plate are the same thing — a colour as a function of
//! position — and they were being written twice, once against the cell
//! buffer and once against the pixel framebuffer. Two copies of the
//! cover fit, the mirror and the beat punch is two places for them to
//! drift apart, and the second copy had already started lying: the pixel
//! table held entries whose function pointer was never called.
//!
//! So the sampling lives here, and the tiers get thin adapters that
//! differ only in what they do with a colour: pack two into a halfblock
//! cell, pick a ramp glyph, or write a pixel. Glyph-native effects
//! (rain, collapse, the text overlay) are *not* sources in this sense —
//! they choose characters, not colours — and stay where they are.

use image::RgbImage;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use crate::effects::ramp_glyph;
use crate::graphics::Framebuffer;
use crate::pass::luma;

/// How a cell tier should turn a sampled colour into a cell.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CellStyle {
    /// Two samples per cell in `▀`, full colour.
    Halfblock,
    /// One sample, luminance to glyph density, grey.
    Ascii,
    /// One sample, luminance to glyph density, colour kept.
    AsciiColour,
}

/// Where an image lands in the target, resolved once for both tiers.
///
/// Cover fit, plus an optional camera offset so a plate's Ken Burns
/// drift and a bare camera frame land through the same equation rather
/// than through two that merely resemble each other.
#[derive(Clone, Copy)]
pub struct Fit {
    zoom: f64,
    sw: f64,
    sh: f64,
    tw: f64,
    th: f64,
    mirror: bool,
    /// Where in the source the target's centre lands, in source pixels.
    drift_x: f64,
    drift_y: f64,
}

impl Fit {
    /// `target` is in the sampling grid's own units: pixels for the
    /// pixel tier, and cells-by-half-rows for the cell tier, which is
    /// why the caller passes it rather than a `Rect`.
    pub fn cover(src: (f64, f64), target: (f64, f64), punch: f64, mirror: bool) -> Option<Self> {
        let (sw, sh) = src;
        let (tw, th) = target;
        if sw < 1.0 || sh < 1.0 || tw < 1.0 || th < 1.0 {
            return None;
        }
        Some(Self {
            zoom: (tw / sw).max(th / sh) * punch,
            sw,
            sh,
            tw,
            th,
            mirror,
            drift_x: 0.0,
            drift_y: 0.0,
        })
    }

    /// Take a resolved placement instead of a bare cover fit — this is
    /// how a plate's framing and camera reach the sampler.
    pub fn placed(
        src: (f64, f64),
        target: (f64, f64),
        scale: f64,
        drift: (f64, f64),
        mirror: bool,
    ) -> Option<Self> {
        let (sw, sh) = src;
        let (tw, th) = target;
        if sw < 1.0 || sh < 1.0 || tw < 1.0 || th < 1.0 || scale <= 0.0 {
            return None;
        }
        Some(Self {
            zoom: scale,
            sw,
            sh,
            tw,
            th,
            mirror,
            drift_x: drift.0,
            drift_y: drift.1,
        })
    }

    /// The zoom this fit resolved to, for callers that shear or offset
    /// in target space and need the same scale.
    #[allow(dead_code)]
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    /// Sample the image at a point in target space.
    pub fn at(&self, img: &RgbImage, x: f64, y: f64) -> (u8, u8, u8) {
        let px = if self.mirror { self.tw - x } else { x };
        let sx = ((px - self.tw / 2.0) / self.zoom + self.sw / 2.0 + self.drift_x)
            .clamp(0.0, self.sw - 1.0);
        let sy = ((y - self.th / 2.0) / self.zoom + self.sh / 2.0 + self.drift_y)
            .clamp(0.0, self.sh - 1.0);
        let p = img.get_pixel(sx as u32, sy as u32).0;
        (p[0], p[1], p[2])
    }
}

/// The beat punch every image source applies. One definition, so a
/// picture pulses identically whichever tier is drawing it and whether
/// it came from a file or a camera.
///
/// The depth is the plates' 0.10 rather than a rounder number: they were
/// ported against that figure, and the camera had drifted to 0.08 by
/// being written second.
pub fn punch(gbeat: f64, intensity: f64) -> f64 {
    1.0 + 0.10 * gbeat * (0.3 + 0.7 * intensity)
}

/// Draw an image into the cell grid.
pub fn draw_cells(
    buf: &mut Buffer,
    area: Rect,
    img: &RgbImage,
    style: CellStyle,
    gbeat: f64,
    intensity: f64,
    mirror: bool,
) {
    if area.width < 1 || area.height < 1 {
        return;
    }
    // The cell grid samples at half-row resolution, which is what makes
    // halfblock worth having.
    let target = (area.width as f64, area.height as f64 * 2.0);
    let Some(fit) = Fit::cover(
        (img.width() as f64, img.height() as f64),
        target,
        punch(gbeat, intensity),
        mirror,
    ) else {
        return;
    };

    for cy in 0..area.height {
        for cx in 0..area.width {
            let cell = &mut buf[(area.x + cx, area.y + cy)];
            match style {
                CellStyle::Halfblock => {
                    let t = fit.at(img, cx as f64 + 0.5, cy as f64 * 2.0 + 0.5);
                    let b = fit.at(img, cx as f64 + 0.5, cy as f64 * 2.0 + 1.5);
                    cell.set_char('▀');
                    cell.set_fg(Color::Rgb(t.0, t.1, t.2));
                    cell.set_bg(Color::Rgb(b.0, b.1, b.2));
                }
                CellStyle::Ascii | CellStyle::AsciiColour => {
                    // One glyph per cell, so sample its middle rather
                    // than either half.
                    let (r, g, b) = fit.at(img, cx as f64 + 0.5, cy as f64 * 2.0 + 1.0);
                    let l = luma(r as f64, g as f64, b as f64) / 255.0;
                    cell.set_char(ramp_glyph(l));
                    if style == CellStyle::Ascii {
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

/// Draw an image into the pixel framebuffer. Same fit, same punch — the
/// post passes downstream then apply to it exactly as they do to a
/// generated field.
pub fn draw_pixels(fb: &mut Framebuffer, img: &RgbImage, gbeat: f64, intensity: f64, mirror: bool) {
    let Some(fit) = Fit::cover(
        (img.width() as f64, img.height() as f64),
        (fb.w as f64, fb.h as f64),
        punch(gbeat, intensity),
        mirror,
    ) else {
        return;
    };
    for y in 0..fb.h {
        for x in 0..fb.w {
            let (r, g, b) = fit.at(img, x as f64 + 0.5, y as f64 + 0.5);
            fb.set(x, y, r, g, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> RgbImage {
        // A left-to-right red ramp, so mirroring is detectable.
        RgbImage::from_fn(w, h, |x, _| image::Rgb([(x * 255 / w.max(1)) as u8, 0, 0]))
    }

    #[test]
    fn the_two_tiers_agree_on_what_they_sampled() {
        // The whole point of this module: a camera drawn as pixels and
        // the same camera drawn as cells must be the same picture, not
        // two implementations that happen to look similar.
        let im = img(64, 36);
        let mut fb = Framebuffer::new(40, 20);
        draw_pixels(&mut fb, &im, 0.0, 0.5, false);

        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        draw_cells(&mut buf, area, &im, CellStyle::Halfblock, 0.0, 0.5, false);

        // A cell's top half is pixel row 2y, so row 0 of each must match.
        for x in 0..40u16 {
            let i = (x as usize) * 3;
            let px = (fb.px[i], fb.px[i + 1], fb.px[i + 2]);
            let Color::Rgb(r, g, b) = buf[(x, 0)].fg else {
                panic!("expected rgb")
            };
            assert_eq!(px, (r, g, b), "tiers disagree at column {x}");
        }
    }

    #[test]
    fn a_placement_and_a_cover_agree_when_there_is_no_camera() {
        // The plate tier builds its fit from a resolved Placement while
        // the pixel tier builds a bare cover. With no camera offset they
        // must be the same equation — this is the assertion that would
        // have caught the plate sampler drifting away from the camera's.
        let im = img(64, 36);
        let target = (40.0, 20.0);
        let cover = Fit::cover((64.0, 36.0), target, 1.0, false).unwrap();
        let placed = Fit::placed((64.0, 36.0), target, cover.zoom(), (0.0, 0.0), false).unwrap();
        for x in 0..40 {
            for y in 0..20 {
                let (a, b) = (x as f64 + 0.5, y as f64 + 0.5);
                assert_eq!(
                    cover.at(&im, a, b),
                    placed.at(&im, a, b),
                    "the two construction paths disagree at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn one_punch_serves_every_image_source() {
        // The bug this guards: a camera and a plate pulsing by different
        // amounts because each had its own copy of the formula.
        let a = punch(0.7, 0.6);
        let b = crate::source::punch(0.7, 0.6);
        assert_eq!(a, b);
        assert!(punch(0.0, 1.0) == 1.0);
    }

    #[test]
    fn mirror_flips_the_picture() {
        let im = img(64, 36);
        let mut a = Framebuffer::new(40, 20);
        let mut b = Framebuffer::new(40, 20);
        draw_pixels(&mut a, &im, 0.0, 0.5, false);
        draw_pixels(&mut b, &im, 0.0, 0.5, true);
        // The ramp rises left to right unmirrored, and falls mirrored.
        let left = a.px[0];
        let right = a.px[((39u32) * 3) as usize];
        assert!(right > left, "unmirrored ramp should rise");
        let mleft = b.px[0];
        let mright = b.px[((39u32) * 3) as usize];
        assert!(mright < mleft, "mirrored ramp should fall");
    }

    #[test]
    fn cover_never_leaves_a_gap() {
        // Cover is a guarantee: every target pixel must come from the
        // image, whatever the aspect mismatch.
        for src in [(64u32, 36u32), (36, 64), (100, 10), (10, 100)] {
            let im = img(src.0, src.1);
            let mut fb = Framebuffer::new(37, 21);
            draw_pixels(&mut fb, &im, 0.0, 0.0, false);
            assert!(
                fb.px.chunks(3).all(|c| c[1] == 0 && c[2] == 0),
                "src {src:?} sampled outside the image"
            );
        }
    }

    #[test]
    fn the_punch_is_one_definition() {
        assert_eq!(punch(0.0, 1.0), 1.0, "no beat, no punch");
        assert!(punch(1.0, 1.0) > punch(1.0, 0.0), "intensity deepens it");
    }

    #[test]
    fn degenerate_sizes_do_not_panic() {
        let im = img(4, 4);
        let mut fb = Framebuffer::new(0, 0);
        draw_pixels(&mut fb, &im, 0.0, 0.5, false);
        for (w, h) in [(0u16, 0u16), (1, 1), (1, 9), (9, 1)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            draw_cells(&mut buf, area, &im, CellStyle::Ascii, 0.0, 0.5, false);
        }
        // And a zero-sized image against a real target.
        let empty = RgbImage::new(0, 0);
        let mut fb2 = Framebuffer::new(8, 8);
        draw_pixels(&mut fb2, &empty, 0.0, 0.5, false);
    }
}
