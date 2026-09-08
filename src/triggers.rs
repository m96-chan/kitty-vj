//! Momentary pad triggers — held, not toggled: hit it on the drop, let
//! go. Each is the visual counterpart of the audio FX on the same pad
//! (see kitty-vj.conf), drawn as post-passes over the channel mix.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use crate::drive::Drive;
use crate::pass::rgb;

pub const PADS: usize = 8;
pub const PAD_NAMES: [&str; PADS] = [
    "roll", "sweep", "flanger", "vbreak", "backspin", "trans", "reverb", "echo",
];

const ROLL: usize = 0;
const SWEEP: usize = 1;
const FLANGER: usize = 2;
const VBREAK: usize = 3;
const BACKSPIN: usize = 4;
const TRANS: usize = 5;
const REVERB: usize = 6;
const ECHO: usize = 7;

pub struct Triggers {
    pub active: [bool; PADS],
    press_beat: [f64; PADS],
    hold: Option<Buffer>,  // roll: the frozen frame
    trail: Option<Buffer>, // reverb: decaying smear
    echo: Option<Buffer>,  // echo: last beat-line snapshot
    echo_beat: f64,
}

fn scale(c: Color, k: f64) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f64 * k).min(255.0) as u8,
            (g as f64 * k).min(255.0) as u8,
            (b as f64 * k).min(255.0) as u8,
        ),
        other => other,
    }
}

/// Beat hits ported from EasyPngVJ's `hits` pool — momentary, driven by
/// the grid pulses rather than raw phase so the numbers match.
///
/// `invertflash` inverts on a strong beat, `colorflash` washes the accent
/// in on the bar, `strobe` fires on the first fraction of each eighth.
pub fn beat_hits(
    buf: &mut Buffer,
    area: Rect,
    d: &Drive,
    beat: f64,
    intensity: f64,
    accent: (u8, u8, u8),
) {
    let invert = d.gbeat() > 0.72;
    let flash = 0.35 * d.gbar() * intensity;
    // First 14% of each eighth, as over there.
    let eighth = (beat * 2.0).rem_euclid(1.0);
    let strobe = if eighth < 0.14 {
        (0.10 + 0.22 * intensity) * intensity
    } else {
        0.0
    };
    if !invert && flash < 0.01 && strobe < 0.01 {
        return;
    }
    for y in 0..area.height {
        for x in 0..area.width {
            let cell = &mut buf[(area.x + x, area.y + y)];
            cell.fg = hit_colour(cell.fg, invert, flash, strobe, accent);
            cell.bg = hit_colour(cell.bg, invert, flash, strobe, accent);
        }
    }
}

fn hit_colour(c: Color, invert: bool, flash: f64, strobe: f64, accent: (u8, u8, u8)) -> Color {
    let Color::Rgb(r, g, b) = c else { return c };
    let (mut r, mut g, mut b) = (r as f64, g as f64, b as f64);
    if invert {
        r = 255.0 - r;
        g = 255.0 - g;
        b = 255.0 - b;
    }
    if flash > 0.0 {
        // Additive accent, the `lighter` composite over there.
        r += accent.0 as f64 * flash;
        g += accent.1 as f64 * flash;
        b += accent.2 as f64 * flash;
    }
    if strobe > 0.0 {
        r += 255.0 * strobe;
        g += 255.0 * strobe;
        b += 255.0 * strobe;
    }
    rgb(r, g, b)
}

fn blank(buf: &Buffer, x: u16, y: u16) -> bool {
    let c = &buf[(x, y)];
    c.symbol() == " " && c.bg == Color::Reset
}

impl Triggers {
    pub fn new() -> Self {
        Self {
            active: [false; PADS],
            press_beat: [0.0; PADS],
            hold: None,
            trail: None,
            echo: None,
            echo_beat: 0.0,
        }
    }

    pub fn press(&mut self, i: usize, beat: f64) {
        if i < PADS && !self.active[i] {
            self.active[i] = true;
            self.press_beat[i] = beat;
            if i == ECHO {
                self.echo_beat = beat.floor();
            }
        }
    }

    pub fn release(&mut self, i: usize) {
        if i < PADS {
            self.active[i] = false;
            match i {
                ROLL => self.hold = None,
                REVERB => self.trail = None,
                ECHO => self.echo = None,
                _ => {}
            }
        }
    }

    /// Backspin bends the beat the effects see: from the press point,
    /// time runs backwards, accelerating — the platter slowing your
    /// hand dragged past zero.
    pub fn warp_beat(&self, real: f64) -> f64 {
        if self.active[BACKSPIN] {
            let held = real - self.press_beat[BACKSPIN];
            self.press_beat[BACKSPIN] - held * held * 2.0
        } else {
            real
        }
    }

    /// Post-passes over the composited mix, in fixed order.
    pub fn post(&mut self, buf: &mut Buffer, area: Rect, beat: f64) {
        let tick16 = (beat.max(0.0) * 16.0) as u64;

        // ROLL — freeze the frame, re-trigger flicker on 16ths.
        if self.active[ROLL] {
            match &self.hold {
                None => {
                    let mut h = Buffer::empty(area);
                    copy_region(buf, &mut h, area);
                    self.hold = Some(h);
                }
                Some(h) => {
                    copy_region(h, buf, area);
                    let k = if tick16.is_multiple_of(2) { 1.0 } else { 0.8 };
                    if k < 1.0 {
                        dim_region(buf, area, k);
                    }
                }
            }
        }

        // VBREAK — the picture sags downward and dims, braking to black.
        if self.active[VBREAK] {
            let held = (beat - self.press_beat[VBREAK]).max(0.0);
            let src = buf.clone();
            let h = area.height as f64;
            for y in (0..area.height).rev() {
                let sag = (held * held * 0.6 * (y as f64 + 1.0) / h) as u16;
                let sy = y.saturating_sub(sag);
                for x in 0..area.width {
                    buf[(area.x + x, area.y + y)] = src[(area.x + x, area.y + sy)].clone();
                }
            }
            dim_region(buf, area, (1.0 - held * 0.12).max(0.2));
        }

        // FLANGER — comb-filter stripes: rows displaced by a sine wave.
        if self.active[FLANGER] {
            let src = buf.clone();
            for y in 0..area.height {
                let off = ((y as f64 * 0.55 + beat * std::f64::consts::TAU).sin() * 3.0) as i32;
                for x in 0..area.width {
                    let sx = (x as i32 - off).rem_euclid(area.width as i32) as u16;
                    buf[(area.x + x, area.y + y)] = src[(area.x + sx, area.y + y)].clone();
                }
            }
        }

        // TRANS — 16th-note gate to black.
        if self.active[TRANS] && !tick16.is_multiple_of(2) {
            for y in 0..area.height {
                for x in 0..area.width {
                    buf[(area.x + x, area.y + y)].reset();
                }
            }
        }

        // SWEEP — a bright band sweeps across over two beats, the rest ducks.
        if self.active[SWEEP] {
            let pos = (beat * 0.5).fract() * area.width as f64;
            for y in 0..area.height {
                for x in 0..area.width {
                    let d = (x as f64 - pos).abs();
                    let k = if d < 4.0 { 1.7 } else { 0.4 };
                    let cell = &mut buf[(area.x + x, area.y + y)];
                    cell.fg = scale(cell.fg, k);
                    cell.bg = scale(cell.bg, k);
                }
            }
        }

        // REVERB — decaying trail under the live frame.
        if self.active[REVERB] {
            let trail = self.trail.get_or_insert_with(|| Buffer::empty(area));
            if trail.area != area {
                *trail = Buffer::empty(area);
            }
            for y in 0..area.height {
                for x in 0..area.width {
                    let (ax, ay) = (area.x + x, area.y + y);
                    if !blank(buf, ax, ay) {
                        trail[(ax, ay)] = buf[(ax, ay)].clone();
                    } else if !blank(trail, ax, ay) {
                        let t = &mut trail[(ax, ay)];
                        t.fg = scale(t.fg, 0.90);
                        t.bg = scale(t.bg, 0.90);
                        if let Color::Rgb(r, g, b) = t.fg
                            && r.max(g).max(b) < 18
                        {
                            t.reset();
                        } else {
                            buf[(ax, ay)] = t.clone();
                        }
                    }
                }
            }
        }

        // ECHO — the frame from the last beat line, ghosted where the
        // live frame is blank.
        if self.active[ECHO] {
            if beat.floor() > self.echo_beat {
                self.echo_beat = beat.floor();
                self.echo = None; // re-snapshot below
            }
            match &self.echo {
                None => {
                    let mut e = Buffer::empty(area);
                    copy_region(buf, &mut e, area);
                    self.echo = Some(e);
                }
                Some(e) => {
                    for y in 0..area.height {
                        for x in 0..area.width {
                            let (ax, ay) = (area.x + x, area.y + y);
                            if blank(buf, ax, ay) && !blank(e, ax, ay) {
                                let mut c = e[(ax, ay)].clone();
                                c.fg = scale(c.fg, 0.5);
                                c.bg = scale(c.bg, 0.5);
                                buf[(ax, ay)] = c;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn hud(&self) -> String {
        let held: Vec<&str> = (0..PADS)
            .filter(|&i| self.active[i])
            .map(|i| PAD_NAMES[i])
            .collect();
        if held.is_empty() {
            String::new()
        } else {
            format!(" │ ▶{}", held.join("+"))
        }
    }
}

fn copy_region(src: &Buffer, dst: &mut Buffer, area: Rect) {
    for y in 0..area.height {
        for x in 0..area.width {
            let (ax, ay) = (area.x + x, area.y + y);
            dst[(ax, ay)] = src[(ax, ay)].clone();
        }
    }
}

fn dim_region(buf: &mut Buffer, area: Rect, k: f64) {
    for y in 0..area.height {
        for x in 0..area.width {
            let cell = &mut buf[(area.x + x, area.y + y)];
            cell.fg = scale(cell.fg, k);
            cell.bg = scale(cell.bg, k);
        }
    }
}
