mod assets;
mod audio;
mod clock;
mod config;
mod drive;
mod effects;
mod font;
mod generate;
mod graphics;
mod jog;
mod link;
mod looks;
mod lyrics;
mod midi;
mod pixfx;
mod rng;
mod scfx;
mod triggers;

use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use clock::{ClockSource, InternalClock, TapTempo};
use effects::{
    Collapse, Cube, Effect, FrameCtx, PlateFx, Pulse, Rain, Sparks, TextOverlay, Tunnel,
};

#[derive(Clone, Copy, PartialEq)]
enum LearnTarget {
    Off,
    Intensity,
    /// One of the four channel faders.
    Ch(usize),
}

/// A mixer channel: an effect slot, a vertical fader, and a COLOR knob.
struct Channel {
    slot: usize,
    level: f64,
    cc: Option<(u8, u8)>,
    /// Sound Color FX knob, 0..1, 0.5 = neutral.
    color: f64,
    color_cc: Option<(u8, u8)>,
}

const CHANNELS: usize = 4;

/// Pixel effects available in the graphics tier, with display names.
type PixFn = fn(&mut graphics::Framebuffer, f64, f64);
const PIX_FX: [(&str, PixFn); 3] = [
    ("PLASMA", pixfx::plasma),
    ("TUNNEL", pixfx::tunnel),
    ("STARS", pixfx::starfield),
];

/// Fixed timestep for clock advancement. Real elapsed time is consumed in
/// whole ticks so a run is a pure function of (seed, tick count).
const TICK: f64 = 1.0 / 120.0;
/// Render frame budget (~60fps).
const FRAME: Duration = Duration::from_millis(16);

struct App {
    clock: InternalClock,
    tap: TapTempo,
    effects: Vec<Box<dyn Effect>>,
    overlay: TextOverlay,
    audio: Option<audio::AudioBeat>,
    audio_sync: bool,
    audio_err: Option<String>,
    /// Fold window for audio detection; cycled with 'r'.
    bpm_window: (u32, u32),
    /// Persistent colour treatment over the whole mix ('w' cycles).
    look: looks::Look,
    /// Per-scene random hue for duotone, and the palette accent for lut.
    hue_base: f64,
    accent: (u8, u8, u8),
    /// Drive signals derived from the clock (and audio when running).
    drive: drive::Drive,
    /// Generative plugins; empty until an adapter is registered (#17).
    /// Disabled by default — 'k' is the kill switch either way.
    ai: generate::Registry,
    /// Timed lyric cards, when a .lrc is present.
    lyrics: lyrics::Lyrics,
    /// Jog wheels, one per deck, scrubbing beat time.
    jogs: [jog::Jog; 4],
    jog_bind: [Option<(u8, u8)>; 4],
    jog_touch_bind: [Option<(u8, u8)>; 4],
    midi: Option<midi::MidiIn>,
    midi_clock: midi::MidiClock,
    midi_clock_sync: bool,
    cc_bind_int: Option<(u8, u8)>,
    learn: LearnTarget,
    last_cc: Option<(u8, u8, u8)>,
    link: Option<link::LinkSync>,
    link_sync: bool,
    /// Four mixer channels, composited bottom (1) to top (4).
    channels: [Channel; CHANNELS],
    /// Channel the keyboard is aimed at.
    focus: usize,
    scfx_type: scfx::ScfxType,
    scfx_on: bool,
    scfx_bind: [Option<(u8, u8)>; 6],
    triggers: triggers::Triggers,
    pad_bind: [Vec<(u8, u8)>; triggers::PADS],
    /// Walks pad slots during 'b' learn; None when idle.
    pad_learn: Option<usize>,
    last_note: Option<(u8, u8)>,
    scratch: Vec<ratatui::buffer::Buffer>,
    /// Graphics tier: full-pixel rendering over the kitty protocol.
    gfx: bool,
    gfx_fb: graphics::Framebuffer,
    /// Which pixel effect (index into PIX_FX) when in graphics mode.
    pix: usize,
    /// Stage cell rect from the last draw, for image placement.
    stage_cells: (u16, u16),
    intensity: f64,
    /// HUD off = clean feed for the projector.
    hud_visible: bool,
    render_ms: f64, // EWMA of draw time
    quit: bool,
}

impl App {
    fn new(plates: Vec<assets::Plate>, text: String, lyrics: lyrics::Lyrics) -> Self {
        let bindings = config::load(std::path::Path::new(config::PATH));
        let plates = std::rc::Rc::new(plates);
        let mut effects: Vec<Box<dyn Effect>> = vec![
            Box::new(Pulse),
            Box::new(Rain),
            Box::new(Tunnel),
            Box::new(Collapse),
            Box::new(Cube::new(plates.clone())),
            Box::new(Sparks),
        ];
        if !plates.is_empty() {
            effects.push(Box::new(PlateFx::new(plates)));
        }
        Self {
            clock: InternalClock::new(120.0),
            tap: TapTempo::new(),
            effects,
            overlay: TextOverlay::new(text),
            audio: None,
            audio_sync: false,
            audio_err: None,
            bpm_window: (85, 170),
            look: looks::Look::Plain,
            hue_base: 210.0,
            accent: (0, 255, 213),
            drive: drive::Drive::default(),
            ai: generate::Registry::default(),
            lyrics,
            jogs: Default::default(),
            jog_bind: bindings.jogs,
            jog_touch_bind: bindings.jog_touch,
            // MIDI needs no permission prompt — open at launch.
            midi: midi::MidiIn::open().ok(),
            midi_clock: midi::MidiClock::new(),
            midi_clock_sync: false,
            cc_bind_int: bindings.intensity,
            learn: LearnTarget::Off,
            last_cc: None,
            link: None,
            link_sync: false,
            channels: std::array::from_fn(|i| Channel {
                slot: i,
                level: if i == 0 { 1.0 } else { 0.0 },
                cc: bindings.channels[i],
                color: 0.5,
                color_cc: bindings.colors[i],
            }),
            scfx_type: bindings
                .scfx_selected
                .map(|i| scfx::TYPES[i])
                .unwrap_or(scfx::TYPES[0]),
            scfx_on: bindings.scfx_selected.is_some(),
            scfx_bind: bindings.scfx,
            focus: 0,
            triggers: triggers::Triggers::new(),
            pad_bind: bindings.pads,
            pad_learn: None,
            last_note: None,
            scratch: Vec::new(),
            gfx: false,
            gfx_fb: graphics::Framebuffer::new(1, 1),
            pix: 0,
            stage_cells: (0, 0),
            intensity: 0.5,
            hud_visible: true,
            render_ms: 0.0,
            quit: false,
        }
    }

    fn on_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char(' ') => {
                if let Some(bpm) = self.tap.tap(Instant::now()) {
                    self.clock.set_tempo(bpm);
                }
                self.clock.resync();
            }
            KeyCode::Char('+') | KeyCode::Char('=') => self.clock.nudge_tempo(0.5),
            KeyCode::Char('-') => self.clock.nudge_tempo(-0.5),
            KeyCode::Up => self.intensity = (self.intensity + 0.05).min(1.0),
            KeyCode::Down => self.intensity = (self.intensity - 0.05).max(0.0),
            KeyCode::Left => {
                let ch = &mut self.channels[self.focus];
                ch.level = (ch.level - 0.05).max(0.0);
            }
            KeyCode::Right => {
                let ch = &mut self.channels[self.focus];
                ch.level = (ch.level + 0.05).min(1.0);
            }
            KeyCode::Tab => self.focus = (self.focus + 1) % CHANNELS,
            KeyCode::Char('o') => self.overlay.toggle(),
            KeyCode::Char('h') => self.hud_visible = !self.hud_visible,
            KeyCode::Char('w') => {
                self.look = self.look.next();
                // A fresh hue each time duotone comes round.
                self.hue_base = (self.hue_base + 47.0).rem_euclid(360.0);
            }
            KeyCode::Char('k') => {
                // Kill switch: cuts every generator for the rest of the
                // set. Nothing downstream may block on them anyway.
                self.ai.enabled = !self.ai.enabled;
            }
            KeyCode::Char('y') => {
                // Mark the track start — the lyric clock is track time,
                // not beat time, so it needs its own downbeat.
                if self.lyrics.enabled {
                    self.lyrics.stop();
                } else if !self.lyrics.is_empty() {
                    self.lyrics.start();
                }
            }
            KeyCode::Char('[') => self.lyrics.nudge(-0.25),
            KeyCode::Char(']') => self.lyrics.nudge(0.25),
            KeyCode::Char('g') => self.gfx = !self.gfx,
            KeyCode::Char('p') => self.pix = (self.pix + 1) % PIX_FX.len(),
            KeyCode::Char('x') => {
                // Cycle FILTER → SPACE → DUBECHO → CRUSH → OFF → …
                if !self.scfx_on {
                    self.scfx_on = true;
                    self.scfx_type = scfx::TYPES[0];
                } else if self.scfx_type == *scfx::TYPES.last().unwrap() {
                    self.scfx_on = false;
                } else {
                    self.scfx_type = self.scfx_type.next();
                }
                self.save_bindings();
            }
            KeyCode::Char('r') => {
                // Cycle the tempo fold window: pick one that puts the set
                // in its middle, not against a fold edge.
                const WINDOWS: [(u32, u32); 5] =
                    [(85, 170), (120, 240), (140, 280), (170, 340), (60, 120)];
                let i = WINDOWS
                    .iter()
                    .position(|&w| w == self.bpm_window)
                    .unwrap_or(0);
                self.bpm_window = WINDOWS[(i + 1) % WINDOWS.len()];
                if let Some(a) = &self.audio {
                    a.set_window(self.bpm_window.0, self.bpm_window.1);
                }
            }
            KeyCode::Char('a') => {
                if self.audio.is_none() {
                    match audio::AudioBeat::start(self.bpm_window.0, self.bpm_window.1) {
                        Ok(a) => {
                            self.audio = Some(a);
                            self.audio_sync = true;
                            self.audio_err = None;
                        }
                        Err(e) => self.audio_err = Some(e),
                    }
                } else {
                    self.audio_sync = !self.audio_sync;
                }
                if self.audio_sync {
                    self.link_sync = false;
                    self.midi_clock_sync = false;
                }
            }
            KeyCode::Char('b') => {
                // Pad learn: walks roll → sweep → … — hit the pad on the
                // controller for each slot; 'b' again skips a slot.
                self.pad_learn = match self.pad_learn {
                    None => Some(0),
                    Some(i) if i + 1 < triggers::PADS => Some(i + 1),
                    Some(_) => None,
                };
            }
            KeyCode::F(n @ 1..=8) => {
                // Keyboard fallback (needs kitty's key protocol for release).
                self.triggers.press(n as usize - 1, self.clock.beat());
            }
            KeyCode::Char('m') => {
                if self.midi.is_some() {
                    // Cycle the learn target: off → int → ch1..ch4 → off.
                    self.learn = match self.learn {
                        LearnTarget::Off => LearnTarget::Intensity,
                        LearnTarget::Intensity => LearnTarget::Ch(0),
                        LearnTarget::Ch(i) if i + 1 < CHANNELS => LearnTarget::Ch(i + 1),
                        LearnTarget::Ch(_) => LearnTarget::Off,
                    };
                }
            }
            KeyCode::Char('c') => {
                self.midi_clock_sync = !self.midi_clock_sync;
                if self.midi_clock_sync {
                    self.link_sync = false;
                    self.audio_sync = false;
                }
            }
            KeyCode::Char('l') => {
                if self.link.is_none() {
                    self.link = Some(link::LinkSync::new(self.clock.tempo()));
                    self.link_sync = true;
                } else {
                    self.link_sync = !self.link_sync;
                }
                if self.link_sync {
                    self.audio_sync = false;
                    self.midi_clock_sync = false;
                }
            }
            KeyCode::Char(c @ '1'..='9') => {
                let i = (c as usize) - ('1' as usize);
                if i < self.effects.len() {
                    self.channels[self.focus].slot = i;
                }
            }
            _ => {
                // Unclaimed keys go to the focused channel's effect.
                let target = self.channels[self.focus].slot;
                self.effects[target].on_key(code);
            }
        }
    }

    fn on_key_release(&mut self, code: KeyCode) {
        if let KeyCode::F(n @ 1..=8) = code {
            self.triggers.release(n as usize - 1);
        }
    }

    /// Draw the current lyric line, with the previous/next dimmed above
    /// and below. Enhanced LRC fills the line word by word as it's sung.
    fn render_lyrics(
        &self,
        buf: &mut ratatui::buffer::Buffer,
        area: ratatui::layout::Rect,
        ctx: &FrameCtx,
    ) {
        let Some((line, prog)) = self.lyrics.current() else {
            return;
        };
        // Word timings give a real fill; without them, ride the line's
        // own duration so the text still sweeps in time.
        let lit = if line.words.is_empty() {
            prog
        } else {
            self.lyrics.words_done() as f64 / line.words.len() as f64
        };
        effects::draw_text(buf, area, ctx, &line.text, 0.5, 1.0, lit, true);
    }

    /// Take whatever the generators have ready. Never blocks; a stalled
    /// or dead model simply yields nothing this frame.
    fn consume_ai(&mut self) {
        for a in self.ai.drain() {
            match a {
                generate::Artifact::Text(t) => self.overlay.text = t,
                generate::Artifact::Image(_) => {
                    // Generated plates join the rotation once the plate
                    // effect takes shared ownership (#17).
                }
            }
        }
    }

    /// Combined scrub from every jog, in beats.
    fn jog_offset(&self) -> f64 {
        self.jogs.iter().map(|j| j.offset()).sum()
    }

    /// Coast/recentre the wheels by real elapsed time.
    fn tick_jogs(&mut self, dt: f64) {
        for j in &mut self.jogs {
            j.tick(dt);
        }
    }

    /// Advance the drive signals. Audio contributes kick/groove/onset
    /// when A-mode is running; otherwise the clock alone drives them.
    fn tick_drive(&mut self, dt: f64) {
        let audio = if self.audio_sync {
            self.audio
                .as_ref()
                .and_then(|a| a.estimate())
                .map(|est| drive::AudioDrive {
                    // Confidence stands in for beat presence until the
                    // analyser exposes crest factor directly.
                    thump: (est.confidence - 1.0).clamp(0.0, 1.5) / 1.5,
                    groove: ((est.confidence - 1.0) / 1.5).clamp(0.0, 1.0),
                    hit: false,
                })
        } else {
            None
        };
        let beat = self.clock.beat();
        self.drive.update(dt, beat, audio);
    }

    /// Persist current CC bindings to the gig config next to the app.
    fn save_bindings(&self) {
        let b = config::Bindings {
            intensity: self.cc_bind_int,
            channels: std::array::from_fn(|i| self.channels[i].cc),
            pads: self.pad_bind.clone(),
            colors: std::array::from_fn(|i| self.channels[i].color_cc),
            jogs: self.jog_bind,
            jog_touch: self.jog_touch_bind,
            scfx: self.scfx_bind,
            scfx_selected: self
                .scfx_on
                .then(|| scfx::TYPES.iter().position(|t| *t == self.scfx_type))
                .flatten(),
        };
        let _ = config::save(std::path::Path::new(config::PATH), &b);
    }

    /// Drain MIDI: CC learn/binding, clock ticks, transport.
    fn process_midi(&mut self) {
        let Some(m) = &self.midi else { return };
        for ev in m.drain() {
            match ev {
                midi::MidiEvent::Cc { ch, cc, val } => {
                    self.last_cc = Some((ch, cc, val));
                    match self.learn {
                        LearnTarget::Intensity => {
                            self.cc_bind_int = Some((ch, cc));
                            self.learn = LearnTarget::Off;
                            self.save_bindings();
                        }
                        LearnTarget::Ch(i) => {
                            self.channels[i].cc = Some((ch, cc));
                            self.learn = LearnTarget::Off;
                            self.save_bindings();
                        }
                        LearnTarget::Off => {}
                    }
                    if self.cc_bind_int == Some((ch, cc)) {
                        self.intensity = val as f64 / 127.0;
                    }
                    for (j, bind) in self.jog_bind.iter().enumerate() {
                        if *bind == Some((ch, cc)) {
                            self.jogs[j].cc(val);
                        }
                    }
                    for c in &mut self.channels {
                        if c.cc == Some((ch, cc)) {
                            c.level = val as f64 / 127.0;
                        }
                        if c.color_cc == Some((ch, cc)) {
                            c.color = val as f64 / 127.0;
                        }
                    }
                }
                midi::MidiEvent::NoteOn { ch, note } => {
                    self.last_note = Some((ch, note));
                    if let Some(i) = self.pad_learn {
                        // A note some pad already owns is ignored: one
                        // press can double-fire, and eating two learn
                        // slots is how duplicate binds happened.
                        let owned = self.pad_bind.iter().any(|v| v.contains(&(ch, note)));
                        if !owned {
                            self.pad_bind[i].push((ch, note));
                            self.pad_learn = if i + 1 < triggers::PADS {
                                Some(i + 1)
                            } else {
                                None
                            };
                            self.save_bindings();
                        }
                        continue;
                    }
                    for (j, bind) in self.jog_touch_bind.iter().enumerate() {
                        if *bind == Some((ch, note)) {
                            self.jogs[j].touch(true);
                        }
                    }
                    if let Some(i) = self.scfx_bind.iter().position(|b| *b == Some((ch, note))) {
                        // Hardware semantics: pressing the lit button
                        // turns the section off; the app mirrors that
                        // state and persists it across restarts.
                        if self.scfx_on && self.scfx_type == scfx::TYPES[i] {
                            self.scfx_on = false;
                        } else {
                            self.scfx_type = scfx::TYPES[i];
                            self.scfx_on = true;
                        }
                        self.save_bindings();
                    }
                    if let Some(i) = self.pad_bind.iter().position(|v| v.contains(&(ch, note))) {
                        self.triggers.press(i, self.clock.beat());
                    }
                }
                midi::MidiEvent::NoteOff { ch, note } => {
                    for (j, bind) in self.jog_touch_bind.iter().enumerate() {
                        if *bind == Some((ch, note)) {
                            self.jogs[j].touch(false);
                        }
                    }
                    if let Some(i) = self.pad_bind.iter().position(|v| v.contains(&(ch, note))) {
                        self.triggers.release(i);
                    }
                }
                midi::MidiEvent::Clock(at) => self.midi_clock.tick(at),
                midi::MidiEvent::Start => self.clock.resync(),
            }
        }
    }

    /// External sync, one source at a time (Link > audio > MIDI clock),
    /// always as a pull on the internal clock — never a replacement, so
    /// the instrument keeps playing when a source dies.
    fn sync(&mut self) {
        if self.link_sync {
            if let Some(l) = &mut self.link {
                let (tempo, link_beat) = l.capture();
                self.clock.set_tempo(tempo);
                // Align within the 4-beat quantum, pulled hard.
                let mut err = link_beat.rem_euclid(4.0) - self.clock.phase(4.0);
                err -= (err / 4.0).round() * 4.0;
                self.clock.nudge_beats(err * 0.2);
            }
            return;
        }
        if self.audio_sync {
            let Some(est) = self.audio.as_ref().and_then(|a| a.estimate()) else {
                return;
            };
            if est.confidence < 1.3 {
                return;
            }
            let t = self.clock.tempo();
            self.clock.set_tempo(t + (est.bpm - t) * 0.05);

            let period = 60.0 / self.clock.tempo();
            let target = (est.anchor.elapsed().as_secs_f64() / period).fract();
            let mut err = target - self.clock.phase(1.0);
            err -= err.round(); // wrap into [-0.5, 0.5)
            self.clock.nudge_beats(err * 0.08);
            return;
        }
        if self.midi_clock_sync
            && let Some(bpm) = self.midi_clock.bpm()
        {
            // 0xF8 carries tempo only; the downbeat stays on SPACE.
            let t = self.clock.tempo();
            self.clock.set_tempo(t + (bpm - t) * 0.1);
        }
    }

    /// Render the active pixel effect and ship it to kitty, scaled into
    /// the stage's cell box. Called after ratatui's draw so the image
    /// lands over the blank stage cells, HUD row left as text below.
    fn render_pixels(&mut self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        let (cols, rows) = self.stage_cells;
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        // Cap the long edge; a cell is ~1:2, so the box is cols : rows*2
        // in pixel aspect. Keep circles round by matching that ratio.
        let long = 720u32;
        let (pw, ph) = if cols as u32 >= rows as u32 * 2 {
            (long, (long * rows as u32 * 2 / cols as u32).max(1))
        } else {
            ((long * cols as u32 / (rows as u32 * 2)).max(1), long)
        };
        self.gfx_fb.resize(pw, ph);
        let beat = self.triggers.warp_beat(self.clock.beat());
        PIX_FX[self.pix].1(&mut self.gfx_fb, beat, self.intensity);
        // z=-1: below the cell layer, so default-background cells show it
        // through and the two tiers composite in one pass.
        graphics::transmit_placed(out, &self.gfx_fb, 1, cols, rows, -1)
    }

    fn draw(&mut self, frame: &mut Frame) {
        // HUD hidden: the stage takes the whole screen — a clean feed
        // for the HDMI projector. The operator's info lives on the last
        // row otherwise.
        let hud_h = if self.hud_visible { 1 } else { 0 };
        let [stage, hud] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(hud_h)]).areas(frame.area());

        let beat = self.clock.beat();
        // Backspin bends the beat the effects see; the jogs scrub it
        // continuously on top. The clock itself keeps real time.
        let vbeat = self.triggers.warp_beat(beat) + self.jog_offset();
        let ctx = FrameCtx {
            beat: vbeat,
            phase: vbeat.rem_euclid(1.0),
            bar_phase: vbeat.rem_euclid(4.0),
            intensity: self.intensity,
            drive: self.drive,
        };

        // Graphics tier composites UNDER the cells: the pixel image goes
        // to z=-1 after this draw (see the main loop), and cells left at
        // the default background show it through. So the cell mixer always
        // runs — sparse effects (rain, sparks, pulse glyphs) let the pixel
        // tier through, opaque ones (plate halfblock) cover it.
        self.stage_cells = (stage.width, stage.height);

        // Mix the channels like a mixer sums audio: per cell, every
        // channel that drew something enters a lottery weighted by its
        // fader, with the leftover weight going to background. Four
        // faders at full = a quarter of the cells each; one fader alone
        // at 30% = 30% of its cells. The per-cell hash is fixed, so the
        // allocation is stable frame to frame instead of boiling.
        let active: Vec<usize> = (0..CHANNELS)
            .filter(|&i| self.channels[i].level > 0.004)
            .collect();
        if self.scratch.len() != CHANNELS || self.scratch[0].area != stage {
            self.scratch = (0..CHANNELS)
                .map(|_| ratatui::buffer::Buffer::empty(stage))
                .collect();
        }
        for &i in &active {
            self.scratch[i].reset();
            let slot = self.channels[i].slot;
            self.effects[slot].render(&mut self.scratch[i], stage, &ctx);
        }
        for y in 0..stage.height {
            for x in 0..stage.width {
                let (ax, ay) = (stage.x + x, stage.y + y);
                // Contenders: active channels whose effect touched this
                // cell, top channel first.
                let mut wsum = 0.0;
                let mut parts: [(usize, f64); CHANNELS] = [(0, 0.0); CHANNELS];
                let mut n = 0;
                for &i in active.iter().rev() {
                    let cell = &self.scratch[i][(ax, ay)];
                    if cell.symbol() == " " && cell.bg == Color::Reset {
                        continue; // untouched = transparent
                    }
                    let l = self.channels[i].level;
                    parts[n] = (i, l);
                    n += 1;
                    wsum += l;
                }
                if n == 0 {
                    continue;
                }
                let bg = (1.0 - wsum).max(0.0);
                let r = rng::unit_f64(rng::hash3(x as u64, y as u64, 77)) * (wsum + bg);
                let mut acc = 0.0;
                for &(i, l) in &parts[..n] {
                    acc += l;
                    if r < acc {
                        let mut cell = self.scratch[i][(ax, ay)].clone();
                        // The winning channel's COLOR knob shades its
                        // cells — only while the SCFX section is lit.
                        {
                            cell.fg = looks::apply(
                                cell.fg,
                                self.look,
                                &self.drive,
                                vbeat,
                                y,
                                self.hue_base,
                                self.accent,
                            );
                            cell.bg = looks::apply(
                                cell.bg,
                                self.look,
                                &self.drive,
                                vbeat,
                                y,
                                self.hue_base,
                                self.accent,
                            );
                        }
                        if self.scfx_on {
                            scfx::apply(
                                &mut cell,
                                self.scfx_type,
                                self.channels[i].color,
                                vbeat,
                                x,
                                y,
                                stage.width,
                            );
                        }
                        frame.buffer_mut()[(ax, ay)] = cell;
                        break;
                    }
                }
                // r beyond every contender lands in the background share.
            }
        }
        triggers::beat_hits(
            frame.buffer_mut(),
            stage,
            &self.drive,
            vbeat,
            self.intensity,
            self.accent,
        );
        self.triggers.post(frame.buffer_mut(), stage, vbeat);
        self.render_lyrics(frame.buffer_mut(), stage, &ctx);
        self.overlay.render(frame.buffer_mut(), stage, &ctx);

        if !self.hud_visible {
            return;
        }

        let bar = (beat / 4.0).floor() as i64 + 1;
        let beat_in_bar = ctx.bar_phase as i64 + 1;
        let status = self.effects[self.channels[self.focus].slot]
            .status()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        let mut trig_seg = self.triggers.hud();
        if self.jogs.iter().any(|j| j.active()) {
            trig_seg.push_str(&format!(" ↻{:+.2}", self.jog_offset()));
        }
        let look_seg = if self.look != looks::Look::Plain {
            format!("{} ", self.look.name())
        } else {
            String::new()
        };
        let gfx_seg = if self.gfx {
            format!("▓{} ", PIX_FX[self.pix].0)
        } else {
            String::new()
        };
        let decks: String = format!(
            "{look_seg}{gfx_seg}{}",
            self.channels
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mark = if i == self.focus { '*' } else { ' ' };
                    let col = if (c.color - 0.5).abs() > 0.04 {
                        format!("~{:.0}", c.color * 100.0)
                    } else {
                        String::new()
                    };
                    format!(
                        "{mark}{}:{}·{:.0}{col}",
                        i + 1,
                        self.effects[c.slot].name(),
                        c.level * 100.0
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let aud = if let Some(e) = &self.audio_err {
            format!(" │ AUD! {e}")
        } else if let Some(a) = &self.audio {
            let dev: String = a.device.chars().take(12).collect();
            let (lo, hi) = self.bpm_window;
            match (self.audio_sync, a.estimate()) {
                (true, Some(est)) => {
                    format!(
                        " │ ♪{dev} {:>5.1} c{:.1} [{lo}-{hi}]",
                        est.bpm, est.confidence
                    )
                }
                (true, None) => format!(" │ ♪{dev} ... [{lo}-{hi}]"),
                (false, _) => format!(" │ ♪{dev} off [{lo}-{hi}]"),
            }
        } else {
            String::new()
        };
        let lnk = match (&self.link, self.link_sync) {
            (Some(l), true) => format!(" │ ⇄Link {}p", l.peers()),
            (Some(_), false) => " │ ⇄ off".to_string(),
            (None, _) => String::new(),
        };
        let mid = if let Some(m) = &self.midi {
            let port: String = m
                .ports
                .first()
                .map(|p| p.chars().take(12).collect())
                .unwrap_or_default();
            let cc = match self.last_cc {
                Some((ch, cc, val)) => format!(" cc{ch}.{cc}={val}"),
                None => String::new(),
            };
            let note = match self.last_note {
                Some((ch, n)) => format!(" nt{ch}.{n}"),
                None => String::new(),
            };
            let mut bind = String::new();
            if let Some((ch, cc)) = self.cc_bind_int {
                bind.push_str(&format!(" int←cc{ch}.{cc}"));
            }
            for (i, c) in self.channels.iter().enumerate() {
                if let Some((ch, cc)) = c.cc {
                    bind.push_str(&format!(" c{}←cc{ch}.{cc}", i + 1));
                }
            }
            let learn = match (self.learn, self.pad_learn) {
                (_, Some(i)) => format!(" LEARN→pad.{}", triggers::PAD_NAMES[i]),
                (LearnTarget::Off, _) => String::new(),
                (LearnTarget::Intensity, _) => " LEARN→int".to_string(),
                (LearnTarget::Ch(i), _) => format!(" LEARN→ch{}", i + 1),
            };
            let mclk = match (self.midi_clock_sync, self.midi_clock.bpm()) {
                (true, Some(b)) => format!(" ♻{b:.1}"),
                (true, None) => " ♻--".to_string(),
                _ => String::new(),
            };
            format!(" │ M:{port}{cc}{note}{bind}{learn}{mclk}")
        } else {
            String::new()
        };
        let hud_text = format!(
            " {:>6.1} BPM │ {:>3}.{} │ {}{}{} │ SC:{}{}{}{}{} │ int {:>3.0}% │ {:>4.1}ms │ SPC ± ↑↓ TAB ←→ 1-{} g p w x b o h y k a r m c l q",
            self.clock.tempo(),
            bar,
            beat_in_bar,
            decks,
            status,
            trig_seg,
            if self.scfx_on {
                self.scfx_type.name()
            } else {
                "OFF"
            },
            aud,
            lnk,
            mid,
            self.ai.hud(),
            self.intensity * 100.0,
            self.render_ms,
            self.effects.len(),
        );
        frame.render_widget(
            Paragraph::new(Line::from(hud_text)).style(Style::new().fg(Color::DarkGray)),
            hud,
        );
    }
}

/// Kills the caffeinate child when the app exits.
#[cfg(target_os = "macos")]
struct CaffeinateGuard(std::process::Child);

#[cfg(target_os = "macos")]
impl Drop for CaffeinateGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

fn main() -> std::io::Result<()> {
    let assets_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets".to_string());
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "KITTY-VJ".to_string());
    let plates = assets::load(std::path::Path::new(&assets_dir));
    // Optional third arg names a lyric file: lyrics/<name>.lrc
    let lyrics = match std::env::args().nth(3) {
        Some(name) => lyrics::Lyrics::load(std::path::Path::new("lyrics"), &name),
        None => lyrics::Lyrics::new(Vec::new()),
    };

    // Keep the display awake for the length of the set — a projector
    // going to sleep mid-show is the classic HDMI gig failure. Dies
    // with us since it's a child process.
    #[cfg(target_os = "macos")]
    let _caffeinate = std::process::Command::new("caffeinate")
        .arg("-d")
        .spawn()
        .map(CaffeinateGuard);

    let mut terminal = ratatui::init();
    // kitty's keyboard protocol: real release events, which momentary
    // pad-fallback keys (F1-F8) need. Harmless no-op elsewhere.
    let _ = crossterm::execute!(
        std::io::stdout(),
        event::PushKeyboardEnhancementFlags(event::KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
    );
    let mut app = App::new(plates, text, lyrics);

    let mut last = Instant::now();
    let mut acc = 0.0_f64;
    let mut prev_frame = Instant::now();
    let mut gfx_was_on = false;

    let result = loop {
        if app.quit {
            break Ok(());
        }

        // Drain input.
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                match key.kind {
                    KeyEventKind::Press | KeyEventKind::Repeat => app.on_key(key.code),
                    KeyEventKind::Release => app.on_key_release(key.code),
                }
            }
        }

        // Advance the clock in whole fixed ticks.
        let now = Instant::now();
        acc += now.duration_since(last).as_secs_f64();
        last = now;
        while acc >= TICK {
            app.clock.advance(TICK);
            acc -= TICK;
        }
        app.consume_ai();
        app.process_midi();
        let frame_dt = now.duration_since(prev_frame).as_secs_f64();
        app.tick_jogs(frame_dt);
        app.tick_drive(frame_dt);
        prev_frame = now;
        app.sync();

        let t0 = Instant::now();
        terminal.draw(|f| app.draw(f))?;
        // Graphics tier: place the pixel stage over the blank stage cells
        // ratatui just drew. Leaving gfx mode clears the image once.
        if app.gfx {
            app.render_pixels(&mut std::io::stdout().lock())?;
            gfx_was_on = true;
        } else if gfx_was_on {
            let _ = graphics::clear_all(&mut std::io::stdout().lock());
            gfx_was_on = false;
        }
        let draw_ms = t0.elapsed().as_secs_f64() * 1000.0;
        app.render_ms = if app.render_ms == 0.0 {
            draw_ms
        } else {
            app.render_ms * 0.9 + draw_ms * 0.1
        };

        // Sleep out the remainder of the frame budget, keeping input latency low.
        let spent = t0.elapsed();
        if spent < FRAME && event::poll(FRAME - spent)? {
            // Input arrived — loop immediately to handle it.
        }
    };

    let _ = crossterm::execute!(std::io::stdout(), event::PopKeyboardEnhancementFlags);
    ratatui::restore();
    result
}
