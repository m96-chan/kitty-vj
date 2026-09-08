mod assets;
mod audio;
mod clock;
mod config;
mod effects;
mod font;
mod link;
mod midi;
mod rng;

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

/// A mixer channel: an effect slot and a vertical fader.
struct Channel {
    slot: usize,
    level: f64,
    cc: Option<(u8, u8)>,
}

const CHANNELS: usize = 4;

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
    scratch: Vec<ratatui::buffer::Buffer>,
    intensity: f64,
    render_ms: f64, // EWMA of draw time
    quit: bool,
}

impl App {
    fn new(plates: Vec<assets::Plate>, text: String) -> Self {
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
            }),
            focus: 0,
            scratch: Vec::new(),
            intensity: 0.5,
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
            KeyCode::Char('a') => {
                if self.audio.is_none() {
                    match audio::AudioBeat::start() {
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

    /// Persist current CC bindings to the gig config next to the app.
    fn save_bindings(&self) {
        let b = config::Bindings {
            intensity: self.cc_bind_int,
            channels: std::array::from_fn(|i| self.channels[i].cc),
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
                    for c in &mut self.channels {
                        if c.cc == Some((ch, cc)) {
                            c.level = val as f64 / 127.0;
                        }
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

    fn draw(&mut self, frame: &mut Frame) {
        let [stage, hud] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(frame.area());

        let beat = self.clock.beat();
        let ctx = FrameCtx {
            beat,
            phase: self.clock.phase(1.0),
            bar_phase: self.clock.phase(4.0),
            intensity: self.intensity,
        };

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
                        frame.buffer_mut()[(ax, ay)] = self.scratch[i][(ax, ay)].clone();
                        break;
                    }
                }
                // r beyond every contender lands in the background share.
            }
        }
        self.overlay.render(frame.buffer_mut(), stage, &ctx);

        let bar = (beat / 4.0).floor() as i64 + 1;
        let beat_in_bar = ctx.bar_phase as i64 + 1;
        let status = self.effects[self.channels[self.focus].slot]
            .status()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        let decks: String = self
            .channels
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mark = if i == self.focus { '*' } else { ' ' };
                format!(
                    "{mark}{}:{}·{:.0}",
                    i + 1,
                    self.effects[c.slot].name(),
                    c.level * 100.0
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let aud = if let Some(e) = &self.audio_err {
            format!(" │ AUD! {e}")
        } else if let Some(a) = &self.audio {
            let dev: String = a.device.chars().take(12).collect();
            match (self.audio_sync, a.estimate()) {
                (true, Some(est)) => {
                    format!(" │ ♪{dev} {:>5.1} c{:.1}", est.bpm, est.confidence)
                }
                (true, None) => format!(" │ ♪{dev} ..."),
                (false, _) => format!(" │ ♪{dev} off"),
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
            let mut bind = String::new();
            if let Some((ch, cc)) = self.cc_bind_int {
                bind.push_str(&format!(" int←cc{ch}.{cc}"));
            }
            for (i, c) in self.channels.iter().enumerate() {
                if let Some((ch, cc)) = c.cc {
                    bind.push_str(&format!(" c{}←cc{ch}.{cc}", i + 1));
                }
            }
            let learn = match self.learn {
                LearnTarget::Off => String::new(),
                LearnTarget::Intensity => " LEARN→int".to_string(),
                LearnTarget::Ch(i) => format!(" LEARN→ch{}", i + 1),
            };
            let mclk = match (self.midi_clock_sync, self.midi_clock.bpm()) {
                (true, Some(b)) => format!(" ♻{b:.1}"),
                (true, None) => " ♻--".to_string(),
                _ => String::new(),
            };
            format!(" │ M:{port}{cc}{bind}{learn}{mclk}")
        } else {
            String::new()
        };
        let hud_text = format!(
            " {:>6.1} BPM │ {:>3}.{} │ {}{}{}{}{} │ int {:>3.0}% │ {:>4.1}ms │ SPC ± ↑↓ TAB ←→ 1-{} o a m c l q",
            self.clock.tempo(),
            bar,
            beat_in_bar,
            decks,
            status,
            aud,
            lnk,
            mid,
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

fn main() -> std::io::Result<()> {
    let assets_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets".to_string());
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "KITTY-VJ".to_string());
    let plates = assets::load(std::path::Path::new(&assets_dir));

    let mut terminal = ratatui::init();
    let mut app = App::new(plates, text);

    let mut last = Instant::now();
    let mut acc = 0.0_f64;

    let result = loop {
        if app.quit {
            break Ok(());
        }

        // Drain input.
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                app.on_key(key.code);
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
        app.process_midi();
        app.sync();

        let t0 = Instant::now();
        terminal.draw(|f| app.draw(f))?;
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

    ratatui::restore();
    result
}
