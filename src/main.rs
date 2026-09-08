mod assets;
mod audio;
mod clock;
mod effects;
mod font;
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
    current: usize,
    /// Effect switch requested; applied on the next bar line.
    pending: Option<usize>,
    intensity: f64,
    render_ms: f64, // EWMA of draw time
    quit: bool,
}

impl App {
    fn new(plates: Vec<assets::Plate>, text: String) -> Self {
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
            current: 0,
            pending: None,
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
            KeyCode::Tab => {
                self.pending = Some((self.current + 1) % self.effects.len());
            }
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
            }
            KeyCode::Char(c @ '1'..='9') => {
                let i = (c as usize) - ('1' as usize);
                if i < self.effects.len() {
                    self.pending = Some(i);
                }
            }
            _ => {
                // Aim at the queued effect if a switch is pending — the
                // operator is already playing the thing they just picked.
                let target = self.pending.unwrap_or(self.current);
                self.effects[target].on_key(code);
            }
        }
    }

    /// Soft PLL: slew tempo and pull phase toward the audio estimate.
    /// Runs once per rendered frame; converges in about a second.
    fn audio_pll(&mut self) {
        if !self.audio_sync {
            return;
        }
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
    }

    /// Apply a pending effect switch when the bar line passes.
    fn maybe_switch(&mut self, prev_beat: f64) {
        if let Some(next) = self.pending {
            let crossed_bar = (self.clock.beat() / 4.0).floor() > (prev_beat / 4.0).floor();
            // Immediate if we're idle at the very start; otherwise on the bar.
            if crossed_bar || self.clock.beat() < 0.01 {
                self.current = next;
                self.pending = None;
            }
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
        self.effects[self.current].render(frame.buffer_mut(), stage, &ctx);
        self.overlay.render(frame.buffer_mut(), stage, &ctx);

        let bar = (beat / 4.0).floor() as i64 + 1;
        let beat_in_bar = ctx.bar_phase as i64 + 1;
        let pending = self
            .pending
            .map(|i| format!(" → {}", self.effects[i].name()))
            .unwrap_or_default();
        let status = self.effects[self.current]
            .status()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
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
        let hud_text = format!(
            " {:>6.1} BPM │ {:>3}.{} │ {}{}{}{} │ int {:>3.0}% │ {:>4.1}ms │ SPACE tap  ±bpm  ↑↓ int  1-{}/TAB fx  o txt  a aud  q quit",
            self.clock.tempo(),
            bar,
            beat_in_bar,
            self.effects[self.current].name(),
            status,
            pending,
            aud,
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
        let prev_beat = app.clock.beat();
        while acc >= TICK {
            app.clock.advance(TICK);
            acc -= TICK;
        }
        app.audio_pll();
        app.maybe_switch(prev_beat);

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
