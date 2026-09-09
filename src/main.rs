mod assets;
mod audio;
mod camera;
mod capture;
mod clock;
mod combo;
mod config;
mod drive;
mod effects;
mod font;
mod framing;
mod generate;
mod graphics;
mod jog;
mod link;
mod looks;
mod lyrics;
mod meshcube;
mod meshspeaker;
mod meshwire;
mod midi;
mod pass;
mod pixfx;
mod pixparticles;
mod pixpost;
mod postfx;
mod prolink;
mod raster;
mod rng;
mod scene;
mod scfx;
mod show;
mod source;
mod transition;
mod triggers;
mod units;

use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use clock::{ClockSource, InternalClock, TapTempo};
use effects::{
    CamFx, Collapse, Cube, Effect, FrameCtx, ImgDust, PlateFx, Pulse, Rain, Sparks, TextOverlay,
    Tunnel,
};
use pass::ColorPass;

#[derive(Clone, Copy, PartialEq)]
enum LearnTarget {
    Off,
    Intensity,
    /// One of the four channel faders.
    Ch(usize),
}

/// A mixer channel: an effect slot, a vertical fader, and a COLOR knob.
struct Channel {
    slot: units::Unit,
    level: f64,
    cc: Option<(u8, u8)>,
    /// Sound Color FX knob, 0..1, 0.5 = neutral.
    color: f64,
    color_cc: Option<(u8, u8)>,
}

const CHANNELS: usize = 4;

/// Every playable unit, cells and pixels alike, in the order the digit
/// keys and TAB walk them. Built at startup because how many cell
/// effects exist depends on whether there are plates.
///
/// Names must be unique across the whole list: the scene director names
/// its choices as strings, and a collision would resolve to whichever
/// entry came first — silently, and only for the scenes that picked the
/// loser.
/// Construct the cell effects, in the order `effects::CELL_NAMES`
/// documents. One function so the app and the bridge test cannot drift.
fn build_effects(
    plates: std::rc::Rc<Vec<assets::Plate>>,
    cap: std::rc::Rc<std::cell::RefCell<Option<capture::Capture>>>,
) -> Vec<Box<dyn Effect>> {
    let mut effects: Vec<Box<dyn Effect>> = vec![
        Box::new(Pulse),
        Box::new(Rain),
        Box::new(Tunnel),
        Box::new(Collapse),
        Box::new(Cube::new(plates.clone())),
        Box::new(Sparks),
    ];
    if !plates.is_empty() {
        effects.push(Box::new(ImgDust::new(plates.clone())));
        effects.push(Box::new(PlateFx::new(plates)));
    }
    effects.push(Box::new(CamFx::new(cap)));
    effects
}

/// Visit a slot's drawable parts with the share of the fader each one
/// takes: a base unit is itself at full share, a combo is its recipe.
/// One level of expansion — combos hold base units by construction.
fn each_part(
    combos: &[combo::Combo],
    slot: units::Unit,
    mut f: impl FnMut(units::Unit, f64),
) {
    match slot {
        units::Unit::Combo(i) => {
            for &(u, w) in &combos[i].parts {
                f(u, w);
            }
        }
        u => f(u, 1.0),
    }
}

fn build_unit_list(n_cell: usize, cell_names: &[&'static str]) -> Vec<(&'static str, units::Unit)> {
    let mut v: Vec<(&'static str, units::Unit)> = (0..n_cell)
        .map(|i| (cell_names[i], units::Unit::Cell(i)))
        .collect();
    v.extend(
        units::PIX_UNITS
            .iter()
            .map(|(n, p)| (*n, units::Unit::Pixel(*p))),
    );
    v
}

/// What feeds the pixel tier. A generated field is a function of beat;
/// an image source is a function of position, and `source.rs` draws
/// those for both tiers so the camera cannot drift between them.
/// Cell-grid post passes from the ported set, cycled with '\''.
#[derive(Clone, Copy, PartialEq)]
enum CellPost {
    None,
    Slice,
    Glitch,
    MirrorV,
    MirrorQuad,
    Pixelate,
    ZoomPunch,
    Shake,
}

const CELL_POSTS: [(&str, CellPost); 8] = [
    ("-", CellPost::None),
    ("SLICE", CellPost::Slice),
    ("GLITCH", CellPost::Glitch),
    ("MIRV", CellPost::MirrorV),
    ("MIRQ", CellPost::MirrorQuad),
    ("PIXEL", CellPost::Pixelate),
    ("ZPUNCH", CellPost::ZoomPunch),
    ("SHAKE", CellPost::Shake),
];

/// Pixel post chain entries, cycled with 'P'.
#[derive(Clone, Copy, PartialEq)]
enum PixPost {
    None,
    Warp,
    Kaleido,
    ZoomBlur,
    RgbSplit,
    Edge,
    Bloom,
    Crt,
    Feedback,
}

const PIX_POSTS: [(&str, PixPost); 9] = [
    ("-", PixPost::None),
    ("WARP", PixPost::Warp),
    ("KALEID", PixPost::Kaleido),
    ("ZBLUR", PixPost::ZoomBlur),
    ("RGB", PixPost::RgbSplit),
    ("EDGE", PixPost::Edge),
    ("BLOOM", PixPost::Bloom),
    ("CRT", PixPost::Crt),
    ("FEEDBK", PixPost::Feedback),
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
    /// Scene director — orchestration: it decides which units are
    /// active, and speaks only when the scene changes.
    scenes: scene::SceneDirector,
    /// Scene flags: A/B cut on eighths, and the phrase-top stutter.
    abcut: bool,
    stutter: bool,
    /// Held frame for the stutter, and the subdivision it was taken on.
    stutter_hold: Option<ratatui::buffer::Buffer>,
    stutter_tick: i64,
    /// The visual clock, in seconds. Distinct from beat time on purpose:
    /// a hue cycle at 22 deg/s and a 20-second Ken Burns move are
    /// wall-clock gestures, and making them ride the beat would have
    /// them change speed with the tempo. Frozen while the show holds,
    /// as the original's VT was frozen during silence.
    vt: f64,
    /// Real time the last frame took, handed to effects that own state.
    frame_dt: f64,
    /// Scene changes that had to fire off-grid. A set full of these
    /// means the clock is wrong, so it is worth seeing.
    escapes: u32,
    /// Opening/closing sequences; their exports gate the rest.
    show: show::Show,
    show_state: show::ShowState,
    /// Persistent colour treatments over the whole mix, applied in
    /// order. A scene rolls one or two (45% chance of the second) —
    /// carrying only the first silently dropped half the roll. 'w'
    /// replaces the set with a single manual pick.
    looks: Vec<looks::Look>,
    /// Per-scene random hue for duotone, and the palette accent for lut.
    hue_base: f64,
    accent: (u8, u8, u8),
    accent_b: (u8, u8, u8),
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
    midi: midi::MidiIn,
    midi_clock: midi::MidiClock,
    midi_clock_sync: bool,
    cc_bind_int: Option<(u8, u8)>,
    learn: LearnTarget,
    last_cc: Option<(u8, u8, u8)>,
    link: Option<link::LinkSync>,
    link_sync: bool,
    /// Pro DJ Link — the XDJ-XZ's own network, listened to passively.
    prolink: Option<prolink::ProLink>,
    prolink_sync: bool,
    prolink_err: Option<String>,
    /// Everything a channel can hold, cells and pixels in one list.
    unit_list: Vec<(&'static str, units::Unit)>,
    /// Channel scenes — the combinations `Unit::Combo` indexes into.
    combos: Vec<combo::Combo>,
    /// The config's raw scene lines, kept verbatim so `save_bindings`
    /// (which rebuilds the file from app state on every MIDI learn)
    /// writes them back instead of eating them.
    scene_defs: Vec<(String, String)>,
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
    /// Scratch for blending one pixel unit over the picture below it.
    scratch_fb: graphics::Framebuffer,
    gfx_fb: graphics::Framebuffer,
    /// Which pixel effect (index into PIX_FX) when in graphics mode.
    /// Manual offset into the plate rotation for the pixel PLATE; the
    /// base index advances by phrase in closed form, like the cell one.
    pix_plate: usize,
    /// Ken Burns for the pixel plate, re-rolled when its plate changes.
    pix_ken: camera::KenBurns,
    pix_ken_idx: usize,
    /// Post pass over the pixel frame.
    pix_post: usize,
    /// Live capture, shared with the CAM effect. Opened on demand: a
    /// camera costs a permission prompt, so it must be asked for.
    capture: std::rc::Rc<std::cell::RefCell<Option<capture::Capture>>>,
    capture_err: Option<String>,
    /// Route camera motion into the drive signals.
    motion_drive: bool,
    /// Plates, shared with the effects that sample them — the mesh cube
    /// textures its faces from the same rotation.
    plates: std::rc::Rc<Vec<assets::Plate>>,
    /// Depth buffer for the mesh units, reused across frames so a
    /// channel change costs no allocation.
    depth: raster::DepthBuffer,
    feedback: pixpost::Feedback,
    /// Cell-grid post pass from the ported set.
    cell_post: usize,
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
        let bindings = config::load(&config::path());
        let plates = std::rc::Rc::new(plates);
        // CAM is always available as a channel slot; it simply draws
        // nothing until a capture is opened.
        let cap_shared: std::rc::Rc<std::cell::RefCell<Option<capture::Capture>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let effects = build_effects(plates.clone(), cap_shared.clone());
        let plates_shared = plates;
        let cell_names: Vec<&'static str> = effects.iter().map(|e| e.name()).collect();
        debug_assert!(
            cell_names.iter().all(|n| effects::CELL_NAMES.contains(n)),
            "an effect name is missing from effects::CELL_NAMES"
        );
        let mut unit_list = build_unit_list(effects.len(), &cell_names);
        // Channel scenes join the same list, after the base units so
        // they can never shadow one and the digit keys keep meaning
        // what they meant.
        let combos = combo::build(&bindings.scenes, &unit_list);
        for (i, c) in combos.iter().enumerate() {
            unit_list.push((c.name, units::Unit::Combo(i)));
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
            scenes: scene::SceneDirector::new(scene::Style::Neon, 1),
            vt: 0.0,
            frame_dt: 1.0 / 60.0,
            abcut: false,
            stutter: false,
            stutter_hold: None,
            stutter_tick: i64::MIN,
            escapes: 0,
            show: show::Show::new(),
            show_state: show::ShowState::neutral(),
            looks: vec![looks::Look::Plain],
            hue_base: 210.0,
            accent: (0, 255, 213),
            accent_b: (255, 255, 255),
            drive: drive::Drive::default(),
            ai: generate::Registry::default(),
            lyrics,
            jogs: Default::default(),
            jog_bind: bindings.jogs,
            jog_touch_bind: bindings.jog_touch,
            // Opened at launch and rescanned on a timer, so a controller
            // powered on later still lands.
            midi: midi::MidiIn::open(),
            midi_clock: midi::MidiClock::new(),
            midi_clock_sync: false,
            cc_bind_int: bindings.intensity,
            learn: LearnTarget::Off,
            last_cc: None,
            link: None,
            link_sync: false,
            prolink: None,
            prolink_sync: false,
            prolink_err: None,
            unit_list,
            combos,
            scene_defs: bindings.scenes.clone(),
            channels: std::array::from_fn(|i| Channel {
                slot: units::Unit::Cell(i),
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
            scratch_fb: graphics::Framebuffer::new(1, 1),
            gfx_fb: graphics::Framebuffer::new(1, 1),
            pix_plate: 0,
            pix_ken: camera::KenBurns::roll(0, 0.0),
            pix_ken_idx: usize::MAX,
            pix_post: 0,
            capture: cap_shared,
            capture_err: None,
            motion_drive: false,
            plates: plates_shared,
            depth: raster::DepthBuffer::new(1, 1),
            feedback: pixpost::Feedback::new(1, 1),
            cell_post: 0,
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
            // Digits reach the first nine units; these walk the whole
            // list, which is the only way to reach the pixel ones now
            // that both media share it.
            KeyCode::Char('.') => self.step_unit(1),
            KeyCode::Char(',') => self.step_unit(-1),
            KeyCode::Char('o') => self.overlay.toggle(),
            KeyCode::Char('h') => self.hud_visible = !self.hud_visible,
            KeyCode::Char('S') => self.scenes.request(scene::ChangeReason::Manual),
            KeyCode::Char('D') => {
                // Cycle the artwork style: it changes which pools the
                // next scene draws from, so it takes effect on the roll.
                let next = self.scenes.style().next();
                self.scenes.set_style(next);
                self.scenes.request(scene::ChangeReason::TrackChange);
            }
            KeyCode::Char('A') => self.show.arm(),
            KeyCode::Char('E') => self.show.play_out(),
            KeyCode::Char('w') => {
                let cur = self.looks.first().copied().unwrap_or(looks::Look::Plain);
                self.looks = vec![cur.next()];
                // A fresh hue each time duotone comes round.
                self.hue_base = (self.hue_base + 47.0).rem_euclid(360.0);
            }
            KeyCode::Char('C') => {
                // A camera costs a permission prompt, so it is opened on
                // demand rather than at launch.
                if self.capture.borrow().is_some() {
                    *self.capture.borrow_mut() = None;
                    self.motion_drive = false;
                } else {
                    match capture::Capture::open("0") {
                        Ok(c) => {
                            *self.capture.borrow_mut() = Some(c);
                            self.capture_err = None;
                        }
                        Err(e) => self.capture_err = Some(e),
                    }
                }
            }
            KeyCode::Char('n') if self.capture.borrow().is_some() => {
                // Let the room push the visuals: frame differencing
                // becomes a drive input, which is the one thing a live
                // source can do that a plate cannot.
                self.motion_drive = !self.motion_drive;
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
            KeyCode::Char('P') => self.pix_post = (self.pix_post + 1) % PIX_POSTS.len(),
            KeyCode::Char('\'') => self.cell_post = (self.cell_post + 1) % CELL_POSTS.len(),
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
                    self.prolink_sync = false;
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
                if self.midi.connected() {
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
                    self.prolink_sync = false;
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
                    self.prolink_sync = false;
                }
            }
            KeyCode::Char('j') => {
                // Pro DJ Link (XDJ). First press binds the ports; after
                // that it toggles. A bind failure (usually rekordbox
                // squatting 50000/50001) lands in the HUD, not a panic.
                if self.prolink.is_none() {
                    match prolink::ProLink::start() {
                        Ok(p) => {
                            self.prolink = Some(p);
                            self.prolink_sync = true;
                            self.prolink_err = None;
                        }
                        Err(e) => self.prolink_err = Some(e),
                    }
                } else {
                    self.prolink_sync = !self.prolink_sync;
                }
                if self.prolink_sync {
                    self.link_sync = false;
                    self.audio_sync = false;
                    self.midi_clock_sync = false;
                }
            }
            KeyCode::Char(c @ '1'..='9') => {
                let i = (c as usize) - ('1' as usize);
                if i < self.unit_list.len() {
                    self.channels[self.focus].slot = self.unit_list[i].1;
                }
            }
            _ => {
                // Unclaimed keys go to the focused channel's effect, if
                // that channel is holding one — a pixel unit is a plain
                // function and has nothing to press.
                if let units::Unit::Cell(i) = self.channels[self.focus].slot {
                    self.effects[i].on_key(code);
                }
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

    /// Move the focused channel through the unit list.
    fn step_unit(&mut self, d: i32) {
        let n = self.unit_list.len() as i32;
        if n == 0 {
            return;
        }
        let cur = self
            .unit_list
            .iter()
            .position(|(_, u)| *u == self.channels[self.focus].slot)
            .unwrap_or(0) as i32;
        let next = (cur + d).rem_euclid(n) as usize;
        self.channels[self.focus].slot = self.unit_list[next].1;
    }

    /// Is any channel bringing pixel parts at an audible level? That is
    /// all "the graphics tier is on" ever meant — and with combos in
    /// the list it is a question about a slot's parts, not its variant.
    fn pixel_channels_live(&self) -> bool {
        self.channels.iter().any(|c| {
            if c.level <= 0.004 {
                return false;
            }
            let mut px = false;
            each_part(&self.combos, c.slot, |u, _| {
                px |= matches!(u, units::Unit::Pixel(_));
            });
            px
        })
    }

    /// Draw one pixel-medium unit into the framebuffer. `gain` scales
    /// the units that composite; the ones that establish the picture are
    /// blended by the caller instead, since scaling their brightness is
    /// not the same as mixing them in.
    fn draw_pix_unit(&mut self, kind: units::Pix, beat: f64, gain: f64) {
        let acc = self.accent;
        let acc_b = self.accent_b;
        let i = self.intensity;
        match kind {
            units::Pix::Plasma => pixfx::plasma(&mut self.gfx_fb, beat, i, &self.drive),
            units::Pix::Tunnel => pixfx::tunnel(&mut self.gfx_fb, beat, i, &self.drive),
            units::Pix::Stars => pixfx::starfield(&mut self.gfx_fb, beat, i, &self.drive),
            units::Pix::Cam => {
                if let Some(img) = self.capture.borrow().as_ref().and_then(|c| c.latest()) {
                    source::draw_pixels(&mut self.gfx_fb, &img, self.drive.gbeat(), i, true);
                }
            }
            units::Pix::Plate => {
                // Rotation in closed form — one step per 8 bars, like
                // the cell plate — plus the manual offset. The camera
                // re-rolls when the plate changes, anchored to now.
                let n = self.plates.len().max(1);
                let idx = ((beat.max(0.0) / 32.0) as usize + self.pix_plate) % n;
                if idx != self.pix_ken_idx {
                    self.pix_ken = camera::KenBurns::roll(idx as u64, self.vt);
                    self.pix_ken_idx = idx;
                }
                if let Some(p) = self.plates.get(idx) {
                    let cam = {
                        use camera::Modulator;
                        self.pix_ken.cam(self.vt, &self.drive)
                    };
                    source::draw_pixels_cam(
                        &mut self.gfx_fb,
                        &p.img,
                        self.drive.gbeat(),
                        i,
                        false,
                        cam,
                    );
                }
            }
            units::Pix::Sparks => {
                pixparticles::sparks(&mut self.gfx_fb, beat, i * gain, &self.drive, acc, acc_b)
            }
            units::Pix::PxTunnel => {
                pixparticles::tunnel_px(&mut self.gfx_fb, beat, i * gain, &self.drive, acc, acc_b)
            }
            units::Pix::Floor => {
                pixparticles::grid_floor(&mut self.gfx_fb, beat, i * gain, &self.drive, acc, acc_b)
            }
            units::Pix::Rings => {
                pixparticles::rings(&mut self.gfx_fb, beat, i * gain, &self.drive, acc, acc_b)
            }
            units::Pix::MeshCube => meshcube::cube(
                &mut self.gfx_fb,
                &mut self.depth,
                beat,
                i,
                &self.drive,
                &self.plates,
            ),
            units::Pix::MeshWire => meshwire::wire(
                &mut self.gfx_fb,
                &mut self.depth,
                beat,
                i * gain,
                &self.drive,
                acc,
                acc_b,
            ),
            units::Pix::MeshSpeaker => meshspeaker::speaker(
                &mut self.gfx_fb,
                &mut self.depth,
                beat,
                i,
                &self.drive,
                acc,
                acc_b,
            ),
        }
    }

    /// Run the frame-wide colour passes over the pixel framebuffer, so
    /// the two tiers agree: without this, an outro fade dims the cells
    /// while the plasma underneath stays at full brightness, and no look
    /// or hit ever touches a pixel channel.
    fn apply_passes_over_fb(&mut self, vbeat: f64) {
        let looks_active = self.looks.iter().any(|l| *l != looks::Look::Plain);
        let sc = self.scenes.scene();
        let hits = triggers::HitSet {
            invert: sc.has_hit(scene::Hit::InvertFlash),
            color: sc.has_hit(scene::Hit::ColorFlash),
            strobe: sc.has_hit(scene::Hit::Strobe),
        };
        let sh = self.show_state;
        let probe = pass::CellCtx {
            drive: &self.drive,
            t: self.vt,
            beat: vbeat,
            x: 0,
            y: 0,
            w: self.gfx_fb.w.min(u16::MAX as u32) as u16,
            intensity: self.intensity,
            hue_base: self.hue_base,
            accent: self.accent,
        };
        let hits_on = ColorPass::amount(&hits, &probe) > 0.005;
        let show_on = ColorPass::amount(&sh, &probe) > 0.005;
        if !looks_active && !hits_on && !show_on {
            return;
        }
        let w = self.gfx_fb.w as usize;
        for y in 0..self.gfx_fb.h as usize {
            for x in 0..w {
                let i = 3 * (y * w + x);
                let mut c = Color::Rgb(
                    self.gfx_fb.px[i],
                    self.gfx_fb.px[i + 1],
                    self.gfx_fb.px[i + 2],
                );
                let cctx = pass::CellCtx {
                    x: x.min(u16::MAX as usize) as u16,
                    y: y.min(u16::MAX as usize) as u16,
                    ..probe
                };
                if looks_active {
                    for lk in &self.looks {
                        c = lk.map(c, &cctx);
                    }
                }
                if hits_on {
                    c = hits.map(c, &cctx);
                }
                if show_on {
                    c = sh.map(c, &cctx);
                }
                if let Color::Rgb(r, g, b) = c {
                    self.gfx_fb.px[i] = r;
                    self.gfx_fb.px[i + 1] = g;
                    self.gfx_fb.px[i + 2] = b;
                }
            }
        }
    }

    /// Beat time as the effects see it: the clock, bent by the backspin
    /// pad, scrubbed by the jogs. Derived in one place — the cell and
    /// pixel tiers diverge the moment one of them gains a term.
    fn vbeat(&self) -> f64 {
        self.triggers.warp_beat(self.clock.beat()) + self.jog_offset()
    }

    /// Run one colour pass over the whole stage. This is the mechanism
    /// every frame-wide recolour goes through — the beat hits, the show
    /// gate — so a fourth one is an impl of ColorPass, not another loop.
    /// `amount` is probed once: these passes are position-independent by
    /// contract, and a zero frame costs a single call.
    fn apply_over_stage(
        &self,
        buf: &mut ratatui::buffer::Buffer,
        stage: ratatui::layout::Rect,
        vbeat: f64,
        pass_: &dyn ColorPass,
    ) {
        let probe = pass::CellCtx {
            drive: &self.drive,
            t: self.vt,
            beat: vbeat,
            x: 0,
            y: 0,
            w: stage.width,
            intensity: self.intensity,
            hue_base: self.hue_base,
            accent: self.accent,
        };
        if pass_.amount(&probe) < 0.005 {
            return;
        }
        for y in 0..stage.height {
            for x in 0..stage.width {
                let cctx = pass::CellCtx { x, y, ..probe };
                let cell = &mut buf[(stage.x + x, stage.y + y)];
                pass_.apply(cell, &cctx);
            }
        }
    }

    /// The display name of a unit, whichever medium it is.
    fn unit_name(&self, u: units::Unit) -> &'static str {
        self.unit_list
            .iter()
            .find(|(_, x)| *x == u)
            .map(|(n, _)| *n)
            .unwrap_or("?")
    }

    /// Combined scrub from every jog, in beats.
    fn jog_offset(&self) -> f64 {
        self.jogs.iter().map(|j| j.offset()).sum()
    }

    /// The strongest gesture running on any platter, if any.
    fn jog_spin(&self) -> Option<jog::Spin> {
        self.jogs
            .iter()
            .filter(|j| j.spin() != jog::Spin::None)
            .max_by(|a, b| {
                a.spin_amount()
                    .partial_cmp(&b.spin_amount())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|j| j.spin())
    }

    /// How hard a spin is running, [0,1] — the trail it earns.
    fn jog_spin_amount(&self) -> f64 {
        self.jogs
            .iter()
            .map(|j| j.spin_amount())
            .fold(0.0_f64, f64::max)
    }

    /// Coast/recentre the wheels by real elapsed time.
    fn tick_jogs(&mut self, dt: f64) {
        for j in &mut self.jogs {
            j.tick(dt);
        }
    }

    /// Advance the orchestration and the show sequences. A scene change
    /// lands on a downbeat, so this runs before the draw that would show
    /// it; the show is wall-clock, so it takes real dt.
    fn tick_show(&mut self, dt: f64) {
        let loud = if self.audio_sync {
            self.audio
                .as_ref()
                .and_then(|a| a.estimate())
                .map(|e| (e.confidence - 1.0).clamp(0.0, 1.0))
                .unwrap_or(0.0)
        } else {
            // Without audio there is nothing to arm on, so a standby
            // card would wait forever; treat the clock as the cue.
            1.0
        };
        self.show_state = self.show.update(dt, loud);
        if !self.show_state.hold {
            self.vt += dt;
        }
        // Rotation is expressed in seconds over there, so it has to
        // follow the tempo here or a fast set would rotate twice as often.
        self.scenes.set_tempo(self.clock.tempo());
        // With a controller connected the faders are the scene change;
        // the rotation timer is autopilot for when there isn't one.
        // Hot-plug both ways: unplugging falls back to autopilot.
        self.scenes.set_auto(!self.midi.connected());
        // Automatic scene rotation is suspended outside the live stage,
        // exactly as over there — nothing may interrupt an intro.
        if self.show.is_live()
            && let Some(change) = self.scenes.update(self.clock.beat())
        {
            // A track change reads as a change of chapter and takes a
            // long transition; a rotation is an edit and takes a short
            // one. The set of transitions stays ours; the scene only
            // says which length it wants.
            if let Some(tr) = change.length.pick(&transition::TRANSITIONS)
                && let Some(i) = transition::TRANSITIONS
                    .iter()
                    .position(|t| t.name() == tr.name())
            {
                for e in &mut self.effects {
                    e.on_transition(i);
                }
            }
            if !change.on_downbeat {
                // The escape hatch fired: the grid was unreliable enough
                // that waiting for a downbeat would have stalled the set.
                self.escapes += 1;
            }
            self.apply_scene(&change.scene);
        }
    }

    /// Take a scene's cast list. The director hands over names, not
    /// indices, so the tables stay the app's business — adding an effect
    /// means adding a row here and a name there, not touching scene.rs.
    fn apply_scene(&mut self, sc: &scene::Scene) {
        self.looks = sc.looks.clone();
        self.hue_base = sc.hue_base;
        self.accent = sc.accent;
        self.accent_b = sc.accent_b;
        self.abcut = sc.abcut;
        self.stutter = sc.stutter;
        // Posts: the scene names one of each kind, or none.
        self.cell_post = 0;
        self.pix_post = 0;
        for p in &sc.posts {
            if let Some(n) = p.cell_post()
                && let Some(i) = CELL_POSTS.iter().position(|(name, _)| *name == n)
            {
                self.cell_post = i;
            }
            if let Some(n) = p.pix_post()
                && let Some(i) = PIX_POSTS.iter().position(|(name, _)| *name == n)
            {
                self.pix_post = i;
            }
        }
        // The cast: which unit each of the three lower channels holds.
        // This is the combination itself — the parts are the ported
        // effects, the scene is the recipe. A name that doesn't resolve
        // (no plates loaded) leaves the operator's unit in place, and a
        // manual pick made mid-scene lives until the next change stomps
        // it, same as every other scene choice.
        for (i, name) in sc.cast.iter().enumerate() {
            if let Some(&(_, u)) = self.unit_list.iter().find(|(n, _)| n == name) {
                self.channels[i].slot = u;
            }
        }
        // A scene naming a particle mode puts it on the top channel, so
        // its choice lands somewhere an operator can see and override.
        if let Some(n) = sc.particle.name()
            && let Some(&(_, u)) = self.unit_list.iter().find(|(name, _)| *name == n)
        {
            self.channels[CHANNELS - 1].slot = u;
        }
        // A hit that wants a cell-post slot takes it if the posts left
        // one free — beat-momentary treatments read louder than a
        // persistent one, so they win the slot.
        for h in &sc.hits {
            if let Some(n) = h.cell_post()
                && let Some(i) = CELL_POSTS.iter().position(|(name, _)| *name == n)
            {
                self.cell_post = i;
            }
        }
        // Units that care about the framing hear about it directly.
        let fit = match sc.fit {
            scene::Fit::Cover => framing::Fit::Cover,
            scene::Fit::Contain => framing::Fit::Contain,
        };
        for e in &mut self.effects {
            e.on_scene(fit, sc.seed);
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
        let motion = if self.motion_drive {
            self.capture
                .borrow()
                .as_ref()
                .map(|c| drive::MotionDrive { energy: c.motion() })
        } else {
            None
        };
        let beat = self.clock.beat();
        self.drive.update(dt, beat, audio, motion);
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
            scenes: self.scene_defs.clone(),
        };
        let _ = config::save(&config::path(), &b);
    }

    /// Drain MIDI: CC learn/binding, clock ticks, transport.
    fn process_midi(&mut self) {
        // Look for controllers that appeared or vanished. Doing it here
        // keeps the timer on the render loop instead of a thread.
        if self.midi.due() {
            self.midi.rescan();
        }
        for ev in self.midi.drain() {
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

    /// External sync, one source at a time (Pro DJ Link > Link > audio >
    /// MIDI clock), always as a pull on the internal clock — never a
    /// replacement, so the instrument keeps playing when a source dies.
    fn sync(&mut self) {
        if self.prolink_sync {
            let Some(p) = &self.prolink else { return };
            // Beat packets arrive ON the beat, so the arrival instant is
            // the beat instant and the packet says which beat of the bar
            // it was. A stale beat (deck paused, track ended) pulls
            // nothing — the internal clock free-runs at the last tempo.
            let Some((ev, at, _)) = p.beat() else { return };
            let age = at.elapsed().as_secs_f64();
            if age > 4.0 {
                return;
            }
            let t = self.clock.tempo();
            self.clock.set_tempo(t + (ev.bpm - t) * 0.2);

            let period = 60.0 / self.clock.tempo();
            let target = (ev.beat_in_bar - 1) as f64 + age / period;
            let mut err = target.rem_euclid(4.0) - self.clock.phase(4.0);
            err -= (err / 4.0).round() * 4.0;
            self.clock.nudge_beats(err * 0.2);
            return;
        }
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
        let beat = self.vbeat();

        // Pixel-medium parts, channels bottom to top, each scaled by
        // fader × its share of the slot (a combo brings several parts,
        // a base unit brings itself at full share). A unit that
        // establishes the picture is blended in; one that composites
        // (particles, the wire rig) is drawn straight on top, because
        // that is what additive means.
        self.gfx_fb.px.fill(0);
        if self.scratch_fb.w != pw || self.scratch_fb.h != ph {
            self.scratch_fb = graphics::Framebuffer::new(pw, ph);
        }
        let mut jobs: Vec<(usize, units::Pix, f64)> = Vec::new();
        for i in 0..CHANNELS {
            let level = self.channels[i].level;
            if level <= 0.004 {
                continue;
            }
            each_part(&self.combos, self.channels[i].slot, |u, w| {
                if let units::Unit::Pixel(kind) = u {
                    jobs.push((i, kind, level * w));
                }
            });
        }
        for (i, kind, level) in jobs {
            // Solid geometry ducks what is already there so it carries
            // the frame, exactly as the mode used to.
            let dim = kind.dim_behind();
            if dim < 0.999 {
                for p in self.gfx_fb.px.iter_mut() {
                    *p = (*p as f64 * dim) as u8;
                }
            }
            if kind.additive() {
                // Composites over the frame, so it can draw straight in
                // and the fader scales how hard it hits. The show's
                // particles export rides along: the intro fades the
                // fields in and the outro bursts then collapses them.
                self.draw_pix_unit(kind, beat, level * self.show_state.particles);
            } else {
                self.scratch_fb.px.fill(0);
                std::mem::swap(&mut self.gfx_fb, &mut self.scratch_fb);
                self.draw_pix_unit(kind, beat, 1.0);
                std::mem::swap(&mut self.gfx_fb, &mut self.scratch_fb);
                // The channel's COLOR knob shades its own contribution,
                // exactly as it shades a cell channel's winning cells —
                // parity the knob did not have on pixel units before.
                if self.scfx_on && (self.channels[i].color - 0.5).abs() > 0.04 {
                    let sc = scfx::Scfx {
                        kind: self.scfx_type,
                        knob: self.channels[i].color,
                    };
                    let w = self.scratch_fb.w as usize;
                    let probe = pass::CellCtx {
                        drive: &self.drive,
                        t: self.vt,
                        beat,
                        x: 0,
                        y: 0,
                        w: self.scratch_fb.w.min(u16::MAX as u32) as u16,
                        intensity: self.intensity,
                        hue_base: self.hue_base,
                        accent: self.accent,
                    };
                    for y in 0..self.scratch_fb.h as usize {
                        for x in 0..w {
                            let px = 3 * (y * w + x);
                            let c = Color::Rgb(
                                self.scratch_fb.px[px],
                                self.scratch_fb.px[px + 1],
                                self.scratch_fb.px[px + 2],
                            );
                            let cctx = pass::CellCtx {
                                x: x.min(u16::MAX as usize) as u16,
                                y: y.min(u16::MAX as usize) as u16,
                                ..probe
                            };
                            if let Color::Rgb(r, g, b) = sc.map(c, &cctx) {
                                self.scratch_fb.px[px] = r;
                                self.scratch_fb.px[px + 1] = g;
                                self.scratch_fb.px[px + 2] = b;
                            }
                        }
                    }
                }
                // Blend it over the picture that was there, weighted by
                // the fader.
                let k = level.clamp(0.0, 1.0);
                for (dst, src) in self.gfx_fb.px.iter_mut().zip(self.scratch_fb.px.iter()) {
                    *dst = (*dst as f64 * (1.0 - k) + *src as f64 * k) as u8;
                }
            }
        }

        let t = self.vt;
        match PIX_POSTS[self.pix_post].1 {
            PixPost::None => {}
            PixPost::Warp => pixpost::warp(&mut self.gfx_fb, &self.drive, t, self.intensity),
            PixPost::Kaleido => pixpost::kaleido(&mut self.gfx_fb, &self.drive, t),
            PixPost::ZoomBlur => pixpost::zoomblur(&mut self.gfx_fb, &self.drive, self.intensity),
            PixPost::RgbSplit => pixpost::rgb_split(&mut self.gfx_fb, &self.drive, self.intensity),
            PixPost::Edge => {
                pixpost::edge(&mut self.gfx_fb, &self.drive, self.accent, self.accent_b)
            }
            PixPost::Bloom => pixpost::bloom(&mut self.gfx_fb, &self.drive, self.intensity),
            PixPost::Crt => pixpost::crt(&mut self.gfx_fb, 0.55),
            PixPost::Feedback => self
                .feedback
                .apply(&mut self.gfx_fb, &self.drive, self.intensity),
        }
        // The frame-wide treatments and the pad passes land here too, or
        // the layer under the cells tells a different story from them.
        self.apply_passes_over_fb(beat);
        self.triggers.post_fb(&mut self.gfx_fb, beat);
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
        let vbeat = self.vbeat();
        let ctx = FrameCtx {
            beat: vbeat,
            dt: self.frame_dt,
            vt: self.vt,
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
        // Only the cell-medium parts enter the lottery — a combo brings
        // its cell parts here and its pixel parts to render_pixels, at
        // fader × share each. Pixel parts land under the cells instead.
        // (channel, effect, gain), channels bottom to top.
        let mut contrib: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..CHANNELS {
            let level = self.channels[i].level;
            if level <= 0.004 {
                continue;
            }
            each_part(&self.combos, self.channels[i].slot, |u, w| {
                if let units::Unit::Cell(e) = u {
                    contrib.push((i, e, level * w));
                }
            });
        }
        // Four channels of eight-part combos at most — the lottery's
        // contender array is sized against this.
        contrib.truncate(CHANNELS * combo::MAX_PARTS);
        let need = contrib.len().max(1);
        if self.scratch.len() < need || self.scratch[0].area != stage {
            self.scratch = (0..need)
                .map(|_| ratatui::buffer::Buffer::empty(stage))
                .collect();
        }
        // One scratch per contribution, not per channel: two parts of
        // one combo need two buffers.
        for (k, &(_, e, _)) in contrib.iter().enumerate() {
            self.scratch[k].reset();
            self.effects[e].render(&mut self.scratch[k], stage, &ctx);
        }
        for y in 0..stage.height {
            for x in 0..stage.width {
                let (ax, ay) = (stage.x + x, stage.y + y);
                // Contenders: contributions that touched this cell, top
                // channel's parts first. (scratch idx, channel, gain).
                let mut wsum = 0.0;
                let mut parts: [(usize, usize, f64); CHANNELS * combo::MAX_PARTS] =
                    [(0, 0, 0.0); CHANNELS * combo::MAX_PARTS];
                let mut n = 0;
                for (k, &(i, _, gain)) in contrib.iter().enumerate().rev() {
                    let cell = &self.scratch[k][(ax, ay)];
                    if cell.symbol() == " " && cell.bg == Color::Reset {
                        continue; // untouched = transparent
                    }
                    parts[n] = (k, i, gain);
                    n += 1;
                    wsum += gain;
                }
                if n == 0 {
                    continue;
                }
                let bg = (1.0 - wsum).max(0.0);
                // ABCUT — on eighths the mix re-rolls against a different
                // salt, so the picture hard-cuts between the same set of
                // channels instead of holding one allocation.
                let salt = if self.abcut && self.drive.groove > 0.5 {
                    77 + (vbeat * 2.0).floor().rem_euclid(2.0) as u64
                } else {
                    77
                };
                let r = rng::unit_f64(rng::hash3(x as u64, y as u64, salt)) * (wsum + bg);
                let mut acc = 0.0;
                for &(k, i, l) in &parts[..n] {
                    acc += l;
                    if r < acc {
                        let mut cell = self.scratch[k][(ax, ay)].clone();
                        // The winning channel's COLOR knob shades its
                        // cells — only while the SCFX section is lit.
                        // Both of these are the same shape — a colour in,
                        // a colour out — so they run through one trait.
                        // A look is a scene decision, an SCFX pass is a
                        // knob; the pipeline doesn't care which.
                        let cctx = pass::CellCtx {
                            drive: &self.drive,
                            t: self.vt,
                            beat: vbeat,
                            x,
                            y,
                            w: stage.width,
                            intensity: self.intensity,
                            hue_base: self.hue_base,
                            accent: self.accent,
                        };
                        for lk in &self.looks {
                            lk.apply(&mut cell, &cctx);
                        }
                        if self.scfx_on {
                            let sc = scfx::Scfx {
                                kind: self.scfx_type,
                                knob: self.channels[i].color,
                            };
                            sc.apply(&mut cell, &cctx);
                        }
                        frame.buffer_mut()[(ax, ay)] = cell;
                        break;
                    }
                }
                // r beyond every contender lands in the background share.
            }
        }
        match CELL_POSTS[self.cell_post].1 {
            CellPost::None => {}
            CellPost::Slice => postfx::slice(
                frame.buffer_mut(),
                stage,
                vbeat,
                self.drive.gbeat(),
                self.intensity,
            ),
            CellPost::Glitch => postfx::glitch_rows(
                frame.buffer_mut(),
                stage,
                &self.drive,
                vbeat,
                self.intensity,
            ),
            CellPost::MirrorV => postfx::mirror_v(frame.buffer_mut(), stage),
            CellPost::MirrorQuad => postfx::mirror_quad(frame.buffer_mut(), stage),
            CellPost::Pixelate => {
                postfx::pixelate(frame.buffer_mut(), stage, &self.drive, self.intensity)
            }
            CellPost::ZoomPunch => {
                postfx::zoom_punch_cells(frame.buffer_mut(), stage, &self.drive, self.intensity)
            }
            CellPost::Shake => postfx::shake(
                frame.buffer_mut(),
                stage,
                &self.drive,
                vbeat,
                self.intensity,
            ),
        }
        // A spin earns a smear. Beat time alone moves the picture, but a
        // thrown platter reads as motion only if the frame blurs along
        // the direction it is travelling.
        let smear = self.jog_spin_amount();
        if smear > 0.02 {
            let back = self.jog_spin() == Some(jog::Spin::Back);
            let span = (smear * 4.0) as u16;
            let src = frame.buffer_mut().clone();
            for y in 0..stage.height {
                for x in 0..stage.width {
                    let mut acc = (0.0, 0.0, 0.0);
                    let mut n = 0.0;
                    for k in 0..=span {
                        let sx = if back {
                            (x + k).min(stage.width - 1)
                        } else {
                            x.saturating_sub(k)
                        };
                        if let Color::Rgb(r, g, b) = src[(stage.x + sx, stage.y + y)].fg {
                            let w = 1.0 - k as f64 / (span as f64 + 1.0);
                            acc.0 += r as f64 * w;
                            acc.1 += g as f64 * w;
                            acc.2 += b as f64 * w;
                            n += w;
                        }
                    }
                    if n > 0.0 {
                        let cell = &mut frame.buffer_mut()[(stage.x + x, stage.y + y)];
                        cell.fg = pass::rgb(acc.0 / n, acc.1 / n, acc.2 / n);
                    }
                }
            }
        }

        // The show's hold freezes the frame outright (the outro's first
        // beat). Reuses the stutter's held buffer — it is the same
        // gesture with a different trigger.
        if self.show_state.hold {
            if let Some(h) = &self.stutter_hold
                && h.area == stage
            {
                for y in 0..stage.height {
                    for x in 0..stage.width {
                        let (ax, ay) = (stage.x + x, stage.y + y);
                        frame.buffer_mut()[(ax, ay)] = h[(ax, ay)].clone();
                    }
                }
            } else {
                let mut h = ratatui::buffer::Buffer::empty(stage);
                for y in 0..stage.height {
                    for x in 0..stage.width {
                        let (ax, ay) = (stage.x + x, stage.y + y);
                        h[(ax, ay)] = frame.buffer_mut()[(ax, ay)].clone();
                    }
                }
                self.stutter_hold = Some(h);
            }
        } else if !self.stutter {
            // Neither holder wants the buffer: drop it so a stale frame
            // cannot resurface later.
            if self.stutter_hold.is_some() && !self.show_state.hold {
                self.stutter_hold = None;
            }
        }

        // STUTTER — in the first two beats of a phrase the picture is
        // re-taken at only two subdivisions per beat and the held frame
        // is re-blitted otherwise.
        //
        // Over there this also skipped the pipeline, so the frame rate
        // went up while it held. Here the mix has already run by the
        // time we get to it, so the hold is only visual — the saving is
        // not real, and claiming it would be a lie. Moving the check
        // above the mix would earn it back.
        if self.stutter && self.drive.groove > 0.5 {
            let in_window = vbeat.rem_euclid(16.0) < 2.0;
            let tick = (vbeat * 2.0).floor() as i64;
            if in_window {
                if tick == self.stutter_tick
                    && let Some(h) = &self.stutter_hold
                    && h.area == stage
                {
                    for y in 0..stage.height {
                        for x in 0..stage.width {
                            let (ax, ay) = (stage.x + x, stage.y + y);
                            frame.buffer_mut()[(ax, ay)] = h[(ax, ay)].clone();
                        }
                    }
                } else {
                    self.stutter_tick = tick;
                    let mut h = ratatui::buffer::Buffer::empty(stage);
                    for y in 0..stage.height {
                        for x in 0..stage.width {
                            let (ax, ay) = (stage.x + x, stage.y + y);
                            h[(ax, ay)] = frame.buffer_mut()[(ax, ay)].clone();
                        }
                    }
                    self.stutter_hold = Some(h);
                }
            } else {
                self.stutter_hold = None;
            }
        }

        // The show's exports gate the picture — the same Transform shape
        // as everything else that recolours the frame.
        let sh = self.show_state;
        self.apply_over_stage(frame.buffer_mut(), stage, vbeat, &sh);
        // Only the hits this scene drew are live — that is what makes
        // one scene read differently from the next. Same trait as the
        // looks and SCFX: a colour in, a colour out.
        let sc = self.scenes.scene();
        let hits = triggers::HitSet {
            invert: sc.has_hit(scene::Hit::InvertFlash),
            color: sc.has_hit(scene::Hit::ColorFlash),
            strobe: sc.has_hit(scene::Hit::Strobe),
        };
        self.apply_over_stage(frame.buffer_mut(), stage, vbeat, &hits);
        self.triggers.post(frame.buffer_mut(), stage, vbeat);
        self.render_lyrics(frame.buffer_mut(), stage, &ctx);
        self.overlay.render(frame.buffer_mut(), stage, &ctx);

        if !self.hud_visible {
            return;
        }

        let bar = (beat / 4.0).floor() as i64 + 1;
        let beat_in_bar = ctx.bar_phase as i64 + 1;
        let status = match self.channels[self.focus].slot {
            units::Unit::Cell(i) => self.effects[i].status(),
            units::Unit::Pixel(_) | units::Unit::Combo(_) => None,
        }
        .map(|s| format!(" [{s}]"))
        .unwrap_or_default();
        let mut trig_seg = self.triggers.hud();
        if let Some(sp) = self.jog_spin() {
            trig_seg.push_str(&format!(" ▶{}", sp.name()));
        }
        if self.jogs.iter().any(|j| j.active()) {
            trig_seg.push_str(&format!(" ↻{:+.2}", self.jog_offset()));
        }
        let look_seg = {
            let probe = pass::CellCtx {
                drive: &self.drive,
                t: self.vt,
                beat,
                x: 0,
                y: 0,
                w: stage.width,
                intensity: self.intensity,
                hue_base: self.hue_base,
                accent: self.accent,
            };
            let mut s = String::new();
            for lk in &self.looks {
                if ColorPass::amount(lk, &probe) > 0.0 {
                    s.push_str(ColorPass::name(lk));
                    s.push(' ');
                }
            }
            if self.cell_post > 0 {
                s.push_str(&format!("{} ", CELL_POSTS[self.cell_post].0));
            }
            s.push_str(&format!("{} ", self.scenes.scene().hud()));
            if !self.scenes.auto() {
                // Rotation handed to the operator — worth a word, or a
                // quiet rig at a controller-less rehearsal looks broken.
                s.push_str("MAN ");
            }
            if !self.show.is_live() {
                s.push_str(&format!("[{}] ", self.show.stage().name()));
            }
            if let Some(r) = self.scenes.pending() {
                s.push_str(&format!("→{} ", r.length().name()));
            }
            if self.abcut {
                s.push_str("AB ");
            }
            if self.stutter {
                s.push_str("STU ");
            }
            if self.escapes > 0 {
                s.push_str(&format!("esc{} ", self.escapes));
            }
            s
        };
        // The pixel tier is no longer a mode, so the HUD only reports
        // the post chain over it — which channels are pixel-medium is
        // already visible in the deck list.
        let gfx_seg = if self.pix_post > 0 && self.pixel_channels_live() {
            format!("▓>{} ", PIX_POSTS[self.pix_post].0)
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
                        self.unit_name(c.slot),
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
        let cam = if let Some(e) = &self.capture_err {
            format!(" │ CAM! {e}")
        } else if let Some(c) = self.capture.borrow().as_ref() {
            let m = if self.motion_drive {
                format!(" mot{:.0}%", c.motion() * 100.0)
            } else {
                String::new()
            };
            // A frozen camera is the one failure macOS reports as
            // success, so it earns its own word in the HUD.
            let state = if c.frozen() {
                "FROZEN?"
            } else if c.alive() {
                "live"
            } else {
                "wait"
            };
            format!(" │ CAM {state}{m}")
        } else {
            String::new()
        };
        let pdj = if let Some(e) = &self.prolink_err {
            format!(" │ PDJ! {e}")
        } else if let Some(p) = &self.prolink {
            if !self.prolink_sync {
                " │ ◆PDJ off".to_string()
            } else {
                // One name (the XZ presents three numbers under one
                // name), then followed deck, tempo, beat age — enough to
                // see whether it is heard and whether beats went stale.
                let devs = p.devices();
                let who = devs
                    .first()
                    .map(|d| {
                        let n: String = d.name.chars().take(8).collect();
                        format!("{n}×{}", devs.len())
                    })
                    .unwrap_or_else(|| "0dev".into());
                match p.beat() {
                    Some((ev, at, _)) => format!(
                        " │ ◆PDJ {who} D{} {:>5.1} {:.1}s",
                        ev.device,
                        ev.bpm,
                        at.elapsed().as_secs_f64()
                    ),
                    None => format!(" │ ◆PDJ {who} ..."),
                }
            }
        } else {
            String::new()
        };
        let lnk = match (&self.link, self.link_sync) {
            (Some(l), true) => format!(" │ ⇄Link {}p", l.peers()),
            (Some(_), false) => " │ ⇄ off".to_string(),
            (None, _) => String::new(),
        };
        let mid = if self.midi.connected() {
            let m = &self.midi;
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
            // Say so rather than going quiet: a controller that never
            // arrived and one that was unplugged look the same on stage.
            " │ M:none".to_string()
        };
        let hud_text = format!(
            " {:>6.1} BPM │ {:>3}.{} │ {}{}{} │ SC:{}{}{}{}{}{}{} │ int {:>3.0}% │ {:>4.1}ms │ SPC ± ↑↓ TAB ←→ 1-9 ,. P w x b o h y k a C n r m c l j q",
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
            cam,
            pdj,
            lnk,
            mid,
            self.ai.hud(),
            self.intensity * 100.0,
            self.render_ms,
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
    // The first roll goes live from frame one — before this, the opening
    // scene's cast and palette only landed at the first change, so the
    // rig started on defaults no scene had chosen.
    let first = app.scenes.scene().clone();
    app.apply_scene(&first);

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
        app.frame_dt = frame_dt;
        app.tick_jogs(frame_dt);
        app.tick_drive(frame_dt);
        app.tick_show(frame_dt);
        prev_frame = now;
        app.sync();

        let t0 = Instant::now();
        terminal.draw(|f| app.draw(f))?;
        // Graphics tier: place the pixel stage over the blank stage cells
        // ratatui just drew. Leaving gfx mode clears the image once.
        if app.pixel_channels_live() {
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

#[cfg(test)]
mod app_tests {
    use super::*;

    #[test]
    fn built_effects_carry_the_canonical_names() {
        // The bridge between `effects::CELL_NAMES` and the instances the
        // app actually runs: a rename in an `Effect::name()` or a
        // reorder in `build_effects` must fail here, not resolve a
        // scene's cast to the wrong unit at showtime.
        let plate = assets::Plate {
            name: "T".into(),
            img: image::RgbImage::new(2, 2),
        };
        let cap: std::rc::Rc<std::cell::RefCell<Option<capture::Capture>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let with: Vec<&str> = build_effects(std::rc::Rc::new(vec![plate]), cap.clone())
            .iter()
            .map(|e| e.name())
            .collect();
        assert_eq!(with, effects::CELL_NAMES);
        // Without plates the plate-fed pair is absent and order holds.
        let without: Vec<&str> = build_effects(std::rc::Rc::new(vec![]), cap)
            .iter()
            .map(|e| e.name())
            .collect();
        let expect: Vec<&str> = effects::CELL_NAMES
            .iter()
            .copied()
            .filter(|n| *n != "IMGDUST" && *n != "PLATE")
            .collect();
        assert_eq!(without, expect);
    }
}
