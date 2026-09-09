//! MIDI input. Opens every port it can see (CoreMIDI lets us listen to
//! the DDJ-FLX10 even while rekordbox owns it), forwards events over a
//! channel, and estimates tempo from MIDI clock (0xF8, 24 ppqn) when a
//! device sends it. CC mapping is learn-based: arm a target, move a
//! control, bound.
//!
//! Ports are rescanned on a timer rather than opened once at launch.
//! Controllers are routinely powered on after the software — rekordbox
//! usually gets there first — and a kicked USB cable mid-set must not
//! end the show's control surface for good. Reconnecting is the same
//! code path as connecting, so there is nothing special about recovery.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use midir::{MidiInput, MidiInputConnection};

pub enum MidiEvent {
    Cc { ch: u8, cc: u8, val: u8 },
    NoteOn { ch: u8, note: u8 },
    NoteOff { ch: u8, note: u8 },
    Clock(Instant),
    Start,
}

/// How often to look for controllers that appeared or vanished. Cheap —
/// enumerating CoreMIDI sources is a handful of microseconds — and a
/// second is well inside the time it takes to plug a cable and reach for
/// a fader.
const RESCAN: Duration = Duration::from_secs(1);

pub struct MidiIn {
    rx: Receiver<MidiEvent>,
    tx: Sender<MidiEvent>,
    pub ports: Vec<String>,
    conns: Vec<MidiInputConnection<Sender<MidiEvent>>>,
    last_scan: Instant,
}

impl MidiIn {
    /// Start with whatever is plugged in — possibly nothing. A missing
    /// controller is not a failure: it may be plugged in mid-set, and
    /// the instrument has to survive both that and the cable being
    /// kicked out again.
    pub fn open() -> Self {
        let (tx, rx) = channel();
        let mut me = Self {
            rx,
            tx,
            ports: Vec::new(),
            conns: Vec::new(),
            last_scan: Instant::now() - RESCAN,
        };
        me.rescan();
        me
    }

    /// Reconnect to whatever is present now. Called on a timer, so a
    /// controller powered on after the app — the normal order, since
    /// rekordbox usually claims it first — still lands.
    pub fn rescan(&mut self) {
        self.last_scan = Instant::now();
        let Ok(scan) = MidiInput::new("kitty-vj-scan") else {
            return;
        };
        let names: Vec<String> = scan
            .ports()
            .iter()
            .map(|p| scan.port_name(p).unwrap_or_default())
            .collect();
        if names == self.ports && !self.conns.is_empty() {
            return; // nothing came or went
        }
        // The set changed: drop every connection and take them all
        // again. Reconnecting one port is not obviously cheaper than
        // reconnecting four, and this way there is one code path.
        self.conns.clear();
        self.ports.clear();
        let (conns, names) = Self::connect_all(&self.tx);
        self.conns = conns;
        self.ports = names;
    }

    /// True if a rescan is due. The caller drives this so the timer
    /// lives on the render loop rather than in a thread.
    pub fn due(&self) -> bool {
        self.last_scan.elapsed() >= RESCAN
    }

    pub fn connected(&self) -> bool {
        !self.conns.is_empty()
    }

    fn connect_all(
        tx: &Sender<MidiEvent>,
    ) -> (Vec<MidiInputConnection<Sender<MidiEvent>>>, Vec<String>) {
        let mut conns = Vec::new();
        let mut names = Vec::new();
        let Ok(scan) = MidiInput::new("kitty-vj") else {
            return (conns, names);
        };
        let n_ports = scan.ports().len();
        for i in 0..n_ports {
            let Ok(input) = MidiInput::new("kitty-vj") else {
                continue;
            };
            let Some(port) = input.ports().into_iter().nth(i) else {
                continue;
            };
            let name = input.port_name(&port).unwrap_or_else(|_| format!("#{i}"));
            let conn = input.connect(
                &port,
                "kitty-vj-in",
                move |_stamp, msg, tx| {
                    let ev = match msg {
                        [s, cc, val] if s & 0xf0 == 0xb0 => Some(MidiEvent::Cc {
                            ch: s & 0x0f,
                            cc: *cc,
                            val: *val,
                        }),
                        // Note on with velocity 0 is a release by convention.
                        [s, note, vel] if s & 0xf0 == 0x90 && *vel > 0 => Some(MidiEvent::NoteOn {
                            ch: s & 0x0f,
                            note: *note,
                        }),
                        [s, note, _] if matches!(s & 0xf0, 0x80 | 0x90) => {
                            Some(MidiEvent::NoteOff {
                                ch: s & 0x0f,
                                note: *note,
                            })
                        }
                        [0xf8, ..] => Some(MidiEvent::Clock(Instant::now())),
                        [0xfa, ..] => Some(MidiEvent::Start),
                        _ => None,
                    };
                    if let Some(ev) = ev {
                        let _ = tx.send(ev);
                    }
                },
                tx.clone(),
            );
            if let Ok(c) = conn {
                conns.push(c);
                names.push(name);
            }
        }
        (conns, names)
    }

    pub fn drain(&self) -> Vec<MidiEvent> {
        self.rx.try_iter().collect()
    }
}

/// Tempo from MIDI clock ticks: 24 per quarter note. Phase is NOT
/// carried by 0xF8 — downbeat stays the operator's job (SPACE).
pub struct MidiClock {
    ticks: VecDeque<Instant>,
}

const TICK_WINDOW: usize = 96; // 4 beats

impl MidiClock {
    pub fn new() -> Self {
        Self {
            ticks: VecDeque::with_capacity(TICK_WINDOW + 1),
        }
    }

    pub fn tick(&mut self, at: Instant) {
        if let Some(&last) = self.ticks.back()
            && at.duration_since(last).as_secs_f64() > 0.5
        {
            self.ticks.clear(); // stream stalled — start over
        }
        self.ticks.push_back(at);
        if self.ticks.len() > TICK_WINDOW {
            self.ticks.pop_front();
        }
    }

    /// Current estimate, if enough ticks arrived recently.
    pub fn bpm(&self) -> Option<f64> {
        if self.ticks.len() < 25 {
            return None;
        }
        let first = *self.ticks.front()?;
        let last = *self.ticks.back()?;
        if last.elapsed().as_secs_f64() > 0.5 {
            return None; // stale
        }
        let span = last.duration_since(first).as_secs_f64();
        let intervals = (self.ticks.len() - 1) as f64;
        let bpm = 60.0 / (span / intervals * 24.0);
        (30.0..=400.0).contains(&bpm).then_some(bpm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn opening_with_nothing_plugged_in_is_not_a_failure() {
        // A controller powered on after the app is the normal order —
        // rekordbox usually claims it first — so a missing device must
        // leave a working object that can pick it up later.
        let m = MidiIn::open();
        assert!(m.drain().is_empty());
        // Whether anything is connected depends on the machine; what
        // matters is that we got an object either way and can rescan.
        let _ = m.connected();
    }

    #[test]
    fn rescan_is_rate_limited() {
        let mut m = MidiIn::open();
        assert!(!m.due(), "a fresh scan should not immediately be due");
        m.rescan();
        assert!(!m.due());
    }

    #[test]
    fn midi_clock_tempo_from_ticks() {
        let mut mc = MidiClock::new();
        let base = Instant::now();
        let interval = Duration::from_secs_f64(60.0 / (128.0 * 24.0));
        for i in 0..96 {
            mc.tick(base + interval * i);
        }
        // Ticks land "in the past" relative to now, but within staleness.
        let bpm = mc.bpm().expect("no bpm");
        assert!((bpm - 128.0).abs() < 1.0, "got {bpm}");
    }

    #[test]
    fn midi_clock_stall_resets() {
        let mut mc = MidiClock::new();
        let base = Instant::now();
        let interval = Duration::from_secs_f64(60.0 / (128.0 * 24.0));
        for i in 0..48 {
            mc.tick(base + interval * i);
        }
        // A gap over half a second clears the window.
        mc.tick(base + interval * 48 + Duration::from_secs(1));
        assert!(mc.bpm().is_none());
    }
}
