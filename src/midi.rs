//! MIDI input. Opens every port it can see (CoreMIDI lets us listen to
//! the DDJ-FLX10 even while rekordbox owns it), forwards events over a
//! channel, and estimates tempo from MIDI clock (0xF8, 24 ppqn) when a
//! device sends it. CC mapping is learn-based: arm a target, move a
//! control, bound.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use midir::{MidiInput, MidiInputConnection};

pub enum MidiEvent {
    Cc { ch: u8, cc: u8, val: u8 },
    NoteOn { ch: u8, note: u8 },
    NoteOff { ch: u8, note: u8 },
    Clock(Instant),
    Start,
}

pub struct MidiIn {
    rx: Receiver<MidiEvent>,
    pub ports: Vec<String>,
    _conns: Vec<MidiInputConnection<Sender<MidiEvent>>>,
}

impl MidiIn {
    /// Connect to every available input port.
    pub fn open() -> Result<Self, String> {
        let scan = MidiInput::new("kitty-vj").map_err(|e| e.to_string())?;
        let n_ports = scan.ports().len();
        if n_ports == 0 {
            return Err("no MIDI ports".into());
        }
        let (tx, rx) = channel();
        let mut conns = Vec::new();
        let mut names = Vec::new();

        for i in 0..n_ports {
            let input = MidiInput::new("kitty-vj").map_err(|e| e.to_string())?;
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
        if conns.is_empty() {
            return Err("no MIDI port would open".into());
        }
        Ok(Self {
            rx,
            ports: names,
            _conns: conns,
        })
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
