//! Pro DJ Link — the XDJ-XZ clock source, passive.
//!
//! Beat packets are broadcast to UDP 50001 and keep-alives to 50000, so
//! a listener that never announces itself still gets the device list and
//! per-deck beat events with BPM, pitch and beat-within-bar. What it
//! does NOT get is the tempo-master flag: status packets (port 50002)
//! are unicast only to announced peers. Rather than run a virtual CDJ,
//! v1 follows the deck that is actually beating — sticky, so two decks
//! beating at once doesn't make the clock ping-pong — and the master
//! flag can be added later if the venue proves it necessary.
//!
//! Byte offsets are from Deep Symmetry's DJ Link analysis, cross-checked
//! against beat-link's parsers. XDJ-XZ specifics: one IP presents
//! device numbers 1, 2 (decks) and 33 (mixer), so devices are keyed on
//! (IP, number), never IP alone; and the mixer's beats are a
//! free-running metronome unrelated to the music — a deck must always
//! win over it.
//!
//! Verify on hardware (the venue checklist):
//! - whether both XZ decks emit beat packets (strongly implied, unstated)
//! - the XZ's status packet length, if announcing is ever added
//! - which broadcast address its firmware uses (wildcard bind covers both)
//!
//! rekordbox on the same machine takes these ports and does not share
//! them. That is surfaced as an error string, not worked around.

use std::collections::HashMap;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Every Pro DJ Link packet opens with this ten-byte magic.
const MAGIC: [u8; 10] = [0x51, 0x73, 0x70, 0x74, 0x31, 0x57, 0x6d, 0x4a, 0x4f, 0x4c];

const TYPE_KEEPALIVE: u8 = 0x06;
const TYPE_BEAT: u8 = 0x28;

/// Devices vanish without a goodbye; the ecosystem drops them after
/// ~10 s of silence and so do we.
const DEVICE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the followed deck may go silent before another deck may
/// take over. A track change pauses beats for a few seconds; switching
/// decks on every gap would ping-pong the clock mid-mix.
const FOLLOW_TIMEOUT: Duration = Duration::from_secs(4);

/// Device numbers at and above this are mixers (the XZ's is 33). Their
/// beats are a backup metronome, not the music.
const MIXER_MIN: u8 = 0x10;

/// One beat, as broadcast: the packet arrives ON the beat, so its
/// receive instant is the beat instant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatEvent {
    pub device: u8,
    /// Track BPM × pitch — the tempo actually playing.
    pub bpm: f64,
    /// 1..=4, 1 = downbeat.
    pub beat_in_bar: u8,
}

/// A device seen on the network.
#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub number: u8,
    /// The address the device claims in its keep-alive. The app keys on
    /// the packet's source address instead; the probe prints this so a
    /// mismatch between the two is visible at the venue.
    #[allow(dead_code)]
    pub ip: Ipv4Addr,
}

struct State {
    devices: HashMap<(Ipv4Addr, u8), (Device, Instant)>,
    /// The deck the clock follows, and its latest beat.
    followed: Option<(Ipv4Addr, u8)>,
    beat: Option<(BeatEvent, Instant)>,
    /// Counts every accepted beat, so the app can detect a fresh one.
    beats_seen: u64,
}

pub struct ProLink {
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
}

impl Drop for ProLink {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl ProLink {
    /// Bind the two broadcast ports and start listening. Wildcard binds:
    /// an interface-specific bind silently misses subnet-directed
    /// broadcasts, and the XZ may be on a 169.254.x.x link-local subnet.
    pub fn start() -> Result<Self, String> {
        let beats = UdpSocket::bind(("0.0.0.0", 50001)).map_err(|e| {
            format!("port 50001: {e} (rekordbox or another Link tool running?)")
        })?;
        let keepalive = UdpSocket::bind(("0.0.0.0", 50000)).map_err(|e| {
            format!("port 50000: {e} (rekordbox or another Link tool running?)")
        })?;
        beats
            .set_read_timeout(Some(Duration::from_millis(250)))
            .map_err(|e| e.to_string())?;
        keepalive
            .set_read_timeout(Some(Duration::from_millis(250)))
            .map_err(|e| e.to_string())?;

        let state = Arc::new(Mutex::new(State {
            devices: HashMap::new(),
            followed: None,
            beat: None,
            beats_seen: 0,
        }));
        let stop = Arc::new(AtomicBool::new(false));

        {
            let (state, stop) = (state.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut buf = [0u8; 512];
                while !stop.load(Ordering::Relaxed) {
                    // recv timeouts just loop back to check stop.
                    if let Ok((n, src)) = beats.recv_from(&mut buf)
                        && let (Some(ev), std::net::IpAddr::V4(ip)) =
                            (parse_beat(&buf[..n]), src.ip())
                    {
                        let mut st = state.lock().unwrap();
                        st.note_beat(ip, ev, Instant::now());
                    }
                }
            });
        }
        {
            let (state, stop) = (state.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut buf = [0u8; 512];
                while !stop.load(Ordering::Relaxed) {
                    if let Ok((n, src)) = keepalive.recv_from(&mut buf)
                        && let (Some(dev), std::net::IpAddr::V4(ip)) =
                            (parse_keepalive(&buf[..n]), src.ip())
                    {
                        let mut st = state.lock().unwrap();
                        let key = (ip, dev.number);
                        st.devices.insert(key, (dev, Instant::now()));
                    }
                }
            });
        }

        Ok(Self { state, stop })
    }

    /// Devices seen within the timeout.
    pub fn devices(&self) -> Vec<Device> {
        let mut st = self.state.lock().unwrap();
        let now = Instant::now();
        st.devices
            .retain(|_, (_, seen)| now.duration_since(*seen) < DEVICE_TIMEOUT);
        st.devices.values().map(|(d, _)| d.clone()).collect()
    }

    /// The latest beat from the followed deck, with its arrival instant
    /// and the running count that lets the caller spot a new one.
    pub fn beat(&self) -> Option<(BeatEvent, Instant, u64)> {
        let st = self.state.lock().unwrap();
        st.beat.map(|(ev, at)| (ev, at, st.beats_seen))
    }

}

impl State {
    /// Sticky follow: decks always beat mixers, the followed deck keeps
    /// the clock while it is beating, and another deck takes over only
    /// after the follow timeout — so a mix across two beating decks
    /// doesn't ping-pong the phase.
    fn note_beat(&mut self, ip: Ipv4Addr, ev: BeatEvent, now: Instant) {
        let is_deck = ev.device < MIXER_MIN;
        let key = (ip, ev.device);

        let followed_fresh = self
            .beat
            .map(|(_, at)| now.duration_since(at) < FOLLOW_TIMEOUT)
            .unwrap_or(false);

        match self.followed {
            Some(f) if f == key => {
                self.beat = Some((ev, now));
                self.beats_seen += 1;
            }
            Some(f) => {
                let followed_is_mixer = f.1 >= MIXER_MIN;
                // A deck displaces a mixer immediately; anything else
                // waits for the follow timeout.
                if (is_deck && followed_is_mixer) || !followed_fresh {
                    self.followed = Some(key);
                    self.beat = Some((ev, now));
                    self.beats_seen += 1;
                }
            }
            None => {
                self.followed = Some(key);
                self.beat = Some((ev, now));
                self.beats_seen += 1;
            }
        }
    }
}

fn u16_be(b: &[u8], off: usize) -> u16 {
    ((b[off] as u16) << 8) | b[off + 1] as u16
}

fn u32_be(b: &[u8], off: usize) -> u32 {
    ((b[off] as u32) << 24) | ((b[off + 1] as u32) << 16) | ((b[off + 2] as u32) << 8)
        | b[off + 3] as u32
}

/// Pitch field: 0x100000 is ±0%, 0x200000 is +100%. Returns the
/// multiplier to apply to the track BPM.
fn pitch_multiplier(raw: u32) -> f64 {
    raw as f64 / 0x100000 as f64
}

/// Parse a beat packet (type 0x28, 96 bytes). Anything malformed is
/// None — on a broadcast port, log-and-drop is the only sane policy.
pub fn parse_beat(buf: &[u8]) -> Option<BeatEvent> {
    if buf.len() < 0x60 || buf[..10] != MAGIC || buf[0x0a] != TYPE_BEAT {
        return None;
    }
    let device = buf[0x21];
    let bpm_track = u16_be(buf, 0x5a) as f64 / 100.0;
    let pitch = pitch_multiplier(u32_be(buf, 0x54));
    let beat_in_bar = buf[0x5c];
    let bpm = bpm_track * pitch;
    // The effective tempo must be one a clock could follow: a stopped
    // platter (pitch 0) or a garbage field is not a beat.
    if !(1..=4).contains(&beat_in_bar) || !(20.0..=999.0).contains(&bpm) {
        return None;
    }
    Some(BeatEvent {
        device,
        bpm,
        beat_in_bar,
    })
}

/// Parse a keep-alive (type 0x06, 54 bytes): name, number, IP.
pub fn parse_keepalive(buf: &[u8]) -> Option<Device> {
    if buf.len() < 0x36 || buf[..10] != MAGIC || buf[0x0a] != TYPE_KEEPALIVE {
        return None;
    }
    let name = String::from_utf8_lossy(&buf[0x0c..0x20])
        .trim_end_matches('\0')
        .to_string();
    let number = buf[0x24];
    let ip = Ipv4Addr::new(buf[0x2c], buf[0x2d], buf[0x2e], buf[0x2f]);
    Some(Device { name, number, ip })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a beat packet from the spec's byte table.
    fn beat_packet(device: u8, bpm_x100: u16, pitch: u32, bb: u8) -> Vec<u8> {
        let mut p = vec![0u8; 0x60];
        p[..10].copy_from_slice(&MAGIC);
        p[0x0a] = TYPE_BEAT;
        p[0x0b..0x11].copy_from_slice(b"XDJ-XZ");
        p[0x21] = device;
        p[0x22] = 0x00;
        p[0x23] = 0x3c;
        p[0x54..0x58].copy_from_slice(&pitch.to_be_bytes());
        p[0x5a..0x5c].copy_from_slice(&bpm_x100.to_be_bytes());
        p[0x5c] = bb;
        p[0x5f] = device;
        p
    }

    fn keepalive_packet(name: &str, device: u8, ip: [u8; 4]) -> Vec<u8> {
        let mut p = vec![0u8; 0x36];
        p[..10].copy_from_slice(&MAGIC);
        p[0x0a] = TYPE_KEEPALIVE;
        p[0x0c..0x0c + name.len()].copy_from_slice(name.as_bytes());
        p[0x20] = 0x01;
        p[0x21] = 0x02;
        p[0x22] = 0x00;
        p[0x23] = 0x36;
        p[0x24] = device;
        p[0x25] = 0x01;
        p[0x2c..0x30].copy_from_slice(&ip);
        p
    }

    #[test]
    fn parses_the_spec_example() {
        // 0x319c = 127.00 BPM at neutral pitch, beat 3 of the bar.
        let ev = parse_beat(&beat_packet(2, 0x319c, 0x0010_0000, 3)).unwrap();
        assert_eq!(ev.device, 2);
        assert!((ev.bpm - 127.0).abs() < 1e-9);
        assert_eq!(ev.beat_in_bar, 3);
    }

    #[test]
    fn pitch_scales_the_track_bpm() {
        // +100% pitch doubles the tempo; the field carries TRACK bpm.
        let ev = parse_beat(&beat_packet(1, 12800, 0x0020_0000, 1)).unwrap();
        assert!((ev.bpm - 256.0).abs() < 1e-9);
        // -100% (stopped platter) is 0 — rejected as not a tempo.
        assert!(parse_beat(&beat_packet(1, 12800, 0, 1)).is_none());
    }

    #[test]
    fn malformed_packets_are_dropped_not_parsed() {
        let good = beat_packet(1, 12800, 0x0010_0000, 1);
        assert!(parse_beat(&good[..0x50]).is_none(), "truncated");
        let mut bad_magic = good.clone();
        bad_magic[0] = 0;
        assert!(parse_beat(&bad_magic).is_none(), "magic");
        let mut bad_type = good.clone();
        bad_type[0x0a] = 0x29;
        assert!(parse_beat(&bad_type).is_none(), "type");
        let mut bad_bb = good.clone();
        bad_bb[0x5c] = 5;
        assert!(parse_beat(&bad_bb).is_none(), "beat-in-bar range");
    }

    #[test]
    fn keepalive_yields_the_device() {
        let d = parse_keepalive(&keepalive_packet("XDJ-XZ", 33, [169, 254, 12, 34])).unwrap();
        assert_eq!(d.name, "XDJ-XZ");
        assert_eq!(d.number, 33);
        assert_eq!(d.ip, Ipv4Addr::new(169, 254, 12, 34));
    }

    #[test]
    fn a_deck_always_displaces_the_mixer_metronome() {
        // The XZ's mixer (33) beats continuously whether or not music
        // plays; the moment a deck beats, the deck owns the clock.
        let ip = Ipv4Addr::new(169, 254, 1, 1);
        let mut st = State {
            devices: HashMap::new(),
            followed: None,
            beat: None,
            beats_seen: 0,
        };
        let t0 = Instant::now();
        st.note_beat(ip, BeatEvent { device: 33, bpm: 120.0, beat_in_bar: 1 }, t0);
        assert_eq!(st.followed, Some((ip, 33)), "mixer holds it when alone");
        st.note_beat(ip, BeatEvent { device: 1, bpm: 128.0, beat_in_bar: 1 }, t0);
        assert_eq!(st.followed, Some((ip, 1)), "a deck displaces the mixer");
        assert!((st.beat.unwrap().0.bpm - 128.0).abs() < 1e-9);
    }

    #[test]
    fn the_followed_deck_is_sticky_across_a_two_deck_mix() {
        // Both XZ decks beat during a blend; hopping between them would
        // ping-pong the phase. The second deck waits for the timeout.
        let ip = Ipv4Addr::new(169, 254, 1, 1);
        let mut st = State {
            devices: HashMap::new(),
            followed: None,
            beat: None,
            beats_seen: 0,
        };
        let t0 = Instant::now();
        st.note_beat(ip, BeatEvent { device: 1, bpm: 128.0, beat_in_bar: 1 }, t0);
        st.note_beat(ip, BeatEvent { device: 2, bpm: 140.0, beat_in_bar: 1 }, t0);
        assert_eq!(st.followed, Some((ip, 1)), "deck 1 keeps it while fresh");
        // Deck 1 goes silent past the follow timeout; deck 2 takes over.
        st.note_beat(
            ip,
            BeatEvent { device: 2, bpm: 140.0, beat_in_bar: 2 },
            t0 + FOLLOW_TIMEOUT + Duration::from_millis(100),
        );
        assert_eq!(st.followed, Some((ip, 2)), "handover after silence");
    }

    #[test]
    fn beats_seen_counts_only_accepted_beats() {
        let ip = Ipv4Addr::new(169, 254, 1, 1);
        let mut st = State {
            devices: HashMap::new(),
            followed: None,
            beat: None,
            beats_seen: 0,
        };
        let t0 = Instant::now();
        st.note_beat(ip, BeatEvent { device: 1, bpm: 128.0, beat_in_bar: 1 }, t0);
        st.note_beat(ip, BeatEvent { device: 2, bpm: 140.0, beat_in_bar: 1 }, t0);
        st.note_beat(ip, BeatEvent { device: 1, bpm: 128.0, beat_in_bar: 2 }, t0);
        // Deck 2's beat was ignored (deck 1 fresh), so 2 accepted.
        assert_eq!(st.beats_seen, 2);
    }
}
