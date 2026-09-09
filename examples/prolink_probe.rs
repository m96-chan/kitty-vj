//! Venue probe for Pro DJ Link. Plug into the XDJ-XZ's LAN port (or its
//! switch), then:
//!
//!     cargo run --release --example prolink_probe
//!
//! Prints every device keep-alive once (then again if it changes) and
//! every beat packet with the gap since that device's previous beat, so
//! clock jitter is visible directly. Unknown packet types are dumped as
//! a hex head — that is the venue-day evidence for extending the parser.
//! Ctrl-C to stop.
//!
//! If binding fails, rekordbox (or another Link tool) owns the ports —
//! quit it; the protocol does not share.

#[allow(dead_code)]
#[path = "../src/prolink.rs"]
mod prolink;

use std::collections::HashMap;
use std::net::UdpSocket;
use std::time::{Duration, Instant};

fn hex_head(buf: &[u8]) -> String {
    buf.iter()
        .take(24)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let beats = match UdpSocket::bind(("0.0.0.0", 50001)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("port 50001: {e} — is rekordbox running?");
            return;
        }
    };
    let keepalive = match UdpSocket::bind(("0.0.0.0", 50000)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("port 50000: {e} — is rekordbox running?");
            return;
        }
    };
    beats
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    keepalive
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();

    println!("listening on 50000 (keep-alive) + 50001 (beats)… Ctrl-C to stop");
    let start = Instant::now();
    // Last announced identity per source, so keep-alive spam (one per
    // ~1.5 s per device) prints only when something actually changes.
    let mut seen: HashMap<(std::net::IpAddr, u8), String> = HashMap::new();
    // Previous beat instant per (ip, device) — the printed gap is the
    // measured beat period, and its wobble is the network jitter.
    let mut last_beat: HashMap<(std::net::IpAddr, u8), Instant> = HashMap::new();
    let mut buf = [0u8; 1500];

    loop {
        let t = start.elapsed().as_secs_f64();
        if let Ok((n, src)) = keepalive.recv_from(&mut buf) {
            match prolink::parse_keepalive(&buf[..n]) {
                Some(d) => {
                    let line = format!("{} #{} ip={}", d.name, d.number, d.ip);
                    let key = (src.ip(), d.number);
                    if seen.get(&key) != Some(&line) {
                        println!("[{t:8.3}] DEVICE  {line} (from {src})");
                        seen.insert(key, line);
                    }
                }
                None => println!(
                    "[{t:8.3}] 50000 ? len={n} from {src}: {}",
                    hex_head(&buf[..n])
                ),
            }
        }
        if let Ok((n, src)) = beats.recv_from(&mut buf) {
            match prolink::parse_beat(&buf[..n]) {
                Some(ev) => {
                    let key = (src.ip(), ev.device);
                    let now = Instant::now();
                    let gap = last_beat
                        .insert(key, now)
                        .map(|p| format!("{:6.1}ms", now.duration_since(p).as_secs_f64() * 1e3))
                        .unwrap_or_else(|| "  first".into());
                    let kind = if ev.device >= 0x10 { "mixer" } else { "deck " };
                    println!(
                        "[{t:8.3}] BEAT    {kind} D{} bb{} {:6.2} BPM  Δ{gap}",
                        ev.device, ev.beat_in_bar, ev.bpm
                    );
                }
                None => println!(
                    "[{t:8.3}] 50001 ? len={n} from {src}: {}",
                    hex_head(&buf[..n])
                ),
            }
        }
    }
}
