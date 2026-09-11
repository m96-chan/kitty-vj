//! Desk snapshots — the whole picture, nameable and recallable.
//!
//! `scene.rs` orchestrates *rolled* scenes: treatments drawn from style
//! pools, right for unattended rotation and useless for "that one,
//! again". A snapshot is the other half of #22: everything the
//! operator's hands have set, captured as one value — which unit sits
//! on each channel, the fader and COLOR knob positions, the global
//! intensity, the looks, both post slots, the palette and the flags.
//! Deliberately outside it: the text overlay and lyrics (they follow
//! the track, not the picture), the armed beat hits (those belong to
//! the rolled scene), and the SCFX section (that is a hand on a knob,
//! not a picture).
//!
//! Persisted as one flat line per slot in the gig config, in the same
//! hand-editable spirit as the MIDI map, referencing units, looks and
//! posts **by name** so adding an effect never invalidates a saved
//! snapshot:
//!
//! ```text
//! snap.1 = DROP; ch=PLASMA:0.80:0.50,ACID:1.00:0.50,MCUBE:0.00:0.50,
//!          SPARKS3D:0.30:0.50; int=0.90; looks=HUE+SATP; cpost=SLICE;
//!          ppost=KALEID; hue=212; acc=0,255,213; accb=255,0,200; flags=ab
//! ```
//!
//! Parsing is forgiving the way the combo recipes are: an unknown look
//! or unit name drops out and the rest of the snapshot lands, because
//! on stage a half-recalled picture beats a refusal.

use crate::looks::{LOOKS, Look};

/// How many recall slots the bank holds — one pad page.
pub const SLOTS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub name: String,
    /// Per channel, bottom to top: unit name, fader, COLOR knob.
    pub channels: [(String, f64, f64); 4],
    pub intensity: f64,
    pub looks: Vec<Look>,
    /// Names in the app's `CELL_POSTS` / `PIX_POSTS`; "-" is none.
    pub cell_post: String,
    pub pix_post: String,
    pub hue_base: f64,
    pub accent: (u8, u8, u8),
    pub accent_b: (u8, u8, u8),
    pub abcut: bool,
    pub stutter: bool,
}

fn fmt_rgb(c: (u8, u8, u8)) -> String {
    format!("{},{},{}", c.0, c.1, c.2)
}

fn parse_rgb(s: &str) -> Option<(u8, u8, u8)> {
    let mut it = s.split(',').map(|v| v.trim().parse::<u8>());
    match (it.next(), it.next(), it.next(), it.next()) {
        (Some(Ok(r)), Some(Ok(g)), Some(Ok(b)), None) => Some((r, g, b)),
        _ => None,
    }
}

impl Snapshot {
    /// One config line, after the `snap.<slot> = `.
    pub fn serialize(&self) -> String {
        let ch = self
            .channels
            .iter()
            .map(|(n, l, c)| format!("{n}:{l:.2}:{c:.2}"))
            .collect::<Vec<_>>()
            .join(",");
        let looks = self
            .looks
            .iter()
            .map(|l| l.name())
            .collect::<Vec<_>>()
            .join("+");
        let mut flags = Vec::new();
        if self.abcut {
            flags.push("ab");
        }
        if self.stutter {
            flags.push("st");
        }
        format!(
            "{}; ch={ch}; int={:.2}; looks={looks}; cpost={}; ppost={}; hue={:.1}; acc={}; accb={}; flags={}",
            self.name,
            self.intensity,
            self.cell_post,
            self.pix_post,
            self.hue_base,
            fmt_rgb(self.accent),
            fmt_rgb(self.accent_b),
            flags.join(",")
        )
    }

    /// Parse a config line. `None` only when there is nothing usable at
    /// all — individual bad fields fall back to the defaults below.
    pub fn parse(line: &str) -> Option<Snapshot> {
        let mut parts = line.split(';');
        let name = parts.next()?.trim();
        if name.is_empty() {
            return None;
        }
        let mut s = Snapshot {
            name: name.to_string(),
            channels: std::array::from_fn(|_| (String::from("-"), 0.0, 0.5)),
            intensity: 1.0,
            looks: Vec::new(),
            cell_post: "-".into(),
            pix_post: "-".into(),
            hue_base: 0.0,
            accent: (255, 255, 255),
            accent_b: (255, 255, 255),
            abcut: false,
            stutter: false,
        };
        for kv in parts {
            let Some((k, v)) = kv.split_once('=') else {
                continue;
            };
            let v = v.trim();
            match k.trim() {
                "ch" => {
                    for (i, tok) in v.split(',').take(4).enumerate() {
                        let mut f = tok.split(':');
                        let name = f.next().unwrap_or("-").trim().to_uppercase();
                        let lvl = f.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
                        let col = f.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0.5);
                        s.channels[i] = (name, clamp01(lvl), clamp01(col));
                    }
                }
                "int" => s.intensity = v.parse().map(clamp01).unwrap_or(1.0),
                "looks" => {
                    s.looks = v
                        .split('+')
                        .filter_map(|n| {
                            let n = n.trim().to_uppercase();
                            LOOKS.iter().copied().find(|l| l.name() == n)
                        })
                        .collect();
                }
                "cpost" => s.cell_post = v.to_uppercase(),
                "ppost" => s.pix_post = v.to_uppercase(),
                "hue" => {
                    s.hue_base = v.parse::<f64>().ok().filter(|v| v.is_finite()).unwrap_or(0.0)
                }
                "acc" => s.accent = parse_rgb(v).unwrap_or(s.accent),
                "accb" => s.accent_b = parse_rgb(v).unwrap_or(s.accent_b),
                "flags" => {
                    s.abcut = v.split(',').any(|f| f.trim() == "ab");
                    s.stutter = v.split(',').any(|f| f.trim() == "st");
                }
                _ => {}
            }
        }
        Some(s)
    }
}

/// Soft-takeover: may an incoming physical value `v` take the control
/// over from the recalled `held` value, given the previous physical
/// value seen on that control?
///
/// A recall puts the picture at positions the hardware is not at; a
/// fader that then teleported the level on its first wiggle would undo
/// the recall by accident. So the physical control picks up only when
/// it reaches the held value — close enough to it, or crossing it
/// between two messages. The standard hardware answer, and the reason
/// it is a pure function here is that it needs testing more than it
/// needs context.
pub fn picks_up(held: f64, prev: Option<f64>, v: f64) -> bool {
    if (v - held).abs() < 0.03 {
        return true;
    }
    match prev {
        Some(p) => (p - held).signum() != (v - held).signum(),
        None => false,
    }
}

fn clamp01(v: f64) -> f64 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Snapshot {
        Snapshot {
            name: "DROP".into(),
            channels: [
                ("PLASMA".into(), 0.8, 0.5),
                ("ACID".into(), 1.0, 0.5),
                ("MCUBE".into(), 0.0, 0.5),
                ("SPARKS3D".into(), 0.3, 0.75),
            ],
            intensity: 0.9,
            looks: vec![Look::HueCycle, Look::SatPump],
            cell_post: "SLICE".into(),
            pix_post: "KALEID".into(),
            hue_base: 212.0,
            accent: (0, 255, 213),
            accent_b: (255, 0, 200),
            abcut: true,
            stutter: false,
        }
    }

    #[test]
    fn a_snapshot_survives_the_config_round_trip() {
        let s = sample();
        let back = Snapshot::parse(&s.serialize()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn bad_fields_degrade_and_junk_lines_refuse() {
        // A typo'd look and a broken colour fall back; the rest lands.
        let s = Snapshot::parse(
            "X; ch=PLASMA:0.5:0.5; looks=HUE+NOPE; acc=notacolour; hue=90",
        )
        .unwrap();
        assert_eq!(s.name, "X");
        assert_eq!(s.looks, vec![Look::HueCycle]);
        assert_eq!(s.accent, (255, 255, 255));
        assert_eq!(s.channels[0].0, "PLASMA");
        assert_eq!(s.channels[1].0, "-", "missing channels default to keep");
        // Nothing usable at all.
        assert!(Snapshot::parse("").is_none());
        assert!(Snapshot::parse("   ; ch=A:1:1").is_none());
    }

    #[test]
    fn levels_are_clamped_on_the_way_in() {
        let s = Snapshot::parse("X; ch=A:9.0:-2.0; int=42").unwrap();
        assert_eq!(s.channels[0].1, 1.0);
        assert_eq!(s.channels[0].2, 0.0);
        assert_eq!(s.intensity, 1.0);
    }

    #[test]
    fn non_finite_hues_fall_back_to_default() {
        for value in ["NaN", "inf", "-inf", "1e999"] {
            let s = Snapshot::parse(&format!("X; hue={value}")).unwrap();
            assert_eq!(s.hue_base, 0.0);
        }
        assert_eq!(Snapshot::parse("X; hue=212.5").unwrap().hue_base, 212.5);
    }

    #[test]
    fn pickup_requires_reaching_or_crossing_the_held_value() {
        // Far away, no history: hold.
        assert!(!picks_up(0.8, None, 0.2));
        // Wiggling below the held value: still hold.
        assert!(!picks_up(0.8, Some(0.2), 0.3));
        // Close enough: pick up.
        assert!(picks_up(0.8, Some(0.5), 0.79));
        // Sweeping through it between two messages: pick up.
        assert!(picks_up(0.8, Some(0.7), 0.9));
    }
}
