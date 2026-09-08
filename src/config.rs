//! Gig config — MIDI bindings live in a file, not in muscle memory.
//! Learn ('m') is the discovery tool; every successful bind rewrites
//! the file, and the file is loaded on startup. It's also hand-editable:
//! the HUD shows every incoming CC as `cc<ch>.<cc>=<val>`, so the value
//! to write down is right there.
//!
//! Format (one binding per line, `#` comments):
//! ```text
//! intensity = 0.23
//! ch1 = 0.19
//! ch2 = 0.20
//! ```

use std::path::Path;

pub const PATH: &str = "kitty-vj.conf";

#[derive(Default, Clone, PartialEq)]
pub struct Bindings {
    pub intensity: Option<(u8, u8)>,
    pub channels: [Option<(u8, u8)>; 4],
    /// Pad triggers by (midi channel, note) — see triggers::PAD_NAMES.
    /// A pad can hold several bindings (left and right deck send the
    /// same pads on different MIDI channels); comma-separated in the file.
    pub pads: [Vec<(u8, u8)>; crate::triggers::PADS],
    /// SOUND COLOR FX knobs per channel (CC).
    pub colors: [Option<(u8, u8)>; 4],
    /// SCFX type select buttons (notes), hardware order left to right.
    pub scfx: [Option<(u8, u8)>; 6],
    /// Remembered SCFX state: Some(index into scfx::TYPES) or None = off.
    /// The knob only carries depth — which FX is lit is state the app
    /// must keep, and it must survive a restart.
    pub scfx_selected: Option<usize>,
}

pub const SCFX_KEYS: [&str; 6] = [
    "scfx.space",
    "scfx.dubecho",
    "scfx.sweep",
    "scfx.noise",
    "scfx.crush",
    "scfx.filter",
];
const SCFX_NAMES: [&str; 6] = ["space", "dubecho", "sweep", "noise", "crush", "filter"];

fn parse_cc(s: &str) -> Option<(u8, u8)> {
    let (ch, cc) = s.trim().split_once('.')?;
    Some((ch.trim().parse().ok()?, cc.trim().parse().ok()?))
}

fn fmt_cc((ch, cc): (u8, u8)) -> String {
    format!("{ch}.{cc}")
}

pub fn parse(text: &str) -> Bindings {
    let mut b = Bindings::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let cc = parse_cc(val);
        match key.trim() {
            "intensity" => b.intensity = cc,
            "ch1" => b.channels[0] = cc,
            "ch2" => b.channels[1] = cc,
            "ch3" => b.channels[2] = cc,
            "ch4" => b.channels[3] = cc,
            "color1" => b.colors[0] = cc,
            "color2" => b.colors[1] = cc,
            "color3" => b.colors[2] = cc,
            "color4" => b.colors[3] = cc,
            k if SCFX_KEYS.contains(&k) => {
                let i = SCFX_KEYS.iter().position(|s| *s == k).unwrap();
                b.scfx[i] = cc;
            }
            "scfx_type" => {
                let v = val.trim();
                b.scfx_selected = SCFX_NAMES.iter().position(|n| *n == v);
            }
            k => {
                if let Some(name) = k.strip_prefix("pad.")
                    && let Some(i) = crate::triggers::PAD_NAMES.iter().position(|n| *n == name)
                {
                    b.pads[i] = val.split(',').filter_map(parse_cc).collect();
                }
            }
        }
    }
    b
}

pub fn load(path: &Path) -> Bindings {
    std::fs::read_to_string(path)
        .map(|t| parse(&t))
        .unwrap_or_default()
}

pub fn save(path: &Path, b: &Bindings) -> std::io::Result<()> {
    let mut out = String::from(
        "# kitty-vj MIDI bindings — <midi channel>.<cc number>, as the HUD shows them (cc0.23=…)\n",
    );
    if let Some(cc) = b.intensity {
        out.push_str(&format!("intensity = {}\n", fmt_cc(cc)));
    }
    for (i, ch) in b.channels.iter().enumerate() {
        if let Some(cc) = ch {
            out.push_str(&format!("ch{} = {}\n", i + 1, fmt_cc(*cc)));
        }
    }
    for (i, c) in b.colors.iter().enumerate() {
        if let Some(cc) = c {
            out.push_str(&format!("color{} = {}\n", i + 1, fmt_cc(*cc)));
        }
    }
    for (i, s) in b.scfx.iter().enumerate() {
        if let Some(cc) = s {
            out.push_str(&format!("{} = {}  # note\n", SCFX_KEYS[i], fmt_cc(*cc)));
        }
    }
    if let Some(i) = b.scfx_selected {
        out.push_str(&format!("scfx_type = {}\n", SCFX_NAMES[i]));
    }
    for (i, pad) in b.pads.iter().enumerate() {
        if !pad.is_empty() {
            let list: Vec<String> = pad.iter().map(|cc| fmt_cc(*cc)).collect();
            out.push_str(&format!(
                "pad.{} = {}  # note\n",
                crate::triggers::PAD_NAMES[i],
                list.join(", ")
            ));
        }
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut b = Bindings {
            intensity: Some((0, 23)),
            channels: [Some((0, 19)), Some((0, 20)), None, Some((1, 7))],
            ..Default::default()
        };
        b.pads[0] = vec![(7, 16), (9, 16)]; // both decks
        b.pads[7] = vec![(7, 21)];
        let dir = std::env::temp_dir().join("kitty-vj-conf-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PATH);
        save(&path, &b).unwrap();
        assert!(load(&path) == b);
    }

    #[test]
    fn tolerates_junk_and_comments() {
        let b = parse("# hey\nintensity = 0.23 # fader\nnope\nch3 = 1.44\nch9 = 0.1\n");
        assert_eq!(b.intensity, Some((0, 23)));
        assert_eq!(b.channels[2], Some((1, 44)));
        assert_eq!(b.channels[0], None);
    }
}
