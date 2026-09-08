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

#[derive(Default, Clone, Copy, PartialEq)]
pub struct Bindings {
    pub intensity: Option<(u8, u8)>,
    pub channels: [Option<(u8, u8)>; 4],
    /// Pad triggers by (midi channel, note) — see triggers::PAD_NAMES.
    pub pads: [Option<(u8, u8)>; crate::triggers::PADS],
}

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
            k => {
                if let Some(name) = k.strip_prefix("pad.")
                    && let Some(i) = crate::triggers::PAD_NAMES.iter().position(|n| *n == name)
                {
                    b.pads[i] = cc;
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
    for (i, pad) in b.pads.iter().enumerate() {
        if let Some(cc) = pad {
            out.push_str(&format!(
                "pad.{} = {}  # note\n",
                crate::triggers::PAD_NAMES[i],
                fmt_cc(*cc)
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
        b.pads[0] = Some((7, 0));
        b.pads[7] = Some((7, 21));
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
