//! Channel scenes — a named combination a mixer channel can hold.
//!
//! The port took EasyPngVJ's composite effects apart into parts; this
//! is where parts recombine. A scene here is a recipe: base units with
//! a share each, e.g. `PLASMA*0.6 + MCUBE + SPARKS3D*0.5`. Put it on a
//! channel and the fader rides the whole combination, so four faders
//! can blend four combinations — which is the instrument the four
//! vertical faders were always reaching for.
//!
//! Defined in the gig config as `scene.<NAME> = <recipe>` lines, on top
//! of a handful of built-in samples. A config scene with a sample's
//! name replaces it. Parts are base units only — a scene inside a
//! scene resolves to nothing, deliberately: one level of recipe is an
//! instrument, two is a debugging session on stage.
//!
//! Resolution is forgiving by design. An unknown part (a typo, a
//! PLATE with no plates loaded) is dropped and the rest of the scene
//! plays; a scene with nothing left is dropped whole; a scene named
//! after an existing unit is dropped so it cannot shadow it. On stage
//! a half-working recipe beats a refusal to start.

use crate::units::Unit;

pub struct Combo {
    pub name: &'static str,
    /// Base units only, in recipe order, with the share of the
    /// channel's fader each part takes.
    pub parts: Vec<(Unit, f64)>,
}

/// The built-in deck: one sample per mood, cells and pixels mixed in
/// one recipe because the render route no longer cares. Overridable
/// from the config by name.
pub const SAMPLES: &[(&str, &str)] = &[
    ("ACID", "PLASMA*0.6 + MCUBE + SPARKS3D*0.5"),
    ("VOID", "STARS*0.8 + MWIRE + TUBE*0.4"),
    ("BOOTH", "FLOOR + MSPKR + RINGS*0.5"),
    ("PAPER", "PLATE + IMGDUST*0.5"),
    ("GRID", "RAIN*0.7 + PXTUNNEL + CUBE*0.6"),
    ("LIVE", "PXCAM + SPARKS*0.5"),
];

/// A recipe never expands past this many parts; beyond it a "scene" is
/// a screensaver fighting itself, and the per-cell lottery buffer is
/// sized against it.
pub const MAX_PARTS: usize = 8;

/// `PLASMA*0.6 + MCUBE` → [("PLASMA", 0.6), ("MCUBE", 1.0)]. Garbage
/// tokens vanish rather than veto the line.
pub fn parse_def(rhs: &str) -> Vec<(String, f64)> {
    rhs.split('+')
        .filter_map(|tok| {
            let tok = tok.trim();
            if tok.is_empty() {
                return None;
            }
            let (name, w) = match tok.split_once('*') {
                Some((n, w)) => (n.trim(), w.trim().parse::<f64>().ok()?),
                None => (tok, 1.0),
            };
            if name.is_empty() || !w.is_finite() {
                return None;
            }
            Some((name.to_uppercase(), w.clamp(0.05, 1.0)))
        })
        .take(MAX_PARTS)
        .collect()
}

/// Samples plus the config's `scene.` lines, resolved against the base
/// unit list. Config wins name collisions among defs; the unit list
/// wins collisions against defs, because a scene that shadows a unit
/// would quietly change what a digit key has always selected.
pub fn build(config_scenes: &[(String, String)], base: &[(&'static str, Unit)]) -> Vec<Combo> {
    let mut defs: Vec<(String, String)> = SAMPLES
        .iter()
        .map(|(n, d)| (n.to_string(), d.to_string()))
        .collect();
    for (name, def) in config_scenes {
        let name = name.trim().to_uppercase();
        if name.is_empty() {
            continue;
        }
        match defs.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = def.clone(),
            None => defs.push((name, def.clone())),
        }
    }
    let mut out: Vec<Combo> = Vec::new();
    for (name, def) in defs {
        if base.iter().any(|(n, _)| *n == name) || out.iter().any(|c| c.name == name) {
            continue;
        }
        let parts: Vec<(Unit, f64)> = parse_def(&def)
            .into_iter()
            .filter_map(|(pn, w)| {
                base.iter()
                    .find(|(n, _)| *n == pn)
                    .map(|&(_, u)| (u, w))
            })
            .collect();
        if parts.is_empty() {
            continue;
        }
        // Leaked once per scene per launch — names live as long as the
        // unit list they join.
        out.push(Combo {
            name: Box::leak(name.into_boxed_str()),
            parts,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{PIX_UNITS, Pix};

    /// The real castable surface: canonical cell names + pixel units.
    fn base() -> Vec<(&'static str, Unit)> {
        let mut v: Vec<(&'static str, Unit)> = crate::effects::CELL_NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, Unit::Cell(i)))
            .collect();
        v.extend(PIX_UNITS.iter().map(|(n, p)| (*n, Unit::Pixel(*p))));
        v
    }

    #[test]
    fn a_recipe_parses_names_and_shares() {
        let p = parse_def(" plasma*0.6 + MCUBE +  sparks3d * 0.5 ");
        assert_eq!(
            p,
            vec![
                ("PLASMA".to_string(), 0.6),
                ("MCUBE".to_string(), 1.0),
                ("SPARKS3D".to_string(), 0.5),
            ]
        );
        // Garbage tokens vanish; shares clamp to something playable.
        let p = parse_def("PLASMA*nan + *0.3 + STARS*99 + + TUBE*0.001");
        assert_eq!(
            p,
            vec![("STARS".to_string(), 1.0), ("TUBE".to_string(), 0.05)]
        );
    }

    #[test]
    fn every_builtin_sample_resolves_fully() {
        // A sample naming a unit that no longer exists is dead weight
        // shipped to every gig — each part must resolve against the
        // real tables (PAPER's plate pair included: the cell list is
        // the with-plates one).
        let combos = build(&[], &base());
        assert_eq!(combos.len(), SAMPLES.len());
        for ((name, def), c) in SAMPLES.iter().zip(&combos) {
            assert_eq!(c.name, *name);
            assert_eq!(c.parts.len(), parse_def(def).len(), "{name} lost a part");
        }
        // And the deck is genuinely cross-media somewhere.
        assert!(combos.iter().any(|c| {
            c.parts.iter().any(|(u, _)| matches!(u, Unit::Cell(_)))
                && c.parts.iter().any(|(u, _)| matches!(u, Unit::Pixel(_)))
        }));
    }

    #[test]
    fn a_config_scene_overrides_a_sample_by_name() {
        let cfg = vec![
            ("acid".to_string(), "STARS*0.9".to_string()),
            ("MINE".to_string(), "PLASMA + RINGS*0.5".to_string()),
        ];
        let combos = build(&cfg, &base());
        let acid = combos.iter().find(|c| c.name == "ACID").unwrap();
        assert_eq!(acid.parts, vec![(Unit::Pixel(Pix::Stars), 0.9)]);
        let mine = combos.iter().find(|c| c.name == "MINE").unwrap();
        assert_eq!(mine.parts.len(), 2);
    }

    #[test]
    fn broken_scenes_degrade_instead_of_vetoing() {
        let cfg = vec![
            // A typo'd part plays without it.
            ("HALF".to_string(), "PLASMA + NOPE*0.5".to_string()),
            // Nothing resolvable: dropped whole.
            ("GONE".to_string(), "NOPE + NADA".to_string()),
            // Shadowing a real unit: dropped, the digit keys keep
            // meaning what they meant.
            ("PLASMA".to_string(), "STARS".to_string()),
        ];
        let combos = build(&cfg, &base());
        let half = combos.iter().find(|c| c.name == "HALF").unwrap();
        assert_eq!(half.parts, vec![(Unit::Pixel(Pix::Plasma), 1.0)]);
        assert!(combos.iter().all(|c| c.name != "GONE"));
        assert_eq!(
            combos.iter().filter(|c| c.name == "PLASMA").count(),
            0,
            "a scene must not shadow a unit"
        );
    }

    #[test]
    fn a_scene_cannot_contain_a_scene() {
        // ACID exists as a sample, but a recipe resolves against base
        // units only — one level of recipe, deliberately.
        let cfg = vec![("META".to_string(), "ACID + PLASMA".to_string())];
        let combos = build(&cfg, &base());
        let meta = combos.iter().find(|c| c.name == "META").unwrap();
        assert_eq!(meta.parts, vec![(Unit::Pixel(Pix::Plasma), 1.0)]);
    }

    #[test]
    fn a_runaway_recipe_is_truncated() {
        let long = vec!["PLASMA"; 20].join(" + ");
        assert_eq!(parse_def(&long).len(), MAX_PARTS);
    }
}
