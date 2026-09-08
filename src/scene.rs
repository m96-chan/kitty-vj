//! Scenes — which units are on, and when that changes.
//!
//! `pass.rs` sorts every visual unit by how data moves through it:
//! Source, Transform, Mixer, Modulator. A scene is none of the four, and
//! that is worth saying out loud rather than filing it under the nearest
//! one. It reads no frame and writes no pixel. It is not a Modulator
//! either: a modulator exports numbers every frame and a scene speaks
//! only at the moments it changes, and what it hands over is not a
//! parameter but a cast list — which look is on, which hits are armed,
//! which post chain is loaded, which particle mode is stacked. That is
//! **orchestration**, a shape that sits above the taxonomy instead of
//! inside it: the four shapes describe units, this one chooses them.
//!
//! Over there the same job was done by a `STYLES` table. Each artwork
//! style declared pools — looks, hits, post, accents — and a scene was a
//! draw from its style's pools: 1-2 looks, 2-3 hits, 1-2 post, an accent
//! and a hue. Everything here is that, kept as flat data tables for the
//! same reason it worked over there: at ~88 units the only thing that
//! keeps a pool manageable is that adding to it is an edit to a table,
//! not to a function.
//!
//! ## Why a change waits for the downbeat
//!
//! A scene change fired the instant the operator asks for it lands
//! wherever the hand landed, which on a projector reads as a glitch —
//! the picture broke — rather than as an edit. Fired on the next bar
//! line it reads as a decision. The wait is at most four beats, under
//! two seconds at club tempo, and nothing on screen is idle during it,
//! so the cost is nothing and the difference is the whole illusion that
//! somebody is playing this.
//!
//! ## Why there is an escape hatch
//!
//! The bar line comes from the beat grid, and the grid is the least
//! trustworthy thing in the building: audio detection loses the plot in
//! a breakdown, a resync yanks beat time sideways, a jog wheel drags it
//! backwards. A queue that waits for a downbeat that never arrives
//! stalls the set — the one failure the audience actually notices. So
//! the queue also watches how much beat time has gone past in either
//! direction, and after four seconds' worth it fires anyway. A change
//! landing off the grid is a bad edit; a change that never lands is a
//! broken instrument.

use crate::looks::Look;
use crate::rng::{hash3, unit_f64};
use crate::transition::Transition;

/// Artwork style. Everything a scene draws comes from this style's
/// pools, which is what stops a scene reading as a shuffle of unrelated
/// effects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum Style {
    Neon = 0,
    Deck,
    Ink,
    Poster,
}

/// A momentary, beat-driven treatment. The first three are the ported
/// hits `triggers::beat_hits` fires; the rest are the beat-keyed cell
/// posts, which over there lived in the same pool because the audience
/// cannot tell a flash from a displacement — both are things that
/// happen *on* a beat rather than treatments that stay on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hit {
    InvertFlash,
    ColorFlash,
    Strobe,
    Shake,
    ZoomPunch,
    Slice,
    Glitch,
}

/// A persistent post pass, either tier. One pool covers both because a
/// scene picks by how the frame should look, not by which tier can
/// deliver it; the caller resolves the name against `CELL_POSTS` or
/// `PIX_POSTS`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Post {
    MirrorV,
    MirrorQuad,
    Pixelate,
    Warp,
    Kaleido,
    ZoomBlur,
    RgbSplit,
    Edge,
    Bloom,
    Crt,
    Feedback,
}

/// Which particle mode is stacked over the base pixel effect, if any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Particle {
    Off,
    Sparks,
    PxTunnel,
    Floor,
    Rings,
}

/// How artwork meets the stage. Cover fills and crops, contain shows
/// the whole plate and lets the background through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fit {
    Cover,
    Contain,
}

impl Hit {
    pub fn name(&self) -> &'static str {
        match self {
            Hit::InvertFlash => "INVERT",
            Hit::ColorFlash => "CFLASH",
            Hit::Strobe => "STROBE",
            Hit::Shake => "SHAKE",
            Hit::ZoomPunch => "ZPUNCH",
            Hit::Slice => "SLICE",
            Hit::Glitch => "GLITCH",
        }
    }

    /// The `CELL_POSTS` entry this hit is wired to, when it is one.
    /// `None` means it is one of the three `beat_hits` treatments, which
    /// run unconditionally over the mix and need no slot.
    pub fn cell_post(&self) -> Option<&'static str> {
        match self {
            Hit::InvertFlash | Hit::ColorFlash | Hit::Strobe => None,
            Hit::Shake => Some("SHAKE"),
            Hit::ZoomPunch => Some("ZPUNCH"),
            Hit::Slice => Some("SLICE"),
            Hit::Glitch => Some("GLITCH"),
        }
    }
}

impl Post {
    pub fn name(&self) -> &'static str {
        match self {
            Post::MirrorV => "MIRV",
            Post::MirrorQuad => "MIRQ",
            Post::Pixelate => "PIXEL",
            Post::Warp => "WARP",
            Post::Kaleido => "KALEID",
            Post::ZoomBlur => "ZBLUR",
            Post::RgbSplit => "RGB",
            Post::Edge => "EDGE",
            Post::Bloom => "BLOOM",
            Post::Crt => "CRT",
            Post::Feedback => "FEEDBK",
        }
    }

    /// Name in `CELL_POSTS`, for the passes that run on the cell grid.
    pub fn cell_post(&self) -> Option<&'static str> {
        match self {
            Post::MirrorV | Post::MirrorQuad | Post::Pixelate => Some(self.name()),
            _ => None,
        }
    }

    /// Name in `PIX_POSTS`, for the passes that run on the pixel frame.
    /// A scene may hand back a pixel post while the graphics tier is
    /// off; the caller decides whether it can honour it.
    pub fn pix_post(&self) -> Option<&'static str> {
        match self {
            Post::MirrorV | Post::MirrorQuad | Post::Pixelate => None,
            _ => Some(self.name()),
        }
    }
}

impl Particle {
    /// Name in `PIX_PARTICLES`; `None` when the scene wants no particles.
    pub fn name(&self) -> Option<&'static str> {
        match self {
            Particle::Off => None,
            Particle::Sparks => Some("SPARKS"),
            Particle::PxTunnel => Some("PXTUNNEL"),
            Particle::Floor => Some("FLOOR"),
            Particle::Rings => Some("RINGS"),
        }
    }
}

/// One style's pools. Flat data on purpose: a new effect joins a style
/// by being typed into a list here, which is the only reason a pool this
/// wide stayed editable during a gig.
struct Pools {
    name: &'static str,
    looks: &'static [Look],
    hits: &'static [Hit],
    posts: &'static [Post],
    particles: &'static [Particle],
    fits: &'static [Fit],
    accents: &'static [(u8, u8, u8)],
    /// Hue window the scene's random base is drawn from: (start, span)
    /// in degrees. A style is a colour decision before it is anything
    /// else, so the window is part of the table.
    hue: (f64, f64),
}

const STYLES: [Pools; 4] = [
    // NEON — saturated, moving, everything on. The style that takes the
    // pixel tier's whole post chain.
    Pools {
        name: "NEON",
        looks: &[Look::HueCycle, Look::SatPump, Look::Lut, Look::Duotone],
        hits: &[Hit::Strobe, Hit::ColorFlash, Hit::ZoomPunch, Hit::Glitch],
        posts: &[
            Post::Kaleido,
            Post::Bloom,
            Post::RgbSplit,
            Post::Feedback,
            Post::MirrorQuad,
        ],
        particles: &[Particle::Sparks, Particle::Rings, Particle::PxTunnel],
        fits: &[Fit::Cover],
        accents: &[(0, 255, 213), (255, 0, 200), (120, 80, 255)],
        hue: (150.0, 210.0),
    },
    // DECK — warm, club-lit, hits over treatment.
    Pools {
        name: "DECK",
        looks: &[Look::Duotone, Look::SatPump, Look::Scan, Look::Plain],
        hits: &[Hit::InvertFlash, Hit::Strobe, Hit::Shake, Hit::Slice],
        posts: &[Post::RgbSplit, Post::Crt, Post::ZoomBlur, Post::MirrorV],
        particles: &[Particle::Sparks, Particle::Floor, Particle::Off],
        fits: &[Fit::Cover, Fit::Contain],
        accents: &[(255, 140, 0), (255, 40, 40), (255, 220, 120)],
        hue: (0.0, 60.0),
    },
    // INK — monochrome, cold, sparse. No colour flash: there is no
    // colour to flash.
    Pools {
        name: "INK",
        looks: &[Look::HardMono, Look::Scan, Look::Plain],
        hits: &[Hit::InvertFlash, Hit::Slice, Hit::Glitch],
        posts: &[Post::Edge, Post::Crt, Post::Pixelate],
        particles: &[Particle::Off, Particle::Floor],
        fits: &[Fit::Contain, Fit::Cover],
        accents: &[(235, 235, 235), (150, 170, 190), (90, 110, 130)],
        hue: (190.0, 40.0),
    },
    // POSTER — flat graphic blocks. Deliberately holds no mirror and no
    // punch-in: a poster reads as a composition, and both of those break
    // the composition rather than treating it. Same exclusion as over
    // there.
    Pools {
        name: "POSTER",
        looks: &[Look::Lut, Look::Duotone, Look::HardMono],
        hits: &[Hit::ColorFlash, Hit::InvertFlash, Hit::Shake],
        posts: &[Post::Pixelate, Post::Bloom, Post::Warp],
        particles: &[Particle::Off, Particle::Sparks],
        fits: &[Fit::Contain],
        accents: &[(255, 60, 50), (250, 205, 40), (40, 90, 220)],
        hue: (0.0, 360.0),
    },
];

/// Chance a scene runs two looks rather than one.
const P_TWO_LOOKS: f64 = 0.45;
/// Chance a scene arms a third hit.
const P_THIRD_HIT: f64 = 0.5;
/// Chance a scene loads a second post pass.
const P_TWO_POSTS: f64 = 0.40;
const P_ABCUT: f64 = 0.18;
const P_STUTTER: f64 = 0.22;

// Disjoint salts so no two draws share a hash input.
const SALT_SCENE: u64 = 0x5ce9_e000;
const SALT_COUNTS: u64 = 1;
const SALT_FLAGS: u64 = 2;
const SALT_PALETTE: u64 = 3;
const SALT_ROTATE: u64 = 4;
const SALT_LOOKS: u64 = 10;
const SALT_HITS: u64 = 20;
const SALT_POSTS: u64 = 30;

impl Style {
    pub fn name(&self) -> &'static str {
        STYLES[*self as usize].name
    }

    pub fn next(&self) -> Style {
        match self {
            Style::Neon => Style::Deck,
            Style::Deck => Style::Ink,
            Style::Ink => Style::Poster,
            Style::Poster => Style::Neon,
        }
    }
}

/// Draw `n` distinct entries from a pool, deterministically. Without
/// replacement: a scene running the same look twice is a wasted slot.
fn draw<T: Copy>(pool: &[T], n: usize, seed: u64, salt: u64) -> Vec<T> {
    let mut bag: Vec<T> = pool.to_vec();
    let n = n.min(bag.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let k = unit_f64(hash3(seed, salt, i as u64)) * bag.len() as f64;
        out.push(bag.remove((k as usize).min(bag.len() - 1)));
    }
    out
}

/// One entry from a pool, chosen by a unit random.
fn one<T: Copy>(pool: &[T], u: f64) -> T {
    pool[((u * pool.len() as f64) as usize).min(pool.len() - 1)]
}

/// One rolled set of choices. Everything the renderer needs to know
/// about "what is on right now" that is not an operator's hand.
#[derive(Clone)]
pub struct Scene {
    pub style: Style,
    /// 1-2 persistent colour treatments, applied in order.
    pub looks: Vec<Look>,
    /// 2-3 armed beat hits.
    pub hits: Vec<Hit>,
    /// 1-2 persistent post passes.
    pub posts: Vec<Post>,
    pub particle: Particle,
    pub fit: Fit,
    pub accent: (u8, u8, u8),
    /// Degrees, for the looks that map luminance onto one hue.
    pub hue_base: f64,
    /// Cut back and forth between two sources on the bar.
    pub abcut: bool,
    /// Re-trigger the frame on sixteenths.
    pub stutter: bool,
    /// The seed this scene was rolled from — enough to reproduce it.
    pub seed: u64,
}

impl Scene {
    /// Roll a scene from a style's pools. Pure: same style and seed give
    /// the same scene on every machine and every run, which is what lets
    /// a set be replayed from a seed instead of a recording.
    pub fn roll(style: Style, seed: u64) -> Scene {
        let p = &STYLES[style as usize];
        let s = hash3(seed, style as u64, SALT_SCENE);
        let r = |k: u64| unit_f64(hash3(s, SALT_COUNTS, k));
        let f = |k: u64| unit_f64(hash3(s, SALT_FLAGS, k));
        let c = |k: u64| unit_f64(hash3(s, SALT_PALETTE, k));

        let n_looks = if r(0) < P_TWO_LOOKS { 2 } else { 1 };
        let n_hits = if r(1) < P_THIRD_HIT { 3 } else { 2 };
        let n_posts = if r(2) < P_TWO_POSTS { 2 } else { 1 };

        Scene {
            style,
            looks: draw(p.looks, n_looks, s, SALT_LOOKS),
            hits: draw(p.hits, n_hits, s, SALT_HITS),
            posts: draw(p.posts, n_posts, s, SALT_POSTS),
            particle: one(p.particles, c(1)),
            fit: one(p.fits, c(2)),
            accent: one(p.accents, c(3)),
            hue_base: (p.hue.0 + c(0) * p.hue.1).rem_euclid(360.0),
            abcut: f(0) < P_ABCUT,
            stutter: f(1) < P_STUTTER,
            seed,
        }
    }

    /// The scene's primary look — the one a single-look pipeline uses.
    pub fn look(&self) -> Look {
        self.looks.first().copied().unwrap_or(Look::Plain)
    }

    pub fn has_hit(&self, h: Hit) -> bool {
        self.hits.contains(&h)
    }

    /// One HUD segment, and the readable fingerprint of a roll.
    pub fn hud(&self) -> String {
        let join = |v: Vec<&str>| v.join("+");
        let mut s = format!(
            "{} {} {} {}",
            self.style.name(),
            join(self.looks.iter().map(|l| l.name()).collect()),
            join(self.hits.iter().map(|h| h.name()).collect()),
            join(self.posts.iter().map(|p| p.name()).collect()),
        );
        if let Some(n) = self.particle.name() {
            s.push_str(&format!(" +{n}"));
        }
        s.push_str(&format!(" h{:.0}", self.hue_base));
        if self.abcut {
            s.push_str(" AB");
        }
        if self.stutter {
            s.push_str(" ST");
        }
        s
    }
}

/// Why a change was asked for. It decides nothing about what the scene
/// contains — only how long the cut over to it should take.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChangeReason {
    /// New track. A chapter break, and it should look like one.
    TrackChange,
    /// The rotation timer came round on its own.
    Rotation,
    /// The operator asked.
    Manual,
}

/// How long the cut wants to be. The scene names the length; the caller
/// picks the actual `Transition` from `TRANSITIONS`, because which
/// transitions exist is not a scene's business.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Length {
    /// A chapter change — the audience should notice.
    Long,
    /// An edit — over before it registers as a change of state.
    Short,
}

impl ChangeReason {
    pub fn length(&self) -> Length {
        match self {
            ChangeReason::TrackChange => Length::Long,
            ChangeReason::Rotation | ChangeReason::Manual => Length::Short,
        }
    }
}

impl Length {
    pub fn name(&self) -> &'static str {
        match self {
            Length::Long => "LONG",
            Length::Short => "SHORT",
        }
    }

    /// Pick the transition in `set` that best matches this length, by
    /// the duration each transition asks for. The set stays the
    /// caller's; this only expresses the preference.
    pub fn pick<'a>(&self, set: &[&'a dyn Transition]) -> Option<&'a dyn Transition> {
        let cmp =
            |a: &&&dyn Transition, b: &&&dyn Transition| a.beats().partial_cmp(&b.beats()).unwrap();
        match self {
            Length::Long => set.iter().max_by(cmp).copied(),
            Length::Short => set.iter().min_by(cmp).copied(),
        }
    }
}

/// A change that has fired.
pub struct SceneChange {
    pub scene: Scene,
    /// Why the change fired. Carried for the log and for anything that
    /// wants to treat a track change differently from a rotation; the
    /// length it implies is already resolved in `length`.
    #[allow(dead_code)]
    pub reason: ChangeReason,
    pub length: Length,
    /// True when it landed on a bar line, false when the escape hatch
    /// pushed it out. Worth surfacing: a set full of escapes means the
    /// grid is wrong, and that is the operator's problem to fix.
    pub on_downbeat: bool,
}

struct Pending {
    reason: ChangeReason,
    /// Beat time the request was stamped at, for the downbeat test.
    at: f64,
    /// Beat time travelled since the request, in either direction.
    waited: f64,
}

/// Beats per bar. The downbeat is the only grid line a change is allowed
/// to land on.
const BAR: f64 = 4.0;
/// Escape hatch, in seconds.
const ESCAPE_SECS: f64 = 4.0;
/// Longest a scene runs before rotating itself out, in seconds.
const ROTATE_SECS: f64 = 50.0;
/// Tempo the beat-denominated defaults are derived from until the caller
/// says otherwise.
const REF_TEMPO: f64 = 120.0;

/// Owns the current scene and the queue. The only stateful thing in this
/// module — a scene itself is a value.
pub struct SceneDirector {
    scene: Scene,
    style: Style,
    seed: u64,
    /// Scene index; rolled into every seed so consecutive scenes differ.
    counter: u64,
    pending: Option<Pending>,
    last_beat: f64,
    /// Beat the rotation timer next comes round at.
    next_rotation: f64,
    escape_beats: f64,
    rotate_beats: f64,
}

impl SceneDirector {
    pub fn new(style: Style, seed: u64) -> Self {
        let scene = Scene::roll(style, hash3(seed, 0, SALT_SCENE));
        let mut d = Self {
            scene,
            style,
            seed,
            counter: 0,
            pending: None,
            last_beat: 0.0,
            next_rotation: 0.0,
            escape_beats: ESCAPE_SECS * REF_TEMPO / 60.0,
            rotate_beats: ROTATE_SECS * REF_TEMPO / 60.0,
        };
        d.schedule_rotation(0.0);
        d
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn style(&self) -> Style {
        self.style
    }

    /// Which style the next roll draws from. Takes effect on the next
    /// change, not immediately: swapping pools under a live scene is how
    /// a frame ends up half in one style and half in another.
    pub fn set_style(&mut self, style: Style) {
        self.style = style;
    }

    /// Re-derive both timers from the running tempo. Everything here is
    /// counted in beats — beats are the only clock the rest of the
    /// instrument agrees on — but the durations came from the original
    /// in seconds, so this is where the two meet.
    pub fn set_tempo(&mut self, bpm: f64) {
        let bpm = bpm.max(1.0);
        self.escape_beats = ESCAPE_SECS * bpm / 60.0;
        self.rotate_beats = ROTATE_SECS * bpm / 60.0;
    }

    /// Longest a scene may run before rotating itself, in beats.
    /// Tuning knobs for a room: a support slot wants faster rotation
    /// than a three-hour set. Not wired to a key yet.
    #[allow(dead_code)]
    pub fn set_rotate_beats(&mut self, beats: f64) {
        self.rotate_beats = beats.max(1.0);
        self.schedule_rotation(self.last_beat);
    }

    #[allow(dead_code)]
    pub fn set_escape_beats(&mut self, beats: f64) {
        self.escape_beats = beats.max(0.0);
    }

    /// Beat the rotation timer fires at.
    /// How far off the next automatic change is — for a HUD countdown.
    #[allow(dead_code)]
    pub fn next_rotation(&self) -> f64 {
        self.next_rotation
    }

    /// What is queued, if anything.
    pub fn pending(&self) -> Option<ChangeReason> {
        self.pending.as_ref().map(|p| p.reason)
    }

    /// Queue a change. Never fires here — see the module note on why the
    /// bar line is worth waiting for. A second request while one is
    /// queued upgrades the reason (a track change outranks a rotation)
    /// but keeps the original stamp, so asking twice cannot reset the
    /// escape hatch and strand the queue.
    pub fn request(&mut self, reason: ChangeReason) {
        match &mut self.pending {
            Some(p) => {
                if reason == ChangeReason::TrackChange {
                    p.reason = reason;
                }
            }
            None => {
                self.pending = Some(Pending {
                    reason,
                    at: self.last_beat,
                    waited: 0.0,
                })
            }
        }
    }

    /// Advance to `beat`. Returns a change on the frame it fires.
    pub fn update(&mut self, beat: f64) -> Option<SceneChange> {
        let prev = self.last_beat;
        let step = beat - prev;
        self.last_beat = beat;

        // A bar line crossed since the last call. Backwards motion is
        // not a crossing: a resync that drags beat time back over a bar
        // line has not played a downbeat, it has rewritten one.
        let crossed = if step > 0.0 {
            let db = (beat / BAR).floor() * BAR;
            (db > prev).then_some(db)
        } else {
            None
        };

        if self.pending.is_none() && beat >= self.next_rotation {
            self.request(ChangeReason::Rotation);
        }

        let p = self.pending.as_mut()?;
        p.waited += step.abs();
        let on_downbeat = crossed.is_some_and(|db| db >= p.at);
        if !on_downbeat && p.waited < self.escape_beats {
            return None;
        }

        let reason = p.reason;
        self.pending = None;
        self.counter += 1;
        self.scene = Scene::roll(self.style, hash3(self.seed, self.counter, SALT_SCENE));
        self.schedule_rotation(beat);
        Some(SceneChange {
            scene: self.scene.clone(),
            reason,
            length: reason.length(),
            on_downbeat,
        })
    }

    /// Next self-rotation, somewhere in the back half of the window —
    /// "up to 50 seconds" over there, never metronomic, because a scene
    /// change that lands on a predictable clock stops reading as a
    /// decision.
    fn schedule_rotation(&mut self, from: f64) {
        let u = unit_f64(hash3(self.seed, self.counter, SALT_ROTATE));
        self.next_rotation = from + self.rotate_beats * (0.5 + 0.5 * u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transition::TRANSITIONS;

    const STYLE_LIST: [Style; 4] = [Style::Neon, Style::Deck, Style::Ink, Style::Poster];

    fn director() -> SceneDirector {
        let mut d = SceneDirector::new(Style::Neon, 1234);
        // Pin the timers so a test is not at the mercy of a rolled
        // interval: 8 beats of escape, 64 of rotation.
        d.set_escape_beats(8.0);
        d.set_rotate_beats(64.0);
        d
    }

    #[test]
    fn a_roll_draws_the_documented_counts() {
        for style in STYLE_LIST {
            for seed in 0..500u64 {
                let s = Scene::roll(style, seed);
                assert!(
                    (1..=2).contains(&s.looks.len()),
                    "{} looks {}",
                    style.name(),
                    s.looks.len()
                );
                assert!(
                    (2..=3).contains(&s.hits.len()),
                    "{} hits {}",
                    style.name(),
                    s.hits.len()
                );
                assert!(
                    (1..=2).contains(&s.posts.len()),
                    "{} posts {}",
                    style.name(),
                    s.posts.len()
                );
                // Without replacement: a repeated slot is a wasted one.
                for (i, h) in s.hits.iter().enumerate() {
                    assert!(!s.hits[i + 1..].contains(h), "duplicate hit {}", h.name());
                }
                for (i, p) in s.posts.iter().enumerate() {
                    assert!(!s.posts[i + 1..].contains(p), "duplicate post {}", p.name());
                }
                assert!((0.0..360.0).contains(&s.hue_base));
            }
        }
    }

    #[test]
    fn the_second_look_lands_near_its_stated_odds() {
        let n = 4000;
        let two = (0..n)
            .filter(|&i| Scene::roll(Style::Neon, i).looks.len() == 2)
            .count() as f64
            / n as f64;
        assert!(
            (0.40..0.50).contains(&two),
            "two-look share {two} is not the documented 45%"
        );
        let two_post = (0..n)
            .filter(|&i| Scene::roll(Style::Neon, i).posts.len() == 2)
            .count() as f64
            / n as f64;
        assert!(
            (0.35..0.45).contains(&two_post),
            "two-post share {two_post} is not the documented 40%"
        );
    }

    #[test]
    fn flags_fire_at_their_stated_odds() {
        let n = 4000;
        let ab = (0..n)
            .filter(|&i| Scene::roll(Style::Deck, i).abcut)
            .count() as f64
            / n as f64;
        let st = (0..n)
            .filter(|&i| Scene::roll(Style::Deck, i).stutter)
            .count() as f64
            / n as f64;
        assert!((0.13..0.23).contains(&ab), "abcut {ab}");
        assert!((0.17..0.27).contains(&st), "stutter {st}");
    }

    #[test]
    fn the_same_seed_reproduces_exactly() {
        for seed in [0u64, 7, 99, 0xdead_beef] {
            for style in STYLE_LIST {
                let a = Scene::roll(style, seed);
                let b = Scene::roll(style, seed);
                assert_eq!(a.hud(), b.hud());
                assert_eq!(a.seed, seed, "a scene must carry the seed that made it");
                assert!(a.look() == b.look());
                assert_eq!(a.accent, b.accent);
                assert_eq!(a.fit, b.fit);
                assert_eq!(a.hue_base.to_bits(), b.hue_base.to_bits());
                assert_eq!(a.abcut, b.abcut);
                assert_eq!(a.stutter, b.stutter);
            }
        }
    }

    #[test]
    fn different_seeds_give_different_scenes() {
        let hud: std::collections::HashSet<String> = (0..200u64)
            .map(|s| Scene::roll(Style::Neon, s).hud())
            .collect();
        assert!(hud.len() > 20, "only {} distinct scenes in 200", hud.len());
    }

    #[test]
    fn styles_draw_from_their_own_pools() {
        for seed in 0..400u64 {
            let p = Scene::roll(Style::Poster, seed);
            // A poster is a composition; mirrors and punch-ins break one
            // rather than treating it, exactly as over there.
            assert!(
                !p.posts.contains(&Post::MirrorV) && !p.posts.contains(&Post::MirrorQuad),
                "poster drew a mirror: {}",
                p.hud()
            );
            assert!(!p.has_hit(Hit::ZoomPunch), "poster drew a punch-in");
            // Ink has no colour to flash and no colour post.
            let i = Scene::roll(Style::Ink, seed);
            assert!(!i.has_hit(Hit::ColorFlash), "ink drew a colour flash");
            assert!(!i.posts.contains(&Post::RgbSplit), "ink drew an rgb split");
            assert!(i.looks.iter().all(|l| *l != Look::HueCycle));
        }
        // And the pools are not just exclusions: neon reaches things
        // nobody else has.
        let neon: Vec<Scene> = (0..400u64).map(|s| Scene::roll(Style::Neon, s)).collect();
        assert!(neon.iter().any(|s| s.posts.contains(&Post::Kaleido)));
        assert!(neon.iter().any(|s| s.has_hit(Hit::ZoomPunch)));
    }

    #[test]
    fn pooled_names_match_the_wiring_tables() {
        // The caller resolves these strings against CELL_POSTS,
        // PIX_POSTS and PIX_PARTICLES in main.rs. A typo here is a
        // silent no-op at showtime, so it is caught here instead.
        const CELL: [&str; 7] = [
            "SLICE", "GLITCH", "MIRV", "MIRQ", "PIXEL", "ZPUNCH", "SHAKE",
        ];
        const PIX: [&str; 8] = [
            "WARP", "KALEID", "ZBLUR", "RGB", "EDGE", "BLOOM", "CRT", "FEEDBK",
        ];
        const PART: [&str; 4] = ["SPARKS", "PXTUNNEL", "FLOOR", "RINGS"];
        for style in STYLE_LIST {
            let p = &STYLES[style as usize];
            for h in p.hits {
                if let Some(n) = h.cell_post() {
                    assert!(CELL.contains(&n), "{n} is not a CELL_POSTS entry");
                }
            }
            for post in p.posts {
                let n = post.name();
                assert!(
                    CELL.contains(&n) || PIX.contains(&n),
                    "{n} is in no post table"
                );
                assert!(post.cell_post().is_some() != post.pix_post().is_some());
            }
            for part in p.particles {
                if let Some(n) = part.name() {
                    assert!(PART.contains(&n), "{n} is not a PIX_PARTICLES entry");
                }
            }
        }
    }

    #[test]
    fn a_request_fires_on_the_next_downbeat_and_not_before() {
        let mut d = director();
        d.update(1.0);
        d.request(ChangeReason::Manual);
        for b in [1.5, 2.0, 3.0, 3.99] {
            assert!(d.update(b).is_none(), "fired early at beat {b}");
        }
        let c = d.update(4.05).expect("should fire on the bar line");
        assert!(c.on_downbeat);
        assert!(d.pending().is_none());
    }

    #[test]
    fn a_request_made_on_a_downbeat_waits_for_the_next_one() {
        let mut d = director();
        d.update(8.0); // crosses the bar line at 8
        d.request(ChangeReason::Manual);
        assert!(d.update(8.5).is_none());
        assert!(d.update(11.9).is_none());
        assert!(d.update(12.1).is_some(), "next bar line should fire it");
    }

    #[test]
    fn the_escape_hatch_fires_when_no_downbeat_arrives() {
        // An unreliable grid: beat time jitters back and forth just shy
        // of the bar line, so a crossing never happens. The set must not
        // stall on it.
        let mut d = director();
        d.update(3.5);
        d.request(ChangeReason::Rotation);
        let mut fired = None;
        for i in 0..40 {
            let b = if i % 2 == 0 { 3.9 } else { 3.4 };
            if let Some(c) = d.update(b) {
                fired = Some((i, c));
                break;
            }
        }
        let (i, c) = fired.expect("escape hatch never fired");
        assert!(!c.on_downbeat, "that was not a downbeat");
        // Eight beats of travel at half a beat a step.
        assert!(i >= 15, "fired after only {i} steps");
    }

    #[test]
    fn a_track_change_is_long_and_a_rotation_is_short() {
        let mut d = director();
        d.update(1.0);
        d.request(ChangeReason::TrackChange);
        let c = d.update(4.1).unwrap();
        assert_eq!(c.reason, ChangeReason::TrackChange);
        assert_eq!(c.length, Length::Long);

        d.request(ChangeReason::Rotation);
        let c = d.update(8.1).unwrap();
        assert_eq!(c.length, Length::Short);

        d.request(ChangeReason::Manual);
        let c = d.update(12.1).unwrap();
        assert_eq!(c.length, Length::Short);
    }

    #[test]
    fn a_track_change_outranks_a_queued_rotation() {
        let mut d = director();
        d.update(1.0);
        d.request(ChangeReason::Rotation);
        d.request(ChangeReason::TrackChange);
        let c = d.update(4.1).unwrap();
        assert_eq!(c.length, Length::Long);
    }

    #[test]
    fn length_picks_the_transition_it_asks_for() {
        // Deliberately not tied to which transitions exist — the set is
        // the caller's, and it grows.
        let hi = TRANSITIONS
            .iter()
            .map(|t| t.beats())
            .fold(f64::MIN, f64::max);
        let lo = TRANSITIONS
            .iter()
            .map(|t| t.beats())
            .fold(f64::MAX, f64::min);
        let long = Length::Long.pick(&TRANSITIONS).unwrap();
        let short = Length::Short.pick(&TRANSITIONS).unwrap();
        assert_eq!(long.beats(), hi, "{} is not the longest", long.name());
        assert_eq!(short.beats(), lo, "{} is not the shortest", short.name());
        assert!(long.beats() > short.beats(), "the set has one length only");
        assert!(Length::Short.pick(&[]).is_none());
    }

    #[test]
    fn the_rotation_timer_comes_round_on_its_own() {
        let mut d = director();
        let due = d.next_rotation();
        // In the back half of the 64-beat window, never metronomic.
        assert!((32.0..=64.0).contains(&due), "rotation due at {due}");
        let mut b = 0.0;
        while b < due - 0.5 {
            b += 0.5;
            assert!(d.update(b).is_none(), "rotated early at {b}");
            assert!(d.pending().is_none(), "queued early at {b}");
        }
        d.update(due);
        assert_eq!(
            d.pending(),
            Some(ChangeReason::Rotation),
            "the timer should queue, not fire"
        );
        // ... and then land on the bar line like any other change.
        let next_bar = (due / BAR).floor() * BAR + BAR;
        let c = d.update(next_bar + 0.01).expect("rotation should fire");
        assert_eq!(c.length, Length::Short);
        assert!(d.next_rotation() > next_bar, "the timer should rearm");
    }

    #[test]
    fn a_change_rolls_a_new_scene() {
        let mut d = director();
        let before = d.scene().hud();
        d.update(1.0);
        d.request(ChangeReason::Manual);
        let c = d.update(4.1).unwrap();
        assert_eq!(c.scene.hud(), d.scene().hud());
        // Not a hard guarantee for one draw, but over a run of changes
        // the director must not be stuck on one scene.
        let mut seen = std::collections::HashSet::new();
        seen.insert(before);
        for i in 0..20 {
            d.request(ChangeReason::Manual);
            let c = d.update(8.1 + i as f64 * 4.0).unwrap();
            seen.insert(c.scene.hud());
        }
        assert!(seen.len() > 5, "director stuck: {} scenes", seen.len());
    }

    #[test]
    fn a_director_run_is_reproducible() {
        let run = || {
            let mut d = SceneDirector::new(Style::Deck, 42);
            let mut log = Vec::new();
            for i in 0..400 {
                let b = i as f64 * 0.25;
                if i == 40 {
                    d.request(ChangeReason::TrackChange);
                }
                if let Some(c) = d.update(b) {
                    log.push(format!("{b:.2} {} {}", c.length.name(), c.scene.hud()));
                }
            }
            log
        };
        assert_eq!(run(), run());
        assert!(!run().is_empty(), "nothing changed in 100 beats");
    }

    #[test]
    fn a_new_style_takes_effect_on_the_next_change_only() {
        let mut d = SceneDirector::new(Style::Neon, 5);
        assert_eq!(d.style(), Style::Neon);
        d.update(1.0);
        d.set_style(Style::Ink);
        assert_eq!(
            d.scene().style,
            Style::Neon,
            "the live scene must not change pools under itself"
        );
        d.request(ChangeReason::TrackChange);
        let c = d.update(4.1).unwrap();
        assert_eq!(c.scene.style, Style::Ink);
    }

    #[test]
    fn the_timers_follow_the_tempo() {
        let mut d = SceneDirector::new(Style::Neon, 5);
        // Four seconds and fifty seconds, counted in beats.
        d.set_tempo(120.0);
        assert!((d.escape_beats - 8.0).abs() < 1e-9);
        assert!((d.rotate_beats - 100.0).abs() < 1e-9);
        d.set_tempo(174.0);
        assert!((d.escape_beats - 11.6).abs() < 1e-9);
        assert!((d.rotate_beats - 145.0).abs() < 1e-9);
        // A stopped clock must not divide by zero into an infinite wait.
        d.set_tempo(0.0);
        assert!(d.escape_beats.is_finite() && d.escape_beats > 0.0);
    }

    #[test]
    fn styles_cycle() {
        let mut s = Style::Neon;
        for _ in 0..4 {
            s = s.next();
        }
        assert_eq!(s, Style::Neon);
    }
}
