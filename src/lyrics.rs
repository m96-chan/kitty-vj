//! Lyric cards. LRC files (`[mm:ss.xx] line`) drive timed text through
//! the big-glyph overlay.
//!
//! Lyric time is TRACK time, not beat time — they diverge the moment the
//! DJ nudges or loops — so the track clock here is started by hand (or by
//! Pro DJ Link later) and runs on wall time, nudgeable to correct drift.

use std::time::Instant;

#[derive(Debug, PartialEq)]
pub struct Line {
    /// Seconds from the start of the track.
    pub at: f64,
    pub text: String,
    /// Enhanced LRC word timings: (seconds, word).
    pub words: Vec<(f64, String)>,
}

/// Parse an LRC file. Tolerates junk lines, metadata tags, and multiple
/// timestamps on one line (a repeated chorus).
pub fn parse(text: &str) -> Vec<Line> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let mut rest = raw.trim();
        let mut stamps = Vec::new();
        // Leading [..] groups: timestamps, or metadata like [ar:...].
        while rest.starts_with('[') {
            let Some(end) = rest.find(']') else { break };
            let inner = &rest[1..end];
            if let Some(t) = parse_stamp(inner) {
                stamps.push(t);
            }
            rest = rest[end + 1..].trim_start();
        }
        if stamps.is_empty() {
            continue;
        }
        // Enhanced LRC: <mm:ss.xx> before each word.
        let (text, words) = parse_words(rest);
        if text.is_empty() {
            continue;
        }
        for at in stamps {
            out.push(Line {
                at,
                text: text.clone(),
                words: words.clone(),
            });
        }
    }
    out.sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap());
    out
}

/// `mm:ss.xx` / `mm:ss` / `h:mm:ss.xx` → seconds.
fn parse_stamp(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let mut secs = 0.0;
    for p in &parts[..parts.len() - 1] {
        secs = secs * 60.0 + p.trim().parse::<f64>().ok()?;
    }
    secs = secs * 60.0 + parts[parts.len() - 1].trim().parse::<f64>().ok()?;
    Some(secs)
}

fn parse_words(s: &str) -> (String, Vec<(f64, String)>) {
    if !s.contains('<') {
        return (s.trim().to_string(), Vec::new());
    }
    let mut words = Vec::new();
    let mut plain = String::new();
    let mut rest = s;
    while let Some(open) = rest.find('<') {
        plain.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        let stamp = &rest[open + 1..open + close];
        rest = &rest[open + close + 1..];
        let word_end = rest.find('<').unwrap_or(rest.len());
        let word = rest[..word_end].to_string();
        if let Some(t) = parse_stamp(stamp) {
            words.push((t, word.trim().to_string()));
        }
        plain.push_str(&word);
        rest = &rest[word_end..];
    }
    plain.push_str(rest);
    (plain.trim().to_string(), words)
}

/// Playback state over a parsed lyric set.
pub struct Lyrics {
    lines: Vec<Line>,
    /// When the track started, if running.
    started: Option<Instant>,
    /// Manual offset in seconds, for drift correction.
    offset: f64,
    pub enabled: bool,
}

impl Lyrics {
    pub fn new(lines: Vec<Line>) -> Self {
        Self {
            lines,
            started: None,
            offset: 0.0,
            enabled: false,
        }
    }

    /// Load `lyrics/<name>.lrc`; empty set if there's no file.
    pub fn load(dir: &std::path::Path, name: &str) -> Self {
        let path = dir.join(format!("{name}.lrc"));
        let lines = std::fs::read_to_string(path)
            .map(|t| parse(&t))
            .unwrap_or_default();
        Self::new(lines)
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Mark the downbeat of the track — the lyric clock starts here.
    pub fn start(&mut self) {
        self.started = Some(Instant::now());
        self.offset = 0.0;
        self.enabled = true;
    }

    pub fn stop(&mut self) {
        self.started = None;
        self.enabled = false;
    }

    /// Shift the lyric clock (positive = lyrics come sooner).
    pub fn nudge(&mut self, secs: f64) {
        self.offset += secs;
    }

    pub fn elapsed(&self) -> Option<f64> {
        self.started
            .map(|s| s.elapsed().as_secs_f64() + self.offset)
    }

    /// The line to show now, plus progress through it [0,1].
    pub fn current(&self) -> Option<(&Line, f64)> {
        let t = self.elapsed()?;
        let i = self.lines.partition_point(|l| l.at <= t).checked_sub(1)?;
        let line = &self.lines[i];
        let end = self.lines.get(i + 1).map(|n| n.at).unwrap_or(line.at + 4.0);
        let span = (end - line.at).max(0.01);
        Some((line, ((t - line.at) / span).clamp(0.0, 1.0)))
    }

    /// How many words of the current line have been sung (enhanced LRC).
    pub fn words_done(&self) -> usize {
        let Some(t) = self.elapsed() else { return 0 };
        self.current()
            .map(|(l, _)| l.words.iter().filter(|(at, _)| *at <= t).count())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_lrc() {
        let l = parse("[ar:Someone]\n[00:12.50]first line\n[01:05.00]second\njunk\n");
        assert_eq!(l.len(), 2);
        assert!((l[0].at - 12.5).abs() < 1e-9);
        assert_eq!(l[0].text, "first line");
        assert!((l[1].at - 65.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_timestamps_expand() {
        let l = parse("[00:10.00][00:40.00]chorus\n");
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].text, "chorus");
        assert!((l[1].at - 40.0).abs() < 1e-9);
    }

    #[test]
    fn parses_enhanced_word_timings() {
        let l = parse("[00:05.00]<00:05.00>hello <00:05.50>world\n");
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].text, "hello world");
        assert_eq!(l[0].words.len(), 2);
        assert_eq!(l[0].words[1].1, "world");
        assert!((l[0].words[1].0 - 5.5).abs() < 1e-9);
    }

    #[test]
    fn picks_the_current_line() {
        let mut ly = Lyrics::new(parse("[00:00.00]one\n[00:10.00]two\n[00:20.00]three\n"));
        assert!(ly.current().is_none(), "not started = nothing to show");
        ly.start();
        ly.nudge(12.0); // pretend 12s in
        let (line, prog) = ly.current().unwrap();
        assert_eq!(line.text, "two");
        assert!(prog > 0.15 && prog < 0.35, "progress {prog}");
    }

    #[test]
    fn missing_file_is_empty_not_a_crash() {
        let ly = Lyrics::load(std::path::Path::new("/nonexistent"), "nope");
        assert!(ly.is_empty());
        assert!(ly.current().is_none());
    }
}
