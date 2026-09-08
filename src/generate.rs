//! Generative plugin boundary — LLM, image, video, music behind one
//! interface.
//!
//! The rule that shapes everything here: **generation never blocks a
//! frame.** Every generator runs off the render thread; the loop only
//! ever reads the latest *ready* artifact. A model that hangs, dies, or
//! returns garbage costs the set nothing. This is the README's
//! llama.cpp-as-separate-process rule generalized to all four modalities.
//!
//! The boundary and its guarantees are here and tested; the adapters
//! (llama.cpp, image, video, audio) plug in behind it — hence the types
//! nothing constructs yet.
#![allow(dead_code)]

use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

/// What a generator is allowed to know about the show.
#[derive(Clone, Copy)]
pub struct GenCtx {
    pub beat: f64,
    pub tempo: f64,
    pub intensity: f64,
}

/// A finished piece of generated content.
pub enum Artifact {
    /// DSL diff / caption text.
    Text(String),
    /// A plate, ready to drop into the rotation.
    Image(image::RgbImage),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Health {
    /// Nothing requested yet.
    Idle,
    /// A request is out; the deadline hasn't passed.
    Working,
    /// An artifact is waiting to be taken.
    Ready,
    /// Timed out or the worker died. Harmless — nothing is blocked.
    Failed,
}

/// One generator: request and poll, never block.
pub trait Generator {
    fn kind(&self) -> Modality;
    /// Fire a request. Must return immediately.
    fn request(&mut self, ctx: &GenCtx);
    /// Take the latest ready artifact, if any.
    fn poll(&mut self) -> Option<Artifact>;
    fn health(&self) -> Health;
}

/// Wraps a worker that sends artifacts down a channel, adding the
/// timeout and the health tracking every generator owes the instrument.
/// A worker is free to be a thread, a subprocess, or an HTTP client —
/// this side only sees the channel.
pub struct Worker {
    kind: Modality,
    rx: Receiver<Artifact>,
    pending: Option<Instant>,
    timeout: Duration,
    ready: Option<Artifact>,
    health: Health,
}

impl Worker {
    pub fn new(kind: Modality, rx: Receiver<Artifact>, timeout: Duration) -> Self {
        Self {
            kind,
            rx,
            pending: None,
            timeout,
            ready: None,
            health: Health::Idle,
        }
    }

    /// Called by the app once per frame. Never blocks: a try_recv, a
    /// clock comparison, nothing else.
    pub fn pump(&mut self) {
        match self.rx.try_recv() {
            Ok(a) => {
                self.ready = Some(a);
                self.pending = None;
                self.health = Health::Ready;
            }
            Err(TryRecvError::Empty) => {
                // Still waiting — but only until the deadline.
                if let Some(started) = self.pending
                    && started.elapsed() > self.timeout
                {
                    self.pending = None;
                    self.health = Health::Failed;
                }
            }
            Err(TryRecvError::Disconnected) => {
                // The worker is gone. The show does not care.
                self.pending = None;
                if self.ready.is_none() {
                    self.health = Health::Failed;
                }
            }
        }
    }

    /// Mark a request as sent (the caller does the actual sending).
    pub fn mark_requested(&mut self) {
        self.pending = Some(Instant::now());
        if self.health != Health::Ready {
            self.health = Health::Working;
        }
    }
}

impl Generator for Worker {
    fn kind(&self) -> Modality {
        self.kind
    }

    fn request(&mut self, _ctx: &GenCtx) {
        self.mark_requested();
    }

    fn poll(&mut self) -> Option<Artifact> {
        let a = self.ready.take();
        if a.is_some() {
            self.health = if self.pending.is_some() {
                Health::Working
            } else {
                Health::Idle
            };
        }
        a
    }

    fn health(&self) -> Health {
        self.health
    }
}

/// All active generators. The kill switch is the point: one call and the
/// whole subsystem is inert for the rest of the set.
#[derive(Default)]
pub struct Registry {
    gens: Vec<Box<dyn Generator>>,
    pub enabled: bool,
}

impl Registry {
    pub fn add(&mut self, g: Box<dyn Generator>) {
        self.gens.push(g);
    }

    pub fn is_empty(&self) -> bool {
        self.gens.is_empty()
    }

    /// Pull every ready artifact. Called once per frame; cheap and
    /// non-blocking by construction.
    pub fn drain(&mut self) -> Vec<Artifact> {
        if !self.enabled {
            return Vec::new();
        }
        self.gens.iter_mut().filter_map(|g| g.poll()).collect()
    }

    /// Compact status for the HUD: one letter per generator.
    pub fn hud(&self) -> String {
        if self.gens.is_empty() {
            return String::new();
        }
        let s: String = self
            .gens
            .iter()
            .map(|g| match (self.enabled, g.health()) {
                (false, _) => '-',
                (_, Health::Idle) => '·',
                (_, Health::Working) => '*',
                (_, Health::Ready) => '+',
                (_, Health::Failed) => '!',
            })
            .collect();
        format!(" │ AI:{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn ready_artifact_is_taken_once() {
        let (tx, rx) = channel();
        let mut w = Worker::new(Modality::Text, rx, Duration::from_secs(1));
        w.request(&ctx());
        tx.send(Artifact::Text("hi".into())).unwrap();
        w.pump();
        assert_eq!(w.health(), Health::Ready);
        assert!(matches!(w.poll(), Some(Artifact::Text(t)) if t == "hi"));
        assert!(w.poll().is_none(), "artifact taken twice");
    }

    #[test]
    fn slow_worker_times_out_without_blocking() {
        let (_tx, rx) = channel::<Artifact>();
        let mut w = Worker::new(Modality::Image, rx, Duration::from_millis(1));
        w.request(&ctx());
        std::thread::sleep(Duration::from_millis(5));
        w.pump();
        assert_eq!(w.health(), Health::Failed);
        assert!(w.poll().is_none());
    }

    #[test]
    fn dead_worker_is_survivable() {
        let (tx, rx) = channel::<Artifact>();
        drop(tx); // the model process died
        let mut w = Worker::new(Modality::Text, rx, Duration::from_secs(1));
        w.request(&ctx());
        w.pump();
        assert_eq!(w.health(), Health::Failed);
        assert!(w.poll().is_none());
    }

    #[test]
    fn kill_switch_silences_everything() {
        let (tx, rx) = channel();
        let mut w = Worker::new(Modality::Text, rx, Duration::from_secs(1));
        tx.send(Artifact::Text("x".into())).unwrap();
        w.pump();
        let mut reg = Registry::default();
        reg.add(Box::new(w));
        assert!(reg.drain().is_empty(), "disabled registry must stay silent");
        reg.enabled = true;
        assert_eq!(reg.drain().len(), 1);
    }

    fn ctx() -> GenCtx {
        GenCtx {
            beat: 0.0,
            tempo: 128.0,
            intensity: 0.5,
        }
    }
}
