//! Ableton Link session (rusty_link). rekordbox 6+ in PERFORMANCE mode
//! speaks Link natively — over loopback on one machine, no LAN needed.
//! If Link is silent, suspect UDP multicast blocked by the firewall.

use rusty_link::{AblLink, SessionState};

pub struct LinkSync {
    link: AblLink,
    state: SessionState,
}

impl LinkSync {
    pub fn new(bpm: f64) -> Self {
        let link = AblLink::new(bpm);
        link.enable(true);
        Self {
            link,
            state: SessionState::new(),
        }
    }

    pub fn peers(&self) -> u64 {
        self.link.num_peers()
    }

    /// Session tempo and beat (quantum 4) right now.
    pub fn capture(&mut self) -> (f64, f64) {
        self.link.capture_app_session_state(&mut self.state);
        let t = self.link.clock_micros();
        (self.state.tempo(), self.state.beat_at_time(t, 4.0))
    }
}

impl Drop for LinkSync {
    fn drop(&mut self) {
        self.link.enable(false);
    }
}
