use rusty_link::{AblLink, SessionState};
fn main() {
    let link = AblLink::new(120.0);
    link.enable(true);
    let mut st = SessionState::new();
    for i in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        link.capture_app_session_state(&mut st);
        let t = link.clock_micros();
        println!(
            "{}: peers={} tempo={:.2} beat={:.2}",
            i,
            link.num_peers(),
            st.tempo(),
            st.beat_at_time(t, 4.0)
        );
    }
}
