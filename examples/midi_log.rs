use midir::MidiInput;
use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel::<String>();
    let mut conns = Vec::new();
    let scan = MidiInput::new("kvj-log").unwrap();
    let n = scan.ports().len();
    for i in 0..n {
        let input = MidiInput::new("kvj-log").unwrap();
        let port = input.ports().into_iter().nth(i).unwrap();
        let name = input.port_name(&port).unwrap_or_default();
        let txc = tx.clone();
        if let Ok(c) = input.connect(
            &port,
            "kvj-log-in",
            move |_t, msg, txc| {
                if let [s, d1, d2] = msg {
                    let kind = match s & 0xf0 {
                        0x90 if *d2 > 0 => "on ",
                        0x90 | 0x80 => "off",
                        0xb0 => "cc ",
                        _ => return,
                    };
                    let _ = txc.send(format!("{kind} ch{} note{} vel{}", s & 0x0f, d1, d2));
                }
            },
            txc,
        ) {
            conns.push((c, name));
        }
    }
    eprintln!("listening on {} ports", conns.len());
    let start = std::time::Instant::now();
    while start.elapsed().as_secs() < 60 {
        if let Ok(line) = rx.recv_timeout(std::time::Duration::from_millis(200)) {
            println!("{:6.2}s {line}", start.elapsed().as_secs_f64());
        }
    }
}
