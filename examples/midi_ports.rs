use midir::MidiInput;
fn main() {
    let mi = MidiInput::new("kitty-vj-scan").unwrap();
    for p in mi.ports() {
        println!("port: {}", mi.port_name(&p).unwrap_or_default());
    }
}
