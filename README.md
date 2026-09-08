# kitty-vj

A VJ instrument that outputs to a terminal. ASCII and full pixels from the same scene graph,
clocked to a DJ set. Some parameters are live-coded by a small local LLM.

**kitty only.** Not a portability goal — a design decision. See [Why kitty only](#why-kitty-only).

## Status

**Phase 0 — internal-clock PoC. Nothing runs yet.**

Ableton Link, MIDI, and the LLM are stubbed behind interfaces. The first job is getting the
skeleton to hold: clock → scene → renderer.

## Why

ORCA, Bonzomatic, and the Algorave scene already put text on the projector. What's thin is
treating the terminal itself as a serious VJ output. The kitty graphics protocol removed the
reason to stay inside an 80x24 character grid, and nobody really went through that door.

Frames that don't land on time can be hidden or shown. Which one is a performance decision,
not a bug.

## Why kitty only

Supporting every terminal means targeting the intersection of their capabilities, which is
roughly "print some text and hope." kitty has four things that matter here, and giving them up
to gain portability is a bad trade:

- **Shared-memory graphics transfer** (`t=s`) — pixel frames never go through the escape-sequence
  bottleneck at all. This is the difference between full-pixel VJ being viable and not.
- **Synchronized output** — frames update atomically. No tearing mid-composite.
- **Unicode placeholders** — images can be positioned in the text grid, so the ASCII layer and
  the pixel layer composite in a single pass instead of fighting over the screen.
- **The keyboard protocol** — real key press/release events with modifiers, which live coding needs
  and which a normal TTY cannot give you.

sixel was considered and dropped. It saturates on escape-sequence bandwidth, and the workarounds
(drop resolution, drop framerate) are exactly the constraints kitty removes.

## Architecture

```
                  ┌─────────────┐
   ClockSource ──→│             │
                  │ SceneGraph  │──→ Renderer ──→ kitty
   MIDI (CC)  ───→│             │      ├─ ascii
                  └─────────────┘      ├─ halfblock (▀ fg/bg)
                         ↑             └─ graphics protocol
                         │
                    LLM (DSL diff)
```

### ClockSource

Everything sits on top of this. It's a trait so the source can be swapped.

```rust
trait ClockSource {
    fn beat(&self) -> f64;              // cumulative beats; fractional part is phase
    fn tempo(&self) -> f64;
    fn phase(&self, quantum: f64) -> f64;
}
```

The signature matches Ableton Link's session state, so moving over is one more impl.

| Impl | Status | Notes |
|---|---|---|
| `InternalClock` | planned | `beat = elapsed * tempo / 60.0`. Tempo tapped or nudged from the keyboard |
| `LinkClock` | not started | rekordbox 6+ supports Link natively. The workhorse on a DDJ rig |
| `ProLinkClock` | someday | Standalone CDJ/XDJ, or XDJ-XZ/RX3 with rekordbox in PERFORMANCE mode |

While the clock is internal, **time can run backwards.** Keep it deterministic: fixed timestep
plus a fixed seed should reproduce a frame exactly. Collapse-oriented visuals are undebuggable
without this, and the property is lost the moment Link is attached.

### Renderer

One scene graph, swappable renderers, three tiers of granularity:

1. **ascii** — the character grid, unadorned
2. **halfblock** — `▀` with separate fg/bg, doubling vertical resolution at the same cell count
3. **graphics** — full pixels over the kitty graphics protocol, shared-memory transfer

Animation frames (`a=f`) are worth exploring for preloaded loops — push frames once, cycle them
terminal-side, spend no bandwidth per frame.

### LLM

llama.cpp runs as a **separate process behind HTTP.** If the model dies, the instrument keeps
playing.

- **Grammar-constrained decoding** (GBNF). A small hand-written DSL grammar makes syntax errors
  structurally impossible. A 2B model will write bad code; it should not be able to write
  *unparseable* code.
- Emit **diffs against the scene graph** — add a layer, rewrite a parameter — never whole programs.
- Stream the tokens to screen as they arrive. The jitter in decode speed becomes a groove.
- Map `temperature` / `top_p` / `repeat_penalty` to physical faders. Low temp is a stable loop,
  high temp is collapse. That's a buildup you can perform with your hands.
- `logit_bias` to skew the glyph set — effectively a character-set switch.
- **Hide latency with double buffering.** Render the current scene while generating the next one,
  hot-swap on the bar line. Surface the "thinking" state only when it's wanted on screen.

A feedback loop — downsample the current frame to ASCII, feed it back as prompt — is a separate
switch, for when the point is to lose control.

## Roadmap

- [ ] **1. Clock** — internal clock with a PLL; print bars and beats to stdout
- [ ] **2. Renderer** — three tiers, switching on the clock
- [ ] **3. MIDI** — CCs wired straight to renderer parameters
- [ ] **4. LLM** — inject DSL diffs

**Stop at step 3 and make it giggable.** It's insurance against the model misbehaving on stage,
and it might reveal that the LLM was never the good part.

Start the DSL at ~10 verbs and grind on "rewrite exactly one line" before adding anything.

## Stack

Rust.

| | crate |
|---|---|
| TUI | `ratatui` |
| MIDI | `midir` |
| Ableton Link | `rusty_link` |
| Audio in | `cpal` |
| LLM | llama.cpp (subprocess / HTTP) |

Python has all of this too, but pushing large escape sequences every frame runs into the GIL
and I/O. Fine for ASCII, not for full pixels at 60fps.

## Rig (DDJ)

```
rekordbox (Ableton Link) ──LAN──┐
                                 ├─→ kitty-vj ─→ kitty
dedicated MIDI controller ──USB─┘
```

rekordbox owns the DDJ's faders, so they don't reach the VJ machine. **Add a separate controller
for VJ duty** — nanoKONTROL2, Midi Fighter Twister, whatever. Splitting the DJ's hands from the
VJ's hands is easier to operate anyway.

Link works over loopback on a single machine. No physical LAN required.

### Gotchas

- If Link goes silent, suspect **UDP multicast blocked by the firewall** before anything else.
- Beat packets jitter by a few milliseconds. Triggering directly off them shakes; run a local PLL
  and pull phase toward the packets.
- (Pro DJ Link) CDJs use `169.254.x.x` link-local. Bind explicitly or you'll listen on the wrong
  interface and receive nothing, silently.

## Someday

rekordbox export sticks carry `PIONEER/rekordbox/export.pdb`, which is reverse-engineered
(`crate-digger`, `rekordcrate`, `pyrekordbox`). It holds:

- the full beat grid, start to end
- memory cues and hot cues — **the drops are known in advance**
- 3-band color waveforms (`.EXT`) — band energy without running an FFT

A 2B model is slow, but this means **everything can be baked offline.** Feed it the whole track
list, pre-generate scenes per track and per section, and let Pro DJ Link say "this track, this
position" at showtime. Switching to a buildup scene 16 bars before the drop becomes deterministic
rather than reactive.

Requires Pro DJ Link, so it's dead on a DDJ rig. Revisit after moving to XDJ.

## Prior art

| | |
|---|---|
| ORCA (Hundred Rabbits) | Grid-based live coding that reads as visuals on its own |
| Bonzomatic | Shader battles where the code on screen is the show |
| TidalCycles / Hydra | Projecting the editor is the convention |
| notcurses | The unhinged end of terminal output; does video playback |
| chafa / libcaca / cava | |
