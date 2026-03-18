# Plan: Floating Overlay Window for Closed Captioning

## Summary

System-wide floating overlay window that displays real-time closed captioning. Always-on-top, draggable, semi-transparent dark background with white text. Works with both single-stream and dual-delay modes. New `--overlay` CLI flag.

## GUI Crate: egui + eframe (Recommended)

- Immediate-mode GUI, perfect for text-only display updating every 80ms
- Cross-platform (Windows + Linux) out of the box
- `egui::ViewportBuilder` has `.with_always_on_top()`, `.with_transparent(true)`, `.with_decorations(false)`
- Built-in font rendering, text wrapping, scroll areas
- Runs event loop on a separate thread, receives data via mpsc channel
- Binary size increase ~5-10MB on top of existing ~35MB (negligible given 9GB model weights)
- Use `eframe::Renderer::Glow` (OpenGL) backend for faster compile time than wgpu

## Architecture

### Communication: mpsc channel, inference thread never blocks

```
Inference thread (main loop)          Overlay thread (egui event loop)
        |                                       |
    emit_token()                                |
        |                                       |
    OutputSink::Overlay                         |
        |----> mpsc::Sender<OverlayMessage> --->|
        |      (non-blocking send)              |
        |                                   try_recv() loop in update()
        |                                   updates display text
        |                                   egui repaints
```

### Message Protocol

```rust
enum OverlayMessage {
    /// Append confirmed text (single-stream, or slow stream in dual-delay)
    AppendConfirmed(String),
    /// Set speculative text (fast-ahead dim text, replaced each tick)
    SetSpeculative(String),
    /// Paragraph break (silence detected)
    NewParagraph,
    /// Clear all text (on pause/restart)
    Clear,
}
```

### Thread Model

1. **Main/inference thread** — existing streaming loop unchanged. `OutputSink::Overlay` sends `OverlayMessage` via `mpsc::Sender`.
2. **Overlay GUI thread** — spawned before streaming loop. Runs `eframe::run_native()`. Each frame drains `mpsc::Receiver` and renders 2-3 lines of text on a semi-transparent panel.
3. **No shared mutable state** — channel is the only connection. Inference thread never waits on overlay.

## Files to Create/Modify

### New: `src/overlay.rs` (~200 lines)

- `OverlayMessage` enum
- `OverlayApp` struct implementing `eframe::App`
- `spawn_overlay()` — creates channel, spawns GUI thread, returns `Sender`

### Modify: `src/streaming.rs`

- Add `Overlay(mpsc::Sender<overlay::OverlayMessage>)` variant to `OutputSink`
- Implement `emit_text()` / `emit_newline()` for Overlay variant
- Create `DualDisplayOverlay` for dual-delay overlay mode (same `compute_speculative()` logic, sends channel messages instead of ANSI codes)
- In `run_streaming()`: if overlay mode, call `overlay::spawn_overlay()`, construct `OutputSink::Overlay(sender)`
- In `run_dual_streaming()`: branch for overlay vs terminal DualDisplay

### Modify: `src/main.rs`

- Add `--overlay` CLI flag to `Cli` struct
- Validate: `--overlay` and `--type` are mutually exclusive
- Add `overlay: bool` to `StreamConfig`
- Add `mod overlay;`
- Print in config table if enabled

### Modify: `Cargo.toml`

- Add `eframe = "0.30"` dependency

## Implementation Steps

### Step 1: Add dependency and module skeleton

1. Add `eframe = "0.30"` to `Cargo.toml`
2. Create `src/overlay.rs` with `OverlayMessage` enum and placeholder `spawn_overlay()`
3. Add `mod overlay;` to `main.rs`
4. Verify it compiles

### Step 2: Implement the overlay window

In `src/overlay.rs`:

**State:**
```rust
struct OverlayApp {
    rx: mpsc::Receiver<OverlayMessage>,
    lines: VecDeque<String>,  // rolling buffer of completed lines
    current_line: String,      // in-progress line
    speculative: String,       // dim text (dual-delay)
    max_visible_lines: usize,  // 2-3
}
```

**eframe::App::update():**
1. Drain all pending messages via `try_recv()` loop
2. `AppendConfirmed(text)` → append to `current_line`; word-wrap if > ~80 chars (push first part to `lines`, keep remainder)
3. `SetSpeculative(text)` → replace `speculative`
4. `NewParagraph` → push `current_line` to `lines`, start new line
5. `Clear` → empty everything
6. Cap `lines` to 5-6 entries, drop oldest

**Rendering:**
- `egui::CentralPanel` with custom `egui::Frame`:
  - Fill: `Color32::from_rgba_premultiplied(15, 15, 15, 210)` (dark semi-transparent)
  - Rounding: `Rounding::same(8.0)`
- Completed lines: `RichText::new(text).color(Color32::WHITE).size(20.0)`
- Current line: white confirmed + gray speculative: `RichText::new(spec).color(Color32::from_rgb(140, 140, 140)).size(20.0)`

**spawn_overlay():**
```rust
pub fn spawn_overlay() -> Result<mpsc::Sender<OverlayMessage>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let options = eframe::NativeOptions {
            renderer: eframe::Renderer::Glow,
            viewport: egui::ViewportBuilder::default()
                .with_always_on_top()
                .with_transparent(true)
                .with_decorations(false)
                .with_inner_size([600.0, 100.0])
                .with_position([100.0, 800.0]),
            ..Default::default()
        };
        let _ = eframe::run_native(
            "Voicet CC",
            options,
            Box::new(move |_cc| Ok(Box::new(OverlayApp::new(rx)))),
        );
    });
    Ok(tx)
}
```

**Dragging (no title bar):**
- In `update()`, detect drag on the panel area via `egui::Response::drag_delta()`
- Apply via `ctx.send_viewport_cmd(ViewportCommand::SetOuterPosition(...))`

### Step 3: Integrate OutputSink::Overlay into single-stream mode

In `src/streaming.rs`:

```rust
enum OutputSink {
    Stdout,
    Keyboard(enigo::Enigo),
    Overlay(mpsc::Sender<overlay::OverlayMessage>),
}
```

Implement emit methods:
```rust
OutputSink::Overlay(tx) => {
    let _ = tx.send(OverlayMessage::AppendConfirmed(text.to_string()));
}
```

In `run_streaming()`, before the loop:
```rust
let mut sink = if config.overlay {
    let tx = overlay::spawn_overlay()?;
    OutputSink::Overlay(tx)
} else if config.type_mode {
    OutputSink::Keyboard(...)
} else {
    OutputSink::Stdout
};
```

### Step 4: Add CLI flag

In `src/main.rs`:
```rust
/// Display transcription in a floating overlay window
#[arg(long)]
overlay: bool,
```

Validation:
```rust
if cli.overlay && cli.type_mode {
    anyhow::bail!("--overlay and --type cannot be used together");
}
if cli.overlay && cli.dual_delay {
    // This can work, just note it in config table
}
```

Add `overlay: cli.overlay` to `StreamConfig`.

### Step 5: Handle dual-delay overlay mode

Create `DualDisplayOverlay` that mirrors `DualDisplay` but sends channel messages:

```rust
struct DualDisplayOverlay {
    slow_text: String,
    fast_text: String,
    tx: mpsc::Sender<OverlayMessage>,
    last_confirmed_len: usize,
}

impl DualDisplayOverlay {
    fn push_fast_token(&mut self, token: u32, tok: &Tokenizer) { /* same as DualDisplay */ }
    fn push_slow_token(&mut self, token: u32, tok: &Tokenizer) { /* same */ }

    fn refresh(&mut self) {
        // Send new confirmed text
        if self.last_confirmed_len < self.slow_text.len() {
            let new = self.slow_text[self.last_confirmed_len..].to_string();
            let _ = self.tx.send(OverlayMessage::AppendConfirmed(new));
            self.last_confirmed_len = self.slow_text.len();
        }
        // Send speculative (reuses same compute_speculative algorithm)
        let spec = self.compute_speculative();
        let _ = self.tx.send(OverlayMessage::SetSpeculative(spec));
    }

    fn compute_speculative(&self) -> String { /* same reverse-matching algorithm */ }

    fn emit_newline(&mut self) {
        let _ = self.tx.send(OverlayMessage::NewParagraph);
        self.slow_text.clear();
        self.fast_text.clear();
        self.last_confirmed_len = 0;
    }
}
```

In `run_dual_streaming()`, branch on `config.overlay`:
```rust
if config.overlay {
    let tx = overlay::spawn_overlay()?;
    let mut display = DualDisplayOverlay::new(tx);
    // ... use display
} else {
    let mut display = DualDisplay::new();
    // ... existing code
}
```

### Step 6: Visual polish

- Dark semi-transparent background with rounded corners
- White confirmed text, gray speculative text, ~20px font
- Show only last 2-3 lines (drop oldest from `lines` deque)
- No scroll bar needed for 2-3 lines
- Window size: ~600x100, positioned bottom-center of screen by default
- Prevent accidental close: intercept close event in egui `update()`, require Ctrl+C from terminal to quit

## Platform Notes

**Windows (primary):**
- eframe handles transparency and always-on-top via ViewportBuilder
- Does NOT interfere with `RegisterHotKey` in `hotkey.rs` (separate message loops)
- Use `Glow` renderer for faster compile

**Linux (future):**
- Same `src/overlay.rs` works on both platforms, no code changes
- X11: always-on-top via `_NET_WM_STATE_ABOVE`; transparency requires compositor
- Wayland: works but may not stay above fullscreen apps

## Challenges to Watch For

1. **`eframe::run_native()` blocks** — must be spawned on a dedicated thread (not main thread)
2. **Overlay closes early** — if user Alt-F4s the overlay, `tx.send()` silently fails, program continues without overlay. Consider intercepting close.
3. **Compile time** — eframe + glow adds ~30-60s to first build. Cached thereafter.
4. **DualDisplay code duplication** — `compute_speculative()` is duplicated between `DualDisplay` (terminal) and `DualDisplayOverlay`. Consider extracting to a shared function if maintaining both becomes burdensome.

## Future Enhancements (out of scope)

- Font size hotkey
- Opacity slider
- Click-through mode toggle (`WS_EX_TRANSPARENT` on Windows)
- Position persistence (save to config file)
- Custom color scheme
- `--type` mode integration (type confirmed text, overlay shows speculative)
