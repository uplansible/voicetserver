# PLAN1-UE.md -- System Tray UI for Voicet

## Overview

System tray icon and settings window for voicet. Tray icon shows recording state (3 icons: Active=green, Paused=red, Loading=grey). Left-click toggles Active/Paused. Right-click opens settings. Settings persist to `settings.ini`. Console window hidden via `windows_subsystem = "windows"`.

## Architecture

```
                    ┌──────────────────────────┐
                    │     Tray icon thread      │
                    │  (tray-icon event loop)   │
                    │                           │
                    │  Left-click → toggle      │
                    │    AtomicU8 state          │
                    │  Right-click → spawn      │
                    │    settings subprocess     │
                    └─────────┬────────────────┘
                              │ writes to shared atomics
    ┌─────────────┐    ┌──────┴──────────┐    ┌───────────────┐
    │ Audio thread │───►│ Inference thread │◄───│ Hotkey thread  │
    │   (cpal)     │    │     (main)       │    │ (RegisterHotKey│
    └─────────────┘    └─────────────────┘    │  or rdev)      │
                              │                └───────────────┘
                              ▼
                         OutputSink
                    (Keyboard / Discard)
```

**Threads:**
1. Audio thread (cpal callback)
2. Inference thread (main thread, runs `run_streaming`)
3. Hotkey thread (RegisterHotKey / rdev)
4. Tray thread (`tray-icon` event loop)

**Settings window** runs as a separate subprocess (`voicet.exe --settings-ui`), not a thread. This avoids winit's EventLoop re-creation limitation (can only be created once per process).

**States:** `STATE_PAUSED=0`, `STATE_ACTIVE=1`, `STATE_LOADING=2`

**Startup sequence:**
```
1. Load settings.ini → IniValues (defaults for missing keys)
2. Parse CLI args → override IniValues where provided
3. Construct SharedSettings with STATE_LOADING
4. Spawn tray thread (icon visible during model load)
5. Load model
6. Set state to STATE_ACTIVE
7. Call run_streaming (spawns hotkey thread internally)
```

---

## Phase 0: Dependencies and project setup — COMPLETED

### Step 0.1: Add crate dependencies — COMPLETED

```toml
tray-icon = "0.19"
eframe = { version = "0.31", default-features = false, features = ["default_fonts", "glow"] }
winit = "0.30"
```

### Step 0.2: Create icon assets — COMPLETED

`assets/icon_active.rgba` (green), `assets/icon_paused.rgba` (grey), `assets/icon_unloaded.rgba` (red). 32x32 raw RGBA, embedded via `include_bytes!()`.

### Step 0.3: Hide console window — COMPLETED

`#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`

---

## Phase 1: Shared settings infrastructure — COMPLETED

### Step 1.1: `src/settings.rs` — SharedSettings struct — COMPLETED

7 GUI-adjustable settings as atomics in an `Arc`-shared struct. `AtomicF32` via bit reinterpretation. `IniValues` intermediate struct for INI + CLI merging.

### Step 1.2: Settings.ini parser/writer — COMPLETED

`load_ini(path)` / `save_settings(path, settings)` / `reload_from_file(settings, path, hotkey_thread_id)`. Missing keys use defaults. File location: same directory as executable.

### Step 1.3: `src/streaming.rs` — read from SharedSettings — COMPLETED

`run_streaming()` takes `&Arc<SharedSettings>`. Atomic loads in the streaming loop. Output mode swap check per iteration.

### Step 1.4: `src/hotkey.rs` — runtime hotkey change — COMPLETED

Windows: `RegisterHotKey` + `WM_USER` message to change hotkey at runtime. Linux: `rdev::listen` reads hotkey mutex each keypress. `SUPPORTED_KEYS` array as single source of truth. ~~`delay_up_key`/`delay_down_key` removed.~~

---

## Phase 2: System tray icon — COMPLETED

### Step 2.1: `src/tray.rs` — COMPLETED

No menu — right-click directly opens settings subprocess. Left-click toggles state. Polls state every ~100ms, updates icon if changed. Windows: raw `PeekMessage`/`DispatchMessage` loop.

Icon mapping: `STATE_ACTIVE` → green, `STATE_PAUSED` → red, `STATE_LOADING` → grey.

### Step 2.2: Spawn tray thread — COMPLETED

Spawned BEFORE model loading so icon is visible during load (shows grey). Transitions to green once model is ready.

---

## Phase 3: Settings window (egui) — COMPLETED

### Step 3.1: `src/settings_window.rs` — COMPLETED

Runs as a standalone subprocess (`voicet.exe --settings-ui`). Reads `settings.ini` on open, writes on OK. No shared atomics — just file I/O. Parent reloads file when subprocess exits.

```
┌─ voicet settings ─────────────────┐
│  Delay              [▼ 4 ▲]      │
│  Silence threshold  [▼ 0.006 ▲]  │
│  Paragraph break    [▼ 18 ▲]     │
│  Min speech         [▼ 12 ▲]     │
│  EMA smoothing      [▼ 0.30 ▲]   │
│  Hotkey             [ F9      ▼ ] │
│  Output mode     [Type ○ / ○ None]│
│                                   │
│  [Quit Voicet]      [  OK  ][Cancel]│
└───────────────────────────────────┘
```

No title bar (decorations off), always on top, positioned lower-right above taskbar. OK saves and closes. Cancel discards. Quit Voicet saves and exits the entire app (exit code 99).

### Step 3.2: Hotkey selector — COMPLETED

ComboBox dropdown using `hotkey::SUPPORTED_KEYS`. Hotkey change applied when parent reloads `settings.ini` after subprocess exits.

### ~~Step 3.3: Connect to tray menu~~ — DELETED

No tray menu. Right-click spawns settings subprocess directly. Child process handle tracks whether settings is already open.

### Step 3.4: Save settings — COMPLETED

OK writes to `settings.ini`. Cancel discards. Drop saves only if dirty (safety net for Alt+F4).

---

## Phase 4: Integration — COMPLETED

### Step 4.1: main.rs restructure — COMPLETED

Early `--settings-ui` check before clap parsing (subprocess returns immediately). ~~`StreamConfig` removed~~, replaced by `SharedSettings`. `OutputSink::Discard` added.

---

## Files created

| File | Purpose |
|---|---|
| `src/settings.rs` | SharedSettings, AtomicF32, INI parser/writer, reload_from_file |
| `src/tray.rs` | Tray icon, event loop, settings subprocess management |
| `src/settings_window.rs` | Standalone egui settings window (subprocess) |
| `assets/*.rgba` | Tray icons (green, grey, red) |

## Files modified

| File | Changes |
|---|---|
| `Cargo.toml` | Add tray-icon, eframe, winit |
| `src/main.rs` | windows_subsystem, --settings-ui subprocess mode, tray spawn before model load |
| `src/streaming.rs` | SharedSettings replaces StreamConfig, OutputSink::Discard |
| `src/hotkey.rs` | SharedSettings, runtime hotkey change, SUPPORTED_KEYS, state constants |

## Files NOT modified

All ML pipeline files (`encoder.rs`, `decoder.rs`, `adapter.rs`, `mel.rs`, `common.rs`, `tokenizer.rs`), `candle-fork/`, `build.rs`.
