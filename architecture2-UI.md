# Voicet UI Architecture

## Overview

System tray icon and settings window. Tray icon shows recording state (Active=green, Paused=red, Loading=grey). Left-click toggles Active/Paused. Right-click opens settings. Settings persist to `settings.ini`. Console window hidden in release builds.

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

## Settings Window — Subprocess Model

The settings window runs as a separate process (`voicet.exe --settings-ui`), not a thread. This avoids winit's EventLoop limitation (can only be created once per process — closing and reopening the window would fail).

- Parent saves current `settings.ini` before spawning
- Subprocess reads `settings.ini`, shows egui window, writes on OK
- Parent reloads `settings.ini` when subprocess exits
- Child process handle prevents opening multiple windows
- Exit code 99 = "Quit Voicet" button → parent shuts down

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

No title bar, always on top, positioned lower-right above taskbar.

## State Machine

`STATE_PAUSED=0`, `STATE_ACTIVE=1`, `STATE_LOADING=2`

```
    startup ──► Loading (grey) ──model loaded──► Active (green)
                                                    │
                                              hotkey / click
                                                    │
                                                    ▼
                                               Paused (red)
                                                    │
                                              hotkey / click
                                                    │
                                                    ▼
                                               Active (green)
                                                  ...
```

Toggle ignored while Loading. Tray icon polls state every 100ms.

## Startup Sequence

```
1. Check --settings-ui flag → if present, run settings window and exit
2. Load settings.ini → IniValues (defaults for missing keys)
3. Parse CLI args → override IniValues where provided
4. Construct SharedSettings with STATE_LOADING
5. Spawn tray thread (icon visible during model load)
6. Load model
7. Set state to STATE_ACTIVE
8. Call run_streaming (spawns hotkey thread internally)
```

## Files

| File | Purpose |
|---|---|
| `src/settings.rs` | SharedSettings (atomics), AtomicF32, INI parser/writer, reload_from_file |
| `src/tray.rs` | Tray icon event loop, settings subprocess management |
| `src/settings_window.rs` | Standalone egui settings window (subprocess mode) |
| `assets/*.rgba` | 32x32 tray icons (green, grey, red) embedded via include_bytes |
