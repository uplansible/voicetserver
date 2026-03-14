// Global hotkey listener for toggling dictation on/off.
// Uses rdev for cross-platform low-level keyboard hooking.

use anyhow::{bail, Result};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const STATE_READY: u8 = 0;
pub const STATE_ACTIVE: u8 = 1;
pub const STATE_PAUSED: u8 = 2;

/// Parse a hotkey name string into an rdev::Key.
pub fn parse_hotkey(s: &str) -> Result<rdev::Key> {
    match s {
        "F1" => Ok(rdev::Key::F1),
        "F2" => Ok(rdev::Key::F2),
        "F3" => Ok(rdev::Key::F3),
        "F4" => Ok(rdev::Key::F4),
        "F5" => Ok(rdev::Key::F5),
        "F6" => Ok(rdev::Key::F6),
        "F7" => Ok(rdev::Key::F7),
        "F8" => Ok(rdev::Key::F8),
        "F9" => Ok(rdev::Key::F9),
        "F10" => Ok(rdev::Key::F10),
        "F11" => Ok(rdev::Key::F11),
        "F12" => Ok(rdev::Key::F12),
        "ScrollLock" => Ok(rdev::Key::ScrollLock),
        "Pause" => Ok(rdev::Key::Pause),
        "PrintScreen" => Ok(rdev::Key::PrintScreen),
        _ => bail!(
            "Unknown hotkey '{}'. Valid keys: F1-F12, ScrollLock, Pause, PrintScreen",
            s
        ),
    }
}

/// Spawn a background thread that listens for the given key and toggles
/// the shared state between ACTIVE and PAUSED. Includes 200ms debounce.
pub fn spawn_hotkey_listener(key: rdev::Key, state: Arc<AtomicU8>) {
    std::thread::spawn(move || {
        let mut last_press = Instant::now();
        let debounce = std::time::Duration::from_millis(200);

        if let Err(e) = rdev::listen(move |event| {
            if let rdev::EventType::KeyPress(k) = event.event_type {
                if k == key && last_press.elapsed() >= debounce {
                    last_press = Instant::now();
                    let current = state.load(Ordering::SeqCst);
                    if current == STATE_ACTIVE {
                        state.store(STATE_PAUSED, Ordering::SeqCst);
                        eprint!("[paused]\n");
                    } else {
                        state.store(STATE_ACTIVE, Ordering::SeqCst);
                        eprint!("[recording]\n");
                    }
                }
            }
        }) {
            eprintln!("Hotkey listener error: {:?}", e);
        }
    });
}
