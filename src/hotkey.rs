// Global hotkey listener.
//
// On Windows: uses RegisterHotKey (no keyboard hook at all). rdev's WH_KEYBOARD_LL
// hook called AttachThreadInput + GetKeyboardState + ToUnicodeEx for every keyboard
// event, which interfered with enigo's SendInput and caused character reordering.
// RegisterHotKey avoids this entirely — no hook, no interference.
//
// On Linux: uses rdev::listen().

use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::Arc;

use crate::settings::SharedSettings;

pub const STATE_PAUSED: u8 = 0;
pub const STATE_ACTIVE: u8 = 1;
pub const STATE_LOADING: u8 = 2;

/// Supported hotkeys — single source of truth for parse/stringify/UI.
pub const SUPPORTED_KEYS: &[(&str, rdev::Key)] = &[
    ("F1", rdev::Key::F1),
    ("F2", rdev::Key::F2),
    ("F3", rdev::Key::F3),
    ("F4", rdev::Key::F4),
    ("F5", rdev::Key::F5),
    ("F6", rdev::Key::F6),
    ("F7", rdev::Key::F7),
    ("F8", rdev::Key::F8),
    ("F9", rdev::Key::F9),
    ("F10", rdev::Key::F10),
    ("F11", rdev::Key::F11),
    ("F12", rdev::Key::F12),
    ("ScrollLock", rdev::Key::ScrollLock),
    ("Pause", rdev::Key::Pause),
    ("PrintScreen", rdev::Key::PrintScreen),
];

/// Toggle recording state: Active ↔ Paused. Ignored while loading.
pub fn toggle_state(state: &AtomicU8, source: &str) {
    let current = state.load(Ordering::SeqCst);
    if current == STATE_LOADING { return; }
    if current == STATE_ACTIVE {
        state.store(STATE_PAUSED, Ordering::SeqCst);
        eprintln!("[paused]{}", source);
    } else {
        state.store(STATE_ACTIVE, Ordering::SeqCst);
        eprintln!("[recording]{}", source);
    }
}

pub fn parse_hotkey(s: &str) -> Result<rdev::Key> {
    SUPPORTED_KEYS
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, key)| *key)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown hotkey '{}'. Valid keys: F1-F12, ScrollLock, Pause, PrintScreen",
                s
            )
        })
}

/// Convert an rdev::Key to its string name (reverse of parse_hotkey).
pub fn key_to_string(key: rdev::Key) -> String {
    SUPPORTED_KEYS
        .iter()
        .find(|(_, k)| *k == key)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

// ============================================================
// Windows: RegisterHotKey (no keyboard hook)
// ============================================================

#[cfg(target_os = "windows")]
fn rdev_key_to_vk(key: rdev::Key) -> u32 {
    match key {
        rdev::Key::F1 => 0x70,
        rdev::Key::F2 => 0x71,
        rdev::Key::F3 => 0x72,
        rdev::Key::F4 => 0x73,
        rdev::Key::F5 => 0x74,
        rdev::Key::F6 => 0x75,
        rdev::Key::F7 => 0x76,
        rdev::Key::F8 => 0x77,
        rdev::Key::F9 => 0x78,
        rdev::Key::F10 => 0x79,
        rdev::Key::F11 => 0x7A,
        rdev::Key::F12 => 0x7B,
        rdev::Key::ScrollLock => 0x91,
        rdev::Key::Pause => 0x13,
        rdev::Key::PrintScreen => 0x2C,
        _ => 0,
    }
}

#[cfg(target_os = "windows")]
pub fn spawn_listener(
    _running: Arc<AtomicBool>,
    settings: Arc<SharedSettings>,
    hotkey_thread_id: Arc<AtomicU32>,
) {
    std::thread::spawn(move || {
        #[repr(C)]
        struct Msg { hwnd: isize, message: u32, wparam: usize, lparam: isize, time: u32, pt: [i32; 2] }
        extern "system" {
            fn RegisterHotKey(hwnd: isize, id: i32, mods: u32, vk: u32) -> i32;
            fn UnregisterHotKey(hwnd: isize, id: i32) -> i32;
            fn GetMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
            fn GetCurrentThreadId() -> u32;
        }
        const MOD_NOREPEAT: u32 = 0x4000;

        unsafe {
            hotkey_thread_id.store(GetCurrentThreadId(), Ordering::SeqCst);

            let initial_key = settings.hotkey.lock().unwrap().clone();
            if let Some(key) = initial_key {
                if RegisterHotKey(0, 1, MOD_NOREPEAT, rdev_key_to_vk(key)) == 0 {
                    eprintln!("Failed to register hotkey");
                }
            }

            loop {
                let mut msg = std::mem::zeroed::<Msg>();
                if GetMessageW(&mut msg, 0, 0, 0) <= 0 { break; }
                match msg.message {
                    0x0312 /* WM_HOTKEY */ if msg.wparam as i32 == 1 => {
                        toggle_state(&settings.state, "");
                    }
                    0x0400 /* WM_USER */ => {
                        UnregisterHotKey(0, 1);
                        if msg.wparam != 0 {
                            RegisterHotKey(0, 1, MOD_NOREPEAT, msg.wparam as u32);
                        }
                    }
                    _ => {}
                }
            }
        }
    });
}

/// Post a message to the hotkey thread to change the registered hotkey.
#[cfg(target_os = "windows")]
pub fn change_hotkey(thread_id: &AtomicU32, new_key: Option<rdev::Key>) {
    let tid = thread_id.load(Ordering::SeqCst);
    if tid == 0 { return; }
    let vk = new_key.map(|k| rdev_key_to_vk(k) as usize).unwrap_or(0);
    extern "system" {
        fn PostThreadMessageW(tid: u32, msg: u32, wparam: usize, lparam: isize) -> i32;
    }
    unsafe { PostThreadMessageW(tid, 0x0400, vk, 0); }
}

// ============================================================
// Linux: rdev::listen()
// ============================================================

#[cfg(not(target_os = "windows"))]
pub fn spawn_listener(
    running: Arc<AtomicBool>,
    settings: Arc<SharedSettings>,
    _hotkey_thread_id: Arc<AtomicU32>,
) {
    use std::time::Instant;

    std::thread::spawn(move || {
        let mut ctrl_held = false;
        let mut last_toggle = Instant::now();
        let debounce = std::time::Duration::from_millis(200);

        if let Err(e) = rdev::listen(move |event| match event.event_type {
            rdev::EventType::KeyPress(k) => {
                if matches!(k, rdev::Key::ControlLeft | rdev::Key::ControlRight) {
                    ctrl_held = true;
                }
                if k == rdev::Key::KeyC && ctrl_held {
                    running.store(false, Ordering::SeqCst);
                }
                let current_hotkey = settings.hotkey.lock().unwrap().clone();
                if let Some(hk) = current_hotkey {
                    if k == hk && last_toggle.elapsed() >= debounce {
                        last_toggle = Instant::now();
                        toggle_state(&settings.state, "");
                    }
                }
            }
            rdev::EventType::KeyRelease(k) => {
                if matches!(k, rdev::Key::ControlLeft | rdev::Key::ControlRight) {
                    ctrl_held = false;
                }
            }
            _ => {}
        }) {
            eprintln!("Keyboard listener error: {:?}", e);
        }
    });
}

/// No-op on Linux: rdev listener reads settings.hotkey mutex each keypress.
#[cfg(not(target_os = "windows"))]
pub fn change_hotkey(_thread_id: &AtomicU32, _new_key: Option<rdev::Key>) {}
