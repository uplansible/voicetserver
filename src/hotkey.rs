// Global hotkey listener.
//
// On Windows: uses RegisterHotKey (no keyboard hook at all). rdev's WH_KEYBOARD_LL
// hook called AttachThreadInput + GetKeyboardState + ToUnicodeEx for every keyboard
// event, which interfered with enigo's SendInput and caused character reordering.
// RegisterHotKey avoids this entirely — no hook, no interference.
//
// On Linux: uses rdev::listen().

use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

pub const STATE_READY: u8 = 0;
pub const STATE_ACTIVE: u8 = 1;
pub const STATE_PAUSED: u8 = 2;

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
    hotkey: Option<rdev::Key>,
    state: Option<Arc<AtomicU8>>,
) {
    let (Some(key), Some(state)) = (hotkey, state) else { return };
    let vk = rdev_key_to_vk(key);

    std::thread::spawn(move || {
        #[repr(C)]
        struct Msg { hwnd: isize, message: u32, wparam: usize, lparam: isize, time: u32, pt: [i32; 2] }
        extern "system" {
            fn RegisterHotKey(hwnd: isize, id: i32, mods: u32, vk: u32) -> i32;
            fn GetMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
        }
        const MOD_NOREPEAT: u32 = 0x4000;

        unsafe {
            if RegisterHotKey(0, 1, MOD_NOREPEAT, vk) == 0 {
                eprintln!("Failed to register hotkey");
                return;
            }
            loop {
                let mut msg = std::mem::zeroed::<Msg>();
                if GetMessageW(&mut msg, 0, 0, 0) <= 0 { break; }
                if msg.message == 0x0312 /* WM_HOTKEY */ {
                    if state.load(Ordering::SeqCst) == STATE_ACTIVE {
                        state.store(STATE_PAUSED, Ordering::SeqCst);
                        eprintln!("[paused]");
                    } else {
                        state.store(STATE_ACTIVE, Ordering::SeqCst);
                        eprintln!("[recording]");
                    }
                }
            }
        }
    });
}

// ============================================================
// Linux: rdev::listen()
// ============================================================

#[cfg(not(target_os = "windows"))]
pub fn spawn_listener(
    running: Arc<AtomicBool>,
    hotkey: Option<rdev::Key>,
    state: Option<Arc<AtomicU8>>,
) {
    use std::time::Instant;

    std::thread::spawn(move || {
        let mut ctrl_held = false;
        let mut last_press = Instant::now();
        let debounce = std::time::Duration::from_millis(200);

        if let Err(e) = rdev::listen(move |event| match event.event_type {
            rdev::EventType::KeyPress(k) => {
                if matches!(k, rdev::Key::ControlLeft | rdev::Key::ControlRight) {
                    ctrl_held = true;
                }
                if k == rdev::Key::KeyC && ctrl_held {
                    running.store(false, Ordering::SeqCst);
                }
                if let (Some(hk), Some(ref st)) = (hotkey, &state) {
                    if k == hk && last_press.elapsed() >= debounce {
                        last_press = Instant::now();
                        if st.load(Ordering::SeqCst) == STATE_ACTIVE {
                            st.store(STATE_PAUSED, Ordering::SeqCst);
                            eprintln!("[paused]");
                        } else {
                            st.store(STATE_ACTIVE, Ordering::SeqCst);
                            eprintln!("[recording]");
                        }
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
