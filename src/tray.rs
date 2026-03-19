// System tray icon — shows recording state.
//
// Left-click: toggle Active ↔ Paused
// Right-click: open settings subprocess directly
// Settings subprocess reads/writes settings.ini; parent reloads on exit.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use tray_icon::{Icon, TrayIconBuilder, TrayIconEvent};

use crate::hotkey;
use crate::settings::{self, SharedSettings};

const ICON_ACTIVE: &[u8] = include_bytes!("../assets/icon_active.rgba");
const ICON_PAUSED: &[u8] = include_bytes!("../assets/icon_paused.rgba");
const ICON_UNLOADED: &[u8] = include_bytes!("../assets/icon_unloaded.rgba");

fn load_icon(rgba: &[u8]) -> Icon {
    Icon::from_rgba(rgba.to_vec(), 32, 32).expect("Failed to load tray icon")
}

fn icon_for_state(state: u8, active: &Icon, paused: &Icon, unloaded: &Icon) -> Icon {
    match state {
        hotkey::STATE_ACTIVE => active.clone(),   // green
        hotkey::STATE_PAUSED => unloaded.clone(),  // red
        _ => paused.clone(),                       // grey (loading)
    }
}

/// Run the tray icon event loop (blocking). Call from a dedicated thread.
pub fn run_tray(
    settings: Arc<SharedSettings>,
    running: Arc<AtomicBool>,
    hotkey_thread_id: Arc<AtomicU32>,
) {
    let icon_active = load_icon(ICON_ACTIVE);
    let icon_paused = load_icon(ICON_PAUSED);
    let icon_unloaded = load_icon(ICON_UNLOADED);

    let initial_state = settings.state.load(Ordering::SeqCst);
    let initial_icon = icon_for_state(initial_state, &icon_active, &icon_paused, &icon_unloaded);

    let tray = match TrayIconBuilder::new()
        .with_tooltip("Voicet")
        .with_icon(initial_icon)
        .build()
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create tray icon: {}", e);
            return;
        }
    };

    let tray_rx = TrayIconEvent::receiver();
    let mut last_state = initial_state;
    let settings_path = settings::settings_path();
    let mut settings_child: Option<std::process::Child> = None;

    while running.load(Ordering::SeqCst) {
        #[cfg(target_os = "windows")]
        pump_messages();

        // Tray click events
        while let Ok(event) = tray_rx.try_recv() {
            if let TrayIconEvent::Click {
                button,
                button_state: tray_icon::MouseButtonState::Up,
                ..
            } = event
            {
                match button {
                    tray_icon::MouseButton::Left => {
                        hotkey::toggle_state(&settings.state, " (tray)");
                    }
                    tray_icon::MouseButton::Right => {
                        let already_open = settings_child
                            .as_mut()
                            .map_or(false, |c| c.try_wait().ok().flatten().is_none());
                        if !already_open {
                            settings::save_settings(&settings_path, &settings);
                            if let Ok(exe) = std::env::current_exe() {
                                settings_child = std::process::Command::new(exe)
                                    .arg("--settings-ui")
                                    .spawn()
                                    .ok();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Reload settings when the subprocess exits
        if let Some(child) = &mut settings_child {
            match child.try_wait() {
                Ok(Some(status)) => {
                    settings::reload_from_file(&settings, &settings_path, &hotkey_thread_id);
                    settings_child = None;
                    if status.code() == Some(99) {
                        running.store(false, Ordering::SeqCst);
                    }
                }
                Err(_) => {
                    settings_child = None;
                }
                Ok(None) => {}
            }
        }

        // Update icon on state change
        let current_state = settings.state.load(Ordering::SeqCst);
        if current_state != last_state {
            let _ = tray.set_icon(Some(icon_for_state(
                current_state,
                &icon_active,
                &icon_paused,
                &icon_unloaded,
            )));
            last_state = current_state;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Clean up settings subprocess
    if let Some(mut child) = settings_child {
        let _ = child.kill();
    }
}

/// Pump the Win32 message loop so tray icon events are dispatched.
#[cfg(target_os = "windows")]
fn pump_messages() {
    #[repr(C)]
    struct Msg {
        hwnd: isize,
        message: u32,
        wparam: usize,
        lparam: isize,
        time: u32,
        pt: [i32; 2],
    }
    extern "system" {
        fn PeekMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32, remove: u32) -> i32;
        fn TranslateMessage(msg: *const Msg) -> i32;
        fn DispatchMessageW(msg: *const Msg) -> i32;
    }
    const PM_REMOVE: u32 = 0x0001;
    unsafe {
        let mut msg = std::mem::zeroed::<Msg>();
        while PeekMessageW(&mut msg, 0, 0, 0, PM_REMOVE) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}
