// Settings window — egui-based settings dialog.
//
// Runs as a standalone subprocess (--settings-ui). Reads settings.ini on open,
// writes on OK. Parent reloads the file when the subprocess exits.

use std::path::PathBuf;

use eframe::egui;

use crate::hotkey;
use crate::settings;

struct SettingsApp {
    path: PathBuf,
    delay: usize,
    silence_threshold: f32,
    silence_chunks: usize,
    min_speech_chunks: usize,
    rms_ema_alpha: f32,
    hotkey_index: usize,
    type_mode: bool,
    dirty: bool,
}

impl SettingsApp {
    fn new(path: PathBuf) -> Self {
        let vals = settings::load_ini(&path);
        let delay = vals.delay;
        let silence_chunks = vals.silence_chunks.unwrap_or(delay + 14);

        let hotkey_index = match vals.hotkey {
            None => 0,
            Some(key) => hotkey::SUPPORTED_KEYS
                .iter()
                .position(|(_, k)| *k == key)
                .map(|i| i + 1)
                .unwrap_or(0),
        };

        Self {
            path,
            delay,
            silence_threshold: vals.silence_threshold,
            silence_chunks,
            min_speech_chunks: vals.min_speech_chunks,
            rms_ema_alpha: vals.rms_ema_alpha,
            hotkey_index,
            type_mode: vals.type_mode,
            dirty: false,
        }
    }

    fn write_ini(&self) {
        let hotkey_str = if self.hotkey_index == 0 {
            "none".to_string()
        } else {
            hotkey::key_to_string(hotkey::SUPPORTED_KEYS[self.hotkey_index - 1].1)
        };
        let output_mode = if self.type_mode { "type" } else { "none" };
        let content = format!(
            "delay={}\nsilence_threshold={}\nsilence_chunks={}\nmin_speech_chunks={}\nrms_ema_alpha={}\nhotkey={}\noutput_mode={}\n",
            self.delay, self.silence_threshold, self.silence_chunks,
            self.min_speech_chunks, self.rms_ema_alpha, hotkey_str, output_mode,
        );
        let _ = std::fs::write(&self.path, content);
    }

    fn selected_label(&self) -> &str {
        if self.hotkey_index == 0 {
            "None"
        } else {
            hotkey::SUPPORTED_KEYS[self.hotkey_index - 1].0
        }
    }
}

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut changed = false;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("voicet settings");
            ui.add_space(4.0);

            egui::Grid::new("settings_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Delay");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.delay).range(1..=30))
                        .changed();
                    ui.end_row();

                    ui.label("Silence threshold");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.silence_threshold)
                                .range(0.001..=0.1)
                                .speed(0.001)
                                .fixed_decimals(4),
                        )
                        .changed();
                    ui.end_row();

                    ui.label("Paragraph break");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.silence_chunks).range(1..=100))
                        .changed();
                    ui.end_row();

                    ui.label("Min speech");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.min_speech_chunks).range(1..=100))
                        .changed();
                    ui.end_row();

                    ui.label("EMA smoothing");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.rms_ema_alpha)
                                .range(0.01..=1.0)
                                .speed(0.01)
                                .fixed_decimals(2),
                        )
                        .changed();
                    ui.end_row();

                    ui.label("Hotkey");
                    let prev_hotkey = self.hotkey_index;
                    egui::ComboBox::from_id_salt("hotkey_combo")
                        .selected_text(self.selected_label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.hotkey_index, 0, "None");
                            for (i, (label, _)) in hotkey::SUPPORTED_KEYS.iter().enumerate() {
                                ui.selectable_value(&mut self.hotkey_index, i + 1, *label);
                            }
                        });
                    changed |= self.hotkey_index != prev_hotkey;
                    ui.end_row();

                    ui.label("Output mode");
                    let prev_type = self.type_mode;
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.type_mode, true, "Type");
                        ui.radio_value(&mut self.type_mode, false, "None");
                    });
                    changed |= self.type_mode != prev_type;
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Quit Voicet").clicked() {
                    self.write_ini();
                    std::process::exit(99);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel").clicked() {
                        self.dirty = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("  OK  ").clicked() {
                        self.write_ini();
                        self.dirty = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        if changed {
            self.dirty = true;
        }
    }
}

impl Drop for SettingsApp {
    fn drop(&mut self) {
        if self.dirty {
            self.write_ini();
        }
    }
}

/// Run the settings window as a standalone process. Called via --settings-ui.
pub fn run_standalone() {
    let path = settings::settings_path();
    let win_w: f32 = 251.0;
    let win_h: f32 = 235.0;

    // Position at lower-right of the primary monitor
    let (x, y) = {
        #[cfg(target_os = "windows")]
        {
            extern "system" {
                fn GetSystemMetrics(index: i32) -> i32;
            }
            let (sw, sh) = unsafe { (GetSystemMetrics(0) as f32, GetSystemMetrics(1) as f32) };
            (sw - win_w - 8.0, sh - win_h - 48.0 - 8.0) // 48 = taskbar
        }
        #[cfg(not(target_os = "windows"))]
        {
            (800.0, 400.0) // fallback; Linux tray typically top-right
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([win_w, win_h])
            .with_position([x, y])
            .with_resizable(false)
            .with_decorations(false)
            .with_always_on_top(),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "Voicet Settings",
        options,
        Box::new(move |_cc| Ok(Box::new(SettingsApp::new(path)))),
    );
}
