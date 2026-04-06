// Shared settings infrastructure for server configuration.
//
// SharedSettings holds all adjustable inference parameters as atomics, shared
// between the model loading thread and per-connection streaming tasks.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

pub const SILENCE_CHUNKS_DEFAULT: usize = 20;

/// Atomic f32 via bit reinterpretation (no std AtomicF32).
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }
    pub fn store(&self, v: f32, order: Ordering) {
        self.0.store(v.to_bits(), order);
    }
}

/// Read-only snapshot of startup parameters — included in GET /config responses.
/// These cannot be changed at runtime; a server restart is required.
#[derive(serde::Serialize, Clone)]
pub struct StartupSnapshot {
    pub model_dir:    String,
    pub device:       usize,
    pub port:         u16,
    pub bind_addr:    String,
    pub tls_enabled:  bool,
    pub lora_adapter: Option<String>,
    pub venv_path:    Option<String>,
}

/// Inference parameters shared across all WebSocket connections.
pub struct SharedSettings {
    pub silence_threshold: AtomicF32,
    pub silence_chunks: AtomicUsize,
    pub paragraph_delay_offset: AtomicUsize,
    pub min_speech_chunks: AtomicUsize,
    pub rms_ema_alpha: AtomicF32,
    pub delay_tokens: AtomicUsize,
    /// Server state: STATE_READY / STATE_LOADING (no hotkey toggle in server mode).
    pub state: AtomicU8,
}

pub const STATE_LOADING: u8 = 2;
pub const STATE_READY: u8 = 1;

impl SharedSettings {
    pub fn new(vals: IniValues, silence_chunks: usize) -> Self {
        Self {
            silence_threshold: AtomicF32::new(vals.silence_threshold),
            silence_chunks: AtomicUsize::new(silence_chunks),
            paragraph_delay_offset: AtomicUsize::new(vals.paragraph_delay_offset),
            min_speech_chunks: AtomicUsize::new(vals.min_speech_chunks),
            rms_ema_alpha: AtomicF32::new(vals.rms_ema_alpha),
            delay_tokens: AtomicUsize::new(vals.delay),
            state: AtomicU8::new(STATE_LOADING),
        }
    }
}

/// Intermediate non-atomic settings for CLI argument merging.
pub struct IniValues {
    pub delay: usize,
    pub silence_threshold: f32,
    pub silence_chunks: Option<usize>,
    pub paragraph_delay_offset: usize,
    pub min_speech_chunks: usize,
    pub rms_ema_alpha: f32,
}

impl Default for IniValues {
    fn default() -> Self {
        Self {
            delay: 4,
            silence_threshold: 0.006,
            silence_chunks: None,
            paragraph_delay_offset: 4,
            min_speech_chunks: 15,
            rms_ema_alpha: 0.3,
        }
    }
}
