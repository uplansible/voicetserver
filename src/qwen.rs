// Qwen3-ASR engine handle — second inference backend alongside VoxtralModel.
//
// Wraps the vendored `qwen3_asr::AsrInference` behind the same
// `tokio::sync::Mutex<Option<…>>` unload pattern as `VoxtralModel::inner`:
// setting the Option to `None` drops the engine (~1.5 GB VRAM) so LoRA
// training can reuse the memory; `load()` rebuilds it from `model_dir`.
//
// The inner value is an `Arc<AsrInference>` (not the bare engine) because
// streaming sessions run inference via `spawn_blocking` and need a handle
// that outlives the brief lock: a session locks `inner`, clones the Arc,
// releases the lock, then does GPU work serialised by AsrInference's own
// internal mutex. This is what gives each engine its own GPU lock — Voxtral
// and Qwen sessions can run concurrently.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use candle_core::Device;

/// The Qwen3-ASR engine shared across WebSocket connections.
///
/// Present in AppState only when `qwen_model_dir` is configured; sessions
/// requesting `?model=qwen` on a server without it get an error frame.
pub struct QwenEngine {
    /// `None` while unloaded for LoRA training (same pattern as `VoxtralModel::inner`).
    pub inner: tokio::sync::Mutex<Option<Arc<qwen3_asr::AsrInference>>>,
    /// Model directory — used to rebuild the engine when reloading after training.
    pub model_dir: String,
    /// Device shared with the Voxtral engine (one CUDA context for both).
    pub device: Device,
}

impl QwenEngine {
    /// Load the engine from `model_dir` (model.safetensors, config.json, tokenizer.json).
    pub fn load(model_dir: &str, device: Device) -> Result<Self> {
        let engine = load_inference(model_dir, &device)?;
        Ok(Self {
            inner: tokio::sync::Mutex::new(Some(Arc::new(engine))),
            model_dir: model_dir.to_string(),
            device,
        })
    }

    /// Rebuild the inner engine after a training unload, optionally re-applying a
    /// LoRA adapter. Sync (model load takes seconds) — call from `spawn_blocking`
    /// or before the tokio runtime starts; `blocking_lock` panics in async context.
    pub fn reload_blocking(&self, lora_dir: Option<&Path>) -> Result<()> {
        let engine = load_inference(&self.model_dir, &self.device)?;
        // Warn-and-continue on LoRA failure so a bad adapter never leaves the
        // engine unloaded (same semantics as the Voxtral load_enc_dec reload).
        if let Some(dir) = lora_dir {
            if let Err(e) = engine.load_lora(dir) {
                eprintln!("Warning: Qwen LoRA reload failed ({}): {}", dir.display(), e);
            }
        }
        *self.inner.blocking_lock() = Some(Arc::new(engine));
        Ok(())
    }

    /// Apply a LoRA adapter to the currently loaded engine. Sync — same calling
    /// constraints as `reload_blocking`. No-op error if the engine is unloaded.
    pub fn apply_lora_blocking(&self, adapter_dir: &Path) -> Result<()> {
        match self.inner.blocking_lock().as_ref() {
            Some(engine) => engine.load_lora(adapter_dir)
                .map_err(|e| anyhow::anyhow!("Qwen LoRA load failed ({}): {}", adapter_dir.display(), e)),
            None => anyhow::bail!("Qwen engine is unloaded (training in progress)"),
        }
    }

    /// Clone out the engine handle, or None if unloaded (training in progress).
    pub async fn get(&self) -> Option<Arc<qwen3_asr::AsrInference>> {
        self.inner.lock().await.clone()
    }
}

fn load_inference(model_dir: &str, device: &Device) -> Result<qwen3_asr::AsrInference> {
    qwen3_asr::AsrInference::load(Path::new(model_dir), device.clone())
        .map_err(|e| anyhow::anyhow!("Qwen3-ASR model load failed ({}): {}", model_dir, e))
}
