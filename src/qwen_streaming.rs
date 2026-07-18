// Per-connection streaming inference state for the Qwen3-ASR engine
// (ported from schmidiscribe src/streaming.rs).
//
// Each WebSocket client on `?model=qwen` gets its own StreamingState wrapping
// a qwen3-asr streaming session. Unlike the Voxtral path there is no shared
// GPU mutex to juggle here: the session holds an `Arc<AsrInference>` and the
// engine serialises GPU work behind its own internal lock, so qwen sessions
// run concurrently with Voxtral sessions.
//
// Silence detection triggers a finish/reset cycle so each qwen session stays
// short (inference cost grows with session length), carrying the last 200
// chars of transcript over as initial_text for continuity.

use anyhow::Result;
use qwen3_asr::{AsrInference, StreamingOptions, StreamingState as QwenState};
use std::sync::Arc;

use crate::settings::SharedSettings;

// ---- Output type ----

/// Result of processing one audio chunk.
pub enum ChunkOutput {
    /// Cumulative session text update. Send as partial to client.
    Token(String),
    /// Silence detected and session flushed. Caller should call take_text_buf().
    Silence,
}

// ---- Silence Detector ----

/// Two-stage silence detection: arms after sufficient speech, fires after
/// consecutive silent 100ms ticks reach the threshold.
///
/// Separate from the Voxtral `SilenceDetector` (src/streaming.rs): that one is
/// tied to the Voxtral decoder's token-delay/paragraph machinery and has no
/// `has_speech()` gate — which the qwen path needs to suppress hallucinated
/// partials ("mhh", "ich habe") during silent stretches.
pub struct SilenceDetector {
    silence_counter: usize,
    speech_counter:  usize,
    smoothed_rms:    f32,
    silence_emitted: bool,
}

impl SilenceDetector {
    pub fn new() -> Self {
        Self {
            silence_counter: 0,
            speech_counter:  0,
            smoothed_rms:    0.0,
            silence_emitted: true,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// True once enough speech has been detected to arm the silence trigger.
    pub fn has_speech(&self) -> bool {
        !self.silence_emitted
    }

    /// Process one 100ms chunk of audio. Returns true if a silence event should fire.
    pub fn process_chunk(&mut self, rms: f32, settings: &SharedSettings) -> bool {
        use std::sync::atomic::Ordering;
        let sil_thresh = settings.silence_threshold.load(Ordering::Relaxed);
        let rms_alpha  = settings.rms_ema_alpha.load(Ordering::Relaxed);
        self.smoothed_rms = rms_alpha * rms + (1.0 - rms_alpha) * self.smoothed_rms;

        if settings.silence_chunks.load(Ordering::Relaxed) == 0 {
            return false;
        }

        if rms < sil_thresh {
            self.silence_counter += 1;
        } else {
            self.silence_counter = 0;
        }

        if self.smoothed_rms >= sil_thresh {
            self.speech_counter += 1;
            if self.speech_counter >= settings.min_speech_chunks.load(Ordering::Relaxed) {
                self.silence_emitted = false;
            }
        } else {
            self.speech_counter = 0;
        }

        if !self.silence_emitted
            && self.silence_counter >= settings.silence_chunks.load(Ordering::Relaxed)
        {
            self.silence_emitted = true;
            return true;
        }
        false
    }
}

// ---- Per-connection streaming state ----

/// All per-connection mutable state for one qwen WebSocket session.
pub struct StreamingState {
    qwen_state:             QwenState,
    silence:                SilenceDetector,
    sample_buf_for_silence: Vec<f32>,
    text_buf:               String,
    language:               Option<String>,
    /// Hotword/context text injected into the system prompt for vocabulary biasing
    /// (custom medical terms + per-session hotwords + patient name). Persists
    /// across silence resets.
    context:                Option<String>,
    engine:                 Arc<AsrInference>,
}

// 100ms silence ticks: 16000 Hz / 10 = 1600 samples
const SAMPLES_PER_SILENCE_TICK: usize = 16000 / 10;

impl StreamingState {
    pub fn new(engine: Arc<AsrInference>, language: Option<String>, context: Option<String>) -> Self {
        let mut opts = StreamingOptions::default();
        if let Some(ref lang) = language {
            opts = opts.with_language(lang.clone());
        }
        if let Some(ref ctx) = context {
            opts = opts.with_context(ctx.clone());
        }
        Self {
            qwen_state: engine.init_streaming(opts),
            silence:                SilenceDetector::new(),
            sample_buf_for_silence: Vec::new(),
            text_buf:               String::new(),
            language,
            context,
            engine,
        }
    }

    /// Process a block of 16kHz mono PCM samples. Returns ChunkOutput events.
    pub fn process_chunk(&mut self, pcm: &[f32], settings: &SharedSettings) -> Result<Vec<ChunkOutput>> {
        self.sample_buf_for_silence.extend_from_slice(pcm);
        let mut outputs = Vec::new();

        // Silence detection in 100ms ticks
        while self.sample_buf_for_silence.len() >= SAMPLES_PER_SILENCE_TICK {
            let chunk = &self.sample_buf_for_silence[..SAMPLES_PER_SILENCE_TICK];
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            self.sample_buf_for_silence.drain(..SAMPLES_PER_SILENCE_TICK);

            if self.silence.process_chunk(rms, settings) {
                let result = self.engine.finish_streaming(&mut self.qwen_state)
                    .map_err(|e| anyhow::anyhow!("finish_streaming: {}", e))?;
                self.text_buf = result.text;
                outputs.push(ChunkOutput::Silence);

                // Reset qwen state for next session, passing tail of transcript as context
                let ctx_tail: String = self.text_buf.chars().rev().take(200)
                    .collect::<String>().chars().rev().collect();
                let mut opts = StreamingOptions::default();
                if let Some(ref lang) = self.language {
                    opts = opts.with_language(lang.clone());
                }
                if !ctx_tail.is_empty() {
                    opts = opts.with_initial_text(ctx_tail);
                }
                // Re-apply hotword/context biasing for the new session.
                if let Some(ref ctx) = self.context {
                    opts = opts.with_context(ctx.clone());
                }
                self.qwen_state = self.engine.init_streaming(opts);
                self.silence.reset();
            }
        }

        // Feed audio to qwen engine; suppress partials until speech is detected
        // to avoid emitting hallucinations ("mhh", "ich habe") during silent pauses.
        if let Some(result) = self.engine.feed_audio(&mut self.qwen_state, pcm)
            .map_err(|e| anyhow::anyhow!("feed_audio: {}", e))?
        {
            if !result.text.is_empty() && self.silence.has_speech() {
                outputs.push(ChunkOutput::Token(result.text.clone()));
            }
        }

        Ok(outputs)
    }

    /// Flush remaining audio and return all pending text (call on stop / WebSocket close).
    ///
    /// Skips finish_streaming() when no speech was detected in the current segment to
    /// avoid hallucinations from the model running on just the system prompt / context.
    pub fn finish(&mut self) -> Result<String> {
        if self.silence.has_speech() {
            let result = self.engine.finish_streaming(&mut self.qwen_state)
                .map_err(|e| anyhow::anyhow!("finish_streaming: {}", e))?;
            if !result.text.is_empty() {
                self.text_buf.push_str(&result.text);
            }
        }
        Ok(std::mem::take(&mut self.text_buf))
    }

    /// Return and clear the accumulated text buffer (called after Silence fires).
    pub fn take_text_buf(&mut self) -> String {
        std::mem::take(&mut self.text_buf)
    }
}
