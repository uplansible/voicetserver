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
// chars of transcript over as initial_text for continuity. Audio only reaches
// the engine while speech is detected, and implausibly long output for the
// amount of speech is dropped (see "Hallucination guards" below).

use anyhow::Result;
use qwen3_asr::{AsrInference, StreamingOptions, StreamingState as QwenState};
use std::collections::VecDeque;
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

// ---- Hallucination guards ----
//
// Qwen3-ASR (like Whisper) does not return nothing on non-speech audio — it
// invents a fluent sentence ("Ich bin nicht der Typ, der gerne mit anderen
// Leuten spricht."). Two guards keep that out of the transcript:
//
// A. Gated feeding: audio is only fed to the engine while the silence detector
//    is armed (speech detected). Before that, 100ms ticks wait in a pre-roll
//    buffer; on arming it is trimmed to LEAD_IN_TICKS before the first loud
//    tick and fed first, so the onset of speech is kept but a long pause
//    between segments never reaches the model.
// B. Plausibility check: a partial/final whose length is out of proportion to
//    the amount of speech in the segment (loud ticks) is dropped.

/// Pre-roll capacity on top of `min_speech_chunks`: arming lags speech onset by
/// at least min_speech ticks (more for choppy speech that resets the counter),
/// so the buffer must reach back further than that.
const PREROLL_EXTRA_TICKS: usize = 30;
/// Silent ticks kept before the first loud tick of the pre-roll (0.5 s lead-in).
const LEAD_IN_TICKS: usize = 5;
/// Upper bound on transcript characters per second of loud audio. Normal German
/// dictation is ~13-15 chars/s of wall time; loud ticks cover only part of
/// real speech (gaps between words fall below threshold), hence the headroom.
const MAX_CHARS_PER_SPEECH_SEC: f32 = 35.0;
/// Characters always allowed regardless of speech duration (short words).
const PLAUSIBLE_SLACK_CHARS: f32 = 15.0;

/// True if `text` is a believable transcript of `speech_ticks` loud 100ms ticks.
fn is_plausible(text: &str, speech_ticks: usize) -> bool {
    let chars = text.trim().chars().count() as f32;
    let speech_secs = speech_ticks as f32 * 0.1;
    chars <= MAX_CHARS_PER_SPEECH_SEC * speech_secs + PLAUSIBLE_SLACK_CHARS
}

// ---- Per-connection streaming state ----

/// All per-connection mutable state for one qwen WebSocket session.
pub struct StreamingState {
    qwen_state:             QwenState,
    silence:                SilenceDetector,
    sample_buf_for_silence: Vec<f32>,
    /// Ticks received while the silence detector is not armed, with their
    /// loudness flag. Fed to the engine (trimmed) once speech arms it.
    preroll:                VecDeque<(Vec<f32>, bool)>,
    /// Loud ticks (raw rms >= silence_threshold) fed into the current segment.
    speech_ticks:           usize,
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
            preroll:                VecDeque::new(),
            speech_ticks:           0,
            text_buf:               String::new(),
            language,
            context,
            engine,
        }
    }

    /// Feed one tick into the current segment; pushes a partial if the engine
    /// produced one. Implausible partials are replaced by an empty one so the
    /// client drops any stale partial instead of inserting it on an empty final.
    fn feed_tick(&mut self, tick: &[f32], loud: bool, outputs: &mut Vec<ChunkOutput>) -> Result<()> {
        if loud {
            self.speech_ticks += 1;
        }
        if let Some(result) = self.engine.feed_audio(&mut self.qwen_state, tick)
            .map_err(|e| anyhow::anyhow!("feed_audio: {}", e))?
        {
            if !result.text.is_empty() {
                let text = if is_plausible(&result.text, self.speech_ticks) {
                    result.text
                } else {
                    String::new()
                };
                outputs.push(ChunkOutput::Token(text));
            }
        }
        Ok(())
    }

    /// Finish the current segment's engine pass and return its text, or an empty
    /// string if the text is implausible for the amount of speech (guard B).
    fn finish_segment(&mut self) -> Result<String> {
        let result = self.engine.finish_streaming(&mut self.qwen_state)
            .map_err(|e| anyhow::anyhow!("finish_streaming: {}", e))?;
        if is_plausible(&result.text, self.speech_ticks) {
            Ok(result.text)
        } else {
            eprintln!("qwen: dropped implausible final ({} chars for {:.1} s of speech)",
                result.text.trim().chars().count(), self.speech_ticks as f32 * 0.1);
            Ok(String::new())
        }
    }

    /// Process a block of 16kHz mono PCM samples. Returns ChunkOutput events.
    pub fn process_chunk(&mut self, pcm: &[f32], settings: &SharedSettings) -> Result<Vec<ChunkOutput>> {
        use std::sync::atomic::Ordering;
        self.sample_buf_for_silence.extend_from_slice(pcm);
        let mut outputs = Vec::new();

        // Silence detection + gated feeding in 100ms ticks
        while self.sample_buf_for_silence.len() >= SAMPLES_PER_SILENCE_TICK {
            let tick: Vec<f32> = self.sample_buf_for_silence.drain(..SAMPLES_PER_SILENCE_TICK).collect();
            let rms = (tick.iter().map(|s| s * s).sum::<f32>() / tick.len() as f32).sqrt();
            let loud = rms >= settings.silence_threshold.load(Ordering::Relaxed);

            let was_armed = self.silence.has_speech();
            let fired = self.silence.process_chunk(rms, settings);

            if was_armed {
                // In a segment (including the tick that fires the flush, so the
                // old session gets its trailing audio before finishing).
                self.feed_tick(&tick, loud, &mut outputs)?;
            } else if self.silence.has_speech() {
                // Just armed: feed the pre-roll from LEAD_IN_TICKS before its
                // first loud tick, then this tick.
                let first_loud = self.preroll.iter().position(|(_, l)| *l)
                    .unwrap_or(self.preroll.len());
                let start = first_loud.saturating_sub(LEAD_IN_TICKS);
                let preroll: Vec<_> = self.preroll.drain(..).skip(start).collect();
                for (t, l) in preroll {
                    self.feed_tick(&t, l, &mut outputs)?;
                }
                self.feed_tick(&tick, loud, &mut outputs)?;
            } else {
                // Not armed: hold back from the engine (guard A).
                self.preroll.push_back((tick, loud));
                let cap = settings.min_speech_chunks.load(Ordering::Relaxed) + PREROLL_EXTRA_TICKS;
                while self.preroll.len() > cap {
                    self.preroll.pop_front();
                }
            }

            if fired {
                self.text_buf = self.finish_segment()?;
                if self.text_buf.is_empty() {
                    // Clear any partial the client still shows — it would
                    // otherwise insert it as the fallback for an empty final.
                    outputs.push(ChunkOutput::Token(String::new()));
                }
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
                self.speech_ticks = 0;
            }
        }

        Ok(outputs)
    }

    /// Flush remaining audio and return all pending text (call on stop / WebSocket close).
    ///
    /// Skips finish_streaming() when no speech was detected in the current segment to
    /// avoid hallucinations from the model running on just the system prompt / context.
    /// An empty return also means any partial the client still shows is stale.
    pub fn finish(&mut self) -> Result<String> {
        if self.silence.has_speech() {
            // Sub-tick remainder (<100 ms) belongs to the armed segment too.
            let rest = std::mem::take(&mut self.sample_buf_for_silence);
            if !rest.is_empty() {
                self.engine.feed_audio(&mut self.qwen_state, &rest)
                    .map_err(|e| anyhow::anyhow!("feed_audio: {}", e))?;
            }
            let text = self.finish_segment()?;
            self.text_buf.push_str(&text);
        }
        Ok(std::mem::take(&mut self.text_buf))
    }

    /// Return and clear the accumulated text buffer (called after Silence fires).
    pub fn take_text_buf(&mut self) -> String {
        std::mem::take(&mut self.text_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plausible_normal_dictation() {
        // ~4 s of loud audio, a typical 60-char sentence.
        assert!(is_plausible("Der Patient klagt über Schmerzen beim Wasserlösen seit gestern.", 40));
    }

    #[test]
    fn implausible_sentence_from_noise() {
        // The reported hallucination from ~1 s of cough/noise.
        assert!(!is_plausible("Ich bin nicht der Typ, der gerne mit anderen Leuten spricht.", 10));
    }

    #[test]
    fn short_word_always_plausible() {
        assert!(is_plausible("Ja.", 0));
    }
}
