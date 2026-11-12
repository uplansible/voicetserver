// Per-connection streaming inference state.
//
// Each WebSocket client gets its own StreamingState instance with independent
// KV caches and SilenceDetector. All forward passes acquire a shared GPU mutex
// before touching Candle tensors.
//
// Timing: 1280 samples = 8 mel frames = 4 encoder frames = 1 decoder token = 80ms.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::adapter::Adapter;
use crate::common::{self, MEL_FRAMES_PER_TOKEN};
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::mel::{self, IncrementalMel};
use crate::settings::SharedSettings;
use crate::tokenizer::{self, Tokenizer};

/// PCM samples per decoder token: MEL_FRAMES_PER_TOKEN × HOP_LENGTH.
pub const SAMPLES_PER_TOKEN: usize = MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

// Incremental conv stem: carry 4 mel frames as context between 80ms ticks.
const CONV_CTX: usize = 4;
const CONV_SKIP: usize = 2;
const NEW_MEL_PER_CHUNK: usize = MEL_FRAMES_PER_TOKEN;

// ---- Output type ----

/// Result of processing one 80ms audio chunk.
pub enum ChunkOutput {
    /// A text token was decoded. Append to buffer and send partial result.
    Token(String),
    /// Silence detected. Carries the raw (uncorrected) accumulated text, taken and
    /// cleared at the moment silence fired so any tokens decoded later in the same
    /// chunk start a fresh buffer. The caller applies word/fuzzy correction.
    Silence(String),
    /// PAD token — no output this tick.
    Pad,
}

// ---- Silence Detector ----

/// Two-stage silence detection and paragraph break state machine.
///
/// Stage 1: count consecutive silent chunks. Once count reaches `silence_chunks`,
///   silence is "detected". Only arms after `min_speech` loud chunks
///   (cumulative since the last event / full silence run, not consecutive).
///
/// Stage 2: once silence detected, a paragraph-delay counter ticks each quiet
///   chunk. When it reaches `delay_tokens + offset`, a Silence event fires and
///   both stages reset. If speech resumes first, the pause is over: the event
///   fires on that loud chunk — before it is decoded, so the resumed words go
///   to the next segment — provided the pause has lasted at least
///   `delay_tokens + offset` chunks (enough for the decoder lag to have emitted
///   everything spoken before it); a shorter pause is cancelled instead.
pub struct SilenceDetector {
    silence_counter: usize,
    speech_counter: usize,
    smoothed_rms: f32,
    silence_emitted: bool,
    silence_detected: bool,
    paragraph_delay_counter: usize,
}

impl SilenceDetector {
    pub fn new() -> Self {
        Self {
            silence_counter: 0,
            speech_counter: 0,
            smoothed_rms: 0.0,
            silence_emitted: true,
            silence_detected: false,
            paragraph_delay_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Process one 80ms chunk of audio. Returns true if a silence/paragraph event should fire.
    ///
    /// `delay_tokens` is the value the *session* was built with, not the live
    /// atomic: prefill, KV sizing and the drain length are all fixed at
    /// construction, so reading the atomic here made a mid-session
    /// `PATCH /config` desync finalization timing from the actual decoder lag.
    /// A changed `delay` takes effect on the next session.
    pub fn process_chunk(&mut self, rms: f32, delay_tokens: usize, settings: &SharedSettings) -> bool {
        use std::sync::atomic::Ordering;
        let sil_thresh = settings.silence_threshold.load(Ordering::Relaxed);
        let rms_alpha = settings.rms_ema_alpha.load(Ordering::Relaxed);
        self.smoothed_rms = rms_alpha * rms + (1.0 - rms_alpha) * self.smoothed_rms;

        if settings.silence_chunks.load(Ordering::Relaxed) == 0 {
            return false;
        }

        // Silence counting
        let mut resumed = false;
        if rms < sil_thresh {
            self.silence_counter += 1;
            if !self.silence_detected && !self.silence_emitted
                && self.silence_counter >= settings.silence_chunks.load(Ordering::Relaxed)
            {
                self.silence_detected = true;
                self.paragraph_delay_counter = 0;
            }
        } else {
            self.silence_counter = 0;
            // Speech resumed during the paragraph delay — handled below.
            resumed = self.silence_detected;
        }

        // Paragraph delay ticks once triggered, for every quiet chunk
        if self.silence_detected && !resumed {
            self.paragraph_delay_counter += 1;
        }

        // Speech tracking — arms silence detection after enough speech. Loud
        // ticks count *cumulatively* (not consecutively): a short utterance or
        // control word with small dips between syllables must still arm, or it
        // never becomes its own final. Only a full silence run while unarmed
        // starts the count over, so isolated blips spread over a long pause do
        // not add up to an arming.
        if self.smoothed_rms >= sil_thresh {
            self.speech_counter += 1;
            if self.speech_counter >= settings.min_speech_chunks.load(Ordering::Relaxed) {
                self.silence_emitted = false;
            }
        } else if self.silence_emitted
            && self.silence_counter >= settings.silence_chunks.load(Ordering::Relaxed)
        {
            self.speech_counter = 0;
        }

        // Check if paragraph break should fire
        if self.silence_detected && !self.silence_emitted {
            let para_delay = {
                use std::sync::atomic::Ordering;
                delay_tokens + settings.paragraph_delay_offset.load(Ordering::Relaxed)
            };
            // Quiet chunks since the pause began: detection happened on the
            // silence_chunks-th one, which already counted 1 in the delay.
            let pause_len = settings.silence_chunks.load(Ordering::Relaxed)
                + self.paragraph_delay_counter - 1;
            let fire = self.paragraph_delay_counter >= para_delay
                || (resumed && pause_len >= para_delay);
            if fire {
                self.silence_emitted = true;
                self.silence_detected = false;
                self.paragraph_delay_counter = 0;
                // The next segment must earn its arming from scratch.
                self.speech_counter = 0;
                return true;
            }
            if resumed {
                // Pause too short for the pre-pause text to be fully decoded:
                // cancel it and keep one segment (still armed).
                self.silence_detected = false;
                self.paragraph_delay_counter = 0;
            }
        }
        false
    }
}

// ---- RoPE position limits ----

/// Past this RoPE position (encoder or decoder, whichever is further) a session
/// is rebuilt via the normal prefill path at its next silence final. The
/// encoder advances 50 positions per second of audio, so this leaves ~9 min of
/// continuous speech before the hard limit — a final always comes long before.
pub const ROPE_SOFT_LIMIT: usize = common::ROPE_MAX_POSITIONS - 30_000;
/// Past this position the session is rebuilt immediately, flushing its text as
/// a final first, even mid-speech (~60 s of audio before the table ends — far
/// more than one WebSocket frame plus the end-of-session drain can consume).
pub const ROPE_HARD_LIMIT: usize = common::ROPE_MAX_POSITIONS - 3_000;

/// Current RoPE position of a streaming session's shared encoder/decoder.
pub fn rope_position(enc: &AudioEncoder, dec: &TextDecoder) -> usize {
    enc.position().max(dec.position())
}

// ---- Startup ----

/// Experimental (german_prime config flag): German text used to prime the decoder
/// prefill instead of pure PADs, biasing the model's language prior toward German.
/// Encoded once at model load (`Tokenizer::encode_greedy`); must stay short enough
/// to fit the LEFT_PAD_TOKENS (32) silence region.
pub const GERMAN_PRIME_TEXT: &str = "Der Patient kommt zur Verlaufskontrolle in die Sprechstunde.";

/// Internal state after startup prefill.
pub struct StartupResult {
    pub last_token: u32,
    pub mel_frames: Vec<[f32; mel::N_MELS]>,
}

/// Run startup prefill: [silence + delay_audio] through batch mel → enc → adapter → decoder.
/// This initialises all KV caches for subsequent streaming.
pub fn run_startup(
    delay_samples: &[f32],
    delay_tokens: usize,
    prime_tokens: &[u32],
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<StartupResult> {
    let left_pad_samples = common::LEFT_PAD_TOKENS * MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

    let mut padded = vec![0.0f32; left_pad_samples];
    padded.extend_from_slice(delay_samples);

    let mel_data = mel::log_mel_spectrogram(&padded, filters);
    let mel_time = mel_data.len() / mel::N_MELS;

    let mut mel_frames = Vec::with_capacity(mel_time);
    for t in 0..mel_time {
        let mut frame = [0.0f32; mel::N_MELS];
        for b in 0..mel::N_MELS {
            frame[b] = mel_data[b * mel_time + t];
        }
        mel_frames.push(frame);
    }

    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    let enc_out = enc.forward(&mel_tensor)?;
    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(delay_tokens as f32, device, dtype)?;
    let prefill_len = decoder::prefill_len(delay_tokens);

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, prime_tokens, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;
    let mut last_token = common::argmax_last(&logits)?;

    for pos in prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;
        last_token = next_token;
        if last_token == tokenizer::EOS_ID { break; }
    }

    Ok(StartupResult { last_token, mel_frames })
}

// ---- Per-connection streaming state ----

/// All per-connection mutable state for one WebSocket session.
///
/// Created once per connection via `new()`, which runs startup prefill.
/// Each 1280-sample audio chunk is processed by `process_chunk()`.
pub struct StreamingState {
    pub last_token: u32,
    pub mel_buffer: Vec<[f32; mel::N_MELS]>,
    pub inc_mel: IncrementalMel,
    pub silence: SilenceDetector,
    pub sample_buf_for_silence: Vec<f32>,
    /// Leading samples of `sample_buf_for_silence` already pushed to `inc_mel`
    /// (an incomplete trailing tick is fed to the mel before its silence check).
    pub mel_fed: usize,
    pub text_buf: String,
    /// Holds raw token bytes that do not yet form a complete UTF-8 sequence.
    /// Tekken is byte-level BPE, so a multi-byte char (umlauts, ß, …) can straddle
    /// two tokens; we buffer the incomplete tail until the next token completes it.
    pub pending_bytes: Vec<u8>,
    // Model references (not owned here — owned by main)
    pub delay_tokens: usize,
    pub dtype: DType,
}

impl StreamingState {
    /// Initialise state and run startup prefill synchronously.
    /// Called while holding the model inner lock — no async, no await.
    pub fn new_sync(
        enc: &mut AudioEncoder,
        adapter: &Adapter,
        dec: &mut TextDecoder,
        prime_tokens: &[u32],
        filters: &[f32],
        device: &Device,
        dtype: DType,
        settings: &SharedSettings,
    ) -> Result<Self> {
        use std::sync::atomic::Ordering;
        let delay_tokens = settings.delay_tokens.load(Ordering::Relaxed);
        let delay_samples_count = (1 + delay_tokens) * SAMPLES_PER_TOKEN;
        let silence_input = vec![0.0f32; delay_samples_count];

        let startup = run_startup(&silence_input, delay_tokens, prime_tokens, enc, adapter, dec, filters, device, dtype)?;

        let mel_len = startup.mel_frames.len();
        let ctx_start = mel_len.saturating_sub(CONV_CTX);
        let mel_buffer = startup.mel_frames[ctx_start..].to_vec();

        Ok(Self {
            last_token: startup.last_token,
            mel_buffer,
            inc_mel: IncrementalMel::new(filters),
            silence: SilenceDetector::new(),
            sample_buf_for_silence: Vec::new(),
            mel_fed: 0,
            text_buf: String::new(),
            pending_bytes: Vec::new(),
            delay_tokens,
            dtype,
        })
    }

    /// Process a block of 16kHz mono PCM samples synchronously.
    /// Called while holding the model inner lock — no async, no await.
    /// Returns one ChunkOutput per 80ms tick completed.
    ///
    /// Silence detection and inference are interleaved per 80 ms tick: each
    /// tick's silence check runs right before that tick's audio is decoded.
    /// Checking the whole WebSocket frame first (3+ ticks live, ~12 for the
    /// 1 s re-transcribe frames) let a final fire while the frame's resumed
    /// speech was already decoded into it, moving the next sentence's first
    /// word to the end of the previous final.
    pub fn process_chunk_sync(
        &mut self,
        pcm: &[f32],
        enc: &mut AudioEncoder,
        adapter: &Adapter,
        dec: &mut TextDecoder,
        tok: &Tokenizer,
        device: &Device,
        settings: &SharedSettings,
    ) -> Result<Vec<ChunkOutput>> {
        self.sample_buf_for_silence.extend_from_slice(pcm);

        let mut outputs = Vec::new();
        let mut checked = 0; // samples of sample_buf_for_silence silence-checked

        while self.sample_buf_for_silence.len() - checked >= SAMPLES_PER_TOKEN {
            let tick_end = checked + SAMPLES_PER_TOKEN;
            // Silence detection for this 80 ms tick, before it is decoded
            let rms = (self.sample_buf_for_silence[checked..tick_end]
                .iter().map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
            if self.silence.process_chunk(rms, self.delay_tokens, settings) {
                // Take + clear the buffer here (not in the caller) so tokens decoded
                // later in this same chunk start fresh instead of being re-sent as
                // part of the next partial.
                let raw = self.take_text_buf();
                outputs.push(ChunkOutput::Silence(raw));
            }
            checked = tick_end;
            // Then feed the tick to the mel and run the inference ticks it enables
            self.feed_mel_up_to(checked);
            self.infer_ticks(enc, adapter, dec, tok, device, &mut outputs)?;
        }

        // An incomplete trailing tick still goes to the mel right away (no added
        // latency); its silence check runs once the tick is complete.
        self.feed_mel_up_to(self.sample_buf_for_silence.len());
        self.infer_ticks(enc, adapter, dec, tok, device, &mut outputs)?;

        self.sample_buf_for_silence.drain(..checked);
        self.mel_fed -= checked;

        Ok(outputs)
    }

    /// Push `sample_buf_for_silence[mel_fed..end]` to the incremental mel.
    fn feed_mel_up_to(&mut self, end: usize) {
        if end > self.mel_fed {
            self.inc_mel.push_samples(&self.sample_buf_for_silence[self.mel_fed..end]);
            self.mel_fed = end;
            self.mel_buffer.extend(self.inc_mel.drain_frames());
        }
    }

    /// Flush the decoder lookahead at end of session by feeding silence.
    ///
    /// The decoder lags the audio by `delay_tokens` ticks (80ms each, plus conv/mel
    /// context), and the client stops sending audio the moment the user presses
    /// stop — without this drain, the last ~delay×80ms of speech would never be
    /// decoded and the final words of a dictation would be truncated. Decoded text
    /// accumulates in `text_buf`; the caller flushes it via `take_text_buf()`.
    /// Silence detection is deliberately skipped (no competing Silence final).
    pub fn drain_sync(
        &mut self,
        enc: &mut AudioEncoder,
        adapter: &Adapter,
        dec: &mut TextDecoder,
        tok: &Tokenizer,
        device: &Device,
    ) -> Result<()> {
        // +3 ticks of margin cover the mel window tail and conv-stem context.
        let zeros = vec![0.0f32; (self.delay_tokens + 3) * SAMPLES_PER_TOKEN];
        self.inc_mel.push_samples(&zeros);
        self.mel_buffer.extend(self.inc_mel.drain_frames());
        let mut outputs = Vec::new();
        self.infer_ticks(enc, adapter, dec, tok, device, &mut outputs)
    }

    /// Run inference ticks (one per 80ms mel chunk) over the buffered mel frames.
    /// Decoded text is appended to `text_buf`; per-token partials are pushed to `outputs`.
    fn infer_ticks(
        &mut self,
        enc: &mut AudioEncoder,
        adapter: &Adapter,
        dec: &mut TextDecoder,
        tok: &Tokenizer,
        device: &Device,
        outputs: &mut Vec<ChunkOutput>,
    ) -> Result<()> {
        let dtype = self.dtype;
        while self.mel_buffer.len() >= CONV_CTX + NEW_MEL_PER_CHUNK {
            let process_len = CONV_CTX + NEW_MEL_PER_CHUNK;

            let mut mel_data = vec![0.0f32; mel::N_MELS * process_len];
            for (frame_idx, frame) in self.mel_buffer[..process_len].iter().enumerate() {
                for mel_bin in 0..mel::N_MELS {
                    mel_data[mel_bin * process_len + frame_idx] = frame[mel_bin];
                }
            }

            let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, process_len), device)?
                .to_dtype(dtype)?
                .unsqueeze(0)?;

            let conv_out = enc.conv_stem(&mel_tensor)?;
            if conv_out.dim(1)? < CONV_SKIP + encoder::CHUNK_SIZE {
                break;
            }
            let new_conv_frames = conv_out.narrow(1, CONV_SKIP, encoder::CHUNK_SIZE)?;

            let enc_out = enc.forward_chunk(&new_conv_frames)?;
            enc.trim_caches();

            let adapter_out = adapter.forward(&enc_out)?;

            let tok_embed = dec.embed_tokens(&[self.last_token], device)?;
            let fused = tok_embed.add(&adapter_out)?;
            let logits = dec.forward(&fused)?;
            let next_token = common::argmax_last(&logits)?;
            dec.trim_caches();

            self.mel_buffer.drain(..NEW_MEL_PER_CHUNK);
            self.last_token = next_token;

            if next_token == tokenizer::EOS_ID {
                break;
            }
            if next_token == tokenizer::STREAMING_PAD_ID || next_token == tokenizer::STREAMING_WORD_ID {
                outputs.push(ChunkOutput::Pad);
            } else if let Some(bytes) = tok.decode_token(next_token) {
                // Append raw bytes and emit only the valid UTF-8 prefix; keep any
                // incomplete trailing bytes for the next token to complete. Without
                // this, a char split across two tokens (e.g. German umlauts) would be
                // silently dropped.
                self.pending_bytes.extend_from_slice(&bytes);
                if drain_valid_utf8(&mut self.pending_bytes, &mut self.text_buf) {
                    outputs.push(ChunkOutput::Token(self.text_buf.clone()));
                }
            }
        }

        Ok(())
    }

    /// Return and clear the accumulated text buffer (used on silence flush + WebSocket
    /// close). Also drops any incomplete trailing UTF-8 bytes so a partial multi-byte
    /// char left over from the previous paragraph cannot prefix the next one with garbage.
    pub fn take_text_buf(&mut self) -> String {
        self.pending_bytes.clear();
        std::mem::take(&mut self.text_buf)
    }
}

/// Move the decodable part of `pending` into `out`; returns true if `out` grew.
///
/// Only a genuinely incomplete trailing sequence (a valid prefix of a
/// multi-byte char) stays in `pending` for the next token. Bytes that can
/// never become valid (`error_len() == Some(_)`, e.g. a lead byte followed by
/// a non-continuation byte) are replaced with U+FFFD and decoding continues —
/// keeping them would block every later token of the segment.
fn drain_valid_utf8(pending: &mut Vec<u8>, out: &mut String) -> bool {
    let start_len = out.len();
    let mut consumed = 0;
    loop {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(s) => {
                out.push_str(s);
                consumed = pending.len();
                break;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // The first `valid` bytes are valid UTF-8 by construction.
                out.push_str(std::str::from_utf8(&pending[consumed..consumed + valid]).unwrap());
                consumed += valid;
                match e.error_len() {
                    Some(bad) => {
                        out.push(char::REPLACEMENT_CHARACTER);
                        consumed += bad;
                    }
                    None => break, // incomplete tail — wait for the next token
                }
            }
        }
    }
    pending.drain(..consumed);
    out.len() > start_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_split_char_waits_for_completion() {
        let (mut p, mut out) = (Vec::new(), String::new());
        p.extend_from_slice(b"F\xC3");
        assert!(drain_valid_utf8(&mut p, &mut out));
        assert_eq!((out.as_str(), p.as_slice()), ("F", &b"\xC3"[..]));
        p.extend_from_slice(b"\xBCr");
        assert!(drain_valid_utf8(&mut p, &mut out));
        assert_eq!(out, "Für");
        assert!(p.is_empty());
    }

    #[test]
    fn utf8_invalid_byte_does_not_block_later_text() {
        let (mut p, mut out) = (Vec::new(), String::new());
        // Lead byte followed by a non-continuation byte: can never become valid.
        p.extend_from_slice(b"a\xC3b");
        assert!(drain_valid_utf8(&mut p, &mut out));
        assert_eq!(out, "a\u{FFFD}b");
        assert!(p.is_empty());
        p.extend_from_slice(" weiter".as_bytes());
        assert!(drain_valid_utf8(&mut p, &mut out));
        assert_eq!(out, "a\u{FFFD}b weiter");
    }

    fn test_settings() -> SharedSettings {
        // silence_threshold 0.006, min_speech 4, paragraph offset 4, no RMS
        // smoothing; silence_flush 5.
        let vals = crate::settings::IniValues { rms_ema_alpha: 1.0, ..Default::default() };
        SharedSettings::new(vals, 5)
    }

    /// Feed `n` ticks of `rms`; true if any of them fired.
    fn feed(d: &mut SilenceDetector, s: &SharedSettings, rms: f32, n: usize) -> bool {
        (0..n).fold(false, |f, _| d.process_chunk(rms, 2, s) || f)
    }

    #[test]
    fn short_choppy_utterance_arms_and_fires() {
        let s = test_settings();
        let mut d = SilenceDetector::new();
        // "Wort löschen": 4 loud ticks with dips in between, never 4 in a row.
        for _ in 0..4 {
            assert!(!feed(&mut d, &s, 0.1, 1));
            assert!(!feed(&mut d, &s, 0.0, 1));
        }
        // silence_flush (5) + delay (2) + paragraph offset (4) quiet ticks.
        assert!(feed(&mut d, &s, 0.0, 12), "cumulative arming must yield a final");
    }

    #[test]
    fn blips_across_a_long_pause_do_not_arm() {
        let s = test_settings();
        let mut d = SilenceDetector::new();
        for _ in 0..6 {
            feed(&mut d, &s, 0.1, 1);
            assert!(!feed(&mut d, &s, 0.0, 10), "isolated blips must not add up");
        }
    }

    #[test]
    fn resumed_speech_fires_before_it_is_decoded() {
        let s = test_settings();
        let mut d = SilenceDetector::new();
        feed(&mut d, &s, 0.1, 4);
        // 7 quiet ticks: detected (5), delay running, not yet fired (needs 10).
        assert!(!feed(&mut d, &s, 0.0, 7));
        // Speech resumes: pause (7) >= delay + offset (6) → fire on this tick.
        assert!(feed(&mut d, &s, 0.1, 1), "resumed speech must cut the final now");
        // Resumed speech must arm from scratch; nothing fires in its own pause.
        assert!(!feed(&mut d, &s, 0.0, 12));
    }

    #[test]
    fn short_pause_with_resumed_speech_is_cancelled() {
        let s = test_settings();
        let mut d = SilenceDetector::new();
        feed(&mut d, &s, 0.1, 4);
        // Just detected (pause 5 < 6) — resuming cancels instead of firing.
        assert!(!feed(&mut d, &s, 0.0, 5));
        assert!(!feed(&mut d, &s, 0.1, 1));
        // Still armed: the next full pause fires as usual (5 + 6 - 1 ticks).
        assert!(!feed(&mut d, &s, 0.0, 9));
        assert!(feed(&mut d, &s, 0.0, 1));
    }

    #[test]
    fn next_segment_must_arm_from_scratch() {
        let s = test_settings();
        let mut d = SilenceDetector::new();
        feed(&mut d, &s, 0.1, 4);
        assert!(feed(&mut d, &s, 0.0, 12));
        // One blip after the final must not re-arm on leftover count.
        feed(&mut d, &s, 0.1, 1);
        assert!(!feed(&mut d, &s, 0.0, 12));
    }
}
