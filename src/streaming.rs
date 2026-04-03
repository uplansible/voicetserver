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
    /// Silence detected. Send final result and clear buffer.
    Silence,
    /// PAD token — no output this tick.
    Pad,
}

// ---- Silence Detector ----

/// Two-stage silence detection and paragraph break state machine.
///
/// Stage 1: count consecutive silent chunks. Once count reaches `silence_chunks`,
///   silence is "detected". Only arms after sufficient speech has occurred.
///
/// Stage 2: once silence detected, a paragraph-delay counter ticks each chunk.
///   When it reaches `delay_tokens + offset`, a Silence event fires and both
///   stages reset.
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
    pub fn process_chunk(&mut self, rms: f32, settings: &SharedSettings) -> bool {
        use std::sync::atomic::Ordering;
        let sil_thresh = settings.silence_threshold.load(Ordering::Relaxed);
        let rms_alpha = settings.rms_ema_alpha.load(Ordering::Relaxed);
        self.smoothed_rms = rms_alpha * rms + (1.0 - rms_alpha) * self.smoothed_rms;

        if settings.silence_chunks.load(Ordering::Relaxed) == 0 {
            return false;
        }

        // Silence counting
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
        }

        // Paragraph delay ticks unconditionally once triggered
        if self.silence_detected {
            self.paragraph_delay_counter += 1;
        }

        // Speech tracking — arms silence detection after enough speech
        if self.smoothed_rms >= sil_thresh {
            self.speech_counter += 1;
            if self.speech_counter >= settings.min_speech_chunks.load(Ordering::Relaxed) {
                self.silence_emitted = false;
            }
        } else {
            self.speech_counter = 0;
        }

        // Check if paragraph break should fire
        if self.silence_detected && !self.silence_emitted {
            let para_delay = {
                use std::sync::atomic::Ordering;
                settings.delay_tokens.load(Ordering::Relaxed)
                    + settings.paragraph_delay_offset.load(Ordering::Relaxed)
            };
            if self.paragraph_delay_counter >= para_delay {
                self.silence_emitted = true;
                self.silence_detected = false;
                self.paragraph_delay_counter = 0;
                return true;
            }
        }
        false
    }
}

// ---- Startup ----

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

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, device, dtype)?;

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
    pub text_buf: String,
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
        filters: &[f32],
        device: &Device,
        dtype: DType,
        settings: &SharedSettings,
    ) -> Result<Self> {
        use std::sync::atomic::Ordering;
        let delay_tokens = settings.delay_tokens.load(Ordering::Relaxed);
        let delay_samples_count = (1 + delay_tokens) * SAMPLES_PER_TOKEN;
        let silence_input = vec![0.0f32; delay_samples_count];

        let startup = run_startup(&silence_input, delay_tokens, enc, adapter, dec, filters, device, dtype)?;

        let mel_len = startup.mel_frames.len();
        let ctx_start = mel_len.saturating_sub(CONV_CTX);
        let mel_buffer = startup.mel_frames[ctx_start..].to_vec();

        Ok(Self {
            last_token: startup.last_token,
            mel_buffer,
            inc_mel: IncrementalMel::new(filters),
            silence: SilenceDetector::new(),
            sample_buf_for_silence: Vec::new(),
            text_buf: String::new(),
            delay_tokens,
            dtype,
        })
    }

    /// Process a block of 16kHz mono PCM samples synchronously.
    /// Called while holding the model inner lock — no async, no await.
    /// Returns one ChunkOutput per 80ms tick completed.
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
        self.inc_mel.push_samples(pcm);
        self.sample_buf_for_silence.extend_from_slice(pcm);
        self.mel_buffer.extend(self.inc_mel.drain_frames());

        let dtype = self.dtype;
        let mut outputs = Vec::new();

        // Silence detection (runs per 80ms chunk of audio)
        while self.sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
            let rms = (self.sample_buf_for_silence[..SAMPLES_PER_TOKEN]
                .iter().map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
            self.sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
            if self.silence.process_chunk(rms, settings) {
                outputs.push(ChunkOutput::Silence);
                self.text_buf.clear();
            }
        }

        // Inference ticks (one per 80ms mel chunk)
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
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    self.text_buf.push_str(s);
                    outputs.push(ChunkOutput::Token(self.text_buf.clone()));
                }
            }
        }

        Ok(outputs)
    }

    /// Return and clear the accumulated text buffer (used on WebSocket close).
    pub fn take_text_buf(&mut self) -> String {
        std::mem::take(&mut self.text_buf)
    }
}
