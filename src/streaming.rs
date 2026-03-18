// Streaming mic capture + real-time inference pipeline
//
// Architecture: cpal audio callback -> mpsc channel -> main thread inference loop
// Timing: 80ms of audio (1280 samples, 8 mel frames) = 1 decoder token
//
// Phase 3 additions:
// - StreamConfig for runtime-configurable parameters
// - State machine: Ready → Active ↔ Paused (hotkey mode)
// - Rolling prebuffer so speech before hotkey press isn't lost
// - OutputSink for stdout vs keyboard injection

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::adapter::Adapter;
use crate::common;
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::hotkey;
use crate::mel::{self, IncrementalMel};
use crate::tokenizer::{self, Tokenizer};

/// Runtime-configurable streaming parameters.
pub struct StreamConfig {
    pub delay_tokens: usize,
    pub silence_threshold: f32,
    pub silence_chunks: usize,
    pub min_speech_chunks: usize,
    pub rms_ema_alpha: f32,
    pub hotkey: Option<rdev::Key>,
    pub delay_up_key: Option<rdev::Key>,
    pub delay_down_key: Option<rdev::Key>,
    pub type_mode: bool,
    pub dual_delay: bool,
    pub slow_delay_tokens: usize,
}

/// PCM samples per decoder token: MEL_FRAMES_PER_TOKEN × HOP_LENGTH.
const SAMPLES_PER_TOKEN: usize = common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;


/// Debug status line at top of terminal (row 1). Uses ANSI save/restore cursor.
fn debug_status(rms: f32, smooth: f32, config: &StreamConfig, silence_ctr: usize, speech_ctr: usize, emitted: bool) {
    let w = 30;
    let max = 0.03f32;
    let level = ((smooth / max).min(1.0) * w as f32) as usize;
    let tp = ((config.silence_threshold / max).min(1.0) * w as f32) as usize;
    let mut bar = String::with_capacity(w);
    for i in 0..w {
        if i == tp { bar.push('|'); }
        else if i < level { bar.push('\u{2588}'); }
        else { bar.push('\u{2591}'); }
    }
    eprint!(
        "\x1b[s\x1b[1;1H\x1b[K [{}] raw={:.4} ema={:.4} sil={}/{} spk={}/{} em={}\x1b[u",
        bar, rms, smooth, silence_ctr, config.silence_chunks,
        speech_ctr, config.min_speech_chunks, emitted,
    );
}

// Incremental conv stem: keep 4 mel frames as context between iterations.
const CONV_CTX: usize = 4;
const CONV_SKIP: usize = 2;
const NEW_MEL_PER_CHUNK: usize = common::MEL_FRAMES_PER_TOKEN;

/// Output destination for transcribed text.
enum OutputSink {
    Stdout,
    Keyboard(enigo::Enigo),
}

impl OutputSink {
    fn emit_text(&mut self, text: &[u8]) {
        match self {
            OutputSink::Stdout => {
                let _ = std::io::stdout().write_all(text);
                let _ = std::io::stdout().flush();
            }
            OutputSink::Keyboard(enigo) => {
                use enigo::Keyboard;
                if let Ok(s) = std::str::from_utf8(text) {
                    let _ = enigo.text(s);
                }
            }
        }
    }

    fn emit_newline(&mut self) {
        match self {
            OutputSink::Stdout => {
                print!("\n\n");
                let _ = std::io::stdout().flush();
            }
            OutputSink::Keyboard(enigo) => {
                use enigo::{Direction, Key, Keyboard};
                let _ = enigo.key(Key::Shift, Direction::Press);
                let _ = enigo.key(Key::Return, Direction::Click);
                let _ = enigo.key(Key::Return, Direction::Click);
                let _ = enigo.key(Key::Shift, Direction::Release);
            }
        }
    }
}

// ---- Dual-delay terminal display ----

/// Manages dual-delay terminal output: confirmed text (normal) + speculative text (dim).
/// Confirmed = slow stream output. Speculative = fast stream words ahead of slow.
struct DualDisplay {
    slow_text: String,
    fast_text: String,
    committed_bytes: usize,     // bytes of slow_text permanently printed
    spec_display_chars: usize,  // visible chars of speculative text on screen
}

impl DualDisplay {
    fn new() -> Self {
        Self {
            slow_text: String::new(),
            fast_text: String::new(),
            committed_bytes: 0,
            spec_display_chars: 0,
        }
    }

    fn push_fast_token(&mut self, token: u32, tok: &Tokenizer) {
        if token == tokenizer::STREAMING_PAD_ID
            || token == tokenizer::STREAMING_WORD_ID
            || token == tokenizer::EOS_ID
        {
            return;
        }
        if let Some(bytes) = tok.decode_token(token) {
            if let Ok(s) = std::str::from_utf8(&bytes) {
                self.fast_text.push_str(s);
            }
        }
    }

    fn push_slow_token(&mut self, token: u32, tok: &Tokenizer) {
        if token == tokenizer::STREAMING_PAD_ID
            || token == tokenizer::STREAMING_WORD_ID
            || token == tokenizer::EOS_ID
        {
            return;
        }
        if let Some(bytes) = tok.decode_token(token) {
            if let Ok(s) = std::str::from_utf8(&bytes) {
                self.slow_text.push_str(s);
            }
        }
    }

    fn erase_speculative(&mut self) {
        if self.spec_display_chars > 0 {
            // Move cursor left, clear to end of line
            print!("\x1b[{}D\x1b[K", self.spec_display_chars);
            self.spec_display_chars = 0;
        }
    }

    /// Rerender: erase old speculative, print new confirmed, print new speculative.
    fn refresh(&mut self) {
        self.erase_speculative();

        // Print any new confirmed text (from slow stream)
        if self.committed_bytes < self.slow_text.len() {
            print!("{}", &self.slow_text[self.committed_bytes..]);
            self.committed_bytes = self.slow_text.len();
        }

        // Compute and print speculative text (fast ahead of slow)
        let speculative = self.compute_speculative();
        if !speculative.is_empty() {
            print!("\x1b[2m{}\x1b[22m", speculative); // dim
            self.spec_display_chars = speculative.chars().count();
        }

        let _ = std::io::stdout().flush();
    }

    /// Find fast-stream words ahead of slow via reverse matching.
    /// Searches backwards from the end of fast_text for the last slow word,
    /// verifying with preceding context. This avoids cascading misalignment
    /// when earlier words differ between the fast and slow streams.
    fn compute_speculative(&self) -> String {
        let slow_words: Vec<&str> = self.slow_text.split_whitespace().collect();
        let fast_words: Vec<&str> = self.fast_text.split_whitespace().collect();

        if slow_words.is_empty() {
            return self.fast_text.clone();
        }

        let last_sw = slow_words.last().unwrap();
        let sw_norm = last_sw
            .trim_matches(|c: char| c.is_ascii_punctuation())
            .to_lowercase();

        // Search backwards in fast for the last slow word, with context verification.
        let mut match_j: Option<usize> = None;
        for j in (0..fast_words.len()).rev() {
            let fw_norm = fast_words[j]
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();

            // Exact match, or partial prefix match for the last (possibly incomplete) word
            let word_matches = sw_norm == fw_norm
                || (fw_norm.starts_with(&sw_norm) && sw_norm.len() >= 2);
            if !word_matches { continue; }

            // Verify: check that preceding slow words also match the preceding fast words.
            // This disambiguates common words like "the" that appear many times.
            let verify_count = slow_words.len().min(3).min(j + 1);
            let mut verified = true;
            for k in 1..verify_count {
                let s = slow_words[slow_words.len() - 1 - k]
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase();
                let f = fast_words[j - k]
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase();
                if s != f { verified = false; break; }
            }

            if verified {
                match_j = Some(j);
                break;
            }
        }

        let match_j = match match_j {
            Some(j) => j,
            None => return String::new(),
        };

        // Carry unmatched tail of the matched fast word.
        // Handles partial words ("Mach" → "machines." carries "ines.")
        // and pending punctuation ("Machines" → "machines." carries ".").
        let fw = fast_words[match_j];
        let tail_carry = if last_sw.len() < fw.len() {
            fw[last_sw.len()..].to_string()
        } else {
            String::new()
        };

        let remaining = match_j + 1;
        if remaining < fast_words.len() {
            if tail_carry.is_empty() {
                format!(" {}", fast_words[remaining..].join(" "))
            } else {
                format!("{} {}", tail_carry, fast_words[remaining..].join(" "))
            }
        } else if !tail_carry.is_empty() {
            tail_carry
        } else {
            String::new()
        }
    }

    fn emit_newline(&mut self) {
        self.erase_speculative();
        // Commit remaining slow text
        if self.committed_bytes < self.slow_text.len() {
            print!("{}", &self.slow_text[self.committed_bytes..]);
        }
        print!("\n\n");
        let _ = std::io::stdout().flush();
        // Reset for new paragraph
        self.slow_text.clear();
        self.fast_text.clear();
        self.committed_bytes = 0;
    }
}

/// Startup result passed to the streaming loop.
struct StartupState {
    last_token: u32,
    mel_frames: Vec<[f32; mel::N_MELS]>,
}

/// Process [silence + delay_audio] through batch mel → enc → adapter → prefill.
fn run_startup(
    delay_samples: &[f32],
    delay_tokens: usize,
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
    sink: &mut OutputSink,
) -> Result<StartupState> {
    let left_pad_samples = common::LEFT_PAD_TOKENS * common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

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
    let total_enc_frames = enc_out.dim(1)?;

    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(delay_tokens as f32, device, dtype)?;
    let prefill_len = decoder::prefill_len(delay_tokens);

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;
    let mut last_token = common::argmax_last(&logits)?;
    emit_token(last_token, tok, sink);

    for pos in prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;
        emit_token(next_token, tok, sink);
        last_token = next_token;
        if last_token == tokenizer::EOS_ID { break; }
    }

    eprintln!("Startup: {} mel frames, {} encoder frames, {} adapter frames",
        mel_time, total_enc_frames, n_adapter);

    Ok(StartupState { last_token, mel_frames })
}

/// Dual-delay startup: shared encoder/adapter, two independent decoder sessions.
fn run_dual_startup(
    fast_delay: usize,
    slow_delay: usize,
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &TextDecoder,
    fast_state: &mut decoder::DecoderState,
    slow_state: &mut decoder::DecoderState,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
    display: &mut DualDisplay,
) -> Result<(u32, u32, Vec<[f32; mel::N_MELS]>)> {
    let left_pad_samples = common::LEFT_PAD_TOKENS * common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;
    // Use slow delay for silence (longer = more encoder context for both streams)
    let delay_samples_count = (1 + slow_delay) * SAMPLES_PER_TOKEN;

    let mut padded = vec![0.0f32; left_pad_samples];
    padded.extend(vec![0.0f32; delay_samples_count]);

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

    // Shared encoder + adapter (runs once)
    let enc_out = enc.forward(&mel_tensor)?;
    let total_enc_frames = enc_out.dim(1)?;
    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    // ---- Fast decoder startup ----
    let fast_t_cond = decoder::sinusoidal_embedding(fast_delay as f32, device, dtype)?;
    let fast_prefill_len = decoder::prefill_len(fast_delay);
    let fast_prefill_embeds = dec.prepare_prefill(&adapter_out, fast_delay, device, dtype)?;

    TextDecoder::reset_state(fast_state);
    dec.precompute_t_cond_into(&fast_t_cond, fast_state)?;
    let logits = dec.forward_with_state(&fast_prefill_embeds, fast_state)?;
    let mut fast_last = common::argmax_last(&logits)?;
    display.push_fast_token(fast_last, tok);

    for pos in fast_prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[fast_last], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward_with_state(&fused, fast_state)?;
        let next_token = common::argmax_last(&logits)?;
        display.push_fast_token(next_token, tok);
        fast_last = next_token;
        if fast_last == tokenizer::EOS_ID { break; }
    }

    // ---- Slow decoder startup ----
    let slow_t_cond = decoder::sinusoidal_embedding(slow_delay as f32, device, dtype)?;
    let slow_prefill_len = decoder::prefill_len(slow_delay);
    let slow_prefill_embeds = dec.prepare_prefill(&adapter_out, slow_delay, device, dtype)?;

    TextDecoder::reset_state(slow_state);
    dec.precompute_t_cond_into(&slow_t_cond, slow_state)?;
    let logits = dec.forward_with_state(&slow_prefill_embeds, slow_state)?;
    let mut slow_last = common::argmax_last(&logits)?;
    display.push_slow_token(slow_last, tok);

    for pos in slow_prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[slow_last], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward_with_state(&fused, slow_state)?;
        let next_token = common::argmax_last(&logits)?;
        display.push_slow_token(next_token, tok);
        slow_last = next_token;
        if slow_last == tokenizer::EOS_ID { break; }
    }

    eprintln!("Dual startup: {} mel, {} enc, {} adapter (fast={}, slow={})",
        mel_time, total_enc_frames, n_adapter, fast_delay, slow_delay);

    Ok((fast_last, slow_last, mel_frames))
}

/// The core streaming processing loop.
fn run_processing_loop(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    state: &mut StartupState,
    mel_buffer: &mut Vec<[f32; mel::N_MELS]>,
    device: &Device,
    dtype: DType,
    sink: &mut OutputSink,
) -> Result<bool> {
    while mel_buffer.len() >= CONV_CTX + NEW_MEL_PER_CHUNK {
        let process_len = CONV_CTX + NEW_MEL_PER_CHUNK;

        // let t0 = std::time::Instant::now();

        let mut mel_data = vec![0.0f32; mel::N_MELS * process_len];
        for (frame_idx, frame) in mel_buffer[..process_len].iter().enumerate() {
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
        mel_buffer.drain(..NEW_MEL_PER_CHUNK);

        let enc_out = enc.forward_chunk(&new_conv_frames)?;
        enc.trim_caches();

        let adapter_out = adapter.forward(&enc_out)?;

        let tok_embed = dec.embed_tokens(&[state.last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;

        emit_token(next_token, tok, sink);
        dec.trim_caches();

        state.last_token = next_token;
        if state.last_token == tokenizer::EOS_ID {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Dual-delay processing loop: shared encoder/adapter, two decoder passes per tick.
fn run_dual_processing_loop(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &TextDecoder,
    tok: &Tokenizer,
    fast_state: &mut decoder::DecoderState,
    slow_state: &mut decoder::DecoderState,
    fast_last_token: &mut u32,
    slow_last_token: &mut u32,
    mel_buffer: &mut Vec<[f32; mel::N_MELS]>,
    device: &Device,
    dtype: DType,
    display: &mut DualDisplay,
) -> Result<bool> {
    while mel_buffer.len() >= CONV_CTX + NEW_MEL_PER_CHUNK {
        let process_len = CONV_CTX + NEW_MEL_PER_CHUNK;

        let mut mel_data = vec![0.0f32; mel::N_MELS * process_len];
        for (frame_idx, frame) in mel_buffer[..process_len].iter().enumerate() {
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
        mel_buffer.drain(..NEW_MEL_PER_CHUNK);

        let enc_out = enc.forward_chunk(&new_conv_frames)?;
        enc.trim_caches();

        let adapter_out = adapter.forward(&enc_out)?;

        // Fast decoder
        let tok_embed = dec.embed_tokens(&[*fast_last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward_with_state(&fused, fast_state)?;
        let fast_token = common::argmax_last(&logits)?;
        display.push_fast_token(fast_token, tok);
        TextDecoder::trim_state(fast_state);
        *fast_last_token = fast_token;

        // Slow decoder
        let tok_embed = dec.embed_tokens(&[*slow_last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward_with_state(&fused, slow_state)?;
        let slow_token = common::argmax_last(&logits)?;
        display.push_slow_token(slow_token, tok);
        TextDecoder::trim_state(slow_state);
        *slow_last_token = slow_token;

        display.refresh();

        if *fast_last_token == tokenizer::EOS_ID && *slow_last_token == tokenizer::EOS_ID {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Live streaming from microphone.
pub fn run_streaming(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
    config: &StreamConfig,
) -> Result<()> {
    let delay_samples_count = (1 + config.delay_tokens) * SAMPLES_PER_TOKEN;

    // Initialize output sink
    let mut sink = if config.type_mode {
        OutputSink::Keyboard(enigo::Enigo::new(&enigo::Settings::default())
            .map_err(|e| anyhow::anyhow!("Failed to initialize keyboard output: {}", e))?)
    } else {
        OutputSink::Stdout
    };

    let running = Arc::new(AtomicBool::new(true));

    // Open mic
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    // Resample state
    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let ratio_int = resample_ratio as usize;
    let mut raw_buf: Vec<f32> = Vec::new();

    // Startup with silence — model is ready, real audio goes through incremental path
    let silence = vec![0.0f32; delay_samples_count];
    eprintln!("Running startup with silence ({} samples)...", silence.len());
    let mut state = run_startup(
        &silence, config.delay_tokens, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
    )?;

    // State machine: with hotkey starts READY (waiting for press),
    // without hotkey starts ACTIVE (recording immediately)
    let initial_state = if config.hotkey.is_some() {
        hotkey::STATE_READY
    } else {
        hotkey::STATE_ACTIVE
    };
    let hotkey_state = Arc::new(AtomicU8::new(initial_state));
    let delay_value = Arc::new(AtomicUsize::new(config.delay_tokens));
    let has_delay_keys = config.delay_up_key.is_some() || config.delay_down_key.is_some();
    hotkey::spawn_listener(
        running.clone(),
        config.hotkey,
        Some(hotkey_state.clone()),
        config.delay_up_key,
        config.delay_down_key,
        if has_delay_keys { Some(delay_value.clone()) } else { None },
    );

    if let Some(key) = config.hotkey {
        eprintln!("Press {:?} to start/stop recording. Ctrl+C to quit.", key);
    } else {
        eprintln!("\n--- Listening (Ctrl+C to stop) ---\n");
    }

    // Keep only last CONV_CTX mel frames as conv stem context
    let mel_len = state.mel_frames.len();
    let ctx_start = mel_len.saturating_sub(CONV_CTX);
    let mut mel_buffer: Vec<[f32; mel::N_MELS]> = state.mel_frames[ctx_start..].to_vec();
    drop(std::mem::take(&mut state.mel_frames));

    let mut inc_mel = IncrementalMel::new(filters);
    let mut prebuffer: VecDeque<f32> = VecDeque::with_capacity(delay_samples_count);
    let mut local_delay = config.delay_tokens;

    // Silence detection state
    let mut silence_counter: usize = 0;
    let mut speech_counter: usize = 0;
    let mut smoothed_rms: f32 = 0.0;
    let mut silence_emitted = true;
    let mut sample_buf_for_silence: Vec<f32> = Vec::new();

    let mut prev_state_val = initial_state;

    while running.load(Ordering::SeqCst) {
        let current_state_val = hotkey_state.load(Ordering::SeqCst);

        // Pull audio from channel
        let mut new_16k = Vec::new();
        match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(chunk) => {
                raw_buf.extend_from_slice(&chunk);
                while let Ok(more) = rx.try_recv() {
                    raw_buf.extend_from_slice(&more);
                }
                resample(&mut raw_buf, &mut new_16k, need_resample, ratio_int);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Check for delay change — restart model with new delay
        let new_delay = delay_value.load(Ordering::SeqCst);
        if new_delay != local_delay {
            local_delay = new_delay;
            let silence = vec![0.0f32; (1 + local_delay) * SAMPLES_PER_TOKEN];
            state = run_startup(
                &silence, local_delay, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
            )?;
            let mel_len = state.mel_frames.len();
            mel_buffer = state.mel_frames[mel_len.saturating_sub(CONV_CTX)..].to_vec();
            drop(std::mem::take(&mut state.mel_frames));
            inc_mel = IncrementalMel::new(filters);
            if current_state_val == hotkey::STATE_ACTIVE { sink.emit_newline(); }
            prev_state_val = current_state_val;
            continue;
        }

        if current_state_val == hotkey::STATE_READY || current_state_val == hotkey::STATE_PAUSED {
            // ---- Ready/Paused: feed rolling prebuffer, don't process ----

            // Detect Active → Paused transition: flush decoder's lookahead
            if prev_state_val == hotkey::STATE_ACTIVE {
                if !new_16k.is_empty() {
                    inc_mel.push_samples(&new_16k);
                    new_16k.clear();
                }

                let flush_samples = (local_delay + 4) * SAMPLES_PER_TOKEN;
                let flush_silence = vec![0.0f32; flush_samples];
                inc_mel.push_samples(&flush_silence);

                mel_buffer.extend(inc_mel.drain_frames());
                let _ = run_processing_loop(
                    enc, adapter, dec, tok, &mut state,
                    &mut mel_buffer, device, dtype, &mut sink,
                )?;

                if mel_buffer.len() > CONV_CTX {
                    mel_buffer.drain(..mel_buffer.len() - CONV_CTX);
                }
                sink.emit_newline();

                // Reset silence detection
                silence_counter = 0;
                speech_counter = 0;
                smoothed_rms = 0.0;
                silence_emitted = true;
                sample_buf_for_silence.clear();
            }

            // Feed prebuffer (capped to current delay)
            let prebuf_cap = (1 + local_delay) * SAMPLES_PER_TOKEN;
            for &s in &new_16k {
                if prebuffer.len() >= prebuf_cap {
                    prebuffer.pop_front();
                }
                prebuffer.push_back(s);
            }
        } else {
            // ---- Active: process audio ----
            if prev_state_val != hotkey::STATE_ACTIVE {
                // Just resumed: flush prebuffer into mel
                let prebuf_samples: Vec<f32> = prebuffer.drain(..).collect();
                if !prebuf_samples.is_empty() {
                    inc_mel.push_samples(&prebuf_samples);
                }
            }

            if !new_16k.is_empty() {
                inc_mel.push_samples(&new_16k);
                sample_buf_for_silence.extend_from_slice(&new_16k);
            }

            mel_buffer.extend(inc_mel.drain_frames());

            // Silence detection
            while sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
                let rms = (sample_buf_for_silence[..SAMPLES_PER_TOKEN].iter()
                    .map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
                sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
                smoothed_rms = config.rms_ema_alpha * rms + (1.0 - config.rms_ema_alpha) * smoothed_rms;
                if rms < config.silence_threshold {
                    silence_counter += 1;
                } else {
                    silence_counter = 0;
                }
                if smoothed_rms >= config.silence_threshold {
                    speech_counter += 1;
                    if speech_counter >= config.min_speech_chunks {
                        silence_emitted = false;
                    }
                } else {
                    speech_counter = 0;
                }
                debug_status(rms, smoothed_rms, config, silence_counter, speech_counter, silence_emitted);
            }

            let eos = run_processing_loop(enc, adapter, dec, tok, &mut state,
                &mut mel_buffer, device, dtype, &mut sink)?;
            if eos { break; }

            if silence_counter >= config.silence_chunks && !silence_emitted {
                sink.emit_newline();
                silence_emitted = true;
            }
        }

        prev_state_val = current_state_val;
    }

    eprintln!("\n--- Stopped ---");
    Ok(())
}

/// Dual-delay live streaming: two decoder sessions on shared encoder/adapter.
pub fn run_dual_streaming(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
    config: &StreamConfig,
) -> Result<()> {
    let fast_delay = config.delay_tokens;
    let slow_delay = config.slow_delay_tokens;

    let mut fast_state = dec.create_state();
    let mut slow_state = dec.create_state();
    let mut display = DualDisplay::new();

    let running = Arc::new(AtomicBool::new(true));

    // Open mic
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let ratio_int = resample_ratio as usize;
    let mut raw_buf: Vec<f32> = Vec::new();

    eprintln!("Running dual startup (fast delay={}, slow delay={})...", fast_delay, slow_delay);
    let (mut fast_last_token, mut slow_last_token, mel_frames) = run_dual_startup(
        fast_delay, slow_delay,
        enc, adapter, dec, &mut fast_state, &mut slow_state,
        tok, filters, device, dtype, &mut display,
    )?;

    // State machine: hotkey mode or always-on
    let initial_state = if config.hotkey.is_some() {
        hotkey::STATE_READY
    } else {
        hotkey::STATE_ACTIVE
    };
    let hotkey_state = Arc::new(AtomicU8::new(initial_state));
    // No delay up/down in dual mode
    hotkey::spawn_listener(
        running.clone(),
        config.hotkey,
        Some(hotkey_state.clone()),
        None, None, None,
    );

    if let Some(key) = config.hotkey {
        eprintln!("Press {:?} to start/stop recording. Ctrl+C to quit.", key);
    } else {
        eprintln!("\n--- Listening (dual: {}ms fast / {}ms slow, Ctrl+C to stop) ---\n",
            fast_delay * 80, slow_delay * 80);
    }

    // Keep only last CONV_CTX mel frames as conv stem context
    let mel_len = mel_frames.len();
    let ctx_start = mel_len.saturating_sub(CONV_CTX);
    let mut mel_buffer: Vec<[f32; mel::N_MELS]> = mel_frames[ctx_start..].to_vec();

    let mut inc_mel = IncrementalMel::new(filters);
    let slow_delay_samples = (1 + slow_delay) * SAMPLES_PER_TOKEN;
    let mut prebuffer: VecDeque<f32> = VecDeque::with_capacity(slow_delay_samples);

    // Silence detection state
    let mut silence_counter: usize = 0;
    let mut speech_counter: usize = 0;
    let mut smoothed_rms: f32 = 0.0;
    let mut silence_emitted = true;
    let mut sample_buf_for_silence: Vec<f32> = Vec::new();

    let mut prev_state_val = initial_state;

    while running.load(Ordering::SeqCst) {
        let current_state_val = hotkey_state.load(Ordering::SeqCst);

        // Pull audio from channel
        let mut new_16k = Vec::new();
        match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(chunk) => {
                raw_buf.extend_from_slice(&chunk);
                while let Ok(more) = rx.try_recv() {
                    raw_buf.extend_from_slice(&more);
                }
                resample(&mut raw_buf, &mut new_16k, need_resample, ratio_int);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if current_state_val == hotkey::STATE_READY || current_state_val == hotkey::STATE_PAUSED {
            // ---- Ready/Paused: feed rolling prebuffer, don't process ----

            // Detect Active → Paused transition: flush both decoders
            if prev_state_val == hotkey::STATE_ACTIVE {
                if !new_16k.is_empty() {
                    inc_mel.push_samples(&new_16k);
                    new_16k.clear();
                }

                let flush_samples = (slow_delay + 4) * SAMPLES_PER_TOKEN;
                let flush_silence = vec![0.0f32; flush_samples];
                inc_mel.push_samples(&flush_silence);

                mel_buffer.extend(inc_mel.drain_frames());
                let _ = run_dual_processing_loop(
                    enc, adapter, dec, tok,
                    &mut fast_state, &mut slow_state,
                    &mut fast_last_token, &mut slow_last_token,
                    &mut mel_buffer, device, dtype, &mut display,
                )?;

                if mel_buffer.len() > CONV_CTX {
                    mel_buffer.drain(..mel_buffer.len() - CONV_CTX);
                }
                display.emit_newline();

                // Reset silence detection
                silence_counter = 0;
                speech_counter = 0;
                smoothed_rms = 0.0;
                silence_emitted = true;
                sample_buf_for_silence.clear();
            }

            // Feed prebuffer (capped to slow delay)
            let prebuf_cap = (1 + slow_delay) * SAMPLES_PER_TOKEN;
            for &s in &new_16k {
                if prebuffer.len() >= prebuf_cap {
                    prebuffer.pop_front();
                }
                prebuffer.push_back(s);
            }
        } else {
            // ---- Active: process audio ----
            if prev_state_val != hotkey::STATE_ACTIVE {
                // Just resumed: flush prebuffer into mel
                let prebuf_samples: Vec<f32> = prebuffer.drain(..).collect();
                if !prebuf_samples.is_empty() {
                    inc_mel.push_samples(&prebuf_samples);
                }
            }

            if !new_16k.is_empty() {
                inc_mel.push_samples(&new_16k);
                sample_buf_for_silence.extend_from_slice(&new_16k);
            }

            mel_buffer.extend(inc_mel.drain_frames());

            // Silence detection
            while sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
                let rms = (sample_buf_for_silence[..SAMPLES_PER_TOKEN].iter()
                    .map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
                sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
                smoothed_rms = config.rms_ema_alpha * rms + (1.0 - config.rms_ema_alpha) * smoothed_rms;
                if rms < config.silence_threshold {
                    silence_counter += 1;
                } else {
                    silence_counter = 0;
                }
                if smoothed_rms >= config.silence_threshold {
                    speech_counter += 1;
                    if speech_counter >= config.min_speech_chunks {
                        silence_emitted = false;
                    }
                } else {
                    speech_counter = 0;
                }
                debug_status(rms, smoothed_rms, config, silence_counter, speech_counter, silence_emitted);
            }

            let eos = run_dual_processing_loop(
                enc, adapter, dec, tok,
                &mut fast_state, &mut slow_state,
                &mut fast_last_token, &mut slow_last_token,
                &mut mel_buffer, device, dtype, &mut display,
            )?;
            if eos { break; }

            if silence_counter >= config.silence_chunks && !silence_emitted {
                display.emit_newline();
                silence_emitted = true;
            }
        }

        prev_state_val = current_state_val;
    }

    eprintln!("\n--- Stopped ---");
    Ok(())
}

// ---- Helpers ----

fn resample(raw_buf: &mut Vec<f32>, out: &mut Vec<f32>, need_resample: bool, ratio_int: usize) {
    if !need_resample {
        out.extend(raw_buf.drain(..));
        return;
    }
    let n_out = raw_buf.len() / ratio_int;
    out.reserve(n_out);
    for i in 0..n_out {
        out.push(raw_buf[i * ratio_int]);
    }
    raw_buf.drain(..n_out * ratio_int);
}

fn emit_token(token: u32, tok: &Tokenizer, sink: &mut OutputSink) {
    if token == tokenizer::STREAMING_PAD_ID { return; }
    if token == tokenizer::STREAMING_WORD_ID { return; }
    if token == tokenizer::EOS_ID { return; }
    if let Some(bytes) = tok.decode_token(token) {
        sink.emit_text(&bytes);
    }
}

fn open_mic(tx: mpsc::Sender<Vec<f32>>) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let input_device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    eprintln!("Input device: {}", input_device.name().unwrap_or_default());

    let default_config = input_device.default_input_config()?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    eprintln!("Native config: {}Hz, {} ch", native_rate, native_channels);

    // Use native channel count — some devices reject mono requests
    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    if native_rate != 16000 {
        eprintln!("Will resample {}Hz -> 16kHz on inference thread", native_rate);
    }

    let ch = native_channels as usize;
    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if ch == 1 {
                let _ = tx.send(data.to_vec());
            } else {
                // Downmix to mono by averaging all channels per frame
                let mut mono = vec![0.0f32; data.len() / ch];
                let scale = 1.0 / ch as f32;
                for (i, frame) in data.chunks_exact(ch).enumerate() {
                    let mut sum = 0.0f32;
                    for &s in frame {
                        sum += s;
                    }
                    mono[i] = sum * scale;
                }
                let _ = tx.send(mono);
            }
        },
        |err| {
            eprintln!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    Ok((stream, native_rate))
}
