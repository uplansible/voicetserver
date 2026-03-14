// Streaming mic capture + real-time inference pipeline for Voicet Phase 2
//
// Architecture: cpal audio callback -> mpsc channel -> main thread inference loop
// Timing: 80ms of audio (1280 samples, 8 mel frames) = 1 decoder token

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::adapter::Adapter;
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::mel::{self, IncrementalMel};
use crate::tokenizer::{self, Tokenizer};

const SILENCE_THRESHOLD: f32 = 0.01;
const SILENCE_CHUNKS: usize = 8; // 8 * 80ms = 640ms
const SAMPLES_PER_TOKEN: usize = 1280; // 8 mel frames * 160 hop

// Number of real-audio adapter frames needed for the delay portion of prefill.
// prefill_len = 1 + 32 + 6 = 39. Silence produces 32 adapter frames.
// Positions 32-38 = 7 frames must come from real audio.
const DELAY_ADAPTER_FRAMES: usize = 7;
// Samples needed: 7 adapter frames * 4 encoder frames * 2 mel-per-conv * 160 hop
const DELAY_SAMPLES: usize = DELAY_ADAPTER_FRAMES * 4 * 2 * mel::HOP_LENGTH; // 8960

/// Feed a WAV file through the streaming pipeline for validation.
pub fn run_stream_test(
    wav_samples: &[f32],
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("=== Stream Test: feeding {} samples ({:.2}s) ===\n",
        wav_samples.len(), wav_samples.len() as f64 / 16000.0);

    // Split: first DELAY_SAMPLES for startup, rest for streaming decode
    let delay_end = DELAY_SAMPLES.min(wav_samples.len());
    let delay_audio = &wav_samples[..delay_end];
    let remaining_audio = &wav_samples[delay_end..];

    let (mut last_token, mut enc_offset, t_cond, startup_mel_time) =
        run_startup(delay_audio, enc, adapter, dec, tok, filters, device, dtype)?;

    println!("Startup first token: {} ({})", last_token,
        if last_token == tokenizer::STREAMING_PAD_ID { "PAD" } else { "???" });

    // Feed remaining WAV samples through the streaming pipeline
    // Use IncrementalMel seeded with left context from the end of startup audio
    let left_ctx_start = delay_audio.len().saturating_sub(mel::N_FFT / 2);
    let left_ctx = &delay_audio[left_ctx_start..];
    let mut inc_mel = IncrementalMel::with_left_context(filters, left_ctx);
    inc_mel.push_samples(remaining_audio);

    // Prepend startup mel frames so conv stem has proper context
    let startup_mel_prefix: Vec<[f32; mel::N_MELS]> =
        vec![[-0.625f32; mel::N_MELS]; startup_mel_time];
    let mut all_mel_frames: Vec<[f32; mel::N_MELS]> = startup_mel_prefix;

    // Actually, we need REAL startup mel frames, not just -0.625 for the delay portion.
    // Let's recompute: the startup batch mel includes silence + delay audio.
    // We need those exact frames for the conv stem context.
    // Simpler approach: recompute startup mel here for the conv stem prefix.
    let mut startup_audio_for_mel = vec![0.0f32; 32 * 8 * mel::HOP_LENGTH]; // 40960 silence
    startup_audio_for_mel.extend_from_slice(delay_audio);
    let startup_mel_data = mel::log_mel_spectrogram(&startup_audio_for_mel, filters);
    let startup_mel_frames_count = startup_mel_data.len() / mel::N_MELS;
    all_mel_frames.clear();
    for t in 0..startup_mel_frames_count {
        let mut frame = [0.0f32; mel::N_MELS];
        for b in 0..mel::N_MELS {
            frame[b] = startup_mel_data[b * startup_mel_frames_count + t];
        }
        all_mel_frames.push(frame);
    }

    let startup_conv_frames = startup_mel_frames_count / 2;
    let mut prev_conv_output_len = startup_conv_frames;
    let mut token_count = 0usize;

    // Add remaining real audio mel frames
    let real_mel_frames = inc_mel.drain_frames();
    println!("Remaining audio: {} mel frames from {} samples\n",
        real_mel_frames.len(), remaining_audio.len());
    all_mel_frames.extend(real_mel_frames);

    // Process tokens
    while all_mel_frames.len() >= prev_conv_output_len * 2 + 8 {
        let total_mel = all_mel_frames.len();
        let mut mel_data = vec![0.0f32; mel::N_MELS * total_mel];
        for (frame_idx, frame) in all_mel_frames.iter().enumerate() {
            for mel_bin in 0..mel::N_MELS {
                mel_data[mel_bin * total_mel + frame_idx] = frame[mel_bin];
            }
        }
        let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, total_mel), device)?
            .to_dtype(dtype)?
            .unsqueeze(0)?;

        let conv_out = enc.conv_stem(&mel_tensor)?;
        let conv_out_len = conv_out.dim(1)?;

        if conv_out_len < prev_conv_output_len + encoder::CHUNK_SIZE {
            break;
        }
        let new_conv_frames = conv_out.narrow(1, prev_conv_output_len, encoder::CHUNK_SIZE)?;

        let enc_out = enc.forward_chunk(&new_conv_frames, enc_offset)?;
        enc_offset += encoder::CHUNK_SIZE;
        prev_conv_output_len += encoder::CHUNK_SIZE;

        let adapter_out = adapter.forward(&enc_out)?;

        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward(&fused, &t_cond)?;
        let next_token = argmax_last(&logits)?;

        token_count += 1;
        let label = if next_token == tokenizer::STREAMING_PAD_ID { "PAD".to_string() }
            else if next_token == tokenizer::STREAMING_WORD_ID { "WORD".to_string() }
            else if next_token == tokenizer::EOS_ID { "EOS".to_string() }
            else { format!("text '{}'", tok.decode(&[next_token])) };
        println!("  token {}: id={} ({})", token_count, next_token, label);

        last_token = next_token;
        if last_token == tokenizer::EOS_ID {
            break;
        }
    }

    println!("\nStream test complete: {} tokens generated", token_count);
    Ok(())
}

pub fn run_streaming(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Open mic BEFORE startup (buffer audio during GPU warm-up)
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    println!("Mic open, buffering audio during startup...");

    // Resample helper
    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let mut raw_buf: Vec<f32> = Vec::new();

    let resample_drain = |raw_buf: &mut Vec<f32>| -> Vec<f32> {
        if !need_resample {
            return raw_buf.drain(..).collect();
        }
        let ratio_int = resample_ratio as usize;
        let mut out = Vec::new();
        if (resample_ratio - ratio_int as f64).abs() < 0.001 {
            while raw_buf.len() >= ratio_int {
                out.push(raw_buf[0]);
                raw_buf.drain(..ratio_int);
            }
        } else {
            while raw_buf.len() as f64 >= resample_ratio {
                out.push(raw_buf[0]);
                let skip = resample_ratio as usize;
                raw_buf.drain(..skip);
            }
        }
        out
    };

    // Wait for enough audio for the delay portion (~560ms) while processing silence
    // The silence GPU processing takes ~2s, during which the mic buffers audio.
    // We need DELAY_SAMPLES (8960) samples at 16kHz.

    // Process silence through encoder (GPU work — mic buffers audio in parallel)
    let silence_samples = 32 * 8 * mel::HOP_LENGTH; // 40960
    let silence = vec![0.0f32; silence_samples];
    let silence_mel_data = mel::log_mel_spectrogram(&silence, filters);
    let silence_mel_time = silence_mel_data.len() / mel::N_MELS;
    let silence_mel_tensor = Tensor::from_vec(
        silence_mel_data.clone(), (mel::N_MELS, silence_mel_time), device)?
        .to_dtype(dtype)?.unsqueeze(0)?;

    enc.reset_caches();
    let silence_conv = enc.conv_stem(&silence_mel_tensor)?;
    let silence_frames = silence_conv.dim(1)?;
    let mut enc_offset = 0usize;
    let mut silence_enc_outputs = Vec::new();
    while enc_offset + encoder::CHUNK_SIZE <= silence_frames {
        let chunk = silence_conv.narrow(1, enc_offset, encoder::CHUNK_SIZE)?;
        let out = enc.forward_chunk(&chunk, enc_offset)?;
        silence_enc_outputs.push(out);
        enc_offset += encoder::CHUNK_SIZE;
    }

    // Now drain mic audio that buffered during silence processing
    while let Ok(chunk) = rx.try_recv() {
        raw_buf.extend_from_slice(&chunk);
    }
    let mut mic_samples_16k: Vec<f32> = resample_drain(&mut raw_buf);

    // Keep collecting until we have enough for the delay period
    while mic_samples_16k.len() < DELAY_SAMPLES {
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(chunk) => {
                raw_buf.extend_from_slice(&chunk);
                while let Ok(more) = rx.try_recv() {
                    raw_buf.extend_from_slice(&more);
                }
                mic_samples_16k.extend(resample_drain(&mut raw_buf));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("Audio stream disconnected during startup");
            }
        }
    }

    // Split mic audio: first DELAY_SAMPLES for startup, rest goes to streaming buffer
    let delay_audio: Vec<f32> = mic_samples_16k.drain(..DELAY_SAMPLES).collect();
    let leftover_audio: Vec<f32> = mic_samples_16k;

    println!("Got {} delay samples from mic, processing startup...", delay_audio.len());

    // Process delay audio through encoder + adapter (continuing from silence)
    let mut startup_audio = vec![0.0f32; silence_samples];
    startup_audio.extend_from_slice(&delay_audio);
    let startup_mel_data = mel::log_mel_spectrogram(&startup_audio, filters);
    let startup_mel_time = startup_mel_data.len() / mel::N_MELS;
    let startup_mel_tensor = Tensor::from_vec(
        startup_mel_data, (mel::N_MELS, startup_mel_time), device)?
        .to_dtype(dtype)?.unsqueeze(0)?;

    // Run conv stem on full startup audio
    let startup_conv = enc.conv_stem(&startup_mel_tensor)?;
    let startup_conv_len = startup_conv.dim(1)?;

    // Process delay conv frames through encoder (continuing from silence KV cache)
    let mut delay_enc_outputs = Vec::new();
    while enc_offset + encoder::CHUNK_SIZE <= startup_conv_len {
        let chunk = startup_conv.narrow(1, enc_offset, encoder::CHUNK_SIZE)?;
        let out = enc.forward_chunk(&chunk, enc_offset)?;
        delay_enc_outputs.push(out);
        enc_offset += encoder::CHUNK_SIZE;
    }

    // Build full adapter output: silence + delay
    let mut all_enc_outputs: Vec<&Tensor> = silence_enc_outputs.iter().collect();
    all_enc_outputs.extend(delay_enc_outputs.iter());
    let full_enc_cat = Tensor::cat(&all_enc_outputs, 1)?;
    let full_adapter_out = adapter.forward(&full_enc_cat)?;
    let n_adapter_frames = full_adapter_out.dim(1)?;

    // Prefill decoder: BOS + 38 PADs fused with adapter frames (matches offline exactly)
    let t_cond = decoder::sinusoidal_embedding(6.0, device, dtype)?;
    let prefill_len = 39usize; // 1 + 32 + 6
    let mut prefill_ids = vec![tokenizer::STREAMING_PAD_ID; prefill_len];
    prefill_ids[0] = tokenizer::BOS_ID;

    let tok_embeds = dec.embed_tokens(&prefill_ids, device)?;
    let audio_slice = full_adapter_out.narrow(1, 0, prefill_len.min(n_adapter_frames))?;
    let audio_slice = if prefill_len <= n_adapter_frames {
        audio_slice
    } else {
        let pad_len = prefill_len - n_adapter_frames;
        let zeros = Tensor::zeros((1, pad_len, decoder::HIDDEN_SIZE), dtype, device)?;
        Tensor::cat(&[&audio_slice, &zeros], 1)?
    };
    let prefill_embeds = tok_embeds.add(&audio_slice)?;

    dec.reset_caches();
    let logits = dec.forward(&prefill_embeds, &t_cond)?;
    let mut last_token = argmax_last(&logits)?;
    emit_token(last_token, tok);

    // Decode remaining adapter frames from startup (positions prefill_len..n_adapter_frames)
    for pos in prefill_len..n_adapter_frames {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let frame = full_adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&frame)?;
        let logits = dec.forward(&fused, &t_cond)?;
        let next_token = argmax_last(&logits)?;
        emit_token(next_token, tok);
        last_token = next_token;
    }

    println!("Startup complete ({} adapter frames, encoder offset {})",
        n_adapter_frames, enc_offset);
    println!("\n--- Listening (Ctrl+C to stop) ---\n");

    // --- Steady-state streaming loop ---
    // Seed IncrementalMel with left context from the end of delay audio
    let left_ctx_start = delay_audio.len().saturating_sub(mel::N_FFT / 2);
    let mut inc_mel = IncrementalMel::with_left_context(filters, &delay_audio[left_ctx_start..]);

    // Feed leftover mic audio that arrived after the delay period
    if !leftover_audio.is_empty() {
        inc_mel.push_samples(&leftover_audio);
    }

    // Build mel frame buffer with startup mel as prefix for conv stem context
    let startup_mel_for_prefix = mel::log_mel_spectrogram(
        &{ let mut a = vec![0.0f32; silence_samples]; a.extend_from_slice(&delay_audio); a },
        filters);
    let startup_mel_prefix_time = startup_mel_for_prefix.len() / mel::N_MELS;
    let mut all_mel_frames: Vec<[f32; mel::N_MELS]> = Vec::with_capacity(startup_mel_prefix_time + 1000);
    for t in 0..startup_mel_prefix_time {
        let mut frame = [0.0f32; mel::N_MELS];
        for b in 0..mel::N_MELS {
            frame[b] = startup_mel_for_prefix[b * startup_mel_prefix_time + t];
        }
        all_mel_frames.push(frame);
    }

    let startup_conv_frames = startup_mel_prefix_time / 2;
    let mut prev_conv_output_len = startup_conv_frames;
    let mut silence_counter: usize = 0;
    let mut sample_buf_for_silence: Vec<f32> = Vec::new();

    while running.load(Ordering::SeqCst) {
        // Pull audio from channel
        match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(chunk) => {
                raw_buf.extend_from_slice(&chunk);
                while let Ok(more) = rx.try_recv() {
                    raw_buf.extend_from_slice(&more);
                }
                let samples = resample_drain(&mut raw_buf);
                if !samples.is_empty() {
                    sample_buf_for_silence.extend_from_slice(&samples);
                    inc_mel.push_samples(&samples);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Drain available mel frames
        let new_frames = inc_mel.drain_frames();
        all_mel_frames.extend_from_slice(&new_frames);

        // Process tokens when we have enough mel frames
        while all_mel_frames.len() >= prev_conv_output_len * 2 + 8 {
            // Silence detection
            if sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
                let block: Vec<f32> = sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN).collect();
                let rms = (block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32).sqrt();
                if rms < SILENCE_THRESHOLD {
                    silence_counter += 1;
                } else {
                    silence_counter = 0;
                }
            }

            let total_mel = all_mel_frames.len();
            let mut mel_data = vec![0.0f32; mel::N_MELS * total_mel];
            for (frame_idx, frame) in all_mel_frames.iter().enumerate() {
                for mel_bin in 0..mel::N_MELS {
                    mel_data[mel_bin * total_mel + frame_idx] = frame[mel_bin];
                }
            }
            let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, total_mel), device)?
                .to_dtype(dtype)?
                .unsqueeze(0)?;

            let conv_out = enc.conv_stem(&mel_tensor)?;
            let conv_out_len = conv_out.dim(1)?;

            if conv_out_len < prev_conv_output_len + encoder::CHUNK_SIZE {
                break;
            }
            let new_conv_frames = conv_out.narrow(1, prev_conv_output_len, encoder::CHUNK_SIZE)?;

            let enc_out = enc.forward_chunk(&new_conv_frames, enc_offset)?;
            enc_offset += encoder::CHUNK_SIZE;
            prev_conv_output_len += encoder::CHUNK_SIZE;

            let adapter_out = adapter.forward(&enc_out)?;

            let tok_embed = dec.embed_tokens(&[last_token], device)?;
            let fused = tok_embed.add(&adapter_out)?;
            let logits = dec.forward(&fused, &t_cond)?;
            let next_token = argmax_last(&logits)?;

            emit_token(next_token, tok);
            last_token = next_token;

            if last_token == tokenizer::EOS_ID {
                println!("\n[EOS]");
                break;
            }

            if silence_counter >= SILENCE_CHUNKS {
                println!();
                silence_counter = 0;
            }
        }

        if last_token == tokenizer::EOS_ID {
            break;
        }
    }

    println!("\n\n--- Stopped ---");
    Ok(())
}

/// Startup: process [silence + delay_audio] through batch mel → encoder → adapter → decoder prefill.
/// This matches the offline pipeline exactly for the startup portion.
/// Returns (first_token, encoder_offset, t_cond, startup_mel_frame_count).
fn run_startup(
    delay_audio: &[f32],
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(u32, usize, Tensor, usize)> {
    let silence_samples = 32 * 8 * mel::HOP_LENGTH; // 40960

    // Batch mel on [silence + delay_audio] — matches offline behavior exactly
    let mut startup_audio = vec![0.0f32; silence_samples];
    startup_audio.extend_from_slice(delay_audio);
    let mel_data = mel::log_mel_spectrogram(&startup_audio, filters);
    let mel_time = mel_data.len() / mel::N_MELS;
    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    println!("Startup mel: [{}, {}] from {} silence + {} delay samples",
        mel::N_MELS, mel_time, silence_samples, delay_audio.len());

    // Full encoder pass (resets caches, processes all chunks)
    let enc_out = enc.forward(&mel_tensor)?;
    let enc_offset = enc_out.dim(1)?; // total encoder frames processed

    // Adapter
    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    // Delay conditioning
    let t_cond = decoder::sinusoidal_embedding(6.0, device, dtype)?;

    // Decoder prefill: BOS + 38 PADs fused with adapter frames
    let prefill_len = 39usize;
    let mut prefill_ids = vec![tokenizer::STREAMING_PAD_ID; prefill_len];
    prefill_ids[0] = tokenizer::BOS_ID;

    let tok_embeds = dec.embed_tokens(&prefill_ids, device)?;
    let audio_slice = if prefill_len <= n_adapter {
        adapter_out.narrow(1, 0, prefill_len)?
    } else {
        let avail = adapter_out.narrow(1, 0, n_adapter)?;
        let pad_len = prefill_len - n_adapter;
        let zeros = Tensor::zeros((1, pad_len, decoder::HIDDEN_SIZE), dtype, device)?;
        Tensor::cat(&[&avail, &zeros], 1)?
    };
    let prefill_embeds = tok_embeds.add(&audio_slice)?;

    dec.reset_caches();
    let logits = dec.forward(&prefill_embeds, &t_cond)?;
    let first_token = argmax_last(&logits)?;
    emit_token(first_token, tok);

    // Decode remaining startup adapter frames (pos 39..n_adapter)
    let mut last_token = first_token;
    for pos in prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&frame)?;
        let logits = dec.forward(&fused, &t_cond)?;
        let next_token = argmax_last(&logits)?;
        emit_token(next_token, tok);
        last_token = next_token;
    }

    println!("Startup complete ({} adapter frames, enc_offset {})", n_adapter, enc_offset);

    Ok((last_token, enc_offset, t_cond, mel_time))
}

fn emit_token(token: u32, tok: &Tokenizer) {
    if token == tokenizer::STREAMING_PAD_ID {
        return;
    }
    if token == tokenizer::STREAMING_WORD_ID {
        print!(" ");
        let _ = std::io::stdout().flush();
        return;
    }
    if token == tokenizer::EOS_ID {
        return;
    }
    if let Some(bytes) = tok.decode_token(token) {
        let _ = std::io::stdout().write_all(&bytes);
        let _ = std::io::stdout().flush();
    }
}

fn argmax_last(logits: &Tensor) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let last = logits_f32.i((0, 0, ..))?;
    let token_id = last.argmax(0)?.to_scalar::<u32>()?;
    Ok(token_id)
}

fn open_mic(tx: mpsc::Sender<Vec<f32>>) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let input_device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    println!("Input device: {}", input_device.name().unwrap_or_default());

    let default_config = input_device.default_input_config()?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    println!("Native config: {}Hz, {} ch", native_rate, native_channels);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    if native_rate != 16000 {
        println!("Will resample {}Hz -> 16kHz on inference thread", native_rate);
    }

    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = tx.send(data.to_vec());
        },
        |err| {
            eprintln!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    Ok((stream, native_rate))
}
