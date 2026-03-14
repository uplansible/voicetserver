mod adapter;
mod common;
mod decoder;
mod encoder;
mod mel;
mod streaming;
mod tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::fs;
use std::time::Instant;

const MODEL_DIR: &str = "Voxtral-Mini-4B-Realtime";

fn main() -> Result<()> {
    // Match PyTorch's default BF16 matmul behavior (reduced precision accumulation)
    candle_core::cuda_backend::set_gemm_reduced_precision_bf16(true);

    let arg1 = std::env::args().nth(1);
    let arg2 = std::env::args().nth(2);
    let stream_test = arg1.as_deref() == Some("--stream-test");
    let wav_path = if stream_test { arg2 } else { arg1 };

    // --- Setup ---
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;

    println!("Loading safetensors...");
    let st_path = format!("{MODEL_DIR}/consolidated.safetensors");
    let st_data = unsafe { memmap2::Mmap::map(&fs::File::open(&st_path)?)? };
    let vb = VarBuilder::from_slice_safetensors(&st_data, dtype, &device)?;

    let t_total = Instant::now();

    println!("Loading tokenizer...");
    let tok = tokenizer::Tokenizer::load(MODEL_DIR)?;

    println!("Loading encoder...");
    let mut enc = encoder::AudioEncoder::load(&vb, &device, dtype)?;

    println!("Loading adapter...");
    let adapter = adapter::Adapter::load(&vb)?;

    println!("Loading decoder...");
    let mut dec = decoder::TextDecoder::load(&vb, &device, dtype)?;

    println!("Total model load: {:.2}s", t_total.elapsed().as_secs_f64());

    let filters = mel::mel_filters(MODEL_DIR);

    if stream_test {
        let path = wav_path.expect("Usage: voicet --stream-test <file.wav>");
        let samples = load_wav(&path)?;
        streaming::run_stream_test(&samples, &mut enc, &adapter, &mut dec, &tok, &filters, &device, dtype)
    } else {
        match wav_path {
            Some(path) => run_offline(&path, &mut enc, &adapter, &mut dec, &tok, &filters, &device, dtype),
            None => {
                println!("\n=== Voicet Streaming Mode ===\n");
                streaming::run_streaming(&mut enc, &adapter, &mut dec, &tok, &filters, &device, dtype)
            }
        }
    }
}

fn run_offline(
    wav_path: &str,
    enc: &mut encoder::AudioEncoder,
    adapter: &adapter::Adapter,
    dec: &mut decoder::TextDecoder,
    tok: &tokenizer::Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("\n=== Voicet Offline Transcription ===\n");

    // --- Load audio ---
    println!("Loading audio: {}", wav_path);
    let samples = load_wav(wav_path)?;
    let audio_secs = samples.len() as f64 / 16000.0;
    println!("Audio: {} samples ({:.2}s at 16kHz)", samples.len(), audio_secs);

    // --- Mel spectrogram ---
    let t0 = Instant::now();
    let n_left_pad_tokens: usize = 32;
    let audio_length_per_tok: usize = 8;
    let left_pad_samples = n_left_pad_tokens * audio_length_per_tok * mel::HOP_LENGTH;
    let mut padded_samples = vec![0.0f32; left_pad_samples];
    padded_samples.extend_from_slice(&samples);
    let mel_data = mel::log_mel_spectrogram(&padded_samples, filters);
    let mel_time = mel_data.len() / mel::N_MELS;
    println!("Mel: [{}, {}] in {:.3}s", mel::N_MELS, mel_time, t0.elapsed().as_secs_f64());

    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    // --- Encoder ---
    let t0 = Instant::now();
    let enc_out = enc.forward(&mel_tensor)?;
    println!("Encoder: {:?} in {:.3}s", enc_out.shape(), t0.elapsed().as_secs_f64());

    // --- Adapter ---
    let t0 = Instant::now();
    let adapter_out = adapter.forward(&enc_out)?;
    println!("Adapter: {:?} in {:.3}s", adapter_out.shape(), t0.elapsed().as_secs_f64());

    // --- Generation ---
    let t0 = Instant::now();
    let n_audio_frames = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(6.0, device, dtype)?;

    let n_left_pad: usize = 32;
    let delay_tokens: usize = 6;
    let prefill_len = 1 + n_left_pad + delay_tokens;

    let mut prefill_ids = vec![tokenizer::STREAMING_PAD_ID; prefill_len];
    prefill_ids[0] = tokenizer::BOS_ID;

    let tok_embeds = dec.embed_tokens(&prefill_ids, device)?;
    let audio_slice = if prefill_len <= n_audio_frames {
        adapter_out.narrow(1, 0, prefill_len)?
    } else {
        let avail = adapter_out.narrow(1, 0, n_audio_frames.min(prefill_len))?;
        let pad_len = prefill_len - n_audio_frames.min(prefill_len);
        if pad_len > 0 {
            let zeros = Tensor::zeros((1, pad_len, decoder::HIDDEN_SIZE), dtype, device)?;
            Tensor::cat(&[&avail, &zeros], 1)?
        } else {
            avail
        }
    };
    let prefill_embeds = tok_embeds.add(&audio_slice)?;

    dec.reset_caches();
    let logits = dec.forward(&prefill_embeds, &t_cond)?;

    let first_token = argmax_last(&logits)?;
    println!("Prefill -> first token: {} ({})", first_token, tok.decode(&[first_token]));
    let mut generated_tokens: Vec<u32> = vec![first_token];

    let max_tokens = n_audio_frames;
    let mut pos = prefill_len;

    loop {
        if pos >= max_tokens {
            break;
        }
        let last_token = *generated_tokens.last().unwrap();
        if last_token == tokenizer::EOS_ID {
            break;
        }

        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = if pos < n_audio_frames {
            adapter_out.narrow(1, pos, 1)?
        } else {
            Tensor::zeros((1, 1, decoder::HIDDEN_SIZE), dtype, device)?
        };
        let input_embed = tok_embed.add(&audio_frame)?;

        let logits = dec.forward(&input_embed, &t_cond)?;
        let next_token = argmax_last(&logits)?;
        generated_tokens.push(next_token);
        pos += 1;
    }

    let gen_time = t0.elapsed().as_secs_f64();
    let n_tokens = generated_tokens.len();
    let text = tok.decode(&generated_tokens);

    println!("\n=== Transcription Result ===");
    println!("{}", text);
    println!("\n--- Stats ---");
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)",
        n_tokens, gen_time, n_tokens as f64 / gen_time);
    println!("Audio duration: {:.2}s", audio_secs);
    println!("Real-time factor: {:.2}x", gen_time / audio_secs);

    let n_pad = generated_tokens.iter().filter(|&&t| t == tokenizer::STREAMING_PAD_ID).count();
    let n_word = generated_tokens.iter().filter(|&&t| t == tokenizer::STREAMING_WORD_ID).count();
    let n_eos = generated_tokens.iter().filter(|&&t| t == tokenizer::EOS_ID).count();
    let n_text = n_tokens - n_pad - n_word - n_eos;
    println!("Tokens: {} text, {} pad, {} word-boundary, {} eos",
        n_text, n_pad, n_word, n_eos);

    println!("\nNon-PAD tokens:");
    for (i, &t) in generated_tokens.iter().enumerate() {
        if t != tokenizer::STREAMING_PAD_ID {
            println!("  pos={}: id={} '{}'", i + prefill_len, t, tok.decode(&[t]));
        }
    }

    Ok(())
}

fn argmax_last(logits: &Tensor) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let last = logits_f32.i((0, 0, ..))?;
    let token_id = last.argmax(0)?.to_scalar::<u32>()?;
    Ok(token_id)
}

fn load_wav(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    println!("  WAV spec: {} Hz, {} ch, {:?}, {} bits",
        spec.sample_rate, spec.channels, spec.sample_format, spec.bits_per_sample);

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max_val).collect()
        }
    };

    let samples = if spec.channels > 1 {
        samples.iter().step_by(spec.channels as usize).copied().collect()
    } else {
        samples
    };

    if spec.sample_rate != 16000 {
        println!("  Resampling from {} Hz to 16000 Hz", spec.sample_rate);
        let ratio = 16000.0 / spec.sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            resampled.push(samples[src_idx.min(samples.len() - 1)]);
        }
        return Ok(resampled);
    }

    Ok(samples)
}
