mod adapter;
mod audio;
mod common;
mod decoder;
mod encoder;
mod macros;
mod mel;
mod session;
mod settings;
mod streaming;
mod tokenizer;

#[cfg(feature = "cuda")]
mod m1_attention;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

use common::MEL_FRAMES_PER_TOKEN;

#[derive(Parser, Clone)]
#[command(name = "voicetserver", about = "SCHMIDIspeech — real-time German medical dictation server")]
pub struct Cli {
    /// WAV file for offline transcription (omit for server mode)
    pub wav_file: Option<String>,

    /// Directory containing model files (consolidated.safetensors, tekken.json, mel_filters.bin)
    #[arg(long, default_value = ".")]
    pub model_dir: String,

    /// CUDA device index
    #[arg(long, default_value_t = 0)]
    pub device: usize,

    /// Delay tokens (1–30; each = 80ms lookahead; higher = more accuracy)
    #[arg(long, default_value_t = 4)]
    pub delay: usize,

    /// Silence RMS threshold for paragraph breaks
    #[arg(long, default_value_t = 0.006)]
    pub silence_threshold: f32,

    /// Consecutive silent chunks before silence is detected (each chunk = 80ms)
    #[arg(long, default_value_t = 20)]
    pub silence_flush: usize,

    /// Minimum speech chunks before silence detection activates (each chunk = 80ms)
    #[arg(long, default_value_t = 15)]
    pub min_speech: usize,

    /// EMA smoothing factor for speech detection (0.0–1.0, lower = smoother)
    #[arg(long, default_value_t = 0.3)]
    pub rms_ema: f32,

    // ---- Server flags ----

    /// WebSocket listen port
    #[arg(long, default_value_t = 8765)]
    pub port: u16,

    /// Bind address (use 0.0.0.0 with Tailscale; 127.0.0.1 for local dev without TLS)
    #[arg(long, default_value = "127.0.0.1")]
    pub bind_addr: String,

    /// Path to TLS certificate (.crt/.pem). Required for wss:// (Tailscale cert).
    #[arg(long)]
    pub tls_cert: Option<String>,

    /// Path to TLS private key (.key). Required for wss://.
    #[arg(long)]
    pub tls_key: Option<String>,

    /// LoRA adapter directory (accepted and logged; not yet wired — Phase 3)
    #[arg(long)]
    pub lora_adapter: Option<String>,
}

/// Mutable model components — serialised by a single tokio::sync::Mutex.
/// All connections share this lock; GPU forward passes run serially.
pub struct ModelInner {
    pub enc: encoder::AudioEncoder,
    pub dec: decoder::TextDecoder,
}

/// All model components shared across WebSocket connections.
/// `inner` is locked per-chunk; `adapter`, `tok`, `filters`, `device`, `dtype` are read-only.
pub struct VoxtralModel {
    pub inner: tokio::sync::Mutex<ModelInner>,
    pub adapter: adapter::Adapter,
    pub tok: tokenizer::Tokenizer,
    pub filters: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(ref adapter_dir) = cli.lora_adapter {
        eprintln!("Note: --lora-adapter '{}' accepted but not yet wired (Phase 3)", adapter_dir);
    }

    // Validate delay
    if cli.delay < 1 || cli.delay > 30 {
        anyhow::bail!("--delay must be between 1 and 30 (got {})", cli.delay);
    }

    // mmap safetensors + spawn readahead (gives 0.5–1s head start before CUDA init)
    println!("Loading safetensors...");
    let st_path = format!("{}/consolidated.safetensors", cli.model_dir);
    let st_data = Arc::new(unsafe { memmap2::Mmap::map(&fs::File::open(&st_path)?)? });
    let readahead = {
        let data = Arc::clone(&st_data);
        std::thread::spawn(move || {
            let mut dummy = 0u8;
            for i in (0..data.len()).step_by(4096) {
                dummy = dummy.wrapping_add(data[i]);
            }
            std::hint::black_box(dummy);
        })
    };

    // Match PyTorch's default BF16 matmul precision
    #[cfg(feature = "cuda")]
    candle_core::cuda_backend::set_gemm_reduced_precision_bf16(true);

    let device = Device::cuda_if_available(cli.device)?;
    let dtype = DType::BF16;

    let is_offline = cli.wav_file.is_some();
    let effective_delay = if is_offline { 20 } else { cli.delay };

    println!("\n{:<28} {}", "Parameter", "Value");
    println!("{:-<28} {:-<32}", "", "");
    println!("{:<28} {} ({}ms){}",
        "Delay tokens", effective_delay, effective_delay * 80,
        if is_offline { " (offline max accuracy)" } else { "" });
    println!("{:<28} {}", "Silence threshold", cli.silence_threshold);
    println!("{:<28} {} ({}ms)", "Silence detection", cli.silence_flush, cli.silence_flush * 80);
    println!("{:<28} {} ({}ms)", "Min speech to activate", cli.min_speech, cli.min_speech * 80);
    println!("{:<28} {}", "RMS EMA alpha", cli.rms_ema);
    println!("{:<28} {:?}", "Compute dtype", dtype);
    if !is_offline {
        println!("{:<28} {}:{}", "Listen", cli.bind_addr, cli.port);
        let tls = cli.tls_cert.is_some() && cli.tls_key.is_some();
        println!("{:<28} {}", "TLS", if tls { "enabled (wss://)" } else { "disabled (ws://)" });
    }
    println!();

    let vb = VarBuilder::from_slice_safetensors(&st_data, dtype, &device)?;

    let t_total = Instant::now();

    println!("Loading tokenizer...");
    let tok = tokenizer::Tokenizer::load(&cli.model_dir)?;

    println!("Loading encoder...");
    let enc = encoder::AudioEncoder::load(&vb, &device, dtype)?;

    println!("Loading adapter...");
    let adapter = adapter::Adapter::load(&vb)?;

    println!("Loading decoder...");
    let dec = decoder::TextDecoder::load(&vb, &device, dtype)?;

    println!("Total model load: {:.2}s", t_total.elapsed().as_secs_f64());
    let _ = readahead.join();

    let filters = mel::mel_filters(&cli.model_dir);

    match cli.wav_file {
        Some(ref path) => {
            run_offline(path, effective_delay, enc, &adapter, dec, &tok, &filters, &device, dtype)
        }
        None => {
            let model = Arc::new(VoxtralModel {
                inner: tokio::sync::Mutex::new(ModelInner { enc, dec }),
                adapter,
                tok,
                filters,
                device,
                dtype,
            });
            server::run(model, Arc::new(cli)).await
        }
    }
}

// ---- Offline WAV transcription ----

fn run_offline(
    wav_path: &str,
    delay_tokens: usize,
    mut enc: encoder::AudioEncoder,
    adapter: &adapter::Adapter,
    mut dec: decoder::TextDecoder,
    tok: &tokenizer::Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("\n=== Voicetserver Offline Transcription ===\n");

    println!("Loading audio: {}", wav_path);
    let samples = load_wav(wav_path)?;
    let audio_secs = samples.len() as f64 / 16000.0;
    println!("Audio: {} samples ({:.2}s at 16kHz)", samples.len(), audio_secs);

    let t0 = Instant::now();
    let left_pad_samples = common::LEFT_PAD_TOKENS * MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;
    let mut padded_samples = vec![0.0f32; left_pad_samples];
    padded_samples.extend_from_slice(&samples);
    let mel_data = mel::log_mel_spectrogram(&padded_samples, filters);
    let mel_time = mel_data.len() / mel::N_MELS;
    println!("Mel: [{}, {}] in {:.3}s", mel::N_MELS, mel_time, t0.elapsed().as_secs_f64());

    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    let t0 = Instant::now();
    let enc_out = enc.forward(&mel_tensor)?;
    println!("Encoder: {:?} in {:.3}s", enc_out.shape(), t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let adapter_out = adapter.forward(&enc_out)?;
    println!("Adapter: {:?} in {:.3}s", adapter_out.shape(), t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let n_audio_frames = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(delay_tokens as f32, device, dtype)?;
    let prefill_len = decoder::prefill_len(delay_tokens);

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;

    let first_token = common::argmax_last(&logits)?;
    println!("Prefill -> first token: {} ({})", first_token, tok.decode(&[first_token]));
    let mut generated_tokens: Vec<u32> = vec![first_token];

    let max_tokens = n_audio_frames;
    let mut pos = prefill_len;

    loop {
        if pos >= max_tokens { break; }
        let last_token = *generated_tokens.last().unwrap();
        if last_token == tokenizer::EOS_ID { break; }

        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = if pos < n_audio_frames {
            adapter_out.narrow(1, pos, 1)?
        } else {
            Tensor::zeros((1, 1, decoder::HIDDEN_SIZE), dtype, device)?
        };
        let input_embed = tok_embed.add(&audio_frame)?;

        let logits = dec.forward(&input_embed)?;
        let next_token = common::argmax_last(&logits)?;
        generated_tokens.push(next_token);
        pos += 1;
        dec.trim_caches();
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

    Ok(())
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

// ---- Server module (inline import) ----
mod server {
    use super::*;
    use crate::settings::{IniValues, SharedSettings, SILENCE_CHUNKS_DEFAULT, STATE_READY};
    use axum::{
        extract::{State, WebSocketUpgrade},
        response::Response,
        routing::get,
        Router,
    };
    use axum::extract::ws::{Message, WebSocket};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub async fn run(model: Arc<VoxtralModel>, cli: Arc<Cli>) -> Result<()> {
        let vals = IniValues {
            delay: cli.delay,
            silence_threshold: cli.silence_threshold,
            silence_chunks: Some(cli.silence_flush),
            paragraph_delay_offset: 4,
            min_speech_chunks: cli.min_speech,
            rms_ema_alpha: cli.rms_ema,
        };
        let settings = Arc::new(SharedSettings::new(vals, cli.silence_flush));
        settings.state.store(STATE_READY, Ordering::SeqCst);

        let connection_count = Arc::new(AtomicUsize::new(0));

        let state = AppState {
            model,
            settings,
            connection_count: connection_count.clone(),
        };

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/asr", get(ws_handler))
            .with_state(Arc::new(state));

        let addr: std::net::SocketAddr = format!("{}:{}", cli.bind_addr, cli.port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;

        let use_tls = cli.tls_cert.is_some() && cli.tls_key.is_some();

        if use_tls {
            let cert_path = cli.tls_cert.as_ref().unwrap();
            let key_path = cli.tls_key.as_ref().unwrap();
            let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_path, key_path).await
                .map_err(|e| anyhow::anyhow!("TLS config error: {}", e))?;
            println!("=== SCHMIDIspeech server listening on wss://{}:{}/asr ===", cli.bind_addr, cli.port);
            axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await?;
        } else {
            println!("=== SCHMIDIspeech server listening on ws://{}:{}/asr (no TLS) ===", cli.bind_addr, cli.port);
            axum_server::bind(addr)
                .serve(app.into_make_service())
                .await?;
        }

        Ok(())
    }

    #[derive(Clone)]
    struct AppState {
        model: Arc<VoxtralModel>,
        settings: Arc<SharedSettings>,
        connection_count: Arc<AtomicUsize>,
    }

    async fn health_handler(State(state): State<Arc<AppState>>) -> axum::response::Json<serde_json::Value> {
        let connections = state.connection_count.load(Ordering::Relaxed);
        axum::response::Json(json!({ "status": "ready", "connections": connections }))
    }

    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<AppState>>,
    ) -> Response {
        ws.on_upgrade(move |socket| handle_socket(socket, state))
    }

    async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
        state.connection_count.fetch_add(1, Ordering::Relaxed);
        eprintln!("New connection (total: {})", state.connection_count.load(Ordering::Relaxed));

        if let Err(e) = handle_asr_session(&mut socket, &state).await {
            eprintln!("ASR session error: {}", e);
        }

        state.connection_count.fetch_sub(1, Ordering::Relaxed);
        eprintln!("Connection closed (total: {})", state.connection_count.load(Ordering::Relaxed));
    }

    async fn handle_asr_session(socket: &mut WebSocket, state: &AppState) -> Result<()> {
        use crate::audio;
        use crate::streaming::{ChunkOutput, StreamingState};
        use std::sync::atomic::Ordering;

        let model = &state.model;
        let settings = &state.settings;

        // Startup prefill: acquire model lock, run synchronously, release before any await.
        let mut stream_state = {
            let mut guard = model.inner.lock().await;
            let inner = &mut *guard; // plain &mut ModelInner — enables disjoint field borrows
            StreamingState::new_sync(
                &mut inner.enc,
                &model.adapter,
                &mut inner.dec,
                &model.filters,
                &model.device,
                model.dtype,
                settings,
            )?
            // guard dropped here — lock released before any network I/O
        };

        loop {
            match socket.recv().await {
                Some(Ok(Message::Binary(data))) => {
                    // Decode raw f32 LE PCM (Phase 1; Opus planned for Phase 3)
                    let pcm = audio::decode_pcm_f32(&data);
                    if pcm.is_empty() { continue; }

                    // Process: acquire GPU lock, do all sync work, release before send
                    let outputs = {
                        let mut guard = model.inner.lock().await;
                        let inner = &mut *guard; // plain &mut ModelInner — enables disjoint field borrows
                        stream_state.process_chunk_sync(
                            &pcm,
                            &mut inner.enc,
                            &model.adapter,
                            &mut inner.dec,
                            &model.tok,
                            &model.device,
                            settings,
                        )?
                        // guard dropped here
                    };

                    // Send results — no lock held
                    for output in outputs {
                        let msg = match output {
                            ChunkOutput::Token(ref text) => {
                                json!({ "type": "partial", "text": text }).to_string()
                            }
                            ChunkOutput::Silence => {
                                let final_text = stream_state.take_text_buf();
                                json!({ "type": "final", "text": final_text }).to_string()
                            }
                            ChunkOutput::Pad => continue,
                        };
                        if socket.send(Message::Text(msg.into())).await.is_err() {
                            return Ok(());
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | None => {
                    // Flush remaining buffer as final
                    let remaining = stream_state.take_text_buf();
                    if !remaining.is_empty() {
                        let msg = json!({ "type": "final", "text": remaining }).to_string();
                        let _ = socket.send(Message::Text(msg.into())).await;
                    }
                    break;
                }
                Some(Ok(_)) => {} // ping/pong/text — ignore
                Some(Err(e)) => {
                    eprintln!("WebSocket error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}
