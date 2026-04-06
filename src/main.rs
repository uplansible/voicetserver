mod adapter;
mod audio;
mod common;
mod config;
mod decoder;
mod encoder;
mod lora;
mod macros;
mod mel;
mod session;
mod settings;
mod streaming;
mod tokenizer;
mod words;

#[cfg(feature = "cuda")]
mod m1_attention;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use common::MEL_FRAMES_PER_TOKEN;
use config::{MergedConfig, ValueSource};

#[derive(Parser, Clone)]
#[command(name = "voicetserver", about = "SCHMIDIspeech — real-time German medical dictation server")]
pub struct Cli {
    /// WAV file for offline transcription (omit for server mode)
    pub wav_file: Option<String>,

    /// Directory containing model files (consolidated.safetensors, tekken.json, mel_filters.bin) [default: "."]
    #[arg(long)]
    pub model_dir: Option<String>,

    /// CUDA device index [default: 0]
    #[arg(long)]
    pub device: Option<usize>,

    /// Delay tokens (1–30; each = 80ms lookahead; higher = more accuracy) [default: 4]
    #[arg(long)]
    pub delay: Option<usize>,

    /// Silence RMS threshold for paragraph breaks [default: 0.006]
    #[arg(long)]
    pub silence_threshold: Option<f32>,

    /// Consecutive silent chunks before silence is detected (each chunk = 80ms) [default: 20]
    #[arg(long)]
    pub silence_flush: Option<usize>,

    /// Minimum speech chunks before silence detection activates (each chunk = 80ms) [default: 15]
    #[arg(long)]
    pub min_speech: Option<usize>,

    /// EMA smoothing factor for speech detection (0.0–1.0, lower = smoother) [default: 0.3]
    #[arg(long)]
    pub rms_ema: Option<f32>,

    // ---- Server flags ----

    /// WebSocket listen port [default: 8765]
    #[arg(long)]
    pub port: Option<u16>,

    /// Bind address (use 0.0.0.0 with Tailscale; 127.0.0.1 for local dev without TLS) [default: "127.0.0.1"]
    #[arg(long)]
    pub bind_addr: Option<String>,

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

/// Validate that a path exists; emit a source-tagged error if not.
fn check_path(path: &str, source: ValueSource, field: &str) -> Result<()> {
    if !Path::new(path).exists() {
        let tag = config::source_tag(source, field);
        anyhow::bail!("{}: path not found: {}", tag, path);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load config file and merge with CLI args (CLI overrides config overrides defaults).
    config::bootstrap_config_dir()?;
    let file_config = config::load_config_file()?;
    let merged = config::merge(&cli, &file_config);

    // LoRA loading is deferred until after model init below.

    // Validate delay
    if merged.delay < 1 || merged.delay > 30 {
        let tag = config::source_tag(
            if cli.delay.is_some() { ValueSource::CliArg } else { ValueSource::ConfigFile },
            "delay",
        );
        anyhow::bail!("{}: must be between 1 and 30 (got {})", tag, merged.delay);
    }

    // Validate model_dir
    check_path(&merged.model_dir.value, merged.model_dir.source, "model_dir")?;

    // Validate TLS paths if provided
    if let Some(ref cert) = merged.tls_cert.value {
        check_path(cert, merged.tls_cert.source, "tls_cert")?;
    }
    if let Some(ref key) = merged.tls_key.value {
        check_path(key, merged.tls_key.source, "tls_key")?;
    }

    // mmap safetensors + spawn readahead (gives 0.5–1s head start before CUDA init)
    println!("Loading safetensors...");
    let st_path = format!("{}/consolidated.safetensors", merged.model_dir.value);
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

    let device = Device::cuda_if_available(merged.device)?;
    let dtype = DType::BF16;

    let is_offline = cli.wav_file.is_some();
    let effective_delay = if is_offline { 20 } else { merged.delay };

    println!("\n{:<28} {}", "Parameter", "Value");
    println!("{:-<28} {:-<32}", "", "");
    println!("{:<28} {} ({}ms){}",
        "Delay tokens", effective_delay, effective_delay * 80,
        if is_offline { " (offline max accuracy)" } else { "" });
    println!("{:<28} {}", "Silence threshold", merged.silence_threshold);
    println!("{:<28} {} ({}ms)", "Silence detection", merged.silence_flush, merged.silence_flush * 80);
    println!("{:<28} {} ({}ms)", "Min speech to activate", merged.min_speech, merged.min_speech * 80);
    println!("{:<28} {}", "RMS EMA alpha", merged.rms_ema);
    println!("{:<28} {:?}", "Compute dtype", dtype);
    if !is_offline {
        println!("{:<28} {}:{}", "Listen", merged.bind_addr.value, merged.port);
        let tls = merged.tls_cert.value.is_some() && merged.tls_key.value.is_some();
        println!("{:<28} {}", "TLS", if tls { "enabled (wss://)" } else { "disabled (ws://)" });
    }
    println!("{:<28} {}", "Config file", config::config_file_path().display());
    println!("{:<28} {}", "Custom words", config::custom_words_path().display());
    println!();

    let vb = VarBuilder::from_slice_safetensors(&st_data, dtype, &device)?;

    let t_total = Instant::now();

    println!("Loading tokenizer...");
    let tok = tokenizer::Tokenizer::load(&merged.model_dir.value)?;

    println!("Loading encoder...");
    let enc = encoder::AudioEncoder::load(&vb, &device, dtype)?;

    println!("Loading adapter...");
    let adapter = adapter::Adapter::load(&vb)?;

    println!("Loading decoder...");
    let mut dec = decoder::TextDecoder::load(&vb, &device, dtype)?;

    println!("Total model load: {:.2}s", t_total.elapsed().as_secs_f64());
    let _ = readahead.join();

    let filters = mel::mel_filters(&merged.model_dir.value);

    // Apply LoRA adapter if configured (before server start, no contention yet)
    if let Some(ref adapter_dir) = merged.lora_adapter {
        println!("Applying LoRA adapter: {}", adapter_dir);
        match lora::load_decoder_lora(std::path::Path::new(adapter_dir), &device, dtype) {
            Ok(Some(dec_lora)) => {
                dec.set_lora(&dec_lora);
                println!("LoRA adapter applied.");
            }
            Ok(None) => eprintln!("Warning: --lora-adapter path does not exist: {}", adapter_dir),
            Err(e)   => eprintln!("Warning: LoRA adapter load failed: {}", e),
        }
    }

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
            let shared_config = Arc::new(tokio::sync::Mutex::new(file_config));
            server::run(model, merged, shared_config).await
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

// ---- Server module (inline) ----
mod server {
    use super::*;
    use crate::config::{save_config_file, SharedConfigFile};
    use crate::settings::{IniValues, SharedSettings, StartupSnapshot, STATE_READY};
    use crate::words::WordsCorrector;
    use axum::{
        extract::{Query, State, WebSocketUpgrade},
        http::StatusCode,
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use axum::extract::ws::{Message, WebSocket};
    use axum::body::Bytes;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower_http::cors::{Any, CorsLayer};

    // ---- Training status ----

    #[derive(Debug, Clone, serde::Serialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    enum TrainingStatusKind {
        Idle,
        Running,
        Done,
        Error,
    }

    struct TrainingStatus {
        status: TrainingStatusKind,
        log: Vec<String>,
    }

    impl TrainingStatus {
        fn new() -> Self { Self { status: TrainingStatusKind::Idle, log: vec![] } }
    }

    pub async fn run(
        model: Arc<VoxtralModel>,
        merged: MergedConfig,
        config_file: SharedConfigFile,
    ) -> Result<()> {
        let vals = IniValues {
            delay: merged.delay,
            silence_threshold: merged.silence_threshold,
            silence_chunks: Some(merged.silence_flush),
            paragraph_delay_offset: 4,
            min_speech_chunks: merged.min_speech,
            rms_ema_alpha: merged.rms_ema,
        };
        let settings = Arc::new(SharedSettings::new(vals, merged.silence_flush));
        settings.state.store(STATE_READY, Ordering::SeqCst);

        let tls_enabled = merged.tls_cert.value.is_some() && merged.tls_key.value.is_some();
        let snapshot = StartupSnapshot {
            model_dir:    merged.model_dir.value.clone(),
            device:       merged.device,
            port:         merged.port,
            bind_addr:    merged.bind_addr.value.clone(),
            tls_enabled,
            lora_adapter: merged.lora_adapter.clone(),
        };

        let connection_count = Arc::new(AtomicUsize::new(0));
        let words_path = crate::config::custom_words_path();
        let words = Arc::new(tokio::sync::RwLock::new(
            WordsCorrector::load(&words_path)
        ));

        let training_status = Arc::new(tokio::sync::Mutex::new(TrainingStatus::new()));

        let state = Arc::new(AppState {
            model,
            settings,
            connection_count,
            startup_snapshot: snapshot,
            config_file,
            words_path,
            words,
            training_status,
        });

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::PATCH,
                axum::http::Method::POST,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers(Any);

        let app = Router::new()
            .route("/health",  get(health_handler))
            .route("/asr",     get(ws_handler))
            .route("/config",  get(config_get_handler).patch(config_patch_handler))
            .route("/words",   get(words_get_handler).post(words_post_handler))
            // Training data collection (Phase 2)
            .route("/training/sentences", get(training_sentences_handler))
            .route("/training/pair",      axum::routing::post(training_pair_handler))
            .route("/training",           get(training_get_handler).delete(training_delete_handler))
            .route("/training/run",       axum::routing::post(training_run_handler))
            .route("/training/status",    get(training_status_handler))
            .layer(cors)
            .with_state(state);

        let addr: std::net::SocketAddr = format!("{}:{}", merged.bind_addr.value, merged.port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;

        if tls_enabled {
            let cert_path = merged.tls_cert.value.as_ref().unwrap();
            let key_path  = merged.tls_key.value.as_ref().unwrap();
            let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_path, key_path).await
                .map_err(|e| anyhow::anyhow!("TLS config error: {}", e))?;
            println!("=== SCHMIDIspeech server listening on wss://{}:{}/asr ===",
                merged.bind_addr.value, merged.port);
            axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await?;
        } else {
            println!("=== SCHMIDIspeech server listening on ws://{}:{}/asr (no TLS) ===",
                merged.bind_addr.value, merged.port);
            axum_server::bind(addr)
                .serve(app.into_make_service())
                .await?;
        }

        Ok(())
    }

    #[derive(Clone)]
    struct AppState {
        model:             Arc<VoxtralModel>,
        settings:          Arc<SharedSettings>,
        connection_count:  Arc<AtomicUsize>,
        startup_snapshot:  StartupSnapshot,
        config_file:       SharedConfigFile,
        words_path:        std::path::PathBuf,
        words:             Arc<tokio::sync::RwLock<WordsCorrector>>,
        training_status:   Arc<tokio::sync::Mutex<TrainingStatus>>,
    }

    // ---- /health ----

    async fn health_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
        let connections = state.connection_count.load(Ordering::Relaxed);
        Json(json!({ "status": "ready", "connections": connections }))
    }

    // ---- /asr WebSocket ----

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
                                let raw = stream_state.take_text_buf();
                                let final_text = state.words.read().await.apply(&raw);
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
                    let raw = stream_state.take_text_buf();
                    if !raw.is_empty() {
                        let corrected = state.words.read().await.apply(&raw);
                        let msg = json!({ "type": "final", "text": corrected }).to_string();
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

    // ---- GET /config ----

    async fn config_get_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
        let s = &state.settings;
        let snap = &state.startup_snapshot;
        Json(json!({
            // Runtime-adjustable — live values from atomics
            "delay":             s.delay_tokens.load(Ordering::Relaxed),
            "silence_threshold": s.silence_threshold.load(Ordering::Relaxed),
            "silence_flush":     s.silence_chunks.load(Ordering::Relaxed),
            "min_speech":        s.min_speech_chunks.load(Ordering::Relaxed),
            "rms_ema":           s.rms_ema_alpha.load(Ordering::Relaxed),
            // Startup-only — from snapshot; changing via PATCH writes config file but requires restart
            "model_dir":         snap.model_dir,
            "device":            snap.device,
            "port":              snap.port,
            "bind_addr":         snap.bind_addr,
            "tls_enabled":       snap.tls_enabled,
            "lora_adapter":      snap.lora_adapter,
            "_startup_only":     ["model_dir", "device", "port", "bind_addr", "tls_cert", "tls_key", "lora_adapter"],
            "_note":             "Changing startup_only fields writes to config file but requires server restart."
        }))
    }

    // ---- PATCH /config ----

    #[derive(serde::Deserialize)]
    struct ConfigPatch {
        // Runtime-adjustable
        delay:             Option<usize>,
        silence_threshold: Option<f32>,
        silence_flush:     Option<usize>,
        min_speech:        Option<usize>,
        rms_ema:           Option<f32>,
        // Startup-only (written to file, restart required)
        model_dir:    Option<String>,
        device:       Option<usize>,
        port:         Option<u16>,
        bind_addr:    Option<String>,
        tls_cert:     Option<String>,
        tls_key:      Option<String>,
        lora_adapter: Option<String>,
    }

    async fn config_patch_handler(
        State(state): State<Arc<AppState>>,
        Json(patch): Json<ConfigPatch>,
    ) -> Response {
        // Validate ranges for runtime fields
        if let Some(d) = patch.delay {
            if d < 1 || d > 30 {
                return (StatusCode::UNPROCESSABLE_ENTITY,
                    format!("delay must be between 1 and 30, got {}", d)).into_response();
            }
        }
        if let Some(r) = patch.rms_ema {
            if !(0.0..=1.0).contains(&r) {
                return (StatusCode::UNPROCESSABLE_ENTITY,
                    format!("rms_ema must be between 0.0 and 1.0, got {}", r)).into_response();
            }
        }

        // Apply runtime params to atomics immediately
        let s = &state.settings;
        if let Some(v) = patch.delay             { s.delay_tokens.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.silence_threshold { s.silence_threshold.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.silence_flush     { s.silence_chunks.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.min_speech        { s.min_speech_chunks.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.rms_ema           { s.rms_ema_alpha.store(v, Ordering::SeqCst); }

        // Update config file under lock and persist to disk
        {
            let mut cfg = state.config_file.lock().await;
            if patch.delay.is_some()             { cfg.delay             = patch.delay; }
            if patch.silence_threshold.is_some() { cfg.silence_threshold = patch.silence_threshold; }
            if patch.silence_flush.is_some()     { cfg.silence_flush     = patch.silence_flush; }
            if patch.min_speech.is_some()        { cfg.min_speech        = patch.min_speech; }
            if patch.rms_ema.is_some()           { cfg.rms_ema           = patch.rms_ema; }
            if patch.model_dir.is_some()         { cfg.model_dir         = patch.model_dir; }
            if patch.device.is_some()            { cfg.device            = patch.device; }
            if patch.port.is_some()              { cfg.port              = patch.port; }
            if patch.bind_addr.is_some()         { cfg.bind_addr         = patch.bind_addr; }
            if patch.tls_cert.is_some()          { cfg.tls_cert          = patch.tls_cert; }
            if patch.tls_key.is_some()           { cfg.tls_key           = patch.tls_key; }
            if patch.lora_adapter.is_some()      { cfg.lora_adapter      = patch.lora_adapter; }

            if let Err(e) = save_config_file(&cfg) {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to persist config: {}", e)).into_response();
            }
        }

        config_get_handler(State(state)).await.into_response()
    }

    // ---- GET /words ----

    async fn words_get_handler(State(state): State<Arc<AppState>>) -> Response {
        let corrector = state.words.read().await;
        Json(json!({ "words": corrector.raw_lines })).into_response()
    }

    // ---- POST /words ----

    #[derive(serde::Deserialize)]
    struct WordsPatch {
        add:    Option<Vec<String>>,
        remove: Option<Vec<String>>,
    }

    async fn words_post_handler(
        State(state): State<Arc<AppState>>,
        Json(body): Json<WordsPatch>,
    ) -> Response {
        // Read current words into a sorted, deduplicated set
        let content = tokio::fs::read_to_string(&state.words_path).await.unwrap_or_default();
        let mut words: std::collections::BTreeSet<String> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|s| s.to_string())
            .collect();

        for w in body.add.unwrap_or_default() { words.insert(w); }
        for w in body.remove.unwrap_or_default() { words.remove(&w); }

        let new_content = words.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n") + "\n";
        match tokio::fs::write(&state.words_path, new_content).await {
            Ok(_) => {
                // Rebuild the in-memory corrector from the updated file
                let new_corrector = WordsCorrector::load(&state.words_path);
                let raw_lines = new_corrector.raw_lines.clone();
                *state.words.write().await = new_corrector;
                Json(json!({ "words": raw_lines })).into_response()
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }

    // ---- Training data collection ----

    // Default German medical sentences for voice calibration (written to disk on first request)
    const DEFAULT_TRAINING_SENTENCES: &str = "\
Miktion, Defäkation und Miktionsbeschwerden
Blutdruck einhundertdreißig zu achtzig Millimeter Quecksilber
Herzfrequenz siebenundsechzig Schläge pro Minute
Der Patient klagt über Dysurie und Pollakisurie
Nierenfunktion mit einer Kreatinin von eins Komma zwei
Echokardiographie zeigt eine gute linksventrikuläre Funktion
Hämoglobin elf Komma vier Gramm pro Deziliter
Die Abdomensonographie ergab keine pathologischen Befunde
Bronchoskopie mit Lavage und Biopsieentnahme
Arterieller Blutgasanalyse im Normbereich
Elektrokardiogramm ohne Zeichen einer Ischämie
Spirometrie ergibt eine leichte obstruktive Ventilationsstörung
Computertomographie des Thorax mit Kontrastmittel
Kolonoskopie bis zum Zökum problemlos durchführbar
Magenspiegelung zeigt eine flache Erosion im Antrum
Liquorpunktion ergab einen klaren Liquor ohne Pleozytose
Schilddrüsenwerte TSH im Normbereich bei unauffälliger Sonographie
Orthopädische Untersuchung der Lendenwirbelsäule mit Bewegungseinschränkung
Neurologischer Status: Hirnnerven intakt, keine Paresen
Postoperativ stabile Vitalparameter, Patient kann mobilisiert werden\
";

    /// GET /training/sentences — returns list of calibration sentences
    async fn training_sentences_handler(State(_state): State<Arc<AppState>>) -> Response {
        let path = crate::config::training_sentences_path();
        // Create default file if not present
        if !path.exists() {
            if let Err(e) = tokio::fs::write(&path, DEFAULT_TRAINING_SENTENCES).await {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to create sentences file: {e}")).into_response();
            }
        }
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        let sentences: Vec<&str> = content
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .collect();
        Json(json!({ "sentences": sentences })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct TrainingPairQuery {
        text: String,
    }

    /// POST /training/pair?text=... — body is raw f32 LE PCM at 16kHz
    async fn training_pair_handler(
        State(_state): State<Arc<AppState>>,
        Query(params): Query<TrainingPairQuery>,
        body: Bytes,
    ) -> Response {
        let pcm = crate::audio::decode_audio_bytes(&body);
        if pcm.is_empty() {
            return (StatusCode::BAD_REQUEST, "Empty audio body").into_response();
        }

        let audio_dir = crate::config::training_audio_dir();
        if let Err(e) = tokio::fs::create_dir_all(&audio_dir).await {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create training dir: {e}")).into_response();
        }

        let pairs_path = crate::config::training_pairs_path();

        // Determine next ID from existing pair count
        let count = count_pairs(&pairs_path).await;
        let id = format!("{:04}", count + 1);
        let wav_path = audio_dir.join(format!("{id}.wav"));
        let duration_s = pcm.len() as f32 / 16000.0;

        // Save WAV (16kHz mono i16)
        if let Err(e) = save_training_wav(&wav_path, &pcm) {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("WAV write error: {e}")).into_response();
        }

        // Append JSONL entry
        let entry = format!("{{\"id\":\"{id}\",\"text\":{},\"duration_s\":{:.3}}}\n",
            serde_json::to_string(&params.text).unwrap_or_default(),
            duration_s);
        {
            use tokio::io::AsyncWriteExt;
            let mut f = match tokio::fs::OpenOptions::new()
                .create(true).append(true).open(&pairs_path).await
            {
                Ok(f) => f,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("JSONL open error: {e}")).into_response(),
            };
            if let Err(e) = f.write_all(entry.as_bytes()).await {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("JSONL write error: {e}")).into_response();
            }
        }

        Json(json!({ "id": id, "duration_s": duration_s, "count": count + 1 })).into_response()
    }

    /// GET /training — summary of collected pairs
    async fn training_get_handler(State(_state): State<Arc<AppState>>) -> Response {
        let pairs_path = crate::config::training_pairs_path();
        let content = tokio::fs::read_to_string(&pairs_path).await.unwrap_or_default();
        let mut count = 0usize;
        let mut duration_s = 0.0f64;
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                count += 1;
                duration_s += v["duration_s"].as_f64().unwrap_or(0.0);
            }
        }
        Json(json!({ "count": count, "duration_sec": duration_s })).into_response()
    }

    /// DELETE /training/pairs — remove all collected training data
    async fn training_delete_handler(State(_state): State<Arc<AppState>>) -> Response {
        let dir = crate::config::training_dir();
        if dir.exists() {
            if let Err(e) = tokio::fs::remove_dir_all(&dir).await {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to delete training data: {e}")).into_response();
            }
        }
        Json(json!({ "deleted": true })).into_response()
    }

    /// POST /training/run — spawn train_lora.py as a subprocess
    async fn training_run_handler(State(state): State<Arc<AppState>>) -> Response {
        {
            let ts = state.training_status.lock().await;
            if ts.status == TrainingStatusKind::Running {
                return (StatusCode::CONFLICT, "Training already running").into_response();
            }
        }

        // Locate train_lora.py: try cwd/tools/ then binary dir/tools/
        let script = find_script("tools/train_lora.py");
        let script = match script {
            Some(p) => p,
            None => return (StatusCode::UNPROCESSABLE_ENTITY,
                "tools/train_lora.py not found — run server from project root").into_response(),
        };

        let data_dir  = crate::config::training_dir();
        let model_dir = state.startup_snapshot.model_dir.clone();
        let output_dir = crate::config::lora_output_dir();

        // Ensure training dir exists before passing it to Python
        if !data_dir.exists() {
            return (StatusCode::UNPROCESSABLE_ENTITY,
                "No training data yet — collect pairs first").into_response();
        }

        {
            let mut ts = state.training_status.lock().await;
            *ts = TrainingStatus { status: TrainingStatusKind::Running, log: vec![] };
        }

        let training_status = Arc::clone(&state.training_status);
        tokio::spawn(async move {
            run_training_subprocess(training_status, script, data_dir, model_dir, output_dir).await;
        });

        (StatusCode::ACCEPTED, "Training started").into_response()
    }

    /// GET /training/status — current training status + last log lines
    async fn training_status_handler(State(state): State<Arc<AppState>>) -> Response {
        let ts = state.training_status.lock().await;
        Json(json!({
            "status": ts.status,
            "log": ts.log,
        })).into_response()
    }

    // ---- Training helpers ----

    async fn count_pairs(pairs_path: &std::path::Path) -> usize {
        let content = tokio::fs::read_to_string(pairs_path).await.unwrap_or_default();
        content.lines().filter(|l| !l.trim().is_empty()).count()
    }

    fn save_training_wav(path: &std::path::Path, pcm: &[f32]) -> anyhow::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &s in pcm {
            writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    }

    fn find_script(rel_path: &str) -> Option<std::path::PathBuf> {
        // Try working directory first
        let cwd_path = std::path::Path::new(rel_path);
        if cwd_path.exists() { return Some(cwd_path.to_path_buf()); }
        // Try directory of the current executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let p = dir.join(rel_path);
                if p.exists() { return Some(p); }
            }
        }
        None
    }

    async fn run_training_subprocess(
        training_status: Arc<tokio::sync::Mutex<TrainingStatus>>,
        script: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        model_dir: String,
        output_dir: std::path::PathBuf,
    ) {
        use tokio::io::AsyncBufReadExt;

        let append_log = |ts: &mut TrainingStatus, line: String| {
            if ts.log.len() >= 200 { ts.log.remove(0); }
            ts.log.push(line);
        };

        let mut child = match tokio::process::Command::new("python3")
            .args([
                script.to_str().unwrap_or("tools/train_lora.py"),
                "--data-dir",   data_dir.to_str().unwrap_or(""),
                "--model-dir",  &model_dir,
                "--output-dir", output_dir.to_str().unwrap_or(""),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let mut ts = training_status.lock().await;
                ts.status = TrainingStatusKind::Error;
                ts.log.push(format!("Failed to spawn python3: {e}"));
                return;
            }
        };

        let stdout = child.stdout.take().map(tokio::io::BufReader::new);
        let stderr = child.stderr.take().map(tokio::io::BufReader::new);

        let mut stdout_lines = stdout.map(|r| r.lines());
        let mut stderr_lines = stderr.map(|r| r.lines());

        loop {
            let next_out = async {
                if let Some(ref mut lines) = stdout_lines { lines.next_line().await }
                else { Ok(None) }
            };
            let next_err = async {
                if let Some(ref mut lines) = stderr_lines { lines.next_line().await }
                else { Ok(None) }
            };

            tokio::select! {
                line = next_out => {
                    match line {
                        Ok(Some(l)) => { let mut ts = training_status.lock().await; append_log(&mut ts, l); }
                        Ok(None)    => { stdout_lines = None; }
                        Err(_)      => { stdout_lines = None; }
                    }
                }
                line = next_err => {
                    match line {
                        Ok(Some(l)) => { let mut ts = training_status.lock().await; append_log(&mut ts, l); }
                        Ok(None)    => { stderr_lines = None; }
                        Err(_)      => { stderr_lines = None; }
                    }
                }
            }
            if stdout_lines.is_none() && stderr_lines.is_none() { break; }
        }

        let exit_status = child.wait().await;
        let mut ts = training_status.lock().await;
        match exit_status {
            Ok(s) if s.success() => {
                ts.status = TrainingStatusKind::Done;
                ts.log.push("Training complete.".to_string());
            }
            Ok(s) => {
                ts.status = TrainingStatusKind::Error;
                ts.log.push(format!("Training exited with status: {s}"));
            }
            Err(e) => {
                ts.status = TrainingStatusKind::Error;
                ts.log.push(format!("Wait error: {e}"));
            }
        }
    }
}
