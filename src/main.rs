mod adapter;
mod audio;
mod common;
mod config;
mod decoder;
mod encoder;
mod lora;
mod macros;
mod mel;
mod qwen;
mod qwen_streaming;
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
use nix::sys::signal::{signal, SigHandler, Signal};
use nix::unistd::{fork, ForkResult};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use common::MEL_FRAMES_PER_TOKEN;
use config::{MergedConfig, ValueSource};

/// `--version` output: package version plus build variant and the compiled-in engines
/// (e.g. "0.1.24 (CUDA 12.6, voxtral+qwen3)" or "0.1.24 (CPU, voxtral+qwen3)").
/// VOICETSERVER_BUILD_VARIANT is emitted by build.rs. With cudarc's dynamic-loading
/// the GPU driver is only opened on first inference, so this prints fine even on a
/// machine with no GPU/driver present.
const LONG_VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), " (", env!("VOICETSERVER_BUILD_VARIANT"), ", voxtral+qwen3)");

#[derive(Parser, Clone)]
#[command(name = "voicetserver", about = "SCHMIDIspeech — real-time German medical dictation server", version = LONG_VERSION)]
pub struct Cli {
    /// WAV file for offline transcription (omit for server mode)
    pub wav_file: Option<String>,

    /// Directory containing model files (consolidated.safetensors, tekken.json, mel_filters.bin) [default: "."]
    #[arg(long)]
    pub model_dir: Option<String>,

    /// Qwen3-ASR model directory (model.safetensors, config.json, tokenizer.json).
    /// Omit to disable the qwen engine.
    #[arg(long)]
    pub qwen_model_dir: Option<String>,

    /// Transcription language for the qwen engine (Voxtral auto-detects) [default: "German"]
    #[arg(long)]
    pub language: Option<String>,

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

    /// LoRA adapter directory
    #[arg(long)]
    pub lora_adapter: Option<String>,

    /// Python venv for LoRA training (e.g. /mnt/ssdupl/voicetserver-venv)
    #[arg(long)]
    pub venv_path: Option<String>,

    /// Base directory for custom_words.txt, training/, lora_adapter/, training_sentences.txt
    /// [default: ~/.config/voicetserver/]
    #[arg(long, value_name = "PATH")]
    pub data_dir: Option<String>,

    /// Daemonize: fork at startup, log to file, return shell prompt immediately
    #[arg(long)]
    pub detach: bool,

    /// Log file path [default: ~/.config/voicetserver/logs/voicetserver.log]
    #[arg(long, value_name = "PATH")]
    pub log_file: Option<String>,

    /// Days to retain rotated log files [default: 7]
    #[arg(long, value_name = "DAYS")]
    pub log_keep_days: Option<u32>,

    /// Stop a running daemon (sends SIGTERM to PID from the PID file)
    #[arg(long)]
    pub stop: bool,

}

/// Mutable model components — serialised by a single tokio::sync::Mutex.
/// All connections share this lock; GPU forward passes run serially.
pub struct ModelInner {
    pub enc: encoder::AudioEncoder,
    pub dec: decoder::TextDecoder,
}

/// All model components shared across WebSocket connections.
/// `inner` is locked per-chunk; `adapter`, `tok`, `filters`, `device`, `dtype` are read-only.
///
/// `inner` is an `Option` so the GPU-resident encoder+decoder (~8 GB) can be temporarily
/// unloaded to free VRAM for the LoRA training subprocess, then reloaded from `model_dir`
/// when training finishes. While unloaded, ASR sessions return a "training in progress" error.
pub struct VoxtralModel {
    pub inner: tokio::sync::Mutex<Option<ModelInner>>,
    pub adapter: adapter::Adapter,
    pub tok: tokenizer::Tokenizer,
    pub filters: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
    /// Model directory — used to rebuild enc+dec when reloading after training.
    pub model_dir: String,
    /// GERMAN_PRIME_TEXT encoded once at load; injected into the prefill when the
    /// german_prime setting is on (experimental language-prior biasing).
    pub prime_ids: Vec<u32>,
}

/// Rebuild the GPU-resident encoder + decoder from `consolidated.safetensors`, optionally
/// re-applying a LoRA adapter. Used to reload the model after it was unloaded for training.
fn load_enc_dec(
    model_dir: &str,
    device: &Device,
    dtype: DType,
    lora_path: Option<&Path>,
) -> Result<ModelInner> {
    let st_path = format!("{model_dir}/consolidated.safetensors");
    let st_data = unsafe { memmap2::Mmap::map(&fs::File::open(&st_path)?)? };
    let vb = VarBuilder::from_slice_safetensors(&st_data, dtype, device)?;
    let enc = encoder::AudioEncoder::load(&vb, device, dtype)?;
    let mut dec = decoder::TextDecoder::load(&vb, device, dtype)?;
    if let Some(p) = lora_path {
        match lora::load_decoder_lora(p, device, dtype) {
            Ok(Some(dec_lora)) => dec.set_lora(&dec_lora),
            Ok(None) => eprintln!("Warning: LoRA path no longer exists on reload: {}", p.display()),
            Err(e)   => eprintln!("Warning: LoRA reload failed: {e}"),
        }
    }
    Ok(ModelInner { enc, dec })
}

/// Return the CUDA stream-ordered memory pool to the OS after the model is dropped.
///
/// cudarc allocates with `cuMemAllocAsync` and frees with `cuMemFreeAsync`, so dropped
/// tensors return memory to the device's default memory pool — but the pool *retains* it
/// in-process, so a separate process (train_lora.py) still sees the GPU as full.
/// `cuMemPoolTrimTo(pool, 0)` forces the pool to release all unused memory to the OS.
/// Caller must `device.synchronize()` first so the async frees have completed.
#[cfg(feature = "cuda")]
fn release_cuda_pool(device: &Device) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let stream    = device.as_cuda_device()?.cuda_stream();
    let cu_device = stream.context().cu_device();
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        sys::cuDeviceGetDefaultMemPool(&mut pool, cu_device)
            .result().map_err(|e| anyhow::anyhow!("cuDeviceGetDefaultMemPool: {e:?}"))?;
        sys::cuMemPoolTrimTo(pool, 0)
            .result().map_err(|e| anyhow::anyhow!("cuMemPoolTrimTo: {e:?}"))?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn release_cuda_pool(_device: &Device) -> Result<()> { Ok(()) }

/// Validate that a path exists; emit a source-tagged error if not.
fn check_path(path: &str, source: ValueSource, field: &str) -> Result<()> {
    if !Path::new(path).exists() {
        let tag = config::source_tag(source, field);
        anyhow::bail!("{}: path not found: {}", tag, path);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    use std::io::IsTerminal;

    let cli = Cli::parse();

    // Handle --stop before any config loading or forking
    if cli.stop {
        return stop_server();
    }

    // Load config file and merge with CLI args (CLI overrides config overrides defaults).
    config::bootstrap_config_dir()?;
    let mut file_config = config::load_config_file()?;
    let merged = config::merge(&cli, &file_config);

    // Ensure data_dir exists (may differ from config_dir when --data-dir / data_dir is set).
    if let Err(e) = std::fs::create_dir_all(&merged.data_dir) {
        anyhow::bail!("Cannot create data_dir {}: {}", merged.data_dir.display(), e);
    }

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

    // Validate qwen_model_dir if set (the qwen engine is optional)
    if let Some(ref dir) = merged.qwen_model_dir.value {
        check_path(dir, merged.qwen_model_dir.source, "qwen_model_dir")?;
    }

    // Validate TLS paths if provided
    if let Some(ref cert) = merged.tls_cert.value {
        check_path(cert, merged.tls_cert.source, "tls_cert")?;
    }
    if let Some(ref key) = merged.tls_key.value {
        check_path(key, merged.tls_key.source, "tls_key")?;
    }

    // Guard against starting a second instance.
    let is_server_mode = cli.wav_file.is_none();

    // Ensure an API key exists (server mode only). Generated + persisted on first start.
    let api_key = if is_server_mode {
        config::ensure_api_key(&mut file_config)?
    } else {
        String::new()
    };

    if is_server_mode {
        if let Some(pid) = read_pid_file() {
            // Verify the cmdline too — after a crash the stale PID may have been
            // recycled by an unrelated process, which must not block startup.
            if is_process_running(pid) && is_our_process(pid) {
                anyhow::bail!(
                    "Server already running (PID {}). Use --stop to stop it.", pid
                );
            }
            // Stale PID file from a previous crash — remove it
            remove_pid_file();
        }
    }

    // Resolve log path before forking (server mode only).
    let log_path = if is_server_mode {
        let p = resolve_log_path(&cli, &merged);
        ensure_log_dir(&p);
        Some(p)
    } else {
        None
    };

    // Fork / daemonize BEFORE tokio starts — forking a multi-threaded process is unsafe.
    let mut watchdog_active = false;
    if is_server_mode {
        if cli.detach {
            daemonize(log_path.as_ref().unwrap());
            // Only child returns from daemonize()
        } else if std::io::stdin().is_terminal() {
            // Ignore SIGINT before fork so the child inherits the ignore.
            // The watchdog parent handles Ctrl+C via raw byte 0x03 instead.
            unsafe { signal(Signal::SIGINT, SigHandler::SigIgn) }.ok();

            if let Some(child_pid) = fork_watchdog() {
                // Parent: watch for 'd' / Ctrl+C keypresses, then exit
                watchdog_loop(child_pid, log_path.as_ref().unwrap());
                std::process::exit(0);
            }
            // Child: redirect stdin to /dev/null (server doesn't need it)
            redirect_stdin_to_null();
            watchdog_active = true;
        }
        // Write PID file immediately after fork so --stop works during model loading
        write_pid_file();
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
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

    let is_offline = cli.wav_file.is_some();
    let effective_delay = if is_offline { 20 } else { merged.delay };

    println!("\n{:<28} {}", "Parameter", "Value");
    println!("{:-<28} {:-<32}", "", "");
    println!("{:<28} {}", "Version", env!("CARGO_PKG_VERSION"));
    println!("{:<28} {} ({}ms){}",
        "Delay tokens", effective_delay, effective_delay * 80,
        if is_offline { " (offline max accuracy)" } else { "" });
    println!("{:<28} {}", "Silence threshold", merged.silence_threshold);
    println!("{:<28} {} ({}ms)", "Silence detection", merged.silence_flush, merged.silence_flush * 80);
    println!("{:<28} {} ({}ms)", "Min speech to activate", merged.min_speech, merged.min_speech * 80);
    println!("{:<28} {}", "RMS EMA alpha", merged.rms_ema);
    println!("{:<28} {}{}", "Fuzzy hotwords",
        if merged.fuzzy_hotwords { "on" } else { "off" },
        if merged.fuzzy_hotwords { format!(" (max ratio {})", merged.fuzzy_max_ratio) } else { String::new() });
    println!("{:<28} {}", "German prime (experimental)",
        if merged.german_prime { "on" } else { "off" });
    println!("{:<28} {}", "Qwen3 engine",
        merged.qwen_model_dir.value.as_deref().unwrap_or("disabled (qwen_model_dir not set)"));
    if merged.qwen_model_dir.value.is_some() {
        println!("{:<28} {}", "Qwen3 language", merged.language);
        println!("{:<28} {}", "Qwen3 context biasing",
            if merged.context_biasing { "on" } else { "off" });
    }
    println!("{:<28} {:?}", "Compute dtype", dtype);
    if !is_offline {
        println!("{:<28} {}:{}", "Listen", merged.bind_addr.value, merged.port);
        let tls = merged.tls_cert.value.is_some() && merged.tls_key.value.is_some();
        println!("{:<28} {}", "TLS", if tls { "enabled (wss://)" } else { "disabled (ws://)" });
        if let Some(ref lp) = log_path {
            println!("{:<28} {}", "Log file", lp.display());
        }
    }
    println!("{:<28} {}", "Config file", config::config_file_path().display());
    println!("{:<28} {}", "Data dir", merged.data_dir.display());
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
            // Offline mode respects the german_prime config flag too, so A/B runs on
            // the same WAV are possible without a server.
            let prime_ids = if merged.german_prime {
                tok.encode_greedy(streaming::GERMAN_PRIME_TEXT)
            } else {
                Vec::new()
            };
            run_offline(path, effective_delay, &prime_ids, enc, &adapter, dec, &tok, &filters, &device, dtype)
        }
        None => {
            // Second engine (Qwen3-ASR) — server mode only; offline WAV mode stays
            // Voxtral-only until phase 3 adds ?model= routing. Shares the candle
            // Device with Voxtral (one CUDA context, separate GPU lock per engine).
            let qwen_engine = match merged.qwen_model_dir.value {
                Some(ref dir) => {
                    println!("Loading Qwen3-ASR engine: {}", dir);
                    let t_qwen = Instant::now();
                    let engine = qwen::QwenEngine::load(dir, device.clone())?;
                    println!("Qwen3-ASR loaded in {:.2}s", t_qwen.elapsed().as_secs_f64());
                    // Apply the qwen LoRA adapter if configured (pre-tokio, so the
                    // blocking lock inside is safe). Warn-and-continue like Voxtral.
                    if let Some(ref adapter_dir) = merged.lora_adapter_qwen {
                        println!("Applying Qwen LoRA adapter: {}", adapter_dir);
                        match engine.apply_lora_blocking(Path::new(adapter_dir)) {
                            Ok(())  => println!("Qwen LoRA adapter applied."),
                            Err(e)  => eprintln!("Warning: {}", e),
                        }
                    }
                    Some(Arc::new(engine))
                }
                None => None,
            };
            let prime_ids = tok.encode_greedy(streaming::GERMAN_PRIME_TEXT);
            let model = Arc::new(VoxtralModel {
                inner: tokio::sync::Mutex::new(Some(ModelInner { enc, dec })),
                adapter,
                tok,
                filters,
                device,
                dtype,
                model_dir: merged.model_dir.value.clone(),
                prime_ids,
            });
            let shared_config = Arc::new(tokio::sync::Mutex::new(file_config));
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(server::run(
                    model,
                    qwen_engine,
                    merged,
                    shared_config,
                    log_path.unwrap_or_default(),
                    watchdog_active,
                    api_key,
                ))
        }
    }
}

// ---- Offline WAV transcription ----

fn run_offline(
    wav_path: &str,
    delay_tokens: usize,
    prime_tokens: &[u32],
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

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, prime_tokens, device, dtype)?;

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
        // Downmix to mono by averaging all channels (not just taking channel 0).
        let ch = spec.channels as usize;
        samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    if samples.is_empty() {
        anyhow::bail!("WAV file contains no decodable samples: {}", path);
    }

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

// ---- PID file helpers ----

fn write_pid_file() {
    std::fs::write(config::pid_file_path(), std::process::id().to_string()).ok();
}

fn read_pid_file() -> Option<u32> {
    std::fs::read_to_string(config::pid_file_path())
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

pub fn remove_pid_file() {
    std::fs::remove_file(config::pid_file_path()).ok();
}

/// Check if a process with this PID is running (Linux: check /proc/<pid>).
fn is_process_running(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

/// Verify the running process is actually voicetserver (guards against stale PID reuse
/// by an unrelated process that happened to be assigned the same PID after a crash).
fn is_our_process(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{}/cmdline", pid))
        .map(|s| s.contains("voicetserver"))
        .unwrap_or(false)
}

/// Send SIGTERM to the running daemon and wait for it to exit.
fn stop_server() -> Result<()> {
    let pid_path = config::pid_file_path();
    let pid_str = std::fs::read_to_string(&pid_path)
        .map_err(|_| anyhow::anyhow!(
            "No PID file found ({}). Is the server running?", pid_path.display()
        ))?;
    let pid: i32 = pid_str.trim().parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID file: {}", pid_path.display()))?;

    if !is_our_process(pid as u32) {
        anyhow::bail!(
            "PID {} does not appear to be voicetserver (process reuse?). Refusing to send SIGTERM.",
            pid
        );
    }

    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid),
        nix::sys::signal::Signal::SIGTERM,
    ).map_err(|e| anyhow::anyhow!("Failed to stop server (PID {}): {}", pid, e))?;

    // Wait up to 5 s for the process to exit
    let mut exited = false;
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if !is_process_running(pid as u32) {
            exited = true;
            break;
        }
    }

    // Only report success (and delete the PID file) if the process actually exited —
    // otherwise a stale "stopped" claim would allow a duplicate instance to start.
    if !exited {
        anyhow::bail!(
            "Server (PID {}) did not exit within 5 s after SIGTERM. PID file kept.", pid
        );
    }

    remove_pid_file();
    println!("Server stopped (PID {}).", pid);
    Ok(())
}

// ---- Detach / watchdog / log-file helpers ----

fn resolve_log_path(cli: &Cli, cfg: &config::MergedConfig) -> std::path::PathBuf {
    cli.log_file
        .as_ref()
        .or(cfg.log_file.as_ref())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| config::config_dir().join("logs").join("voicetserver.log"))
}

fn ensure_log_dir(log_path: &std::path::Path) {
    if let Some(dir) = log_path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
}

/// Fork + setsid + redirect stdin/stdout/stderr to log file. Only the child returns.
fn daemonize(log_path: &std::path::Path) {
    use nix::unistd::setsid;
    match unsafe { fork().expect("daemonize: fork failed") } {
        ForkResult::Parent { child } => {
            println!("Detached. PID {}. Log: {}", child, log_path.display());
            std::process::exit(0);
        }
        ForkResult::Child => {
            setsid().expect("daemonize: setsid failed");
            redirect_stdin_to_null();
            redirect_output_to_log(log_path);
        }
    }
}

/// Fork into a watchdog parent and a server child.
/// Returns Some(child_pid) in the parent, None in the child.
fn fork_watchdog() -> Option<nix::unistd::Pid> {
    match unsafe { fork().expect("fork_watchdog: fork failed") } {
        ForkResult::Parent { child } => Some(child),
        ForkResult::Child => None,
    }
}

/// Watchdog parent loop: child writes its logs directly to the terminal
/// (inherited fds). Parent watches for 'd' or Ctrl+C and signals child accordingly.
fn watchdog_loop(child_pid: nix::unistd::Pid, log_path: &std::path::Path) {
    use nix::sys::termios::{self, SetArg};
    use nix::sys::wait::{WaitPidFlag, WaitStatus};
    use std::os::unix::io::BorrowedFd;

    // Set raw-ish terminal mode: character-by-character input, no echo, no signals.
    // Intentionally keep OPOST set so the child's println! still translates \n → \r\n.
    // cfmakeraw clears OPOST which corrupts the child's output since parent and child
    // share the same tty fd.
    let original_termios = nix::sys::termios::tcgetattr(
        unsafe { BorrowedFd::borrow_raw(0) }
    ).ok();
    if let Some(ref t) = original_termios {
        use nix::sys::termios::{InputFlags, LocalFlags, SpecialCharacterIndices as SCI};
        let mut raw = t.clone();
        raw.input_flags &= !(InputFlags::IGNBRK | InputFlags::BRKINT | InputFlags::PARMRK
            | InputFlags::ISTRIP | InputFlags::INLCR | InputFlags::IGNCR
            | InputFlags::ICRNL  | InputFlags::IXON);
        // OPOST deliberately NOT cleared — keeps \n→\r\n for child's stdout
        raw.local_flags &= !(LocalFlags::ECHO | LocalFlags::ECHONL | LocalFlags::ICANON
            | LocalFlags::ISIG | LocalFlags::IEXTEN);
        raw.control_chars[SCI::VMIN as usize]  = 1;
        raw.control_chars[SCI::VTIME as usize] = 0;
        termios::tcsetattr(unsafe { BorrowedFd::borrow_raw(0) }, SetArg::TCSANOW, &raw).ok();
    }

    // Spawn a thread that reads single bytes from stdin and sends them to the main loop.
    let (tx, rx) = std::sync::mpsc::channel::<u8>();
    std::thread::spawn(move || {
        use std::io::Read;
        use std::mem::ManuallyDrop;
        use std::os::unix::io::FromRawFd;
        // ManuallyDrop prevents closing fd 0 when the File is dropped.
        let mut stdin = ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(0) });
        let mut buf = [0u8; 1];
        loop {
            match stdin.read(&mut buf) {
                Ok(1) => { if tx.send(buf[0]).is_err() { break; } }
                _ => break,
            }
        }
    });

    let restore = |orig: &Option<nix::sys::termios::Termios>| {
        if let Some(t) = orig {
            termios::tcsetattr(
                unsafe { BorrowedFd::borrow_raw(0) },
                SetArg::TCSANOW,
                t,
            ).ok();
        }
    };

    loop {
        // Check if child exited on its own.
        match nix::sys::wait::waitpid(child_pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::Exited(..)) | Ok(WaitStatus::Signaled(..)) => {
                restore(&original_termios);
                return;
            }
            _ => {}
        }

        // Check for keypress.
        match rx.try_recv() {
            Ok(b'd') | Ok(b'D') => {
                // Signal child to redirect its output to the log file.
                nix::sys::signal::kill(child_pid, Signal::SIGUSR1).ok();
                std::thread::sleep(std::time::Duration::from_millis(250));
                restore(&original_termios);
                println!("\nDetached. PID {}. Log: {}", child_pid, log_path.display());
                return;
            }
            Ok(3) => {
                // Ctrl+C in raw mode (byte 0x03 — terminal does not generate SIGINT in raw mode).
                restore(&original_termios);
                eprintln!("\nStopping server...");
                nix::sys::signal::kill(child_pid, Signal::SIGTERM).ok();
                nix::sys::wait::waitpid(child_pid, None).ok();
                return;
            }
            _ => {}
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// Redirect stdout and stderr to the log file (append mode).
pub fn redirect_output_to_log(log_path: &std::path::Path) {
    use std::fs::OpenOptions;
    use std::os::unix::io::IntoRawFd;

    let log_fd = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .expect("redirect_output_to_log: cannot open log file")
        .into_raw_fd();

    nix::unistd::dup2(log_fd, 1).expect("dup2 stdout failed");
    nix::unistd::dup2(log_fd, 2).expect("dup2 stderr failed");
    nix::unistd::close(log_fd).ok();
}

/// Redirect stdin to /dev/null (server does not use stdin).
fn redirect_stdin_to_null() {
    use std::os::unix::io::IntoRawFd;
    if let Ok(null) = std::fs::File::open("/dev/null") {
        let fd = null.into_raw_fd();
        nix::unistd::dup2(fd, 0).ok();
        nix::unistd::close(fd).ok();
    }
}

/// Rotate the log file if it exceeds 20 MB; prune rotated files older than keep_days.
pub fn rotate_log_if_needed(log_path: &std::path::PathBuf, keep_days: u32) {
    const MAX_SIZE: u64 = 20 * 1024 * 1024;

    let meta = match std::fs::metadata(log_path) {
        Ok(m) => m,
        Err(_) => return,
    };
    if meta.len() <= MAX_SIZE {
        return;
    }

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let log_dir = log_path.parent().unwrap_or(std::path::Path::new("."));
    let fname = log_path
        .file_name()
        .map_or("voicetserver.log".into(), |n| n.to_string_lossy().into_owned());
    let rotated = log_dir.join(format!("{}.{}", fname, ts));

    // Rotate by rename + reopen. Rename is atomic, so no lines are lost (the
    // earlier copy+truncate scheme dropped anything written between the two
    // steps). Our own stdout/stderr fds follow the renamed inode, so when they
    // were dup2'd to this log file (detached / after 'd'), reopen the path and
    // dup2 the fresh fd over 1/2 — rotation runs in-process, no SIGHUP needed.
    // In interactive mode stdout is still the terminal and must not be touched.
    let output_redirected = stdout_is_file(&meta);
    if std::fs::rename(log_path, &rotated).is_ok() {
        if output_redirected {
            redirect_output_to_log(log_path);
        }
        prune_old_logs(log_dir, &fname, keep_days);
    }
}

/// Does fd 1 currently point at the same inode as `target` (the live log file)?
fn stdout_is_file(target: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    match nix::sys::stat::fstat(1) {
        Ok(st) => st.st_dev as u64 == target.dev() && st.st_ino as u64 == target.ino(),
        Err(_) => false,
    }
}

/// Delete rotated log files older than keep_days days.
/// Uses the Unix timestamp embedded in the filename ("<log_fname>.<ts>") rather
/// than filesystem mtime, which is unreliable after copies/restores.
fn prune_old_logs(log_dir: &std::path::Path, log_fname: &str, keep_days: u32) {
    let cutoff_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_sub(keep_days as u64 * 86400);

    let entries = match std::fs::read_dir(log_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let prefix = format!("{}.", log_fname);
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Only prune files that look like rotated logs: "<log_fname>.<timestamp>"
        if let Some(ts_str) = name.strip_prefix(prefix.as_str()) {
            if let Ok(ts) = ts_str.parse::<u64>() {
                if ts < cutoff_secs {
                    std::fs::remove_file(&path).ok();
                }
            }
        }
    }
}

// ---- Server module (inline) ----
mod server {
    use super::*;
    use crate::config::{save_config_file, SharedConfigFile};
    use crate::settings::{IniValues, SharedSettings, StartupSnapshot, STATE_READY};
    use crate::words::{AbbrevExpander, FuzzyMatcher, WordsCorrector};
    use axum::{
        extract::{DefaultBodyLimit, Query, State, WebSocketUpgrade},
        http::StatusCode,
        middleware,
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

    // ---- Auth middleware ----

    /// Constant-time string comparison (no early return on first mismatch) to avoid
    /// leaking the API key length/content via timing.
    fn ct_eq(a: &str, b: &str) -> bool {
        if a.len() != b.len() { return false; }
        a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }

    /// Checks the X-Api-Key header (HTTP) or the ?api_key= query param (WebSocket —
    /// browsers cannot send custom headers during the WS upgrade). /health is exempt.
    async fn api_key_auth(
        State(state): State<Arc<AppState>>,
        request: axum::extract::Request,
        next: middleware::Next,
    ) -> Response {
        let header_key = request.headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let query_key = request.uri().query()
            .and_then(|q| {
                q.split('&').find_map(|pair| {
                    let mut parts = pair.splitn(2, '=');
                    let k = parts.next()?;
                    let v = parts.next()?;
                    if k == "api_key" { Some(v) } else { None }
                })
            })
            .unwrap_or("");

        if !ct_eq(header_key, &state.api_key) && !ct_eq(query_key, &state.api_key) {
            return (StatusCode::UNAUTHORIZED, "Invalid or missing API key").into_response();
        }
        next.run(request).await
    }

    pub async fn run(
        model: Arc<VoxtralModel>,
        qwen: Option<Arc<crate::qwen::QwenEngine>>,
        merged: MergedConfig,
        config_file: SharedConfigFile,
        log_path: std::path::PathBuf,
        watchdog_active: bool,
        api_key: String,
    ) -> Result<()> {
        let vals = IniValues {
            delay: merged.delay,
            silence_threshold: merged.silence_threshold,
            silence_chunks: Some(merged.silence_flush),
            paragraph_delay_offset: 4,
            min_speech_chunks: merged.min_speech,
            rms_ema_alpha: merged.rms_ema,
            fuzzy_hotwords: merged.fuzzy_hotwords,
            fuzzy_max_ratio: merged.fuzzy_max_ratio,
            german_prime: merged.german_prime,
            context_biasing: merged.context_biasing,
        };
        let settings = Arc::new(SharedSettings::new(vals, merged.silence_flush));
        settings.state.store(STATE_READY, Ordering::SeqCst);

        let tls_enabled = merged.tls_cert.value.is_some() && merged.tls_key.value.is_some();
        let paths = crate::config::WorkspacePaths::new(&merged.data_dir);
        let snapshot = StartupSnapshot {
            model_dir:    merged.model_dir.value.clone(),
            qwen_model_dir: merged.qwen_model_dir.value.clone(),
            language:     merged.language.clone(),
            device:       merged.device,
            port:         merged.port,
            bind_addr:    merged.bind_addr.value.clone(),
            tls_enabled,
            lora_adapter: merged.lora_adapter.clone(),
            lora_adapter_qwen: merged.lora_adapter_qwen.clone(),
            venv_path:    merged.venv_path.clone(),
            data_dir:     merged.data_dir.to_string_lossy().into_owned(),
        };

        // SIGUSR1: redirect stdout/stderr to log file (triggered by watchdog parent on 'd' press).
        if watchdog_active {
            if let Ok(mut stream) = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::user_defined1()
            ) {
                let lp = log_path.clone();
                tokio::spawn(async move {
                    stream.recv().await;
                    crate::redirect_output_to_log(&lp);
                });
            }
        }
        // SIGTERM: graceful exit (from watchdog Ctrl+C handler, --stop, or systemd stop).
        if let Ok(mut stream) = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate()
        ) {
            tokio::spawn(async move {
                stream.recv().await;
                crate::remove_pid_file();
                std::process::exit(0);
            });
        }
        // Log rotation: check every 5 minutes, rotate at 20 MB, prune files older than log_keep_days.
        if !log_path.as_os_str().is_empty() {
            let lp = log_path.clone();
            let keep_days = merged.log_keep_days;
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
                interval.tick().await; // skip the immediate first tick
                loop {
                    interval.tick().await;
                    crate::rotate_log_if_needed(&lp, keep_days);
                }
            });
        }

        let connection_count = Arc::new(AtomicUsize::new(0));
        let words_path = paths.custom_words.clone();
        let initial_corrector = WordsCorrector::load(&words_path);
        // Fuzzy matcher targets = the plain (non-`=`) terms in custom_words.txt.
        let fuzzy = Arc::new(tokio::sync::RwLock::new(
            FuzzyMatcher::from_corrector(&initial_corrector)
        ));
        // Acronym targets = the uppercase 2–6-letter plain terms (MRI, TUR-B, …).
        let abbrev = Arc::new(tokio::sync::RwLock::new(
            AbbrevExpander::from_corrector(&initial_corrector)
        ));
        let words = Arc::new(tokio::sync::RwLock::new(initial_corrector));

        let training_status = Arc::new(tokio::sync::Mutex::new(TrainingStatus::new()));

        let lora_path = Arc::new(tokio::sync::RwLock::new(
            snapshot.lora_adapter.as_ref().map(std::path::PathBuf::from)
        ));
        // Only meaningful when the qwen engine is enabled; a configured
        // lora_adapter_qwen without an engine must not report lora_active_qwen.
        let qwen_lora_path = Arc::new(tokio::sync::RwLock::new(
            if qwen.is_some() {
                snapshot.lora_adapter_qwen.as_ref().map(std::path::PathBuf::from)
            } else {
                None
            }
        ));

        let state = Arc::new(AppState {
            model,
            qwen,
            settings,
            connection_count,
            startup_snapshot: snapshot,
            config_file,
            words_path,
            words,
            fuzzy,
            abbrev,
            training_status,
            paths,
            lora_path,
            qwen_lora_path,
            api_key,
            pair_write_lock: Arc::new(tokio::sync::Mutex::new(())),
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

        // Protected routes require a valid X-Api-Key header (or ?api_key= for WebSocket).
        let protected = Router::new()
            .route("/asr",     get(ws_handler))
            .route("/config",  get(config_get_handler).patch(config_patch_handler))
            .route("/words",   get(words_get_handler).post(words_post_handler))
            // Training data collection (Phase 2)
            .route("/training/sentences",   get(training_sentences_handler))
            .route("/training/sentence",    axum::routing::post(training_sentence_add_handler)
                                            .patch(training_sentence_edit_handler)
                                            .delete(training_sentence_delete_handler))
            .route("/training/pairs",       get(training_pairs_handler))
            .route("/training/pair",        axum::routing::post(training_pair_handler))
            .route("/training/pair/{id}",   axum::routing::delete(training_pair_delete_handler))
            .route("/training/audio/{id}",  get(training_audio_handler))
            .route("/training",             get(training_get_handler).delete(training_delete_handler))
            .route("/training/run",         axum::routing::post(training_run_handler))
            .route("/training/status",      get(training_status_handler))
            // Dictation review (real dictations as training-pair candidates)
            .route("/training/reviews",             get(review_list_handler))
            .route("/training/review",              axum::routing::post(review_add_handler))
            .route("/training/review/{id}",         axum::routing::delete(review_delete_handler))
            .route("/training/review/{id}/accept",  axum::routing::post(review_accept_handler))
            .route("/training/review/audio/{id}",   get(review_audio_handler))
            .route("/lora/reload",          axum::routing::post(lora_reload_handler))
            .route("/lora",                 axum::routing::delete(lora_clear_handler))
            .route("/log/edit",             axum::routing::post(log_edit_handler))
            .route("/edits/report",         get(edit_report_handler))
            .route_layer(middleware::from_fn_with_state(Arc::clone(&state), api_key_auth));

        let app = Router::new()
            .route("/health", get(health_handler))   // public — no auth required
            .merge(protected)
            // 64 MB cap: review uploads are whole dictations as f32 PCM
            // (64 KB/s → ~16 min head-room; the old 20 MB capped at ~5 min).
            .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
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
            if watchdog_active {
                println!("Press 'd' to detach (log -> {}), Ctrl+C to stop",
                    log_path.display());
            }
            axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await?;
        } else {
            println!("=== SCHMIDIspeech server listening on ws://{}:{}/asr (no TLS) ===",
                merged.bind_addr.value, merged.port);
            if watchdog_active {
                println!("Press 'd' to detach (log -> {}), Ctrl+C to stop",
                    log_path.display());
            }
            axum_server::bind(addr)
                .serve(app.into_make_service())
                .await?;
        }

        Ok(())
    }

    #[derive(Clone)]
    struct AppState {
        model:             Arc<VoxtralModel>,
        /// Second engine (Qwen3-ASR); None when qwen_model_dir is not configured —
        /// sessions requesting `?model=qwen` then get an error frame.
        qwen:              Option<Arc<crate::qwen::QwenEngine>>,
        settings:          Arc<SharedSettings>,
        connection_count:  Arc<AtomicUsize>,
        startup_snapshot:  StartupSnapshot,
        config_file:       SharedConfigFile,
        words_path:        std::path::PathBuf,
        words:             Arc<tokio::sync::RwLock<WordsCorrector>>,
        /// Fuzzy phonetic matcher, rebuilt from `words` whenever custom_words.txt changes.
        fuzzy:             Arc<tokio::sync::RwLock<FuzzyMatcher>>,
        /// Acronym letter-name expander (Em Er I → MRI), rebuilt alongside `fuzzy`.
        abbrev:            Arc<tokio::sync::RwLock<AbbrevExpander>>,
        training_status:   Arc<tokio::sync::Mutex<TrainingStatus>>,
        paths:             crate::config::WorkspacePaths,
        /// Currently active Voxtral LoRA adapter path (None if no adapter loaded).
        /// Updated by POST /lora/reload.
        lora_path:         Arc<tokio::sync::RwLock<Option<std::path::PathBuf>>>,
        /// Currently active Qwen LoRA adapter path (adapters are per-model —
        /// weight-key formats differ). Updated by POST /lora/reload?model=qwen.
        qwen_lora_path:    Arc<tokio::sync::RwLock<Option<std::path::PathBuf>>>,
        /// API key required on every endpoint except GET /health.
        api_key:           String,
        /// Serialises training-file writes: pair upload + delete (so concurrent
        /// requests cannot collide on the same ID or rewrite pairs.jsonl mid-append)
        /// and sentence add/edit/delete (read-modify-write on training_sentences.txt).
        pair_write_lock:   Arc<tokio::sync::Mutex<()>>,
    }

    // ---- /health ----

    async fn health_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
        let connections = state.connection_count.load(Ordering::Relaxed);
        Json(json!({ "status": "ready", "connections": connections }))
    }

    // ---- /asr WebSocket ----

    #[derive(serde::Deserialize, Default)]
    struct WsQuery {
        /// Engine selection: "qwen" → Qwen3-ASR, anything else / absent → Voxtral.
        model:    Option<String>,
        /// Language override for the qwen engine (Voxtral auto-detects; ignored there).
        lang:     Option<String>,
        /// Patient context (qwen prompt biasing only).
        patient:  Option<String>,
        /// Per-session hotwords (comma/semicolon separated) to bias qwen decoding,
        /// in addition to the fixed plain terms from custom_words.txt.
        hotwords: Option<String>,
    }

    /// Collect and deduplicate the hotword/context terms (order-preserving) from
    /// fixed plain terms, optional per-session hotwords, and an optional patient
    /// name. These feed the qwen system-prompt biasing.
    fn collect_terms(
        plain_terms: &[&str],
        hotwords: Option<&str>,
        patient: Option<&str>,
    ) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        let candidates = plain_terms
            .iter()
            .copied()
            .chain(hotwords.into_iter().flat_map(|hw| hw.split([',', ';', '\n'])))
            .chain(patient);
        for c in candidates {
            let c = c.trim();
            if !c.is_empty() && !terms.iter().any(|t| t == c) {
                terms.push(c.to_string());
            }
        }
        terms
    }

    /// Format the system-prompt context string from collected terms.
    /// Returns `None` if there is nothing to bias on.
    fn build_context(terms: &[String]) -> Option<String> {
        if terms.is_empty() {
            None
        } else {
            Some(format!("Eigennamen und Fachbegriffe: {}.", terms.join(", ")))
        }
    }

    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<AppState>>,
        Query(params): Query<WsQuery>,
    ) -> Response {
        ws.on_upgrade(move |socket| handle_socket(socket, state, params))
    }

    async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>, params: WsQuery) {
        state.connection_count.fetch_add(1, Ordering::Relaxed);
        eprintln!("New connection (total: {})", state.connection_count.load(Ordering::Relaxed));

        // Engine dispatch on ?model= — default Voxtral.
        let result = match params.model.as_deref() {
            Some("qwen") => handle_qwen_session(&mut socket, &state, &params).await,
            _            => handle_asr_session(&mut socket, &state).await,
        };
        if let Err(e) = result {
            eprintln!("ASR session error: {}", e);
            let msg = json!({ "type": "error", "text": e.to_string() }).to_string();
            let _ = socket.send(Message::Text(msg.into())).await;
        }

        state.connection_count.fetch_sub(1, Ordering::Relaxed);
        eprintln!("Connection closed (total: {})", state.connection_count.load(Ordering::Relaxed));
    }

    /// Post-process a raw final transcription: literal `wrong=correct` replacements
    /// first, then acronym letter-name expansion (Em Er I → MRI — before the fuzzy
    /// pass so letter-name tokens can't be snapped away), then (if enabled) fuzzy
    /// phonetic snapping onto known hotwords, and finally the unconditional ß→ss
    /// normalization (Swiss orthography).
    async fn finalize_text(state: &AppState, raw: &str) -> String {
        let corrected = state.words.read().await.apply(raw);
        let expanded = {
            let abbrev = state.abbrev.read().await;
            if abbrev.is_empty() { corrected } else { abbrev.expand(&corrected) }
        };
        let snapped = if state.settings.fuzzy_hotwords.load(Ordering::Relaxed) {
            let ratio = state.settings.fuzzy_max_ratio.load(Ordering::Relaxed);
            state.fuzzy.read().await.correct(&expanded, ratio)
        } else {
            expanded
        };
        replace_eszett(&snapped)
    }

    /// Replace every ß with ss (Swiss orthography). Always applied to all output.
    fn replace_eszett(text: &str) -> String {
        text.replace('ß', "ss")
    }

    async fn handle_asr_session(socket: &mut WebSocket, state: &AppState) -> Result<()> {
        use crate::audio;
        use crate::streaming::{ChunkOutput, StreamingState};

        let model = &state.model;
        let settings = &state.settings;

        // Startup prefill: acquire model lock, run synchronously, release before any await.
        let mut stream_state = {
            let mut guard = model.inner.lock().await;
            let inner = guard.as_mut().ok_or_else(|| // None while model is unloaded for LoRA training
                anyhow::anyhow!("Server is training a new voice model — please try again in a few minutes"))?;
            // Experimental german_prime flag: prime the prefill with German text
            // tokens to bias the language prior. Read per-session so PATCH /config
            // A/B toggling applies on the next connection.
            let prime: &[u32] = if settings.german_prime.load(Ordering::Relaxed) {
                &model.prime_ids
            } else {
                &[]
            };
            StreamingState::new_sync(
                &mut inner.enc,
                &model.adapter,
                &mut inner.dec,
                prime,
                &model.filters,
                &model.device,
                model.dtype,
                settings,
            )?
            // guard dropped here — lock released before any network I/O
        };

        // `true` → flush remaining text as a final before returning;
        // `false` → transport error, nothing more to send.
        let flush = loop {
            match socket.recv().await {
                Some(Ok(Message::Binary(data))) => {
                    // Decode raw f32 LE PCM (Phase 1; Opus planned for Phase 3)
                    let pcm = audio::decode_pcm_f32(&data);
                    if pcm.is_empty() { continue; }

                    // Process: acquire GPU lock, do all sync work, release before send
                    let outputs = {
                        let mut guard = model.inner.lock().await;
                        let inner = guard.as_mut().ok_or_else(|| // None while model is unloaded for LoRA training
                            anyhow::anyhow!("Server is training a new voice model — please try again in a few minutes"))?;
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
                                json!({ "type": "partial", "text": replace_eszett(text) }).to_string()
                            }
                            ChunkOutput::Silence(ref raw) => {
                                let final_text = finalize_text(state, raw).await;
                                json!({ "type": "final", "text": final_text }).to_string()
                            }
                            ChunkOutput::Pad => continue,
                        };
                        if socket.send(Message::Text(msg.into())).await.is_err() {
                            return Ok(());
                        }
                    }
                }
                // Graceful stop (same protocol as schmidiscribe, shared frontend):
                // flush remaining text as a final, then close — the client waits
                // for onclose before finalizing, so the final is never lost the
                // way it can be when the client just closes the socket.
                Some(Ok(Message::Text(text))) if text.trim() == "stop" => break true,
                // Fallback: client disconnected without sending "stop" — same flush,
                // though the final may be lost if the client is already gone.
                Some(Ok(Message::Close(_))) | None => break true,
                Some(Ok(_)) => {} // ping/pong/other text — ignore
                Some(Err(e)) => {
                    eprintln!("WebSocket error: {}", e);
                    break false;
                }
            }
        };

        if flush {
            // Drain the decoder lookahead before flushing: the client stops sending
            // audio the moment the user presses stop, but the decoder lags the audio
            // by `delay` tokens — without feeding silence to cover that window, the
            // last words of the dictation would be truncated.
            {
                let mut guard = model.inner.lock().await;
                if let Some(inner) = guard.as_mut() {
                    if let Err(e) = stream_state.drain_sync(
                        &mut inner.enc,
                        &model.adapter,
                        &mut inner.dec,
                        &model.tok,
                        &model.device,
                    ) {
                        eprintln!("End-of-session drain error: {}", e);
                    }
                }
                // guard dropped here — lock released before the final send
            }
            let raw = stream_state.take_text_buf();
            if !raw.is_empty() {
                let corrected = finalize_text(state, &raw).await;
                let msg = json!({ "type": "final", "text": corrected }).to_string();
                let _ = socket.send(Message::Text(msg.into())).await;
            }
        }

        Ok(())
    }

    /// Qwen3-ASR session (`?model=qwen`) — ported from schmidiscribe's handler.
    ///
    /// Inference runs via `spawn_blocking` (the qwen engine serialises GPU work
    /// behind its own internal lock, independent of the Voxtral mutex, so qwen
    /// and Voxtral sessions run concurrently). Finals go through the same
    /// `finalize_text()` pipeline as the Voxtral path, so qwen gains abbrev
    /// expansion on top of schmidiscribe's literal + fuzzy passes.
    async fn handle_qwen_session(socket: &mut WebSocket, state: &AppState, params: &WsQuery) -> Result<()> {
        use crate::audio;
        use crate::qwen_streaming::{ChunkOutput, StreamingState};

        let engine = match &state.qwen {
            Some(qwen) => qwen.get().await.ok_or_else(|| // None while unloaded for LoRA training
                anyhow::anyhow!("Server is training a new voice model — please try again in a few minutes"))?,
            None => anyhow::bail!("Qwen engine not enabled — set qwen_model_dir in config.toml"),
        };

        // Qwen honours a forced language; per-session ?lang= overrides the configured one.
        let language = params.lang.clone()
            .unwrap_or_else(|| state.startup_snapshot.language.clone());

        // Soft system-prompt biasing (custom_words plain terms + ?hotwords= +
        // ?patient=), gated by the runtime `context_biasing` toggle (read at
        // connection start). The finalize_text() correction passes are independent.
        let context = if state.settings.context_biasing.load(Ordering::Relaxed) {
            let terms = collect_terms(
                &state.words.read().await.plain_terms(),
                params.hotwords.as_deref(),
                params.patient.as_deref(),
            );
            build_context(&terms)
        } else {
            None
        };

        let mut stream_state = StreamingState::new(engine, Some(language), context);

        loop {
            match socket.recv().await {
                Some(Ok(Message::Binary(data))) => {
                    let pcm = audio::decode_pcm_f32(&data);
                    if pcm.is_empty() { continue; }

                    // Run inference off the tokio worker thread to keep /health and
                    // other handlers responsive during active dictation sessions.
                    let settings_arc = Arc::clone(&state.settings);
                    let (returned_state, result) = tokio::task::spawn_blocking(move || {
                        let outputs = stream_state.process_chunk(&pcm, &settings_arc);
                        (stream_state, outputs)
                    }).await?;
                    stream_state = returned_state;
                    let outputs = result?;

                    for output in outputs {
                        let msg = match output {
                            ChunkOutput::Token(ref text) => {
                                json!({ "type": "partial", "text": replace_eszett(text) }).to_string()
                            }
                            ChunkOutput::Silence => {
                                let raw = stream_state.take_text_buf();
                                let final_text = finalize_text(state, &raw).await;
                                json!({ "type": "final", "text": final_text }).to_string()
                            }
                        };
                        if socket.send(Message::Text(msg.into())).await.is_err() {
                            return Ok(());
                        }
                    }
                }
                // Graceful stop (shared protocol): finish inference, send final, close.
                Some(Ok(Message::Text(text))) if text.trim() == "stop" => {
                    let result = tokio::task::spawn_blocking(move || stream_state.finish()).await?;
                    let raw = result?;
                    if !raw.is_empty() {
                        let final_text = finalize_text(state, &raw).await;
                        let msg = json!({ "type": "final", "text": final_text }).to_string();
                        let _ = socket.send(Message::Text(msg.into())).await;
                    }
                    break;
                }
                Some(Ok(Message::Close(_))) | None => {
                    // Fallback: client disconnected without sending "stop" — same
                    // flush, though the final may be lost if the client is gone.
                    let result = tokio::task::spawn_blocking(move || stream_state.finish()).await?;
                    let raw = result?;
                    if !raw.is_empty() {
                        let final_text = finalize_text(state, &raw).await;
                        let msg = json!({ "type": "final", "text": final_text }).to_string();
                        let _ = socket.send(Message::Text(msg.into())).await;
                    }
                    break;
                }
                Some(Ok(_)) => {} // ping/pong/other text — ignore
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
        // LoRA state for the userscript "LoRA verwenden" toggle. `lora_active`
        // is the checkbox state; `lora_dir` is the path the toggle re-applies
        // when re-enabled (active path if loaded, else the configured adapter,
        // else the default training output dir — so the toggle still works after
        // DELETE /lora cleared the active path).
        // Per-model LoRA state (adapters are strictly per-model — key formats
        // differ). `lora_dir_*` is the path the userscript "LoRA verwenden"
        // toggle re-applies when re-enabled (active path if loaded, else the
        // configured adapter, else the model's default training output dir —
        // so the toggle still works after DELETE /lora cleared the active path).
        let active_lora = state.lora_path.read().await.clone();
        let lora_dir = active_lora.clone()
            .or_else(|| snap.lora_adapter.as_ref().map(std::path::PathBuf::from))
            .unwrap_or_else(|| state.paths.lora_output_dir.clone());
        let active_lora_qwen = state.qwen_lora_path.read().await.clone();
        let lora_dir_qwen = active_lora_qwen.clone()
            .or_else(|| snap.lora_adapter_qwen.as_ref().map(std::path::PathBuf::from))
            .unwrap_or_else(|| state.paths.lora_output_dir_qwen.clone());
        // Available engines — the frontend hides qwen UI when "qwen" is absent.
        let models: Vec<&str> = if state.qwen.is_some() {
            vec!["voxtral", "qwen"]
        } else {
            vec!["voxtral"]
        };
        Json(json!({
            "version":           env!("CARGO_PKG_VERSION"),
            // Backend identity — lets the shared userscript adapt its UI per engine.
            "server":            "voicetserver",
            "models":            models,
            // Runtime-adjustable — live values from atomics
            "delay":             s.delay_tokens.load(Ordering::Relaxed),
            "silence_threshold": s.silence_threshold.load(Ordering::Relaxed),
            "silence_flush":     s.silence_chunks.load(Ordering::Relaxed),
            "min_speech":        s.min_speech_chunks.load(Ordering::Relaxed),
            "rms_ema":           s.rms_ema_alpha.load(Ordering::Relaxed),
            "fuzzy_hotwords":    s.fuzzy_hotwords.load(Ordering::Relaxed),
            "fuzzy_max_ratio":   s.fuzzy_max_ratio.load(Ordering::Relaxed),
            "german_prime":      s.german_prime.load(Ordering::Relaxed),
            "context_biasing":   s.context_biasing.load(Ordering::Relaxed),
            // Startup-only — from snapshot; changing via PATCH writes config file but requires restart
            "model_dir":         snap.model_dir,
            "qwen_model_dir":    snap.qwen_model_dir,
            "language":          snap.language,
            "device":            snap.device,
            "port":              snap.port,
            "bind_addr":         snap.bind_addr,
            "tls_enabled":       snap.tls_enabled,
            "lora_adapter":      snap.lora_adapter,
            "lora_adapter_qwen": snap.lora_adapter_qwen,
            // Per-model LoRA state; the unsuffixed pair stays as voxtral aliases
            // during the frontend transition (phase 6).
            "lora_active":          active_lora.is_some(),
            "lora_dir":             lora_dir.to_string_lossy(),
            "lora_active_voxtral":  active_lora.is_some(),
            "lora_dir_voxtral":     lora_dir.to_string_lossy(),
            "lora_active_qwen":     active_lora_qwen.is_some(),
            "lora_dir_qwen":        lora_dir_qwen.to_string_lossy(),
            "venv_path":         snap.venv_path,
            "_startup_only":     ["model_dir", "qwen_model_dir", "language", "device", "port", "bind_addr", "tls_cert", "tls_key", "lora_adapter", "lora_adapter_qwen", "venv_path"],
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
        fuzzy_hotwords:    Option<bool>,
        fuzzy_max_ratio:   Option<f32>,
        german_prime:      Option<bool>,
        context_biasing:   Option<bool>,
        // Startup-only (written to file, restart required)
        model_dir:    Option<String>,
        qwen_model_dir: Option<String>,
        language:     Option<String>,
        device:       Option<usize>,
        port:         Option<u16>,
        bind_addr:    Option<String>,
        tls_cert:     Option<String>,
        tls_key:      Option<String>,
        lora_adapter: Option<String>,
        lora_adapter_qwen: Option<String>,
        venv_path:    Option<String>,
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
        if let Some(t) = patch.silence_threshold {
            // RMS of normalized [-1,1] audio lives in [0,1]; a negative value makes
            // every chunk count as speech, so finalization would never fire.
            if !(0.0..=1.0).contains(&t) {
                return (StatusCode::UNPROCESSABLE_ENTITY,
                    format!("silence_threshold must be between 0.0 and 1.0, got {}", t)).into_response();
            }
        }
        if let Some(r) = patch.fuzzy_max_ratio {
            if !(0.0..=1.0).contains(&r) {
                return (StatusCode::UNPROCESSABLE_ENTITY,
                    format!("fuzzy_max_ratio must be between 0.0 and 1.0, got {}", r)).into_response();
            }
        }

        // Apply runtime params to atomics immediately
        let s = &state.settings;
        if let Some(v) = patch.delay             { s.delay_tokens.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.silence_threshold { s.silence_threshold.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.silence_flush     { s.silence_chunks.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.min_speech        { s.min_speech_chunks.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.rms_ema           { s.rms_ema_alpha.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.fuzzy_hotwords    { s.fuzzy_hotwords.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.fuzzy_max_ratio   { s.fuzzy_max_ratio.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.german_prime      { s.german_prime.store(v, Ordering::SeqCst); }
        if let Some(v) = patch.context_biasing   { s.context_biasing.store(v, Ordering::SeqCst); }

        // Update config file under lock and persist to disk
        {
            let mut cfg = state.config_file.lock().await;
            if patch.delay.is_some()             { cfg.delay             = patch.delay; }
            if patch.silence_threshold.is_some() { cfg.silence_threshold = patch.silence_threshold; }
            if patch.silence_flush.is_some()     { cfg.silence_flush     = patch.silence_flush; }
            if patch.min_speech.is_some()        { cfg.min_speech        = patch.min_speech; }
            if patch.rms_ema.is_some()           { cfg.rms_ema           = patch.rms_ema; }
            if patch.fuzzy_hotwords.is_some()    { cfg.fuzzy_hotwords    = patch.fuzzy_hotwords; }
            if patch.fuzzy_max_ratio.is_some()   { cfg.fuzzy_max_ratio   = patch.fuzzy_max_ratio; }
            if patch.german_prime.is_some()      { cfg.german_prime      = patch.german_prime; }
            if patch.context_biasing.is_some()   { cfg.context_biasing   = patch.context_biasing; }
            if patch.model_dir.is_some()         { cfg.model_dir         = patch.model_dir; }
            if patch.qwen_model_dir.is_some()    { cfg.qwen_model_dir    = patch.qwen_model_dir; }
            if patch.language.is_some()          { cfg.language          = patch.language; }
            if patch.device.is_some()            { cfg.device            = patch.device; }
            if patch.port.is_some()              { cfg.port              = patch.port; }
            if patch.bind_addr.is_some()         { cfg.bind_addr         = patch.bind_addr; }
            if patch.tls_cert.is_some()          { cfg.tls_cert          = patch.tls_cert; }
            if patch.tls_key.is_some()           { cfg.tls_key           = patch.tls_key; }
            if patch.lora_adapter.is_some()      { cfg.lora_adapter      = patch.lora_adapter; }
            if patch.lora_adapter_qwen.is_some() { cfg.lora_adapter_qwen = patch.lora_adapter_qwen; }
            if patch.venv_path.is_some()         { cfg.venv_path         = patch.venv_path; }

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
        // Hold the corrector write lock across the whole read-modify-write so two
        // concurrent POST /words requests cannot lose each other's changes.
        let mut words_guard = state.words.write().await;

        // Preserve comment lines (in their original order) and manage only the actual
        // entry lines as a sorted, deduplicated set — folding `# …` comments into the
        // set would scramble them alphabetically among the entries.
        let content = tokio::fs::read_to_string(&state.words_path).await.unwrap_or_default();
        let mut comments: Vec<String> = Vec::new();
        let mut words: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            } else if trimmed.starts_with('#') {
                comments.push(line.to_string());
            } else {
                words.insert(trimmed.to_string());
            }
        }

        for w in body.add.unwrap_or_default() {
            let w = w.trim().to_string();
            if !w.is_empty() { words.insert(w); }
        }
        for w in body.remove.unwrap_or_default() { words.remove(w.trim()); }

        let mut new_content = String::new();
        for c in &comments { new_content.push_str(c); new_content.push('\n'); }
        for w in &words    { new_content.push_str(w); new_content.push('\n'); }
        match tokio::fs::write(&state.words_path, new_content).await {
            Ok(_) => {
                // Rebuild the in-memory corrector + fuzzy matcher + acronym
                // expander from the updated file
                let new_corrector = WordsCorrector::load(&state.words_path);
                let raw_lines = new_corrector.raw_lines.clone();
                *state.fuzzy.write().await = FuzzyMatcher::from_corrector(&new_corrector);
                *state.abbrev.write().await = AbbrevExpander::from_corrector(&new_corrector);
                *words_guard = new_corrector;
                Json(json!({ "words": raw_lines })).into_response()
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }

    // ---- Training data collection ----

    // Stub written to disk on first request — points user to the sample file in the repo
    const DEFAULT_TRAINING_SENTENCES: &str = "\
# SCHMIDIspeech — Kalibrierungssätze
# Dies ist eine Stub-Datei. Beispielsätze findest du in:
# assets/training_sentences.txt im GitHub-Repository
# Füge eigene Sätze hier ein (einen pro Zeile, # für Kommentare).\
";

    /// GET /training/sentences — returns calibration sentences annotated with recording status
    async fn training_sentences_handler(State(state): State<Arc<AppState>>) -> Response {
        let path = &state.paths.training_sentences;
        // Create default file if not present
        if !path.exists() {
            if let Err(e) = tokio::fs::write(path, DEFAULT_TRAINING_SENTENCES).await {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to create sentences file: {e}")).into_response();
            }
        }
        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };

        // Build map: sentence text → list of pair IDs that recorded it
        let pairs_path = &state.paths.training_pairs;
        let mut recorded: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        if let Ok(pairs_content) = tokio::fs::read_to_string(&pairs_path).await {
            for line in pairs_content.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    // Trim so stray whitespace (in the file or the recorded pair text)
                    // cannot break the sentence ↔ pair matching below.
                    let text = v["text"].as_str().unwrap_or("").trim().to_string();
                    let id   = v["id"].as_str().unwrap_or("").to_string();
                    if !text.is_empty() && !id.is_empty() {
                        recorded.entry(text).or_default().push(id);
                    }
                }
            }
        }

        let sentences: Vec<serde_json::Value> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|s| {
                let pair_ids = recorded.get(s).cloned().unwrap_or_default();
                let is_recorded = !pair_ids.is_empty();
                json!({ "text": s, "recorded": is_recorded, "pair_ids": pair_ids })
            })
            .collect();
        Json(json!({ "sentences": sentences })).into_response()
    }

    /// GET /training/audio/{id} — serve a recorded WAV file for playback
    async fn training_audio_handler(
        State(state): State<Arc<AppState>>,
        axum::extract::Path(id): axum::extract::Path<String>,
    ) -> Response {
        // Only allow numeric IDs to prevent path traversal
        if !id.chars().all(|c| c.is_ascii_digit()) {
            return (StatusCode::BAD_REQUEST, "Invalid id").into_response();
        }
        let wav_path = state.paths.training_audio_dir.join(format!("{id}.wav"));
        match tokio::fs::read(&wav_path).await {
            Ok(bytes) => (
                [(axum::http::header::CONTENT_TYPE, "audio/wav")],
                bytes,
            ).into_response(),
            Err(_) => StatusCode::NOT_FOUND.into_response(),
        }
    }

    /// DELETE /training/pair/{id} — remove one recorded pair (WAV + JSONL entry)
    async fn training_pair_delete_handler(
        State(state): State<Arc<AppState>>,
        axum::extract::Path(id): axum::extract::Path<String>,
    ) -> Response {
        if !id.chars().all(|c| c.is_ascii_digit()) {
            return (StatusCode::BAD_REQUEST, "Invalid id").into_response();
        }
        let wav_path  = state.paths.training_audio_dir.join(format!("{id}.wav"));
        let pairs_path = &state.paths.training_pairs;

        if !wav_path.exists() {
            return StatusCode::NOT_FOUND.into_response();
        }

        // Hold the write lock so a concurrent upload can't append to the JSONL while we
        // rewrite it, and so two concurrent deletes don't interleave.
        let _guard = state.pair_write_lock.lock().await;

        if let Err(e) = tokio::fs::remove_file(&wav_path).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to remove WAV: {e}")).into_response();
        }

        // Rewrite pairs.jsonl without the deleted entry
        remove_jsonl_entry(pairs_path, &id).await;

        Json(json!({ "deleted": id })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct DeleteSentenceRequest {
        text: String,
    }

    /// DELETE /training/sentence — remove one sentence from training_sentences.txt
    async fn training_sentence_delete_handler(
        State(state): State<Arc<AppState>>,
        Json(body): Json<DeleteSentenceRequest>,
    ) -> Response {
        let path = &state.paths.training_sentences;
        // Serialise against concurrent sentence edits (read-modify-write on the file).
        let _guard = state.pair_write_lock.lock().await;
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };

        let target = body.text.trim().to_string();
        let mut found = false;
        let new_content: String = content
            .lines()
            .filter(|l| {
                if !found && l.trim() == target.as_str() {
                    found = true;
                    return false;
                }
                true
            })
            .map(|l| format!("{l}\n"))
            .collect();

        if !found {
            return StatusCode::NOT_FOUND.into_response();
        }
        if let Err(e) = tokio::fs::write(&path, new_content).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
        Json(json!({ "deleted": true })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct AddSentenceRequest {
        text: String,
    }

    /// POST /training/sentence — append a new sentence to training_sentences.txt
    async fn training_sentence_add_handler(
        State(state): State<Arc<AppState>>,
        Json(body): Json<AddSentenceRequest>,
    ) -> Response {
        let text = body.text.trim().to_string();
        if text.is_empty() {
            return (StatusCode::BAD_REQUEST, "Empty sentence").into_response();
        }
        let path = &state.paths.training_sentences;
        // Serialise against concurrent sentence edits/deletes rewriting the file.
        let _guard = state.pair_write_lock.lock().await;
        // Ensure file exists (may create default stub first)
        if !path.exists() {
            let _ = tokio::fs::write(&path, DEFAULT_TRAINING_SENTENCES).await;
        }
        use tokio::io::AsyncWriteExt;
        let mut f = match tokio::fs::OpenOptions::new()
            .create(true).append(true).open(&path).await
        {
            Ok(f) => f,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        if let Err(e) = f.write_all(format!("{text}\n").as_bytes()).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
        Json(json!({ "added": true })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct EditSentenceRequest {
        old: String,
        new: String,
    }

    /// PATCH /training/sentence — replace one sentence in training_sentences.txt
    async fn training_sentence_edit_handler(
        State(state): State<Arc<AppState>>,
        Json(body): Json<EditSentenceRequest>,
    ) -> Response {
        let old_text = body.old.trim().to_string();
        let new_text = body.new.trim().to_string();
        if new_text.is_empty() {
            return (StatusCode::BAD_REQUEST, "New text is empty").into_response();
        }
        let path = &state.paths.training_sentences;
        // Serialise against concurrent sentence edits (read-modify-write on the file).
        let _guard = state.pair_write_lock.lock().await;
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };

        let mut found = false;
        let new_content: String = content
            .lines()
            .map(|l| {
                if !found && l.trim() == old_text.as_str() {
                    found = true;
                    format!("{new_text}\n")
                } else {
                    format!("{l}\n")
                }
            })
            .collect();

        if !found {
            return StatusCode::NOT_FOUND.into_response();
        }
        if let Err(e) = tokio::fs::write(&path, new_content).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
        Json(json!({ "updated": true })).into_response()
    }

    /// GET /training/pairs — list all recorded pairs
    async fn training_pairs_handler(State(state): State<Arc<AppState>>) -> Response {
        let pairs_path = &state.paths.training_pairs;
        let content = tokio::fs::read_to_string(&pairs_path).await.unwrap_or_default();
        let mut pairs: Vec<serde_json::Value> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect();
        // Sort numerically by id for stable display order (string compare would put
        // a 5-digit "10000" before the zero-padded 4-digit "9999").
        pairs.sort_by_key(|p| {
            p["id"].as_str().and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX)
        });
        Json(json!({ "pairs": pairs })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct TrainingPairQuery {
        text: String,
    }

    /// Append one JSONL entry `{"id","text","duration_s"}` to `jsonl_path`.
    async fn append_pair_entry(
        jsonl_path: &std::path::Path,
        id: &str,
        text: &str,
        duration_s: f32,
    ) -> anyhow::Result<()> {
        use tokio::io::AsyncWriteExt;
        let entry = format!("{{\"id\":\"{id}\",\"text\":{},\"duration_s\":{:.3}}}\n",
            serde_json::to_string(text).unwrap_or_default(), duration_s);
        let mut f = tokio::fs::OpenOptions::new()
            .create(true).append(true).open(jsonl_path).await?;
        f.write_all(entry.as_bytes()).await?;
        Ok(())
    }

    /// Save a PCM+text pair: WAV to `audio_dir/{id}.wav`, JSONL line appended to
    /// `jsonl_path`. The ID is one past the highest existing ID (never reused after
    /// a delete). Caller must hold `pair_write_lock`.
    /// Returns `(id, duration_s, count_after)`.
    async fn save_pcm_pair(
        audio_dir: &std::path::Path,
        jsonl_path: &std::path::Path,
        text: &str,
        pcm: &[f32],
    ) -> anyhow::Result<(String, f32, usize)> {
        tokio::fs::create_dir_all(audio_dir).await?;
        let (count, next_id) = pairs_stats(jsonl_path).await;
        let id = format!("{:04}", next_id);
        let duration_s = pcm.len() as f32 / 16000.0;
        save_training_wav(&audio_dir.join(format!("{id}.wav")), pcm)?;
        append_pair_entry(jsonl_path, &id, text, duration_s).await?;
        Ok((id, duration_s, count + 1))
    }

    /// POST /training/pair?text=... — body is raw f32 LE PCM at 16kHz
    async fn training_pair_handler(
        State(state): State<Arc<AppState>>,
        Query(params): Query<TrainingPairQuery>,
        body: Bytes,
    ) -> Response {
        let pcm = crate::audio::decode_audio_bytes(&body);
        if pcm.is_empty() {
            return (StatusCode::BAD_REQUEST, "Empty audio body").into_response();
        }

        // Hold the write lock for the entire ID-assignment + file-write sequence so
        // concurrent uploads don't collide on the same ID, and deletions don't rewrite
        // the JSONL while we're appending to it.
        let _guard = state.pair_write_lock.lock().await;

        match save_pcm_pair(&state.paths.training_audio_dir, &state.paths.training_pairs,
                            &params.text, &pcm).await
        {
            Ok((id, duration_s, count)) =>
                Json(json!({ "id": id, "duration_s": duration_s, "count": count })).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Pair save error: {e}")).into_response(),
        }
    }

    /// GET /training — summary of collected pairs
    async fn training_get_handler(State(state): State<Arc<AppState>>) -> Response {
        let pairs_path = &state.paths.training_pairs;
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
    async fn training_delete_handler(State(state): State<Arc<AppState>>) -> Response {
        let dir = &state.paths.training_dir;
        if dir.exists() {
            if let Err(e) = tokio::fs::remove_dir_all(&dir).await {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to delete training data: {e}")).into_response();
            }
        }
        Json(json!({ "deleted": true })).into_response()
    }

    /// POST /training/run?model=voxtral|qwen — spawn the model's trainer script as
    /// a subprocess (default voxtral). The shared pairs pool trains both models'
    /// LoRAs; the scripts and adapter output dirs are per-model.
    async fn training_run_handler(
        State(state): State<Arc<AppState>>,
        Query(scope): Query<ModelQuery>,
    ) -> Response {
        // Validate prerequisites before touching the status, so a failed validation
        // never leaves the status stuck on "running".
        let train_qwen = scope.is_qwen();

        // Locate the trainer: try cwd/tools/, binary dir/tools/, config dir/tools/.
        // Voxtral falls back to the pre-phase-4 script name for existing installs.
        let script = if train_qwen {
            find_script("tools/train_lora_qwen.py")
        } else {
            find_script("tools/train_lora_voxtral.py")
                .or_else(|| find_script("tools/train_lora.py"))
        };
        let script = match script {
            Some(p) => p,
            None => return (StatusCode::UNPROCESSABLE_ENTITY, format!(
                "tools/train_lora_{}.py not found — run server from project root",
                if train_qwen { "qwen" } else { "voxtral" })).into_response(),
        };

        let data_dir   = state.paths.training_dir.clone();
        let model_dir  = if train_qwen {
            match state.startup_snapshot.qwen_model_dir.clone() {
                Some(d) => d,
                None => return (StatusCode::UNPROCESSABLE_ENTITY,
                    "Qwen engine not enabled — set qwen_model_dir in config.toml").into_response(),
            }
        } else {
            state.startup_snapshot.model_dir.clone()
        };
        let output_dir = if train_qwen {
            state.paths.lora_output_dir_qwen.clone()
        } else {
            state.paths.lora_output_dir.clone()
        };
        let venv_path  = state.startup_snapshot.venv_path.clone();

        // Ensure training dir exists before passing it to Python
        if !data_dir.exists() {
            return (StatusCode::UNPROCESSABLE_ENTITY,
                "No training data yet — collect pairs first").into_response();
        }

        // Atomically check-and-set the running flag under a single lock so two
        // concurrent requests cannot both pass the check and double-spawn.
        {
            let mut ts = state.training_status.lock().await;
            if ts.status == TrainingStatusKind::Running {
                return (StatusCode::CONFLICT, "Training already running").into_response();
            }
            *ts = TrainingStatus { status: TrainingStatusKind::Running, log: vec![] };
        }

        // Free the GPU for the training subprocess: unload BOTH engines (simplest
        // safe rule on a 16 GB card — the trainer needs its own copy of whichever
        // model it trains). ASR sessions on either engine return a "training in
        // progress" error until the engines are reloaded. Note: a qwen session
        // already in flight keeps its Arc handle alive (its VRAM frees when the
        // session ends); new sessions are blocked immediately.
        {
            let mut guard = state.model.inner.lock().await;
            *guard = None; // drop enc+dec → frees their VRAM
        }
        if let Some(qwen) = state.qwen.as_ref() {
            *qwen.inner.lock().await = None; // drop the qwen engine (~1.5 GB)
        }
        // Return the freed pool memory to the OS so the separate training process sees it.
        // Drop frees into cudarc's stream-ordered pool; we must synchronize (let the async
        // frees complete) then trim the pool, or the GPU still reads as full to other procs.
        {
            let model = Arc::clone(&state.model);
            let _ = tokio::task::spawn_blocking(move || -> Result<()> {
                model.device.synchronize()?;
                release_cuda_pool(&model.device)?;
                Ok(())
            }).await;
        }
        eprintln!("Models unloaded from VRAM for training.");

        let training_status = Arc::clone(&state.training_status);
        let model           = Arc::clone(&state.model);
        let lora_path       = Arc::clone(&state.lora_path);
        let qwen_engine     = state.qwen.clone();
        let qwen_lora_path  = Arc::clone(&state.qwen_lora_path);
        tokio::spawn(async move {
            run_training_subprocess(training_status, script, data_dir, model_dir, output_dir, venv_path).await;

            // Reload both engines regardless of training outcome, re-applying each
            // model's active LoRA.
            let lora = lora_path.read().await.clone();
            let m = Arc::clone(&model);
            let reloaded = tokio::task::spawn_blocking(move || {
                load_enc_dec(&m.model_dir, &m.device, m.dtype, lora.as_deref())
            }).await;
            match reloaded {
                Ok(Ok(inner)) => {
                    *model.inner.lock().await = Some(inner);
                    eprintln!("Model reloaded into VRAM after training.");
                }
                Ok(Err(e)) => eprintln!("ERROR: model reload after training failed: {e}"),
                Err(e)     => eprintln!("ERROR: model reload task panicked: {e}"),
            }
            if let Some(qwen) = qwen_engine {
                let lora = qwen_lora_path.read().await.clone();
                let reloaded = tokio::task::spawn_blocking(move || {
                    qwen.reload_blocking(lora.as_deref())
                }).await;
                match reloaded {
                    Ok(Ok(()))  => eprintln!("Qwen engine reloaded into VRAM after training."),
                    Ok(Err(e)) => eprintln!("ERROR: qwen reload after training failed: {e}"),
                    Err(e)     => eprintln!("ERROR: qwen reload task panicked: {e}"),
                }
            }
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

    // ---- LoRA hot-reload ----

    /// `?model=` scope for the LoRA endpoints: "qwen" → qwen adapter,
    /// anything else / absent → voxtral (same convention as the WS `?model=`).
    #[derive(serde::Deserialize, Default)]
    struct ModelQuery {
        model: Option<String>,
    }

    impl ModelQuery {
        fn is_qwen(&self) -> bool {
            self.model.as_deref() == Some("qwen")
        }
    }

    /// POST /lora/reload?model=voxtral|qwen — reload (or swap) a model's LoRA
    /// adapter without restarting (default voxtral).
    ///
    /// Optional JSON body: `{"path": "/abs/path/to/adapter_dir"}`.
    /// If omitted, reloads from the currently active path.
    /// If the path does not contain adapter files, the LoRA is cleared (base model).
    async fn lora_reload_handler(
        State(state): State<Arc<AppState>>,
        Query(scope): Query<ModelQuery>,
        body: Option<Json<serde_json::Value>>,
    ) -> impl IntoResponse {
        let requested = body.as_ref()
            .and_then(|b| b.get("path"))
            .and_then(|v| v.as_str())
            .map(std::path::PathBuf::from);

        if scope.is_qwen() {
            return qwen_lora_reload(&state, requested).await;
        }

        let path = if let Some(p) = requested {
            p
        } else {
            match state.lora_path.read().await.clone() {
                Some(p) => p,
                None => return (StatusCode::BAD_REQUEST,
                    Json(json!({"error": "no lora path configured"}))).into_response(),
            }
        };

        let device = state.model.device.clone();
        let dtype  = state.model.dtype;
        let path2  = path.clone();
        let result = tokio::task::spawn_blocking(move || {
            crate::lora::load_decoder_lora(&path2, &device, dtype)
        }).await;

        let lora_opt = match result {
            Ok(Ok(l))  => l,
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()}))).into_response(),
            Err(e)     => return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()}))).into_response(),
        };

        let action = lora_opt.as_ref().map_or("cleared", |_| "applied");
        {
            let mut guard = state.model.inner.lock().await;
            match guard.as_mut() {
                Some(inner) => match lora_opt {
                    Some(ref lora) => inner.dec.set_lora(lora),
                    None           => inner.dec.clear_lora(),
                },
                // Model is unloaded for training; still record the path below so it is
                // re-applied automatically when the model reloads.
                None => {}
            }
        }

        // Record the path only when an adapter was actually applied. Storing it on
        // the "cleared" outcome (no adapter files at the path) would make GET /config
        // report lora_active=true and re-apply a nonexistent adapter after training.
        *state.lora_path.write().await = if lora_opt.is_some() {
            Some(path.clone())
        } else {
            None
        };
        eprintln!("LoRA hot-reload {}: {}", action, path.display());
        (StatusCode::OK, Json(json!({
            "status": "ok",
            "action": action,
            "path":   path.to_string_lossy(),
        }))).into_response()
    }

    /// Qwen branch of POST /lora/reload (`?model=qwen`). Same response shape and
    /// semantics as the voxtral branch: a path without adapter files clears the
    /// LoRA; while the engine is unloaded for training the path is only recorded
    /// and re-applied on the post-training reload.
    async fn qwen_lora_reload(
        state: &Arc<AppState>,
        requested: Option<std::path::PathBuf>,
    ) -> Response {
        let Some(qwen) = state.qwen.as_ref() else {
            return (StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": "Qwen engine not enabled — set qwen_model_dir in config.toml"}))).into_response();
        };

        let path = if let Some(p) = requested {
            p
        } else {
            match state.qwen_lora_path.read().await.clone() {
                Some(p) => p,
                None => return (StatusCode::BAD_REQUEST,
                    Json(json!({"error": "no lora path configured"}))).into_response(),
            }
        };

        // Mirror the voxtral semantics: missing adapter files → clear to base model.
        if !path.join("adapter_model.safetensors").exists() {
            if let Some(engine) = qwen.get().await {
                let result = tokio::task::spawn_blocking(move || engine.clear_lora()).await;
                if let Ok(Err(e)) = result {
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()}))).into_response();
                }
            }
            *state.qwen_lora_path.write().await = None;
            eprintln!("Qwen LoRA hot-reload cleared: {}", path.display());
            return (StatusCode::OK, Json(json!({
                "status": "ok",
                "action": "cleared",
                "path":   path.to_string_lossy(),
            }))).into_response();
        }

        match qwen.get().await {
            Some(engine) => {
                let p2 = path.clone();
                let result = tokio::task::spawn_blocking(move || engine.load_lora(&p2)).await;
                match result {
                    Ok(Ok(()))  => {}
                    Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()}))).into_response(),
                    Err(e)     => return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()}))).into_response(),
                }
            }
            // Engine is unloaded for training; the path recorded below is
            // re-applied automatically when the engine reloads.
            None => {}
        }

        *state.qwen_lora_path.write().await = Some(path.clone());
        eprintln!("Qwen LoRA hot-reload applied: {}", path.display());
        (StatusCode::OK, Json(json!({
            "status": "ok",
            "action": "applied",
            "path":   path.to_string_lossy(),
        }))).into_response()
    }

    // ---- DELETE /lora ----

    /// DELETE /lora?model=voxtral|qwen — unload a model's active LoRA adapter
    /// in-memory (revert to base model) without a restart (default voxtral). The
    /// adapter files on disk are left untouched. Useful to recover quickly from a
    /// bad adapter. No-op (still 200) while the model is unloaded for training;
    /// the cleared path means nothing is re-applied on reload.
    async fn lora_clear_handler(
        State(state): State<Arc<AppState>>,
        Query(scope): Query<ModelQuery>,
    ) -> Response {
        if scope.is_qwen() {
            let Some(qwen) = state.qwen.as_ref() else {
                return (StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({"error": "Qwen engine not enabled — set qwen_model_dir in config.toml"}))).into_response();
            };
            if let Some(engine) = qwen.get().await {
                let result = tokio::task::spawn_blocking(move || engine.clear_lora()).await;
                if let Ok(Err(e)) = result {
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()}))).into_response();
                }
            }
            *state.qwen_lora_path.write().await = None;
            eprintln!("Qwen LoRA cleared (reverted to base model).");
            return Json(json!({ "status": "cleared" })).into_response();
        }

        {
            let mut guard = state.model.inner.lock().await;
            if let Some(inner) = guard.as_mut() {
                inner.dec.clear_lora();
            }
        }
        *state.lora_path.write().await = None;
        eprintln!("LoRA cleared (reverted to base model).");
        Json(json!({ "status": "cleared" })).into_response()
    }

    // ---- POST /log/edit ----

    #[derive(serde::Deserialize)]
    struct EditLogEntry {
        original:  String,
        edited:    String,
        timestamp: String,
    }

    async fn log_edit_handler(
        State(state): State<Arc<AppState>>,
        Json(body): Json<EditLogEntry>,
    ) -> Response {
        use tokio::io::AsyncWriteExt;
        let line = format!("{}\n", serde_json::to_string(&json!({
            "original":  body.original,
            "edited":    body.edited,
            "timestamp": body.timestamp,
        })).unwrap_or_default());
        let mut f = match tokio::fs::OpenOptions::new()
            .create(true).append(true).open(&state.paths.edit_log).await
        {
            Ok(f) => f,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        if let Err(e) = f.write_all(line.as_bytes()).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
        Json(json!({ "status": "ok" })).into_response()
    }

    // ---- GET /edits/report ----

    /// Aggregate edit_log.jsonl (original→edited pairs from commit-mode edits) into
    /// the most frequent word-level corrections — direct candidates for
    /// custom_words `wrong=correct` entries.
    async fn edit_report_handler(State(state): State<Arc<AppState>>) -> Response {
        let content = tokio::fs::read_to_string(&state.paths.edit_log).await.unwrap_or_default();
        let mut counts: std::collections::HashMap<(String, String), usize> =
            std::collections::HashMap::new();
        let mut entries = 0usize;
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let (Some(orig), Some(edit)) = (v["original"].as_str(), v["edited"].as_str()) else {
                continue;
            };
            entries += 1;
            for (o, e) in word_diffs(orig, edit) {
                *counts.entry((o, e)).or_insert(0) += 1;
            }
        }
        let mut list: Vec<_> = counts.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let suggestions: Vec<serde_json::Value> = list.into_iter().take(30)
            .map(|((o, e), c)| json!({ "original": o, "edited": e, "count": c }))
            .collect();
        Json(json!({ "entries": entries, "suggestions": suggestions })).into_response()
    }

    /// Word-level diff of one edit-log entry: LCS over whitespace tokens; changed
    /// regions become (removed run → inserted run) pairs. Pure insertions/deletions
    /// and punctuation-only changes are skipped — they make no wrong=correct pair.
    fn word_diffs(orig: &str, edit: &str) -> Vec<(String, String)> {
        /// Longer replaced runs are rewrites, not word corrections.
        const MAX_RUN_WORDS: usize = 4;

        let a: Vec<&str> = orig.split_whitespace().collect();
        let b: Vec<&str> = edit.split_whitespace().collect();
        // The DP table is O(n·m); skip pathological entries rather than stall.
        if a.is_empty() || b.is_empty() || a.len() * b.len() > 250_000 {
            return Vec::new();
        }

        let (n, m) = (a.len(), b.len());
        let mut dp = vec![vec![0u32; m + 1]; n + 1];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                dp[i][j] = if a[i] == b[j] {
                    dp[i + 1][j + 1] + 1
                } else {
                    dp[i + 1][j].max(dp[i][j + 1])
                };
            }
        }

        fn flush(removed: &mut Vec<&str>, inserted: &mut Vec<&str>,
                 out: &mut Vec<(String, String)>, max_run: usize) {
            if !removed.is_empty() && !inserted.is_empty()
                && removed.len() <= max_run && inserted.len() <= max_run
            {
                let o = trim_punct(&removed.join(" "));
                let e = trim_punct(&inserted.join(" "));
                if !o.is_empty() && !e.is_empty() && o != e {
                    out.push((o, e));
                }
            }
            removed.clear();
            inserted.clear();
        }

        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        let mut removed: Vec<&str> = Vec::new();
        let mut inserted: Vec<&str> = Vec::new();
        while i < n && j < m {
            if a[i] == b[j] {
                flush(&mut removed, &mut inserted, &mut out, MAX_RUN_WORDS);
                i += 1;
                j += 1;
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                removed.push(a[i]);
                i += 1;
            } else {
                inserted.push(b[j]);
                j += 1;
            }
        }
        removed.extend_from_slice(&a[i..]);
        inserted.extend_from_slice(&b[j..]);
        flush(&mut removed, &mut inserted, &mut out, MAX_RUN_WORDS);
        out
    }

    /// Strip leading/trailing punctuation from a phrase (inner punctuation stays).
    fn trim_punct(s: &str) -> String {
        s.trim_matches(|c: char| !c.is_alphanumeric()).to_string()
    }

    // ---- Dictation review (real dictations as training-pair candidates) ----
    //
    // Read-aloud calibration sentences train the LoRA on a different speaking
    // style than free dictation. These endpoints let the client save a finished
    // real dictation (audio + model transcript) as a *candidate*; the user later
    // reviews it (play, re-transcribe, correct the text) and either accepts it
    // into pairs.jsonl or discards it. Candidates live in training/review/ +
    // review.jsonl and are invisible to the trainer until accepted.

    /// POST /training/review?text=... — body raw f32 LE PCM 16kHz, `text` = model
    /// transcript at save time.
    async fn review_add_handler(
        State(state): State<Arc<AppState>>,
        Query(params): Query<TrainingPairQuery>,
        body: Bytes,
    ) -> Response {
        let pcm = crate::audio::decode_audio_bytes(&body);
        if pcm.is_empty() {
            return (StatusCode::BAD_REQUEST, "Empty audio body").into_response();
        }
        let _guard = state.pair_write_lock.lock().await;
        match save_pcm_pair(&state.paths.review_dir, &state.paths.review_jsonl,
                            &params.text, &pcm).await
        {
            Ok((id, duration_s, count)) =>
                Json(json!({ "id": id, "duration_s": duration_s, "count": count })).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Review save error: {e}")).into_response(),
        }
    }

    /// GET /training/reviews — list all dictation candidates awaiting review
    async fn review_list_handler(State(state): State<Arc<AppState>>) -> Response {
        let content = tokio::fs::read_to_string(&state.paths.review_jsonl).await.unwrap_or_default();
        let mut items: Vec<serde_json::Value> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect();
        items.sort_by_key(|p| {
            p["id"].as_str().and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX)
        });
        Json(json!({ "reviews": items })).into_response()
    }

    /// GET /training/review/audio/{id} — serve a candidate WAV for playback /
    /// client-side re-transcription
    async fn review_audio_handler(
        State(state): State<Arc<AppState>>,
        axum::extract::Path(id): axum::extract::Path<String>,
    ) -> Response {
        if !id.chars().all(|c| c.is_ascii_digit()) {
            return (StatusCode::BAD_REQUEST, "Invalid id").into_response();
        }
        let wav_path = state.paths.review_dir.join(format!("{id}.wav"));
        match tokio::fs::read(&wav_path).await {
            Ok(bytes) => (
                [(axum::http::header::CONTENT_TYPE, "audio/wav")],
                bytes,
            ).into_response(),
            Err(_) => StatusCode::NOT_FOUND.into_response(),
        }
    }

    /// DELETE /training/review/{id} — discard a candidate (WAV + JSONL entry)
    async fn review_delete_handler(
        State(state): State<Arc<AppState>>,
        axum::extract::Path(id): axum::extract::Path<String>,
    ) -> Response {
        if !id.chars().all(|c| c.is_ascii_digit()) {
            return (StatusCode::BAD_REQUEST, "Invalid id").into_response();
        }
        let wav_path = state.paths.review_dir.join(format!("{id}.wav"));
        if !wav_path.exists() {
            return StatusCode::NOT_FOUND.into_response();
        }
        let _guard = state.pair_write_lock.lock().await;
        if let Err(e) = tokio::fs::remove_file(&wav_path).await {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to remove WAV: {e}")).into_response();
        }
        remove_jsonl_entry(&state.paths.review_jsonl, &id).await;
        Json(json!({ "deleted": id })).into_response()
    }

    #[derive(serde::Deserialize)]
    struct ReviewAcceptRequest {
        text: String,
    }

    /// POST /training/review/{id}/accept — body `{"text":"…"}` (the corrected
    /// transcript). Moves the WAV into training/audio/ under a fresh pair ID,
    /// appends to pairs.jsonl, and removes the candidate from review.
    async fn review_accept_handler(
        State(state): State<Arc<AppState>>,
        axum::extract::Path(id): axum::extract::Path<String>,
        Json(body): Json<ReviewAcceptRequest>,
    ) -> Response {
        if !id.chars().all(|c| c.is_ascii_digit()) {
            return (StatusCode::BAD_REQUEST, "Invalid id").into_response();
        }
        let text = body.text.trim().to_string();
        if text.is_empty() {
            return (StatusCode::BAD_REQUEST, "Empty text").into_response();
        }
        let src_wav = state.paths.review_dir.join(format!("{id}.wav"));
        if !src_wav.exists() {
            return StatusCode::NOT_FOUND.into_response();
        }

        let _guard = state.pair_write_lock.lock().await;

        // Duration from the review entry; fallback: recompute from the WAV size
        // (16-bit mono 16 kHz, 44-byte header).
        let review_content =
            tokio::fs::read_to_string(&state.paths.review_jsonl).await.unwrap_or_default();
        let duration_s = match review_content.lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|v| v["id"].as_str() == Some(id.as_str()))
            .and_then(|v| v["duration_s"].as_f64())
        {
            Some(d) => d as f32,
            None => {
                let len = tokio::fs::metadata(&src_wav).await.map(|m| m.len()).unwrap_or(0);
                (len.saturating_sub(44) as f32 / 2.0) / 16000.0
            }
        };

        if let Err(e) = tokio::fs::create_dir_all(&state.paths.training_audio_dir).await {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create training dir: {e}")).into_response();
        }
        let (_, next_id) = pairs_stats(&state.paths.training_pairs).await;
        let new_id = format!("{:04}", next_id);
        let dst_wav = state.paths.training_audio_dir.join(format!("{new_id}.wav"));
        if let Err(e) = tokio::fs::rename(&src_wav, &dst_wav).await {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to move WAV: {e}")).into_response();
        }
        if let Err(e) = append_pair_entry(&state.paths.training_pairs, &new_id, &text, duration_s).await {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                format!("Pair append error: {e}")).into_response();
        }
        remove_jsonl_entry(&state.paths.review_jsonl, &id).await;
        Json(json!({ "id": new_id, "duration_s": duration_s })).into_response()
    }

    /// Rewrite a `{"id":…}` JSONL file without the entry matching `id`.
    /// Caller must hold `pair_write_lock`.
    async fn remove_jsonl_entry(jsonl_path: &std::path::Path, id: &str) {
        if let Ok(content) = tokio::fs::read_to_string(jsonl_path).await {
            let new_content: String = content
                .lines()
                .filter(|l| {
                    if l.trim().is_empty() { return false; }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                        return v["id"].as_str() != Some(id);
                    }
                    true
                })
                .map(|l| format!("{l}\n"))
                .collect();
            let _ = tokio::fs::write(jsonl_path, new_content).await;
        }
    }

    // ---- Training helpers ----

    /// Scan pairs.jsonl once and return `(count, next_id)` where `count` is the number
    /// of non-empty lines and `next_id` is one past the highest numeric `id` seen
    /// (so IDs are never reused after a delete). Falls back to 1 for an empty file.
    async fn pairs_stats(pairs_path: &std::path::Path) -> (usize, usize) {
        let content = tokio::fs::read_to_string(pairs_path).await.unwrap_or_default();
        let mut count = 0usize;
        let mut max_id = 0usize;
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            count += 1;
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(id) = v["id"].as_str().and_then(|s| s.parse::<usize>().ok()) {
                    max_id = max_id.max(id);
                }
            }
        }
        (count, max_id + 1)
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
        // 1. Relative to cwd (dev: run from project root)
        let p = std::path::Path::new(rel_path);
        if p.exists() { return Some(p.to_path_buf()); }
        // 2. Relative to binary dir (binary + tools/ deployed together)
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let p = dir.join(rel_path);
                if p.exists() { return Some(p); }
            }
        }
        // 3. ~/.config/voicetserver/tools/ (binary-only install)
        let p = crate::config::config_dir().join(rel_path);
        if p.exists() { return Some(p); }
        None
    }

    /// Locate the Python interpreter to use for training.
    /// Search order:
    ///   1. venv_path from config/CLI (explicit — preferred)
    ///   2. tools/.venv/ relative to cwd  (dev: run from project root)
    ///   3. tools/.venv/ relative to binary dir
    ///   4. system python3
    fn find_python(venv_path: Option<&str>) -> String {
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();

        // 1. Explicitly configured venv
        if let Some(venv) = venv_path {
            candidates.push(std::path::Path::new(venv).join("bin/python3"));
            candidates.push(std::path::Path::new(venv).join("bin/python"));
        }

        // 2. tools/.venv relative to cwd (dev workflow)
        candidates.push(std::path::Path::new("tools/.venv/bin/python3").to_path_buf());

        // 3. tools/.venv relative to binary dir
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("tools/.venv/bin/python3"));
            }
        }

        for p in &candidates {
            if p.exists() { return p.to_string_lossy().into_owned(); }
        }
        "python3".to_string()  // fall back to system python3
    }

    async fn run_training_subprocess(
        training_status: Arc<tokio::sync::Mutex<TrainingStatus>>,
        script: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        model_dir: String,
        output_dir: std::path::PathBuf,
        venv_path: Option<String>,
    ) {
        use tokio::io::AsyncBufReadExt;

        let append_log = |ts: &mut TrainingStatus, line: String| {
            if ts.log.len() >= 200 { ts.log.remove(0); }
            ts.log.push(line);
        };

        let python = find_python(venv_path.as_deref());
        let mut child = match tokio::process::Command::new(&python)
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
                ts.log.push(format!("Failed to spawn {python}: {e}"));
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
