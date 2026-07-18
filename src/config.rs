// Config file loading, merging, and persistence for voicetserver.
//
// Config file:  ~/.config/voicetserver/config.toml
// Custom words: {data_dir}/custom_words.txt  (data_dir defaults to ~/.config/voicetserver/)
//
// Priority: CLI arg > config file value > compiled default

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

// ---------------------------------------------------------------------------
// Value source tracking
// ---------------------------------------------------------------------------

/// Records where a value came from — used in error messages for path-like fields.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ValueSource {
    CliArg,
    ConfigFile,
    Default,
}

/// A value tagged with its provenance.
pub struct Sourced<T> {
    pub value: T,
    pub source: ValueSource,
}

// ---------------------------------------------------------------------------
// Config file struct
// ---------------------------------------------------------------------------

/// On-disk representation of ~/.config/voicetserver/config.toml.
/// All fields are Option<T>: None means "not set in config file".
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ConfigFile {
    // Startup-only (server restart required to apply)
    pub model_dir:    Option<String>,
    pub device:       Option<usize>,
    pub port:         Option<u16>,
    pub bind_addr:    Option<String>,
    pub tls_cert:     Option<String>,
    pub tls_key:      Option<String>,
    pub lora_adapter: Option<String>,
    pub venv_path:    Option<String>,
    pub data_dir:     Option<String>,
    /// Qwen3-ASR model directory. When unset the qwen engine is disabled and
    /// `/asr?model=qwen` sessions get an error frame.
    pub qwen_model_dir: Option<String>,
    /// Transcription language for the qwen engine (Voxtral has no language control).
    pub language:     Option<String>,
    /// API key for authenticating HTTP/WebSocket requests. Auto-generated on first
    /// server start and persisted here. Not a CLI flag — config file only.
    pub api_key:      Option<String>,
    // Runtime-adjustable via PATCH /config
    pub delay:             Option<usize>,
    pub silence_threshold: Option<f32>,
    pub silence_flush:     Option<usize>,
    pub min_speech:        Option<usize>,
    pub rms_ema:           Option<f32>,
    pub fuzzy_hotwords:    Option<bool>,
    pub fuzzy_max_ratio:   Option<f32>,
    pub german_prime:      Option<bool>,
    /// Qwen prompt biasing (custom words + hotwords + patient context) on/off.
    pub context_biasing:   Option<bool>,
    // Log file settings
    pub log_file:      Option<String>,
    pub log_keep_days: Option<u32>,
}

/// Thread-safe handle to the on-disk config (for PATCH /config writes).
pub type SharedConfigFile = Arc<tokio::sync::Mutex<ConfigFile>>;

// ---------------------------------------------------------------------------
// Merged config (CLI + file + compiled defaults)
// ---------------------------------------------------------------------------

/// Result of merging CLI args, config file values, and compiled defaults.
pub struct MergedConfig {
    // Path-like fields: wrapped in Sourced for tagged error messages
    pub model_dir: Sourced<String>,
    pub bind_addr: Sourced<String>,
    pub tls_cert:  Sourced<Option<String>>,
    pub tls_key:   Sourced<Option<String>>,
    /// Qwen3-ASR model directory; None = qwen engine disabled.
    pub qwen_model_dir: Sourced<Option<String>>,
    // Plain merged
    pub device:       usize,
    pub port:         u16,
    pub lora_adapter: Option<String>,
    pub venv_path:    Option<String>,
    /// Base directory for custom_words.txt, training/, lora_adapter/, training_sentences.txt.
    /// Defaults to config_dir() (~/.config/voicetserver/) when not set.
    pub data_dir: PathBuf,
    // Runtime-adjustable
    pub delay:             usize,
    pub silence_threshold: f32,
    pub silence_flush:     usize,
    pub min_speech:        usize,
    pub rms_ema:           f32,
    pub fuzzy_hotwords:    bool,
    pub fuzzy_max_ratio:   f32,
    pub german_prime:      bool,
    pub context_biasing:   bool,
    /// Qwen transcription language (default "German").
    pub language:          String,
    // Log file settings
    pub log_file:      Option<String>,
    pub log_keep_days: u32,
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

pub fn config_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config").join("voicetserver")
}

pub fn config_file_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn pid_file_path() -> PathBuf {
    config_dir().join("voicetserver.pid")
}

// ---------------------------------------------------------------------------
// Workspace paths (data dir — configurable base for user data files)
// ---------------------------------------------------------------------------

/// All data-file paths derived from a single configurable base directory.
///
/// The base defaults to `config_dir()` (~/.config/voicetserver/) so existing
/// setups are fully backward compatible.  Set `data_dir` in config.toml or
/// `--data-dir` on the CLI to move training data, LoRA adapter output,
/// custom_words.txt, and training_sentences.txt to a different location.
///
/// Server-internal files (config.toml, voicetserver.pid, logs/) always live
/// in `config_dir()` regardless of this setting.
#[derive(Clone, Debug)]
pub struct WorkspacePaths {
    pub custom_words:       PathBuf,  // {data_dir}/custom_words.txt
    pub training_dir:       PathBuf,  // {data_dir}/training/
    pub training_audio_dir: PathBuf,  // {data_dir}/training/audio/
    pub training_pairs:     PathBuf,  // {data_dir}/training/pairs.jsonl
    pub training_sentences: PathBuf,  // {data_dir}/training_sentences.txt
    pub lora_output_dir:    PathBuf,  // {data_dir}/lora_adapter/
    pub edit_log:           PathBuf,  // {data_dir}/edit_log.jsonl
    pub review_dir:         PathBuf,  // {data_dir}/training/review/ (candidate WAVs)
    pub review_jsonl:       PathBuf,  // {data_dir}/training/review.jsonl
}

impl WorkspacePaths {
    pub fn new(data_dir: &Path) -> Self {
        let training = data_dir.join("training");
        WorkspacePaths {
            custom_words:       data_dir.join("custom_words.txt"),
            training_dir:       training.clone(),
            training_audio_dir: training.join("audio"),
            training_pairs:     training.join("pairs.jsonl"),
            training_sentences: data_dir.join("training_sentences.txt"),
            lora_output_dir:    data_dir.join("lora_adapter"),
            edit_log:           data_dir.join("edit_log.jsonl"),
            review_dir:         training.join("review"),
            review_jsonl:       training.join("review.jsonl"),
        }
    }
}

// ---------------------------------------------------------------------------
// Bootstrap / load / save
// ---------------------------------------------------------------------------

const CONFIG_TEMPLATE: &str = r#"# voicetserver configuration
# All fields are optional — omit to use the compiled default.
# Restart required for: model_dir, qwen_model_dir, language, device, port, bind_addr, tls_cert, tls_key, lora_adapter
# Runtime-adjustable via PATCH /config: delay, silence_threshold, silence_flush, min_speech, rms_ema, fuzzy_hotwords, fuzzy_max_ratio, german_prime, context_biasing

# model_dir = "/path/to/Voxtral-Mini-4B-Realtime"
# qwen_model_dir = "/path/to/Qwen3-ASR-0.6B"   # second engine (model.safetensors, config.json,
#                                               # tokenizer.json); omit to disable qwen
# language = "German"                           # qwen transcription language (Voxtral auto-detects)
# bind_addr = "127.0.0.1"
# port = 8765
# tls_cert = "/etc/tailscale/certs/host.crt"
# tls_key  = "/etc/tailscale/certs/host.key"
# device = 0
# venv_path = "/mnt/ssdupl/voicetserver-venv"   # Python venv for LoRA training
# data_dir = "/path/to/data"   # base for custom_words.txt, training/, lora_adapter/, training_sentences.txt
#                               # defaults to ~/.config/voicetserver/ when unset

# delay = 6
# silence_threshold = 0.006
# silence_flush = 20
# min_speech = 15
# rms_ema = 0.3

# Fuzzy phonetic hotword correction (snaps transcribed words onto plain terms
# in custom_words.txt that sound the same — Kölner Phonetik + Levenshtein).
# fuzzy_hotwords = true     # set false to disable fuzzy snapping
# fuzzy_max_ratio = 0.34    # max normalized edit distance (0..1); lower = stricter

# Experimental: prime the decoder prefill with a few German text tokens instead of
# pure PADs to bias the model's language prior toward German (A/B test flag).
# german_prime = false

# Qwen prompt biasing: inject custom words + per-session hotwords/patient context
# into the system prompt (qwen engine only; Voxtral has no biasing mechanism).
# context_biasing = true

# log_file = "/path/to/voicetserver.log"   # default: ~/.config/voicetserver/logs/voicetserver.log
# log_keep_days = 7

# api_key = ""   # auto-generated on first server start; copy into the userscript Einstellungen tab
"#;

/// Create ~/.config/voicetserver/ and a commented config.toml template if not present.
pub fn bootstrap_config_dir() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let dir = config_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    let path = config_file_path();
    if !path.exists() {
        std::fs::write(&path, CONFIG_TEMPLATE)?;
        // 0600 — the config file holds the API key.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        eprintln!("Created config: {}", path.display());
    }
    Ok(())
}

/// Parse ~/.config/voicetserver/config.toml.
/// Returns ConfigFile::default() if the file does not exist.
pub fn load_config_file() -> Result<ConfigFile> {
    let path = config_file_path();
    if !path.exists() {
        return Ok(ConfigFile::default());
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("config file ({}): {}", path.display(), e))?;
    toml::from_str::<ConfigFile>(&raw)
        .map_err(|e| anyhow::anyhow!("config file parse error ({}): {}", path.display(), e))
}

/// Serialize and write the config file to disk.
pub fn save_config_file(cfg: &ConfigFile) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let path = config_file_path();
    let toml_str = toml::to_string_pretty(cfg)
        .map_err(|e| anyhow::anyhow!("config serialize error: {}", e))?;
    std::fs::write(&path, toml_str)
        .map_err(|e| anyhow::anyhow!("config write error ({}): {}", path.display(), e))?;
    // 0600 — the config file holds the API key.
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| anyhow::anyhow!("config chmod error ({}): {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// API key management
// ---------------------------------------------------------------------------

/// Read 16 random bytes from /dev/urandom and hex-encode them.
fn generate_api_key() -> String {
    use std::io::Read;
    let mut bytes = [0u8; 16];
    let mut f = std::fs::File::open("/dev/urandom")
        .expect("cannot open /dev/urandom for API key generation");
    f.read_exact(&mut bytes)
        .expect("cannot read /dev/urandom for API key generation");
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Ensure the config file has an API key. Generates and persists one if missing.
/// Returns the key. Prints it to stdout when newly generated.
pub fn ensure_api_key(cfg: &mut ConfigFile) -> Result<String> {
    if let Some(ref key) = cfg.api_key {
        if !key.is_empty() {
            return Ok(key.clone());
        }
    }
    let key = generate_api_key();
    cfg.api_key = Some(key.clone());
    save_config_file(cfg)
        .map_err(|e| anyhow::anyhow!("Failed to save generated API key: {}", e))?;
    // Box interior is 55 chars wide: 2 + 19 ("Generated API key:" + space) + 32 (hex key) + 2.
    println!("┌───────────────────────────────────────────────────────┐");
    println!("│  Generated API key: {}  │", key);
    println!("│  Paste it into the userscript Einstellungen tab.      │");
    println!("└───────────────────────────────────────────────────────┘");
    Ok(key)
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

/// Merge a required-value field: CLI > config file > compiled default.
fn merge_val<T: Clone>(
    cli_opt: &Option<T>,
    file_opt: &Option<T>,
    default: T,
) -> (T, ValueSource) {
    if let Some(v) = cli_opt {
        return (v.clone(), ValueSource::CliArg);
    }
    if let Some(v) = file_opt {
        return (v.clone(), ValueSource::ConfigFile);
    }
    (default, ValueSource::Default)
}

/// Merge an optional-string field (tls_cert, tls_key, lora_adapter).
fn merge_opt_str(
    cli: &Option<String>,
    file: &Option<String>,
) -> (Option<String>, ValueSource) {
    if cli.is_some() {
        return (cli.clone(), ValueSource::CliArg);
    }
    if file.is_some() {
        return (file.clone(), ValueSource::ConfigFile);
    }
    (None, ValueSource::Default)
}

/// Merge CLI args + config file + compiled defaults into a MergedConfig.
pub fn merge(cli: &crate::Cli, file: &ConfigFile) -> MergedConfig {
    let (model_dir_val, model_dir_src) =
        merge_val(&cli.model_dir, &file.model_dir, ".".to_string());
    let (bind_addr_val, bind_addr_src) =
        merge_val(&cli.bind_addr, &file.bind_addr, "127.0.0.1".to_string());
    let (device, _) = merge_val(&cli.device, &file.device, 0usize);
    let (port, _)   = merge_val(&cli.port,   &file.port,   8765u16);
    // Default lookahead raised 4 → 6 (≈480 ms): measurably better accuracy for
    // ~160 ms extra latency. Runtime-tunable via PATCH /config / Einstellungen.
    let (delay, _)             = merge_val(&cli.delay,             &file.delay,             6usize);
    let (silence_threshold, _) = merge_val(&cli.silence_threshold, &file.silence_threshold, 0.006f32);
    let (silence_flush, _)     = merge_val(&cli.silence_flush,     &file.silence_flush,     20usize);
    let (min_speech, _)        = merge_val(&cli.min_speech,        &file.min_speech,        15usize);
    let (rms_ema, _)           = merge_val(&cli.rms_ema,           &file.rms_ema,           0.3f32);
    // Fuzzy phonetic correction: config-file + runtime only (no CLI flag).
    let fuzzy_hotwords  = file.fuzzy_hotwords.unwrap_or(true);
    let fuzzy_max_ratio = file.fuzzy_max_ratio.unwrap_or(0.34f32);
    // Experimental German prefill priming: config-file + runtime only (no CLI flag).
    let german_prime = file.german_prime.unwrap_or(false);
    // Qwen prompt biasing: config-file + runtime only (no CLI flag), like fuzzy_hotwords.
    let context_biasing = file.context_biasing.unwrap_or(true);
    let (language, _) = merge_val(&cli.language, &file.language, "German".to_string());

    let (tls_cert_val, tls_cert_src) = merge_opt_str(&cli.tls_cert, &file.tls_cert);
    let (tls_key_val,  tls_key_src)  = merge_opt_str(&cli.tls_key,  &file.tls_key);
    let (qwen_model_dir_val, qwen_model_dir_src) =
        merge_opt_str(&cli.qwen_model_dir, &file.qwen_model_dir);
    let lora_adapter = cli.lora_adapter.clone().or_else(|| file.lora_adapter.clone());
    let venv_path    = cli.venv_path.clone().or_else(|| file.venv_path.clone());
    let log_file      = cli.log_file.clone().or_else(|| file.log_file.clone());
    let log_keep_days = cli.log_keep_days.or(file.log_keep_days).unwrap_or(7);
    let data_dir = cli.data_dir.clone()
        .or_else(|| file.data_dir.clone())
        .map(PathBuf::from)
        .unwrap_or_else(config_dir);

    MergedConfig {
        model_dir: Sourced { value: model_dir_val, source: model_dir_src },
        bind_addr: Sourced { value: bind_addr_val, source: bind_addr_src },
        tls_cert:  Sourced { value: tls_cert_val,  source: tls_cert_src  },
        tls_key:   Sourced { value: tls_key_val,   source: tls_key_src   },
        qwen_model_dir: Sourced { value: qwen_model_dir_val, source: qwen_model_dir_src },
        device,
        port,
        lora_adapter,
        venv_path,
        data_dir,
        delay,
        silence_threshold,
        silence_flush,
        min_speech,
        rms_ema,
        fuzzy_hotwords,
        fuzzy_max_ratio,
        german_prime,
        context_biasing,
        language,
        log_file,
        log_keep_days,
    }
}

// ---------------------------------------------------------------------------
// Source-tagged error helper
// ---------------------------------------------------------------------------

/// Return a human-readable label for the source of a field's value.
pub fn source_tag(source: ValueSource, field: &str) -> String {
    match source {
        ValueSource::CliArg    => format!("--{} (CLI)", field.replace('_', "-")),
        ValueSource::ConfigFile => format!("{} (from config file)", field),
        ValueSource::Default    => format!("{} (default)", field),
    }
}
