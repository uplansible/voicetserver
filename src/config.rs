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
    // Runtime-adjustable via PATCH /config
    pub delay:             Option<usize>,
    pub silence_threshold: Option<f32>,
    pub silence_flush:     Option<usize>,
    pub min_speech:        Option<usize>,
    pub rms_ema:           Option<f32>,
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
        }
    }
}

// ---------------------------------------------------------------------------
// Bootstrap / load / save
// ---------------------------------------------------------------------------

const CONFIG_TEMPLATE: &str = r#"# voicetserver configuration
# All fields are optional — omit to use the compiled default.
# Restart required for: model_dir, device, port, bind_addr, tls_cert, tls_key, lora_adapter
# Runtime-adjustable via PATCH /config: delay, silence_threshold, silence_flush, min_speech, rms_ema

# model_dir = "/path/to/Voxtral-Mini-4B-Realtime"
# bind_addr = "127.0.0.1"
# port = 8765
# tls_cert = "/etc/tailscale/certs/host.crt"
# tls_key  = "/etc/tailscale/certs/host.key"
# device = 0
# venv_path = "/mnt/ssdupl/voicetserver-venv"   # Python venv for LoRA training
# data_dir = "/path/to/data"   # base for custom_words.txt, training/, lora_adapter/, training_sentences.txt
#                               # defaults to ~/.config/voicetserver/ when unset

# delay = 4
# silence_threshold = 0.006
# silence_flush = 20
# min_speech = 15
# rms_ema = 0.3

# log_file = "/path/to/voicetserver.log"   # default: ~/.config/voicetserver/logs/voicetserver.log
# log_keep_days = 7
"#;

/// Create ~/.config/voicetserver/ and a commented config.toml template if not present.
pub fn bootstrap_config_dir() -> Result<()> {
    let dir = config_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    let path = config_file_path();
    if !path.exists() {
        std::fs::write(&path, CONFIG_TEMPLATE)?;
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
    let path = config_file_path();
    let toml_str = toml::to_string_pretty(cfg)
        .map_err(|e| anyhow::anyhow!("config serialize error: {}", e))?;
    std::fs::write(&path, toml_str)
        .map_err(|e| anyhow::anyhow!("config write error ({}): {}", path.display(), e))
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
    let (delay, _)             = merge_val(&cli.delay,             &file.delay,             4usize);
    let (silence_threshold, _) = merge_val(&cli.silence_threshold, &file.silence_threshold, 0.006f32);
    let (silence_flush, _)     = merge_val(&cli.silence_flush,     &file.silence_flush,     20usize);
    let (min_speech, _)        = merge_val(&cli.min_speech,        &file.min_speech,        15usize);
    let (rms_ema, _)           = merge_val(&cli.rms_ema,           &file.rms_ema,           0.3f32);

    let (tls_cert_val, tls_cert_src) = merge_opt_str(&cli.tls_cert, &file.tls_cert);
    let (tls_key_val,  tls_key_src)  = merge_opt_str(&cli.tls_key,  &file.tls_key);
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
