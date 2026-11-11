# voicetserver — unified two-engine ASR server

One binary serving **both** ASR models on one port (8765), one TLS cert, one API key, one
config file, one data dir:

- **Voxtral Mini 4B Realtime** (~8 GB VRAM, BF16) — primary engine, always loaded.
- **Qwen3-ASR-0.6B** (~1.5 GB VRAM) via vendored `qwen3-asr-rs` — optional second engine;
  enabled by setting `qwen_model_dir` in config
  (unset = disabled, saves the VRAM).

A WebSocket session picks its engine with `?model=voxtral|qwen` (absent/empty = voxtral). The
value is validated everywhere `?model=` is accepted — an unrecognised one is a 400 (HTTP) or an
error frame (WS), never a silent fall-back to voxtral. Each
engine has its own GPU lock, so sessions on different models run concurrently. Both engines
share the training-pair pool, custom words, and the whole HTTP API; LoRA adapters are
strictly per-model (different weight-key formats).

# Build

> **Dev server has CUDA installed** — always use the CUDA build command below.

```bash
export CUDA_PATH=/usr/local/cuda && export PATH=$CUDA_PATH/bin:$PATH && CUDA_COMPUTE_CAP=89 cargo build --release --features cuda 2>&1
```

`cargo check` still works for fast syntax checking without full compile.

**CUTLASS fetch (`cudaforge` ≥ 0.1.6):** the flash-attn build script clones CUTLASS from GitHub
into `~/.cudaforge/git/checkouts/` (overridable via `CUDAFORGE_HOME`), so the first CUDA build
needs network access. In a sandbox where `~/.cudaforge` is not writable the build fails with
`GitOperationFailed("Failed to create lock file: Read-only file system")` — prefix the build
command with `CUDAFORGE_HOME=$PWD/target/cudaforge`.

## Without CUDA (Docker / no GPU)

```bash
apt install -y pkg-config build-essential libssl-dev
cargo build
cargo check   # fast syntax check
```

## CUDA (GPU server, Ubuntu + CUDA 12.x)

```bash
apt install -y pkg-config build-essential libssl-dev
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
# Compute capability: Ada Lovelace (RTX 4000/4090) = 89, Ampere (RTX 3000) = 86, Turing (RTX 2000) = 75
```

Binary: `target/release/voicetserver`

## CUDA driver loading (cudarc `dynamic-loading`)

`candle-fork/Cargo.toml` uses cudarc's **`dynamic-loading`** feature (not `dynamic-linking`):
the GPU driver (`libcuda.so.1`) is `dlopen`ed lazily on first inference rather than linked at
load time. This lets a CUDA build run `--version` / `--help` and start far enough to print
errors on a machine with no GPU/driver. Only the toolkit runtime (`libcudart.so.12`) remains a
load-time dependency, so the CUDA toolkit must still be installed where the binary runs.

Because `dynamic-loading` no longer emits the CUDA library search path, `build.rs`
(`emit_cuda_link_search()`) re-adds `$CUDA_PATH/lib64` (and `/lib`) so candle-kernels /
flash-attn can still resolve `-lcudart`.

`--version` prints `voicetserver <ver> (CUDA <toolkit-ver>)` for CUDA builds or
`voicetserver <ver> (CPU)` for CPU builds. The variant comes from `VOICETSERVER_BUILD_VARIANT`,
emitted by `build.rs` (CUDA version parsed from `nvcc --version`).

Versioned local copies: `releases/v<version>/voicetserver` (not committed, gitignored)

## Reference — moved out

Long-form sections live in **`CLAUDE-reference.md`** (local, not auto-loaded). Read the matching section before working on that topic:

- **Updating prebuilt binaries in tools/**
- **Model language behaviour** — Experimental German prefill priming (`german_prime`)
- **Auto-unload for training (single-GPU VRAM reuse)**
- **Log file**
- **Fuzzy phonetic hotword correction**
- **Abbreviation / acronym expansion (letter-name pass)** — Training (Phase 2 — LoRA voice calibration); LoRA adapter
- **WebSocket reconnect / mic lifecycle**
- **Silence / final message behaviour**
- **Shared LoRA adapter for two users**
- **Concurrent sessions**
- **Proposed per-user folder layout (when `--data-dir` is used)**
- **Key files to modify for full per-user support**

## Offline WAV

```bash
./target/release/voicetserver --model-dir /path/to/Voxtral-Mini-4B-Realtime audio.wav
```

Offline WAV mode is **Voxtral-only** — the qwen engine is server-mode only (use a WS
session with `?model=qwen`, e.g. the userscript's Diktate re-transcribe button, for qwen
A/B runs on the same audio).

## WebSocket server (dev, no TLS)

```bash
./target/release/voicetserver --model-dir /path/to/Voxtral-Mini-4B-Realtime \
  --qwen-model-dir /path/to/Qwen3-ASR-0.6B   # optional second engine
# Listens on ws://127.0.0.1:8765/asr (?model=voxtral|qwen picks the engine)
curl http://127.0.0.1:8765/health
```

## WebSocket server (production, TLS)

```bash
./target/release/voicetserver \
  --model-dir /path/to/Voxtral-Mini-4B-Realtime \
  --bind-addr 0.0.0.0 \
  --tls-cert /etc/tailscale/certs/<host>.crt \
  --tls-key  /etc/tailscale/certs/<host>.key
```

All settings can also be stored in `~/.config/voicetserver/config.toml` instead of passing CLI flags.

# Features

- `default = []` — no CUDA; dev builds always work without nvcc
- `cuda` — enables candle CUDA backend + flash-attn; required for production inference

All CUDA-only code gated behind `#[cfg(feature = "cuda")]` in source files.
`build.rs` checks `CARGO_FEATURE_CUDA` env var (not `#[cfg]` — build scripts don't see feature flags).

## Compute dtype

`BF16` on CUDA, `F32` on CPU — selected automatically at runtime based on device.
CPU candle backend does not support BF16 matmul; using BF16 on CPU causes
`unsupported dtype BF16 for op matmul` on every ASR session.

# Model files

## Voxtral (required, `--model-dir` / `model_dir`)

- `consolidated.safetensors` (~8.9 GB) — model weights
- `tekken.json` — tokenizer
- `mel_filters.bin` — precomputed Slaney mel filterbank (128×201 f32 LE); copy from `assets/mel_filters.bin` or regenerate with `python3 scripts/generate_mel_filters.py <dir>`

Download from HuggingFace: `mistralai/Voxtral-Mini-4B-Realtime-2602`

## Qwen3-ASR (optional, `--qwen-model-dir` / `qwen_model_dir`)

- `model.safetensors` (~1.8 GB) — Qwen3-ASR-0.6B weights
- `config.json` — model config
- `tokenizer.json` — **not published on HuggingFace**; `tools/install.sh` generates it via
  `transformers` (`AutoTokenizer.from_pretrained(dir, trust_remote_code=True).save_pretrained(dir)`)

Download from HuggingFace: `Qwen/Qwen3-ASR-0.6B`. When `qwen_model_dir` is unset the qwen
engine is disabled — `/asr?model=qwen` sessions get an error frame and `GET /config` reports
`"models":["voxtral"]`.

### Runtime-switchable second size (0.6B <-> 1.7B)

`qwen3-asr-rs` is fully `config.json`-driven — layer counts, hidden dims, attention heads all
come from the checkpoint, so any Qwen3-ASR-architecture checkpoint (0.6B, 1.7B, …) works by
pointing at its directory. On top of that, the server supports **two named config slots**
("primary" = `qwen_model_dir`/`qwen_model_size`, "secondary" =
`qwen_model_dir_alt`/`qwen_model_size_alt`), only one of which is ever loaded — switching is a
runtime hot-swap (`POST /qwen/switch?slot=primary|secondary`), not a second engine running
concurrently (VRAM: Voxtral ~8 GB + Qwen 1.7B ~5 GB already leaves little headroom on a 16 GB
card for a third model). Each slot has its own LoRA adapter (`lora_adapter_qwen` /
`lora_adapter_qwen_alt` — weight shapes differ per size, so an adapter is never reusable across
slots) and its own training-output dir (`lora_adapter_qwen/` / `lora_adapter_qwen_alt/` under
`data_dir`). `qwen_active_slot` (persisted on every switch) decides which slot loads at
startup; `POST /training/run?model=qwen` always targets whichever slot is *currently* active,
not always primary. `tools/install.sh` downloads all three checkpoints by default (Voxtral +
Qwen primary + Qwen secondary — the second-size prompt defaults to yes) and, on a re-run,
backfills whichever files are missing from an already-configured model directory instead of
trusting its mere existence, so running the installer again after a partial/interrupted
download or as an "update" fills the gaps (`fetch_missing_files`/`generate_qwen_tokenizer`
helpers, reused for Voxtral, the primary Qwen slot, and the secondary slot); the userscript's
single model dropdown (see Browser client, below) offers Voxtral / Qwen primary / Qwen
secondary and is populated from `GET /config`'s `models`/`qwen_model_dir_alt` fields.

## Two-process watchdog (interactive mode)

Fork happens **before** the tokio runtime starts (forking a multi-threaded process is unsafe).

- **Parent (watchdog):** sets terminal input to raw-minus-OPOST mode (keeps `\n`→`\r\n` so child output renders correctly), spawns stdin-reading thread, watches for `d` / Ctrl+C. Exits when 'd' is pressed or Ctrl+C.
- **Child (server):** runs normally; stdout/stderr go to the terminal. On SIGUSR1 (sent by parent on 'd') → `dup2` stdout/stderr to log file.

Key: `OPOST` must NOT be cleared in the watchdog's termios setup. `cfmakeraw` clears it, which corrupts the child's output. Set `ICANON`, `ECHO`, `ISIG` flags manually.

## --detach flag

Pre-tokio `fork()` → parent prints "Detached. PID X." and exits → child calls `setsid()`, redirects stdin/stdout/stderr to log file, runs server.

## PID file

Written at `~/.config/voicetserver/voicetserver.pid` immediately after fork (before model loading). Deleted on SIGTERM. Prevents duplicate instances: startup checks PID file and errors if the process is still running.

## --stop flag

Reads PID file, verifies `/proc/<pid>/cmdline` contains `voicetserver` (guards against a stale
PID being reused by an unrelated process — refuses to SIGTERM if it doesn't match), sends SIGTERM,
polls `/proc/<pid>` for up to 5 s, deletes PID file.

## Commit workflow

Single `main` branch — no `testing` branch; "ok stable" does not apply here.

1. After the code change is complete, bump the patch version in `Cargo.toml` (+0.0.1) **before** compiling.
2. Run `cargo build --release` so the user can test.
3. On commit ("ok go"): commit and push to `main` (version already bumped — do not bump again).
