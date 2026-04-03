# Build

## Dev (Docker, no GPU)

```bash
apt install -y pkg-config build-essential libssl-dev
cargo build
cargo check   # fast syntax check
```

## Production (GPU server, Ubuntu + CUDA 12.x)

```bash
apt install -y pkg-config build-essential libssl-dev
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
# Compute capability: Ada Lovelace (RTX 4000/4090) = 89, Ampere (RTX 3000) = 86, Turing (RTX 2000) = 75
```

Binary: `target/release/voicetserver`

# Test

## Offline WAV

```bash
./target/release/voicetserver --model-dir /path/to/Voxtral-Mini-4B-Realtime audio.wav
```

## WebSocket server (dev, no TLS)

```bash
./target/release/voicetserver --model-dir /path/to/Voxtral-Mini-4B-Realtime
# Listens on ws://127.0.0.1:8765/asr
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

# Features

- `default = []` — no CUDA; dev builds always work without nvcc
- `cuda` — enables candle CUDA backend + flash-attn; required for production inference

All CUDA-only code gated behind `#[cfg(feature = "cuda")]` in source files.
`build.rs` checks `CARGO_FEATURE_CUDA` env var (not `#[cfg]` — build scripts don't see feature flags).

# Model files

Required in `--model-dir`:
- `consolidated.safetensors` (~8.9 GB) — model weights
- `tekken.json` — tokenizer
- `mel_filters.bin` — precomputed Slaney mel filterbank (128×201 f32 LE); copy from `assets/mel_filters.bin` or regenerate with `python3 scripts/generate_mel_filters.py <dir>`

Download from HuggingFace: `mistralai/Voxtral-Mini-4B-Realtime-2602`

# Architecture

- `src/main.rs` — tokio entry point; `VoxtralModel` (Arc-shared) + `ModelInner` (tokio::sync::Mutex)
- `src/streaming.rs` — `StreamingState`: KV caches, SilenceDetector, mel buffer; `process_chunk_sync`
- `src/audio.rs` — raw f32 LE PCM decode from WebSocket binary frames
- `src/server.rs` — axum 0.8 WebSocket handler + TLS via axum-server 0.8 + rustls
- `src/encoder.rs` / `src/decoder.rs` — Voxtral model; flash-attn behind `#[cfg(feature = "cuda")]`
- `src/session.rs` — Phase 2 stub (patient session vocabulary)
- `src/macros.rs` — Phase 4 stub (macro expansion)
- `candle-fork/` — vendored Candle ML framework; do not update without reason

GPU lock: single `tokio::sync::Mutex<ModelInner>` wrapping both enc and dec.
Acquire → do all sync Candle work → release before every `.await`.
Use `let inner = &mut *guard;` to enable disjoint field borrows (enc, dec).

# Browser client

`schmidispeech.user.js` — Violentmonkey userscript.
Sends raw 16kHz mono f32 LE PCM over WebSocket binary frames.
Receives `{"type":"partial","text":"..."}` / `{"type":"final","text":"..."}`.

Server URL stored via `GM_setValue('server_url', 'wss://...:8765/asr')`.

# Folder structure

- `src/` — Rust source
- `assets/mel_filters.bin` — precomputed mel filterbank (commit this)
- `config/medical_terms_de.txt` — German medical seed terms
- `docs/ubuntu_dependencies.md` — package list by phase
- `scripts/generate_mel_filters.py` — mel filterbank generator (pure Python, no deps)
- `schmidispeech.user.js` — browser userscript
- `candle-fork/` — vendored Candle
- `README.private.md` — private deploy notes (gitignored)

# Versioning

Semantic versioning in `Cargo.toml`. Increment patch on each push.
