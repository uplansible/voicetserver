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

Versioned local copies: `releases/v<version>/voicetserver` (not committed, gitignored)

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

Required in `--model-dir`:
- `consolidated.safetensors` (~8.9 GB) — model weights
- `tekken.json` — tokenizer
- `mel_filters.bin` — precomputed Slaney mel filterbank (128×201 f32 LE); copy from `assets/mel_filters.bin` or regenerate with `python3 scripts/generate_mel_filters.py <dir>`

Download from HuggingFace: `mistralai/Voxtral-Mini-4B-Realtime-2602`

## Model language behaviour

Voxtral Mini 4B Realtime has **no language token mechanism**. It auto-detects language from the audio signal. There is no way to force a specific language — the model will transcribe in whatever language it hears. Confirmed by inspecting `tekken.json` (no `added_tokens`) and the reference C implementation (antirez/voxtral.c).

Known special token IDs (all others in 0–999 are skipped as audio control tokens):
- `BOS = 1`, `EOS = 2`
- `AUDIO = 24`, `BEGIN_AUDIO = 25` — audio frame boundary markers
- `STREAMING_PAD = 32` — silence/padding
- `STREAMING_WORD = 33` — word boundary control token

# Architecture

- `src/main.rs` — tokio entry point; `VoxtralModel` (Arc-shared) + `ModelInner` (tokio::sync::Mutex); inline `server` module with all HTTP + WebSocket handlers
- `src/streaming.rs` — `StreamingState`: KV caches, SilenceDetector, mel buffer; `process_chunk_sync`
- `src/audio.rs` — raw f32 LE PCM decode from WebSocket binary frames
- `src/config.rs` — config file loading (`~/.config/voicetserver/config.toml`), CLI+file merge, source-tagged error messages
- `src/words.rs` — `WordsCorrector`: aho-corasick text replacement from `custom_words.txt`
- `src/settings.rs` — `SharedSettings` (atomic runtime params) + `StartupSnapshot`
- `src/encoder.rs` / `src/decoder.rs` — Voxtral model; flash-attn behind `#[cfg(feature = "cuda")]`
- `src/session.rs` — Phase 2 stub (patient session vocabulary)
- `src/macros.rs` — Phase 4 stub (macro expansion)
- `candle-fork/` — vendored Candle ML framework; do not update without reason

GPU lock: single `tokio::sync::Mutex<ModelInner>` wrapping both enc and dec.
Acquire → do all sync Candle work → release before every `.await`.
Use `let inner = &mut *guard;` to enable disjoint field borrows (enc, dec).

# Config file

`~/.config/voicetserver/config.toml` — created automatically with commented template on first run.
CLI args override config file values. Unknown fields in the config file are silently ignored.

Runtime-adjustable via `PATCH /config` (no restart needed): `delay`, `silence_threshold`, `silence_flush`, `min_speech`, `rms_ema`.

Startup-only (require restart): `model_dir`, `device`, `port`, `bind_addr`, `tls_cert`, `tls_key`, `lora_adapter`.

# Custom words

`~/.config/voicetserver/custom_words.txt` — one entry per line:
- `wrong=correct` — replacement pair: every occurrence of `wrong` in transcribed text is replaced with `correct`
- `PlainTerm` — stored as-is, no correction effect yet (Phase 3 vocab boosting)
- `# comment` — ignored

The aho-corasick automaton is built at startup and hot-reloaded whenever `POST /words` updates the file. No restart required.

Example: add `Migration=Miktion` to correct a model phonetic confusion.

# HTTP API

All endpoints support CORS (`allow_origin: *`).

- `GET /health` — `{"status":"ready","connections":N}`
- `GET /config` — current settings (runtime + startup snapshot)
- `PATCH /config` — update settings; runtime params apply immediately, startup params written to file
- `GET /words` — `{"words":[...]}`
- `POST /words` — `{"add":[...],"remove":[...]}` — updates file + rebuilds corrector
- `ws[s]://host:port/asr` — WebSocket audio stream (raw f32 LE PCM 16kHz mono)

### Training (Phase 2 — LoRA voice calibration)

**Sentence management:**
- `GET /training/sentences` — returns `{"sentences":[{"text","recorded","pair_ids":[…]}]}` annotated with recording status; auto-creates `training_sentences.txt` on first call
- `POST /training/sentence` — `{"text":"…"}` — append new sentence to file
- `PATCH /training/sentence` — `{"old":"…","new":"…"}` — replace one sentence in file
- `DELETE /training/sentence` — `{"text":"…"}` — remove one sentence from file

**Pair collection:**
- `POST /training/pair?text=<url-encoded>` — body: raw f32 LE PCM at 16kHz (always; MediaRecorder path dropped — container bytes caused static); saves 16-bit WAV + appends to `pairs.jsonl`; returns `{"id","duration_s","count"}`
- `GET /training/pairs` — `{"pairs":[{"id","text","duration_s"}]}` sorted by id
- `GET /training/audio/{id}` — serve recorded WAV for playback (numeric id only)
- `DELETE /training/pair/{id}` — remove WAV file + JSONL entry
- `GET /training` — `{"count":N,"duration_sec":F}` summary
- `DELETE /training/pairs` — remove all collected training data

**LoRA:**
- `POST /training/run` — spawn `tools/train_lora.py` subprocess; 202 if started, 409 if already running
- `GET /training/status` — `{"status":"idle"|"running"|"done"|"error","log":[...]}`

Training data stored in `~/.config/voicetserver/training/audio/*.wav` + `pairs.jsonl`.
LoRA adapter output: `~/.config/voicetserver/lora_adapter/` (set `lora_adapter` in config to load at startup).

Python venv install (run once — put venv on a large partition if root fs is tight):
```bash
VENV=/path/to/venv          # e.g. /mnt/ssdupl/voicetserver-venv
python3 -m venv $VENV
mkdir -p $VENV/tempdir
# torch must come from the PyTorch CUDA index (replace cu128 with your CUDA version)
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu128
# remaining deps from PyPI
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  safetensors mistral-common numpy tqdm packaging
```
Deploy training script next to the binary:
```bash
mkdir -p ~/.local/bin/tools && cp tools/train_lora.py ~/.local/bin/tools/
```
Then set `venv_path = "/path/to/venv"` in `~/.config/voicetserver/config.toml`.

### LoRA adapter

`src/lora.rs` — loads `adapter_model.safetensors` + `adapter_config.json` from the adapter dir.
Applied as runtime delta: `proj_output += scale * lora_b @ lora_a @ input` (no weight merging).
Weight key format: `layers.{i}.attention.{wq,wk,wv,wo}.lora_{a,b}.weight`.
Scale = `lora_alpha / r` from `adapter_config.json`.

# Browser client

`schmidispeech.user.js` — Violentmonkey userscript.
Sends raw 16kHz mono f32 LE PCM over WebSocket binary frames.
Receives `{"type":"partial","text":"..."}` / `{"type":"final","text":"..."}` / `{"type":"error","text":"..."}`.

Server URL and hotkey stored via `GM_setValue` (`server_url`, `hotkey`).
Default hotkey: `Ctrl+Shift+D` (configurable via right-click menu → Client tab).
Text is inserted live at cursor on each `final`; trailing partial inserted on stop.
Falls back to clipboard if no editable element was captured.

Right-click → four tabs: **Client** (URL, hotkey), **Server** (runtime params, custom words), **Aufnehmen** (record calibration sentences), **Paare** (review pairs, delete, launch LoRA).

**Aufnehmen tab:** shows only unrecorded sentences (recorded ones are hidden). Navigate ◀/▶, edit/add/remove sentences inline, record via ScriptProcessor (raw f32 LE PCM, same as ASR path), preview recording client-side before saving (Web Audio API, no server round-trip), save to commit pair.

**Paare tab:** scrollable list of all recorded pairs (id, text, duration) with per-pair ▶ playback and ✕ delete. LoRA training run + status log.

## Silence / final message behaviour

`ChunkOutput::Silence` fires after silence threshold; server calls `take_text_buf()` which
returns and clears the accumulated text. Do NOT call `self.text_buf.clear()` before this —
it would cause every silence-triggered final to send empty text.

Custom word correction is applied to the text returned by `take_text_buf()` before sending.

# Folder structure

- `src/` — Rust source
- `assets/mel_filters.bin` — precomputed mel filterbank (commit this)
- `config/medical_terms_de.txt` — German medical seed terms (source list; not auto-loaded — add entries to `~/.config/voicetserver/custom_words.txt`)
- `docs/ubuntu_dependencies.md` — package list by phase
- `scripts/generate_mel_filters.py` — mel filterbank generator (pure Python, no deps)
- `schmidispeech.user.js` — browser userscript
- `candle-fork/` — vendored Candle
- `README.private.md` — private deploy notes (gitignored)
- `releases/` — local versioned binary copies (gitignored)

# Versioning

Semantic versioning in `Cargo.toml`. Increment patch on each push.
