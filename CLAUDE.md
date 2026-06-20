# Build

> **Dev server has CUDA installed** — always use the CUDA build command below.

```bash
export CUDA_PATH=/usr/local/cuda && export PATH=$CUDA_PATH/bin:$PATH && CUDA_COMPUTE_CAP=89 cargo build --release --features cuda 2>&1
```

`cargo check` still works for fast syntax checking without full compile.

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

## Updating prebuilt binaries in tools/

After building, copy the binary to `tools/` so the installer can offer it as a prebuilt
option. Currently only the CUDA (GPU) binary is shipped; add `tools/voicetserver` for a CPU
prebuilt if/when needed.

```bash
# CUDA prebuilt (run on this machine)
export CUDA_PATH=/usr/local/cuda && export PATH=$CUDA_PATH/bin:$PATH
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
cp target/release/voicetserver tools/voicetserver-cuda
git add tools/voicetserver-cuda && git commit -m "chore: update prebuilt CUDA binary (vX.Y.Z)"
```

# Install

`tools/install.sh` — interactive installer. Clone the repo and run it:

```bash
git clone <repo> && cd voicetserver
./tools/install.sh
```

It: creates a Python venv (for LoRA training), installs torch (auto-detects the CUDA index
tag) + `safetensors mistral-common numpy tqdm packaging`, deploys `train_lora.py` to
`~/.config/voicetserver/tools/` (a `find_script()` fallback), downloads the model files
(`tekken.json`, `consolidated.safetensors`) from `mistralai/Voxtral-Mini-4B-Realtime-2602`,
copies `assets/mel_filters.bin` into the model dir (not on HuggingFace), installs the binary to
`~/.local/bin/voicetserver` (prebuilt **gpu** from `tools/voicetserver-cuda`, prebuilt **cpu**
if present, or **compile** from source), ensures `~/.local/bin` is on PATH, and provisions a
Tailscale TLS cert + weekly systemd renewal timer. All choices have sensible defaults; existing
model/venv/config are detected and skipped. Writes/updates `~/.config/voicetserver/config.toml`.

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

- `src/main.rs` — sync `fn main()` → forks/daemonizes before tokio starts → `tokio::runtime::Builder::block_on(server::run(…))`; `VoxtralModel` (Arc-shared) + `ModelInner` (tokio::sync::Mutex); inline `server` module with all HTTP + WebSocket handlers; detach/watchdog/PID/log helpers
- `src/streaming.rs` — `StreamingState`: KV caches, SilenceDetector, mel buffer; `process_chunk_sync`
- `src/audio.rs` — raw f32 LE PCM decode from WebSocket binary frames
- `src/config.rs` — config file loading (`~/.config/voicetserver/config.toml`), CLI+file merge, source-tagged error messages
- `src/words.rs` — `WordsCorrector`: aho-corasick text replacement from `custom_words.txt`; `FuzzyMatcher`: Kölner-Phonetik + Levenshtein fuzzy snapping of transcribed words onto known plain-term hotwords
- `src/settings.rs` — `SharedSettings` (atomic runtime params) + `StartupSnapshot`
- `src/encoder.rs` / `src/decoder.rs` — Voxtral model; flash-attn behind `#[cfg(feature = "cuda")]`
- `src/session.rs` — Phase 2 stub (patient session vocabulary)
- `src/macros.rs` — Phase 4 stub (macro expansion)
- `candle-fork/` — vendored Candle ML framework; do not update without reason

GPU lock: single `tokio::sync::Mutex<Option<ModelInner>>` wrapping both enc and dec.
Acquire → `guard.as_mut()` → do all sync Candle work → release before every `.await`.
Disjoint field borrows (enc, dec) work through the `&mut ModelInner` from `as_mut()`.

`inner` is `Option` so the encoder+decoder (~8 GB VRAM) can be **unloaded during LoRA
training** (see below) and reloaded afterward. While `None`, ASR sessions return a
"Server is training…" error (surfaced to the client as a `{"type":"error"}` frame).
`load_enc_dec()` rebuilds enc+dec from `consolidated.safetensors` and re-applies the
active LoRA on reload.

## Auto-unload for training (single-GPU VRAM reuse)

A 4B model needs ~8 GB resident; a second copy for `train_lora.py` does not fit on a 16 GB
card. So `POST /training/run` **unloads** the model before spawning the trainer:
1. `*model.inner.lock() = None` drops enc+dec → frees ~8 GB.
2. `model.device.synchronize()` returns the freed CUDA pool memory to the OS so the separate
   Python process can allocate it (candle/cudarc use the stream-ordered allocator with the
   default release-threshold of 0, so a sync releases unused pool memory).
3. After the subprocess exits (success or failure), a spawned task reloads the model via
   `load_enc_dec()`, re-applying the LoRA path from `state.lora_path`.

Transcription is unavailable for the duration of training + reload (model is re-read from
disk, ~10–30 s).

# Config file

`~/.config/voicetserver/config.toml` — created automatically with commented template on first run.
CLI args override config file values. Unknown fields in the config file are silently ignored.

Runtime-adjustable via `PATCH /config` (no restart needed): `delay`, `silence_threshold`, `silence_flush`, `min_speech`, `rms_ema`, `fuzzy_hotwords`, `fuzzy_max_ratio`.

Startup-only (require restart): `model_dir`, `device`, `port`, `bind_addr`, `tls_cert`, `tls_key`, `lora_adapter`, `data_dir`, `log_file`, `log_keep_days`.

`data_dir` — base directory for `custom_words.txt`, `training/`, `lora_adapter/`, `training_sentences.txt`. Defaults to `~/.config/voicetserver/`. Config file and PID file always stay in `~/.config/voicetserver/` regardless of this setting.

# Daemon mode / detach / log file

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

Reads PID file, sends SIGTERM, polls `/proc/<pid>` for up to 5 s, deletes PID file.

## Log file

- Default: `~/.config/voicetserver/logs/voicetserver.log`
- Override: `--log-file <path>` or `log_file = "..."` in config.toml
- Rotation: background tokio task checks size every 5 min; rotates at 20 MB → `voicetserver.log.<unix_ts>`
- Pruning: rotated files older than `log_keep_days` days (default 7) are deleted on rotation

# Custom words

`~/.config/voicetserver/custom_words.txt` — one entry per line:
- `wrong=correct` — replacement pair: every occurrence of `wrong` in transcribed text is replaced with `correct`
- `PlainTerm` — fuzzy phonetic target: transcribed words that *sound like* it are snapped onto this canonical spelling (see below)
- `# comment` — ignored

The aho-corasick automaton is built at startup and hot-reloaded whenever `POST /words` updates the file. No restart required.

Example: add `Migration=Miktion` to correct a model phonetic confusion.

## Fuzzy phonetic hotword correction

The model often transcribes unfamiliar proper names / medical terms with a slightly different
(phonetically equivalent) spelling each time (e.g. "Bedmika"/"Bedmiga" for "Betmiga"), so exact
`wrong=correct` pairs cannot keep up. Every **final** transcription is post-processed by a
`FuzzyMatcher` (`src/words.rs`) that snaps phonetically-close words onto the canonical spelling.

- Targets = the **plain terms** in `custom_words.txt` (lines without `=`), exposed via
  `WordsCorrector::plain_terms()`. Rebuilt whenever `POST /words` updates the file.
- Matching: identical **Kölner Phonetik** (Cologne phonetics) code **AND** normalized Levenshtein
  distance ≤ `fuzzy_max_ratio`. Both gates required → low false-positive risk.
- Only single all-alphabetic terms are fuzzy targets (multi-word / hyphenated / digit-bearing
  terms like `TUR-B` are excluded — the word scanner splits on non-alphabetic chars). They still
  work as literal `wrong=correct` pairs.
- Words shorter than 4 chars are never matched (`FUZZY_MIN_LEN`).
- Applied at both final sites (silence flush, Close) **after** the literal `WordsCorrector::apply()`
  pass — see `finalize_text()` in `src/main.rs`.

Pipeline: literal `wrong=correct` replacements → fuzzy phonetic snap.

Runtime-adjustable via `PATCH /config`: `fuzzy_hotwords` (bool, default true),
`fuzzy_max_ratio` (f32 ∈ [0,1], default 0.34 — lower = stricter). No CLI flags; config + runtime only.

# HTTP API

All endpoints support CORS (`allow_origin: *`).

- `GET /health` — `{"status":"ready","connections":N}`
- `GET /config` — current settings (runtime + startup snapshot); includes `data_dir`
- `PATCH /config` — update settings; runtime params apply immediately, startup params written to file. Validates: `delay` ∈ [1,30]; `rms_ema` ∈ [0,1]; `fuzzy_max_ratio` ∈ [0,1].
- `GET /words` — `{"words":[...]}`
- `POST /words` — `{"add":[...],"remove":[...]}` — updates file + rebuilds corrector
- `POST /lora/reload` — hot-reload LoRA adapter without restart; optional JSON body `{"path":"..."}` to specify dir (omit to reload current); returns `{"status":"ok","action":"applied"|"cleared","path":"..."}`
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
- `POST /training/run` — unloads the model from VRAM (frees ~8 GB for the trainer), then spawns `tools/train_lora.py`; 202 if started, 409 if already running. Model is reloaded automatically when training finishes. ASR requests error with "Server is training…" until reload.
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

## WebSocket reconnect / mic lifecycle

`connectWS()` retries up to `MAX_RECONNECT` (3) times on `onerror` and `onclose`.
`reconnectTimer` stores the pending `setTimeout` ID so `stopRecording()` can cancel it.
`ws.onclose` has an `else if (recording)` branch that calls `stopRecording()` when retries
are exhausted — without this, `micStream` tracks stay alive and the next `getUserMedia`
fails with `NotFoundError: requested device not found` until browser restart.
Closing the tab releases the mic automatically; the stuck-mic bug only manifests when
staying on the page after a failed connection.

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

# Multi-User Architecture (Exploration Notes)

Not yet implemented — single-user server. Notes for future reference.

## Shared LoRA adapter for two users

Train a single LoRA on both speakers' audio combined. Both have the same vocabulary; the adapter mainly corrects phonetic patterns. Start with a combined adapter — if quality is unacceptable for one user, train separate adapters and use `POST /lora/reload` to switch.

## Concurrent sessions

**Same LoRA adapter:** Already works. LoRA weights are read-only during inference; all WebSocket sessions serialize on the single `Mutex<Option<ModelInner>>` GPU lock.

**Different LoRA adapters concurrently:** Not supported without architectural change. Would require per-session LoRA deltas applied at inference time inside `process_chunk_sync()` — moving LoRA from `TextDecoder` fields into `StreamingState`. Deferred until needed.

## Proposed per-user folder layout (when `--data-dir` is used)

```
{data_dir}/
├── custom_words.txt            # global fallback
├── training_sentences.txt      # shared sentence pool
├── users/
│   ├── alice/
│   │   ├── custom_words.txt
│   │   ├── lora_adapter/
│   │   └── training/
│   │       ├── audio/*.wav
│   │       └── pairs.jsonl
│   └── bob/
│       ├── custom_words.txt
│       ├── lora_adapter/
│       └── training/
│           ├── audio/*.wav
│           └── pairs.jsonl
```

## Key files to modify for full per-user support

- `src/config.rs` — parameterise `WorkspacePaths::new(user_id)` → user-specific sub-paths
- `src/settings.rs` — add `StartupSnapshot.user_id`; extend API with user-scoped endpoints
- `src/main.rs` — `AppState.paths` → per-user lookup; training/words handlers accept `?user=` query param
- `src/streaming.rs` — `StreamingState` carries per-session `WordsCorrector` + optional LoRA delta
- `src/decoder.rs` — separate LoRA application from `set_lora()` (currently mutates global decoder) into per-call delta in `forward()`
