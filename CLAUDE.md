# voicetserver — unified two-engine ASR server

One binary serving **both** ASR models on one port (8765), one TLS cert, one API key, one
config file, one data dir:

- **Voxtral Mini 4B Realtime** (~8 GB VRAM, BF16) — primary engine, always loaded.
- **Qwen3-ASR-0.6B** (~1.5 GB VRAM) via vendored `qwen3-asr-rs` — optional second engine,
  ported from the schmidiscribe project; enabled by setting `qwen_model_dir` in config
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
tag) + `safetensors mistral-common tokenizers numpy tqdm packaging` (the union of both
trainers' deps), deploys `train_lora_voxtral.py` + `train_lora_qwen.py` to
`~/.config/voicetserver/tools/` (a `find_script()` fallback), downloads the Voxtral model
files (`tekken.json`, `consolidated.safetensors`) from
`mistralai/Voxtral-Mini-4B-Realtime-2602`, copies `assets/mel_filters.bin` into the model dir
(not on HuggingFace), optionally downloads the **Qwen3-ASR-0.6B** files (~1.8 GB) from
`Qwen/Qwen3-ASR-0.6B` and sets `qwen_model_dir` (declining leaves the qwen engine disabled;
`tokenizer.json` is not on HuggingFace and is generated via `transformers` — installed into
the venv on demand), installs the binary to `~/.local/bin/voicetserver` (prebuilt **gpu**
from `tools/voicetserver-cuda`, prebuilt **cpu** if present, or **compile** from source),
ensures `~/.local/bin` is on PATH, and provisions a Tailscale TLS cert + weekly systemd
renewal timer. All choices have sensible defaults; existing model/venv/config are detected
and skipped. Writes/updates `~/.config/voicetserver/config.toml`.

# Test

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

## Model language behaviour

**Voxtral** Mini 4B Realtime has **no language token mechanism**. It auto-detects language from the audio signal. There is no way to force a specific language — the model will transcribe in whatever language it hears. Confirmed by inspecting `tekken.json` (no `added_tokens`) and the reference C implementation (antirez/voxtral.c).

**Qwen3-ASR** *does* support forced language via a language token prepended to the assistant
turn (handled inside `qwen3-asr-rs`'s `build_prompt()`). The server passes the configured
`language` (config key, default `"German"`, startup-only) to every qwen session; a per-session
`?lang=` query param overrides it. Voxtral sessions ignore both.

### Experimental German prefill priming (`german_prime`)

Since the language cannot be forced, an experimental hack biases the LM prior instead: when
`german_prime = true` (config flag, default false; runtime-adjustable via `PATCH /config` and
the userscript Einstellungen tab, applies on the next session), the decoder prefill carries a
short German sentence (`GERMAN_PRIME_TEXT` in `src/streaming.rs`) right after BOS instead of
pure `STREAMING_PAD` tokens — as if the model had already transcribed German text during the
leading silence. Capped to the `LEFT_PAD_TOKENS` (32) silence region.

- Token IDs are produced by `Tokenizer::encode_greedy()` (greedy longest-match over the tekken
  vocab — not true BPE, but valid tokens whose decode equals the text; sufficient for priming).
  Encoded once at model load into `VoxtralModel.prime_ids`.
- Plumbing: `StreamingState::new_sync(prime_tokens)` → `run_startup()` → `TextDecoder::prepare_prefill()`.
- Offline WAV mode respects the flag too, so A/B runs on the same WAV are possible.
- Off by default — A/B test before trusting it; risk: the model repeating/continuing the prime
  text at session start.

Known special token IDs (all others in 0–999 are skipped as audio control tokens):
- `BOS = 1`, `EOS = 2`
- `AUDIO = 24`, `BEGIN_AUDIO = 25` — audio frame boundary markers
- `STREAMING_PAD = 32` — silence/padding
- `STREAMING_WORD = 33` — word boundary control token

# Architecture

- `src/main.rs` — sync `fn main()` → forks/daemonizes before tokio starts → `tokio::runtime::Builder::block_on(server::run(…))`; `VoxtralModel` (Arc-shared) + `ModelInner` (tokio::sync::Mutex); inline `server` module with all HTTP + WebSocket handlers (`handle_socket` dispatches on `?model=` to `handle_asr_session` (voxtral) or `handle_qwen_session`); `api_key_auth` middleware (constant-time `ct_eq`) on a protected sub-router (`/health` is public); `pair_write_lock` serialises training-pair writes; detach/watchdog/PID/log helpers
- `src/streaming.rs` — Voxtral `StreamingState`: KV caches, SilenceDetector, mel buffer; `process_chunk_sync`
- `src/qwen.rs` — `QwenEngine`: wraps vendored `qwen3_asr::AsrInference` behind the same `Mutex<Option<…>>` unload pattern; inner value is `Arc<AsrInference>` so sessions clone the handle, release the lock, and let AsrInference's own internal mutex serialise GPU work
- `src/qwen_streaming.rs` — qwen `StreamingState` (ported from schmidiscribe): SilenceDetector with `has_speech()` gate (suppresses hallucinated partials during silence), silence reset carrying the last 200 chars as initial_text
- `src/audio.rs` — raw f32 LE PCM decode from WebSocket binary frames
- `src/config.rs` — config file loading (`~/.config/voicetserver/config.toml`), CLI+file merge, source-tagged error messages
- `src/words.rs` — `WordsCorrector`: aho-corasick text replacement from `custom_words.txt`; `FuzzyMatcher`: Kölner-Phonetik + Levenshtein fuzzy snapping of transcribed words onto known plain-term hotwords; `AbbrevExpander`: German letter-name runs → acronym targets (Em Er I → MRI)
- `src/settings.rs` — `SharedSettings` (atomic runtime params) + `StartupSnapshot`
- `src/encoder.rs` / `src/decoder.rs` — Voxtral model; flash-attn behind `#[cfg(feature = "cuda")]`
- `src/session.rs` — Phase 2 stub (patient session vocabulary)
- `src/macros.rs` — Phase 4 stub (macro expansion)
- `candle-fork/` — vendored Candle ML framework; do not update without reason
- `qwen3-asr-rs/` — vendored Qwen3-ASR inference library; its candle deps point at `../candle-fork/` paths so there is exactly one candle build / CUDA context

GPU locks — one per engine, so voxtral and qwen sessions run **concurrently**:
- Voxtral: single `tokio::sync::Mutex<Option<ModelInner>>` wrapping both enc and dec.
  Acquire → `guard.as_mut()` → do all sync Candle work → release before every `.await`.
  Disjoint field borrows (enc, dec) work through the `&mut ModelInner` from `as_mut()`.
- Qwen: `QwenEngine.inner` is only locked briefly to clone the `Arc<AsrInference>`;
  inference runs in `spawn_blocking`, serialised by AsrInference's internal mutex.

Both inners are `Option` so the engines can be **unloaded during LoRA training** (see
below) and reloaded afterward. While `None`, ASR sessions return a "Server is training…"
error (surfaced to the client as a `{"type":"error"}` frame). `load_enc_dec()` /
`QwenEngine::reload_blocking()` rebuild the engines from disk and re-apply each model's
active LoRA on reload.

## Auto-unload for training (single-GPU VRAM reuse)

Voxtral needs ~8 GB resident (+~1.5 GB qwen); a trainer's own copy of either model does not
fit alongside them on a 16 GB card. So `POST /training/run` **unloads both engines** before
spawning either trainer (simplest safe rule):
1. `*model.inner.lock() = None` drops enc+dec (~8 GB); `*qwen.inner.lock() = None` drops the
   qwen engine (~1.5 GB). A qwen session already in flight keeps its Arc alive until it ends;
   new sessions are blocked immediately.
2. `model.device.synchronize()` returns the freed CUDA pool memory to the OS so the separate
   Python process can allocate it (candle/cudarc use the stream-ordered allocator with the
   default release-threshold of 0, so a sync releases unused pool memory).
3. After the subprocess exits (success or failure), a spawned task reloads both engines,
   re-applying each model's LoRA path (`state.lora_path` / `state.qwen_lora_path`).

Transcription is unavailable for the duration of training + reload (models are re-read from
disk, ~10–30 s).

# Config file

`~/.config/voicetserver/config.toml` — created automatically with commented template on first run;
file permissions are set to **0600** (owner read/write only) on creation and on every save, since it
contains the API key.
CLI args override config file values. Unknown fields in the config file are silently ignored.

`api_key` — auto-generated (16-byte random hex) on first server start and persisted to config.toml.
Printed prominently at startup when newly generated. Copy into the userscript Einstellungen tab.
Not a CLI flag — set only via config file.

Runtime-adjustable via `PATCH /config` (no restart needed): `delay`, `silence_threshold`, `silence_flush`, `min_speech`, `rms_ema`, `fuzzy_hotwords`, `fuzzy_max_ratio`, `german_prime`, `context_biasing`.

`delay` is **next-session**, not mid-session: `StreamingState` fixes prefill, KV sizing and the
drain length at construction, so an open session keeps its original value (the silence detector
reads the session's captured `delay_tokens`, not the live atomic — otherwise finalization timing
desyncs from the actual decoder lag). New sessions pick the new value up immediately.

Startup-only (require restart): `model_dir`, `qwen_model_dir`, `language`, `device`, `port`, `bind_addr`, `tls_cert`, `tls_key`, `lora_adapter`, `lora_adapter_qwen`, `venv_path`, `data_dir`, `log_file`, `log_keep_days`.

`data_dir` — base directory for `custom_words.txt`, `training/`, `lora_adapter/`, `lora_adapter_qwen/`, `training_sentences.txt`. Defaults to `~/.config/voicetserver/`. Config file and PID file always stay in `~/.config/voicetserver/` regardless of this setting.

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

Reads PID file, verifies `/proc/<pid>/cmdline` contains `voicetserver` (guards against a stale
PID being reused by an unrelated process — refuses to SIGTERM if it doesn't match), sends SIGTERM,
polls `/proc/<pid>` for up to 5 s, deletes PID file.

## Log file

- Default: `~/.config/voicetserver/logs/voicetserver.log`
- Override: `--log-file <path>` or `log_file = "..."` in config.toml
- Rotation: background tokio task checks size every 5 min; rotates at 20 MB → `voicetserver.log.<unix_ts>`. Rotate-by-**rename** (atomic, no lost lines), then — only if fd 1 currently points at the log inode (detached / after 'd'; never in interactive terminal mode) — the path is reopened and `dup2`'d over stdout/stderr in-process (`rotate_log_if_needed` / `stdout_is_file`). The earlier copy+truncate scheme dropped lines written between the two steps.
- Pruning: rotated files older than `log_keep_days` days (default 7) are deleted on rotation; age is taken from the Unix timestamp in the filename (`voicetserver.log.<ts>`), not filesystem mtime (unreliable after copies/restores)

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
- Matching: **near Kölner Phonetik** (Cologne phonetics) code **AND** normalized Levenshtein
  distance ≤ `fuzzy_max_ratio`. Both gates required → low false-positive risk.
  - Phonetic gate is *near*, not *identical*: a leading `0` (edge-vowel code) is stripped before
    comparison, then a Kölner-code edit distance ≤ `FUZZY_MAX_CODE_DIST` (=1) is accepted. Voxtral
    has **no biasing** to keep spellings tight (the qwen3 sister project does), so its output drifts
    further — most often a prepended/dropped edge vowel turning `1264` "Betmiga" into `01264`
    "Epetmika". An identical-code requirement rejected these; the relaxed gate snaps them while the
    orthographic Levenshtein ratio remains the false-positive backstop.
- Only single all-alphabetic terms are fuzzy targets (multi-word / hyphenated / digit-bearing
  terms like `TUR-B` are excluded — the word scanner splits on non-alphabetic chars). They still
  work as literal `wrong=correct` pairs.
- Words shorter than 4 chars are never matched (`FUZZY_MIN_LEN`).
- Applied at both final sites (silence flush, Close) **after** the literal `WordsCorrector::apply()`
  pass — see `finalize_text()` in `src/main.rs`.

Pipeline: literal `wrong=correct` replacements → abbreviation letter-name expansion → fuzzy
phonetic snap (abbrev runs before fuzzy so letter-name tokens can't be snapped away first).

Runtime-adjustable via `PATCH /config`: `fuzzy_hotwords` (bool, default true),
`fuzzy_max_ratio` (f32 ∈ [0,1], default 0.34 — lower = stricter). No CLI flags; config + runtime only.

## Abbreviation / acronym expansion (letter-name pass)

Spelled-out abbreviations are structurally unreachable for the fuzzy matcher: `TUR-B` is excluded
as a fuzzy target (hyphen), and a dictated acronym comes out as German **letter names** — "M-R-I"
→ `Em Er I` / `EM-ER-I` — separate tokens the word-by-word scanner never joins. The
`AbbrevExpander` (`src/words.rs`) handles them:

- **Targets** are auto-detected from the plain (non-`=`) custom_words.txt terms: only
  letters/digits/hyphens with **2–6 letters, all uppercase** (`MRI`, `PSA`, `EKG`, `TUR-B`).
  Matching key = letters only (`TUR-B` → `TURB`). Rebuilt on `POST /words`, like the fuzzy matcher.
- **Scan**: finals are scanned for runs of **≥2 adjacent tokens that are ALL recognized German
  letter names** (`em→M, er→R, es→S, pe→P, te→T, ka→K, …`; a single alphabetic char stands for
  itself, so `E Ka Ge` works too). Adjacent = separated only by spaces/hyphens/periods.
- **Replace**: the run's letters are joined; if they match a target key the whole run is replaced
  by the canonical spelling. Longest run first, shrinking from the right on no match
  (`Em Er I u` still finds `MRI`).
- Requiring every token to be a letter name AND a full-key match keeps false positives low even
  though `er`/`es` are common German words (they only fire inside a matching run).
- The single-word rendering case (`Emery` for MRI) is **not** handled — it would need a phonetic
  fallback that strips leading-zero Kölner codes; deferred.
- Applied in `finalize_text()` between the literal and fuzzy passes. Always on; no-op when no
  acronym-shaped terms exist.

# HTTP API

All endpoints support CORS (`allow_origin: *`).

**Authentication:** all endpoints except `GET /health` require the `X-Api-Key` header. For
WebSocket (browsers cannot send custom headers during the upgrade), pass the key as
`?api_key=<key>` instead. The key is auto-generated (16-byte random hex) on first server start,
persisted to `config.toml` (chmod 0600), and printed prominently at startup when newly generated.
Auth is enforced by the `api_key_auth` middleware (constant-time compare via `ct_eq`) applied as
a `route_layer` to the protected sub-router; `/health` is registered outside it. Not a CLI flag —
set only via config file.

Request bodies are capped at **64 MB** via axum `DefaultBodyLimit` (review uploads are whole
dictations as f32 PCM: 64 KB/s → ~16 min head-room).

- `GET /health` — `{"status":"ready"|"loading","connections":N}` — **public, no auth required**.
  `"loading"` covers startup model loading *and* the whole training window (engines unloaded →
  trainer → reload), so a monitor can tell when ASR sessions would error. A failed reload after
  training keeps it at `"loading"`.
- `GET /config` — union of both engines' settings (runtime + startup snapshot); includes
  `"server":"voicetserver"` (backend identity for the shared userscript) and
  `"models":["voxtral","qwen"]` (or `["voxtral"]` when qwen is disabled — the frontend hides
  qwen UI accordingly). Per-model LoRA state: `lora_active_voxtral`/`lora_active_qwen` (bool —
  is a LoRA currently applied in-memory) and `lora_dir_voxtral`/`lora_dir_qwen` (the path the
  userscript "LoRA verwenden" toggle re-applies on enable: active path if loaded, else configured
  `lora_adapter`/`lora_adapter_qwen`, else the model's default training output dir — so the
  toggle still works after `DELETE /lora`). The unsuffixed `lora_active`/`lora_dir` remain as
  voxtral aliases from the frontend transition.
- `PATCH /config` — update settings (the union — includes `context_biasing`, `language`, `qwen_model_dir`, `lora_adapter_qwen`); runtime params apply immediately (except `delay`, see above), startup params written to file. Validates: `delay` ∈ [1,30]; `rms_ema` ∈ [0,1]; `silence_threshold` ∈ [0,1]; `fuzzy_max_ratio` ∈ [0,1]; `silence_flush` ∈ [1,250]; `min_speech` ∈ [1,250] (`silence_flush = 0` would silently disable silence-triggered finals).
- `GET /words` — `{"words":[...]}`
- `POST /words` — `{"add":[...],"remove":[...]}` — updates file + rebuilds corrector
- `POST /lora/reload?model=voxtral|qwen` — hot-reload the model's LoRA adapter without restart (default voxtral; same for all `?model=` params below); optional JSON body `{"path":"..."}` to specify dir (omit the body entirely — not just the field — to reload current); returns `{"status":"ok","action":"applied"|"cleared","path":"..."}`
- `DELETE /lora?model=voxtral|qwen` — unload the model's active LoRA adapter in-memory (revert to base model) without restart; adapter files on disk are left untouched; clears the model's lora path so nothing is re-applied on the next training reload
- `ws[s]://host:port/asr?model=voxtral|qwen` — WebSocket audio stream (raw f32 LE PCM 16kHz
  mono); `?model=` picks the engine (default voxtral; `?model=qwen` on a server without
  `qwen_model_dir` gets an error frame). **Stop protocol** (identical on both engines):
  client sends text frame `"stop"` → server drains the engine (Voxtral: feeds `delay + 3`
  ticks of silence via `StreamingState::drain_sync` — the decoder lags the audio by `delay`
  tokens, so without this the last ~delay×80ms of speech would be truncated; qwen:
  `finish_streaming()`), then flushes the remaining text as a `{"type":"final"}` and closes;
  client waits for `ws.onclose` (2 s fallback) before finalizing. Fallback: a plain client
  close triggers the same drain + flush, but the final may be lost if the client is already
  gone. Qwen sessions also honour `?hotwords=` + `?patient=` (system-prompt biasing together
  with the custom_words plain terms, gated by the runtime `context_biasing` toggle) and
  `?lang=` (overrides the configured `language`); Voxtral sessions ignore all three.

### Training (Phase 2 — LoRA voice calibration)

**Sentence management:**
- `GET /training/sentences` — returns `{"sentences":[{"text","recorded","pair_ids":[…]}]}` annotated with recording status; auto-creates `training_sentences.txt` on first call
- `POST /training/sentence` — `{"text":"…"}` — append new sentence to file
- `PATCH /training/sentence` — `{"old":"…","new":"…"}` — replace one sentence in file
- `DELETE /training/sentence` — `{"text":"…"}` — remove one sentence from file

**Pair collection:**
- `POST /training/pair?text=<url-encoded>` — body: raw f32 LE PCM at 16kHz (always; MediaRecorder path dropped — container bytes caused static); saves 16-bit WAV + appends to `pairs.jsonl`; returns `{"id","duration_s","count"}`. ID is `max(existing_ids)+1` (not line count) so deletions never cause collisions. Upload + delete are serialised via a `pair_write_lock` mutex so concurrent requests can't collide on an ID or rewrite the JSONL mid-append.
- `GET /training/pairs` — `{"pairs":[{"id","text","duration_s"}]}` sorted by id
- `GET /training/audio/{id}` — serve recorded WAV for playback (numeric id only)
- `DELETE /training/pair/{id}` — remove WAV file + JSONL entry. A missing WAV does not abort the
  delete (the JSONL entry is removed either way, so a lost recording can't leave an undeletable
  entry that breaks the trainer); 404 only when neither existed.
- `GET /training` — `{"count":N,"duration_sec":F}` summary
- `DELETE /training` — remove all collected training **pairs** (`training/audio/` + `pairs.jsonl`).
  Dictation-review candidates (`training/review/` + `review.jsonl`) live under the same training
  dir but are deliberately **kept** — they are a separate, unaccepted pool.

**LoRA:**
- `POST /training/run?model=voxtral|qwen` — unloads **both** engines from VRAM (frees ~9.5 GB for the trainer), then spawns the model's trainer script (`tools/train_lora_voxtral.py` / `tools/train_lora_qwen.py`; voxtral falls back to the legacy `train_lora.py` name for old installs); 202 if started, 409 if already running (one `training_status` — only one training at a time regardless of model). The shared pairs pool trains both models' LoRAs. Both engines are reloaded automatically when training finishes; ASR requests on either engine error with "Server is training…" until then.
- `GET /training/status` — `{"status":"idle"|"running"|"done"|"error","log":[...]}`

**Dictation review (real dictations as training-pair candidates):** read-aloud calibration
sentences train the LoRA on a different speaking style than free dictation. The client can save
a finished real dictation (audio + model transcript) as a *candidate*; the user later plays it,
optionally re-transcribes it, corrects the text, and accepts it into the training set — or
discards it. Candidates live in `training/review/*.wav` + `training/review.jsonl` and are
invisible to the trainer until accepted.

- `POST /training/review?text=<url-encoded>` — body raw f32 LE PCM 16kHz; `text` = model transcript at save time; same ID scheme + `pair_write_lock` as pairs
- `GET /training/reviews` — `{"reviews":[{"id","text","duration_s"}]}` sorted by id
- `GET /training/review/audio/{id}` — serve candidate WAV (playback / client-side re-transcription)
- `POST /training/review/{id}/accept` — `{"text":"…"}` (corrected transcript) — moves the WAV into `training/audio/` under a fresh pair ID, appends to `pairs.jsonl`, removes the candidate
- `DELETE /training/review/{id}` — discard candidate (WAV + JSONL entry)

**Edit-log mining:**
- `POST /log/edit` — `{"original","edited","timestamp"}` — appended to `edit_log.jsonl` by the userscript when a commit-mode dictation is edited before insertion
- `GET /edits/report` — `{"entries":N,"suggestions":[{"original","edited","count"},…]}` — aggregates the edit log into the most frequent word-level corrections (LCS word diff per entry, changed runs ≤4 words paired as removed→inserted, punctuation-only changes skipped, top 30 by count). Direct candidates for custom_words `wrong=correct` entries; surfaced by the userscript's "💡 Vorschläge aus Korrekturen" button in the Eigene Wörter tab.

Training data stored in `~/.config/voicetserver/training/audio/*.wav` + `pairs.jsonl` (one
pool — it trains both models' LoRAs). Adapter output is per-model:
`~/.config/voicetserver/lora_adapter/` (voxtral) and `lora_adapter_qwen/` (qwen); set
`lora_adapter` / `lora_adapter_qwen` in config to load at startup.

`tools/import_pairs.py` — one-time importer that appends another server's collected pairs
(e.g. schmidiscribe's `~/.config/schmidiscribe/training/`) into the pool, re-IDing via the
normal `max(id)+1` path. Run with voicetserver stopped (the `pair_write_lock` is in-process
only); source WAVs are copied, not moved.

Python venv install (run once — put venv on a large partition if root fs is tight; the dep
list is the union of both trainers' requirements):
```bash
VENV=/path/to/venv          # e.g. /mnt/ssdupl/voicetserver-venv
python3 -m venv $VENV
mkdir -p $VENV/tempdir
# torch must come from the PyTorch CUDA index (replace cu128 with your CUDA version)
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu128
# remaining deps from PyPI (mistral-common: voxtral trainer; tokenizers: qwen trainer)
TMPDIR=$VENV/tempdir $VENV/bin/pip install --no-cache-dir \
  safetensors mistral-common tokenizers numpy tqdm packaging
```
Deploy training scripts next to the binary:
```bash
mkdir -p ~/.local/bin/tools && cp tools/train_lora_voxtral.py tools/train_lora_qwen.py ~/.local/bin/tools/
```
Then set `venv_path = "/path/to/venv"` in `~/.config/voicetserver/config.toml`.

### LoRA adapter

Both runtimes load `adapter_model.safetensors` + `adapter_config.json` from the adapter dir
and apply the delta at runtime: `proj_output += scale * lora_b @ lora_a @ input` (no weight
merging). Scale = `lora_alpha / r` from `adapter_config.json`. Weight-key formats **differ
per model**, so adapters are strictly per-model:
- Voxtral (`src/lora.rs`): `layers.{i}.attention.{wq,wk,wv,wo}.lora_{a,b}.weight`
- Qwen (`qwen3-asr-rs`): `q_proj`/`v_proj` attention keys (see `tools/train_lora_qwen.py`)

# Browser client

`schmidispeech.user.js` — Violentmonkey userscript.
Sends raw 16kHz mono f32 LE PCM over WebSocket binary frames.
Receives `{"type":"partial","text":"..."}` / `{"type":"final","text":"..."}` / `{"type":"error","text":"..."}`.

**Unified frontend** (v0.1.16+): one userscript, one server, one URL + API key. The
"Server: [Voxtral][Qwen3]" switcher above the tab bar only selects which **engine** a
session uses (GM value `active_model` → `?model=` on the WS URL); it switches instantly (no
save needed; blocked while recording). Storage keys are the dual-backend era's Voxtral
profile keys (`server_url_voxtral`/`api_key_voxtral`, falling back to the pre-profile
`server_url`/`api_key`; `active_backend` seeds `active_model`), so both upgrade and rollback
keep working; the qwen profile keys stay dormant. The Einstellungen tab is a single pane
driven by the union `GET /config`: qwen rows (`context_biasing`, language) are hidden when
the server's `models` list lacks `"qwen"`, `delay`/`german_prime` rows appear only when the
server reports those fields — so the script still degrades gracefully against an old
single-model server. It replaced schmidiscribe's `schmididict.user.js` (disable that one to
avoid duplicate mic buttons).

WebSocket sessions send `?api_key=` and `?model=` plus `?hotwords=` (Hotwords tab, GM-stored)
and `?patient=` (read from `#schmidi-pat-info-patientendaten` if the host page has it) — used
by the qwen engine's prompt biasing, ignored on Voxtral sessions. Stopping sends the text
frame `"stop"` and waits for the server's final flush + close (2 s fallback). All audio paths
(ASR, Aufnehmen, 2. Durchgang) route the ScriptProcessor through a zero-gain GainNode instead
of connecting to `destination` directly — a direct connection plays the mic through the
speakers and browser AEC then cancels the speech it hears back.

All HTTP requests go through `authFetch()`, which injects `X-Api-Key: <stored key>` on every call
(browsers cannot send headers on the WS upgrade, hence the query param there).
Training-audio playback uses `authFetch` + `URL.createObjectURL(blob)` (not `new Audio(url)`) so the
key is sent. The API key is entered in the Einstellungen tab.
Default hotkey: `Ctrl+Shift+D` (configurable via right-click menu → Einstellungen tab).
Text is inserted live at cursor on each `final`; trailing partial inserted on stop.
Falls back to clipboard if no editable element was captured.

Right-click → seven tabs: **Eigene Wörter** (server-side custom words + "💡 Vorschläge aus
Korrekturen" from `GET /edits/report`), **Hotwords** (client-side GM-stored list sent per session
via `?hotwords=`; biasing only on Qwen3), **Aufnehmen** (record calibration sentences),
**2. Durchgang** (record once-recorded sentences — `pair_ids.length === 1` — a second time),
**Training** (review pairs, delete, LoRA), **Diktate** (real-dictation review, below),
**Einstellungen** (server URL/key, hotkey, runtime params).

**Audio capture (all paths — ASR, Aufnehmen, 2. Durchgang):** `createCaptureContext()` requests
an `AudioContext({ sampleRate: 16000 })` so the **browser** resamples the mic stream with proper
low-pass filtering. The old path decimated the native 48 kHz stream (every 3rd sample), folding
high frequencies into the speech band as aliasing noise. Naive decimation survives only as a
fallback (`captureChunk()`) for browsers that refuse a fixed-rate context.

**Diktate tab (real dictations → training pairs):** every dictation's 16 kHz PCM is kept
client-side (`dictationPcmBuffers` → `lastDictation` snapshot on stop, together with the
transcript — the edited overlay text in commit mode). Saving (overlay 💾 button in commit mode,
or "💾 Letztes Diktat speichern" in the tab) POSTs it to `/training/review`. The tab lists all
candidates with ▶ playback and ✕ delete; selecting one opens a detail area with an editable
transcript, "↻ Voxtral" / "↻ Qwen3" re-transcription (fetches the stored WAV, parses PCM16
client-side, replays it through a normal WS session with the chosen `?model=` —
`transcribePcm()`), and "✓ Als Trainingspaar übernehmen" →
`POST /training/review/{id}/accept` with the corrected text.

**Aufnehmen tab:** shows only unrecorded sentences (recorded ones are hidden). Navigate ◀/▶, edit/add/remove sentences inline, record via ScriptProcessor (raw f32 LE PCM, same as ASR path), preview recording client-side before saving (Web Audio API, no server round-trip), save to commit pair.

**Paare tab:** scrollable list of all recorded pairs (id, text, duration) with per-pair ▶ playback and ✕ delete. LoRA training run + status log (the train/reload buttons act on the currently selected model — `?model=` on `POST /training/run`). **LoRA verwenden** is one checkbox **per model**: checked → load that model's adapter (`POST /lora/reload?model=` with `lora_dir_<model>` from `GET /config`), unchecked → unload (`DELETE /lora?model=`); reflects `lora_active_voxtral`/`_qwen` on tab open (qwen row hidden when the server reports no qwen state). Lets you A/B each adapter against its base model live (useful when an adapter trained on read-aloud audio hurts free-speech accuracy).

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
- `docs/unified_server_migration_plan.md` — the plan that produced the two-engine server
- `scripts/generate_mel_filters.py` — mel filterbank generator (pure Python, no deps)
- `schmidispeech.user.js` — browser userscript
- `candle-fork/` — vendored Candle
- `qwen3-asr-rs/` — vendored Qwen3-ASR inference library (candle deps repointed at `candle-fork/`)
- `tools/` — installer, prebuilt CUDA binary, trainer scripts (`train_lora_voxtral.py`, `train_lora_qwen.py`), `import_pairs.py`
- `README.private.md` — private deploy notes (gitignored)
- `releases/` — local versioned binary copies (gitignored)

# Predecessor projects (deprecation)

The qwen engine was ported from **schmidiscribe** (standalone Qwen3-ASR server, port 8767).
After the unified server has burned in on production: stop/disable the schmidiscribe service,
archive its repo with a pointer here, and optionally import its training pairs via
`tools/import_pairs.py` first. The **schreibot** repo (failed whisper test) is unrelated and
gets nuked independently. Until then schmidiscribe remains deployable standalone; the
userscript degrades gracefully against it (field-presence gating in `GET /config`).

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

## Commit workflow

Single `main` branch — no `testing` branch; "ok stable" does not apply here.

1. After the code change is complete, bump the patch version in `Cargo.toml` (+0.0.1) **before** compiling.
2. Run `cargo build --release` so the user can test.
3. On commit ("ok go"): commit and push to `main` (version already bumped — do not bump again).
