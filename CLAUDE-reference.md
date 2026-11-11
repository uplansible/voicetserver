# voicetserver — reference sections (moved verbatim from CLAUDE.md)

Local, not auto-loaded. `CLAUDE.md` indexes these sections.

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
ensures `~/.local/bin` is on PATH, asks how the server should be reachable, and offers a
`voicetserver.service` systemd unit so it survives a reboot. All choices have sensible
defaults; existing model/venv/config are detected and skipped. Writes/updates
`~/.config/voicetserver/config.toml`.

**Base directory** (default yes): models and training data can live under one folder —
`<base>/models/<checkpoint dirs>` + `<base>/data` (= `data_dir`) — so a single ZFS dataset
covers everything worth replicating (the venv is throwaway and stays outside). If `zfs` is
installed and the base dir isn't a mountpoint yet, the installer offers to `zfs create` a
dataset for it (default name: the dataset of the nearest existing ancestor + `/<basename>`,
explicit `mountpoint=`, via sudo, then chowned to the user). On an existing install it offers
to **move** the configured `model_dir`/`qwen_model_dir`/`qwen_model_dir_alt`/`data_dir` into
the base dir and rewrites those keys (plus `lora_adapter*` paths that pointed into the old data
dir). A running `voicetserver` must be stopped for the move: the installer offers to stop it
(`ensure_server_stopped`: `sudo systemctl stop voicetserver` when the unit is active — a plain
SIGTERM is undone by its `Restart=always` — then `pkill` for hand-started instances) and offers
to start the unit again at the end. When the old data dir is the default
`~/.config/voicetserver`, only the data items move (config, logs, PID, `tools/` stay). The base
dir is not a config key (the server rewrites `config.toml` from its own struct and would drop
it) — a re-run derives it from `data_dir` when that ends in `/data`.

Exposure has two modes:
- **Tailscale Service** (default) — `tailscale serve --service=svc:<name> --https=443` puts
  TLS on a per-service virtual IP proxying to loopback, so the server binds `127.0.0.1` and
  holds no cert (`tls_cert`/`tls_key` are commented out). URL: `wss://<svc>.<tailnet>.ts.net/asr`,
  no port. Needs a tagged node and a service already defined in the admin console — the
  installer only advertises, it cannot create either.
- **Self-hosted TLS** (legacy) — `tailscale cert` for the node FQDN, `bind_addr = 0.0.0.0`,
  plus the weekly `tailscale-cert-renewal.timer`. Exposes the port to the whole LAN.

The unit copies `LD_LIBRARY_PATH` and `PATH` out of the installing shell: a login shell picks
CUDA up from `~/.bashrc`, a systemd service inherits nothing, and without it the server dies
with `CublasError(CUBLAS_STATUS_NOT_INITIALIZED)` — which also happens when a second instance
already holds the card.

# Test

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
- `src/qwen_streaming.rs` — qwen `StreamingState`: SilenceDetector with `has_speech()` gate (suppresses hallucinated partials during silence), silence reset carrying the last 200 chars as initial_text. **Hallucination guards** (Qwen invents fluent sentences on non-speech audio, 1.7B especially): (A) audio reaches the engine only while the detector is armed — unarmed 100 ms ticks wait in a pre-roll (`min_speech + PREROLL_EXTRA_TICKS`, cut back to the lead-in whenever the detector is idle again so it never reaches past a full pause), trimmed on arming to `LEAD_IN_TICKS` (0.5 s) before the first loud tick, so pauses between segments are never fed; (B) `is_plausible()` drops any partial/final longer than `MAX_CHARS_PER_SPEECH_SEC` (35) × loud-tick seconds + `PLAUSIBLE_SLACK_CHARS` (15). A dropped final is preceded by an empty partial (and the stop path sends an empty partial instead of no final) because the userscript inserts `currentPartial` as fallback for an empty final / on stop
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
  **Only one Voxtral session at a time:** the KV caches (encoder attention caches, decoder
  caches) live in the shared `ModelInner` and every session's prefill resets them, so two
  interleaved sessions corrupt each other. `handle_asr_session` holds
  `AppState.voxtral_session` (`try_lock`, RAII guard for the whole session); a second
  `?model=voxtral` session — including a Diktate-tab re-transcription while dictating — gets
  the error frame "Voxtral wird bereits von einer anderen Sitzung verwendet — bitte warten"
  and is closed. Qwen sessions are unaffected.
- Voxtral RoPE limit: the RoPE tables hold `ROPE_MAX_POSITIONS` (131072) positions; the
  encoder advances 4 per 80 ms tick, so one session would overflow after ~44 min. Past
  `ROPE_SOFT_LIMIT` (~34 min) the session's `StreamingState` is rebuilt via the normal
  prefill path at the next silence final; past `ROPE_HARD_LIMIT` (~43 min) immediately. Both
  drain the decoder lag first and send the decoded text as one more final.
- Qwen: `QwenEngine.inner` is only locked briefly to clone the `Arc<AsrInference>`;
  inference runs in `spawn_blocking`, serialised by AsrInference's internal mutex.

Both inners are `Option` so the engines can be **unloaded during LoRA training** (see
below) and reloaded afterward. While `None`, ASR sessions return a "Server is training…"
error (surfaced to the client as a `{"type":"error"}` frame). A Voxtral session that hits
the unload mid-dictation first sends its decoded-so-far text as a final. `load_enc_dec()` /
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
   `training_status` stays `running` until that reload is done (only then `done`/`error`),
   so a second `POST /training/run` or a `POST /qwen/switch` cannot start mid-reload (409).

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

Startup-only (require restart): `model_dir`, `qwen_model_dir`, `qwen_model_size`, `qwen_model_dir_alt`, `qwen_model_size_alt`, `language`, `device`, `port`, `bind_addr`, `tls_cert`, `tls_key`, `lora_adapter`, `lora_adapter_qwen`, `lora_adapter_qwen_alt`, `venv_path`, `data_dir`, `log_file`, `log_keep_days`.

Runtime-switchable, but *not* via `PATCH /config` — its own endpoint instead, since it triggers
a model reload: `qwen_active_slot` (`"primary"|"secondary"`) via `POST /qwen/switch` (see
Runtime-switchable second size, above). Persisted on every switch so a restart resumes there.

`data_dir` — base directory for `custom_words.txt`, `training/`, `lora_adapter/`, `lora_adapter_qwen/`, `lora_adapter_qwen_alt/`, `training_sentences.txt`. Defaults to `~/.config/voicetserver/`. Config file and PID file always stay in `~/.config/voicetserver/` regardless of this setting.

# Daemon mode / detach / log file

## Log file

- Default: `~/.config/voicetserver/logs/voicetserver.log`
- Override: `--log-file <path>` or `log_file = "..."` in config.toml
- Rotation: background tokio task checks size every 5 min; rotates at 20 MB → `voicetserver.log.<unix_ts>`. Rotate-by-**rename** (atomic, no lost lines), then — only if fd 1 currently points at the log inode (detached / after 'd'; never in interactive terminal mode) — the path is reopened and `dup2`'d over stdout/stderr in-process (`rotate_log_if_needed` / `stdout_is_file`). The earlier copy+truncate scheme dropped lines written between the two steps.
- Pruning: rotated files older than `log_keep_days` days (default 7) are deleted on rotation; age is taken from the Unix timestamp in the filename (`voicetserver.log.<ts>`), not filesystem mtime (unreliable after copies/restores)

# Custom words

`~/.config/voicetserver/custom_words.txt` — one entry per line:
- `wrong=correct` — replacement pair: every **whole-word** occurrence of `wrong` in transcribed text is replaced with `correct` (see Whole-word matching, below). The **correction side** is canonical vocabulary too, so it also serves as a fuzzy / acronym target (no need to repeat it as a plain line)
- `PlainTerm` — fuzzy phonetic target: transcribed words that *sound like* it are snapped onto this canonical spelling (see below)
- `# comment` — ignored

The aho-corasick automaton is built at startup and hot-reloaded whenever `POST /words` updates the file. No restart required.

**Whole-word matching:** a literal pair only fires where the characters directly before and
after the match are not alphanumeric (Unicode-aware — ä/ö/ü/ß count as letters), so
`Arm=Harn` fixes "Arm," but leaves "Warm" / "Armband" alone. `WordsCorrector::apply()`
enumerates all overlapping matches (`MatchKind::Standard` + `find_overlapping_iter`),
drops those failing the boundary test, then picks leftmost-longest, non-overlapping — so a
rejected longer candidate never hides a valid shorter one at the same position
(`Arm links=…` + `Arm=Harn` on "Arm linksseitig" → "Harn linksseitig"; `Ura=X` +
`Uratstein=Y` on "Uratstein" → "Y"). A plain `LeftmostLongest` scan cannot do this, it skips
past the rejected match.

Example: add `Migration=Miktion` to correct a model phonetic confusion.

## Fuzzy phonetic hotword correction

The model often transcribes unfamiliar proper names / medical terms with a slightly different
(phonetically equivalent) spelling each time (e.g. "Bedmika"/"Bedmiga" for "Betmiga"), so exact
`wrong=correct` pairs cannot keep up. Every **final** transcription is post-processed by a
`FuzzyMatcher` (`src/words.rs`) that snaps phonetically-close words onto the canonical spelling.

- Targets = the canonical terms of `custom_words.txt` — the **plain terms** (lines without `=`)
  **plus the correction side (RHS) of every `wrong=correct` pair** — exposed via
  `WordsCorrector::fuzzy_terms()` (order-preserving, deduped; malformed pairs with an empty side
  skipped, like `from_str`). The RHS belongs there because the model rarely mishears a term the
  same way twice: `Migration=Miktion` catches only that one variant, while the fuzzy pass also
  snaps `Miktzion`/`Micktion`. `plain_terms()` still exists and feeds the qwen context biasing.
  Rebuilt whenever `POST /words` updates the file.
- Matching: **near Kölner Phonetik** (Cologne phonetics) code **AND** normalized Levenshtein
  distance ≤ `fuzzy_max_ratio`. Both gates required → low false-positive risk.
  - Phonetic gate is *near*, not *identical*: a leading `0` (edge-vowel code) is stripped before
    comparison, then a Kölner-code edit distance ≤ `FUZZY_MAX_CODE_DIST` (=1) is accepted. Voxtral
    has **no biasing** to keep spellings tight (the qwen3 sister project does), so its output drifts
    further — most often a prepended/dropped edge vowel turning `1264` "Betmiga" into `01264`
    "Epetmika". An identical-code requirement rejected these; the relaxed gate snaps them while the
    orthographic Levenshtein ratio remains the false-positive backstop.
- **Inflection-aware**: matching runs on *stems* — the dictated word and each canonical term may
  each shed one German inflection ending (`INFLECTION_SUFFIXES`: `e n s en es er em ns ern innen`;
  a stem must keep ≥ `STEM_MIN_LEN` = 5 chars). Two consequences:
  - Word-stem == target-stem → the word *is* the term in another grammatical form and is returned
    verbatim, so declension survives (`zweizeitigen`, `Miktionen`, `Betmigas` are no longer
    flattened onto `zweizeitige` / `Miktion` / `Betmiga`).
  - A misspelled inflected form is repaired on the stem and keeps its own ending:
    `zweiseitigen` → `zweizeitige` + `n` = `zweizeitigen`.
  - Replacements always come from a **whole** canonical term; a shortened target stem is used only
    for the recognition test above (`Gutwein` minus `n` would otherwise pull `gute` → `Gutwei`).
    The ending is not carried over when the term already ends that way (no `Besten` + `en`),
    and it is re-cased after the canonical term rather than kept as dictated (`MIKTZIONEN` →
    `Miktionen`, not `MiktionEN`).
  - Candidates rank by edit distance, then by shortest stripped word-suffix — so an unsplit match
    wins a tie (`Tovias` → `Toviaz`, not `Toviazs`).
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

- **Targets** are auto-detected from the same canonical terms (`fuzzy_terms()`: plain lines +
  pair RHS, so `Türb=TUR-B` registers `TUR-B`): only letters/digits/hyphens with **2–6 letters,
  all uppercase** (`MRI`, `PSA`, `EKG`, `TUR-B`).
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
- `PATCH /config` — update settings (the union — includes `context_biasing`, `language`, `qwen_model_dir`, `qwen_model_size`, `qwen_model_dir_alt`, `qwen_model_size_alt`, `lora_adapter_qwen`, `lora_adapter_qwen_alt`); runtime params apply immediately (except `delay`, see above), startup params written to file but need a restart (`qwen_active_slot` is the one exception — it changes at runtime via `POST /qwen/switch` instead, never via `PATCH /config`). Validates: `delay` ∈ [1,30]; `rms_ema` ∈ [0,1]; `silence_threshold` ∈ [0,1]; `fuzzy_max_ratio` ∈ [0,1]; `silence_flush` ∈ [1,250]; `min_speech` ∈ [1,250] (`silence_flush = 0` would silently disable silence-triggered finals).
- `GET /words` — `{"words":[...]}`
- `POST /words` — `{"add":[...],"remove":[...]}` — updates file + rebuilds corrector
- `POST /lora/reload?model=voxtral|qwen` — hot-reload the model's LoRA adapter without restart (default voxtral; same for all `?model=` params below); optional JSON body `{"path":"..."}` to specify dir (omit the body entirely — not just the field — to reload current); returns `{"status":"ok","action":"applied"|"cleared","path":"..."}`
- `DELETE /lora?model=voxtral|qwen` — unload the model's active LoRA adapter in-memory (revert to base model) without restart; adapter files on disk are left untouched; clears the model's lora path so nothing is re-applied on the next training reload
- `POST /qwen/switch?slot=primary|secondary` — hot-swap the loaded Qwen checkpoint (e.g. 0.6B <-> 1.7B) without a restart; `slot` is required and explicit, not a toggle. 422 if the target slot isn't configured (`qwen_model_dir`/`qwen_model_dir_alt`). Unloads the current engine, frees its VRAM, loads the target, re-applies that slot's own LoRA if configured, and persists `qwen_active_slot` to config.toml so a restart resumes there. Returns `{"status":"ok","active_slot":...,"model_dir":...,"model_size":...,"lora_active":...}`. While switching (a few seconds), new `?model=qwen` sessions get the same "engine not loaded" error already used during training unload. 409 while a training run (incl. its engine reload) is in progress; concurrent switches are serialised behind `qwen_switch_lock` (held for the whole load + persist).
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
- `POST /training/pair?text=<url-encoded>` — body: raw f32 LE PCM at 16kHz (always; MediaRecorder path dropped — container bytes caused static); saves 16-bit WAV + appends to `pairs.jsonl`; returns `{"id","duration_s","count"}`. IDs come from `allocate_id()`: `max(highest JSONL id, persisted high-water counter)+1`, the counter living in `training/pairs.last_id` (reviews: `training/review.last_id`), so an ID is never reused — not even after deleting the newest entry or `DELETE /training`. Upload + delete are serialised via a `pair_write_lock` mutex so concurrent requests can't collide on an ID or rewrite the JSONL mid-append.
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
- `POST /training/run?model=voxtral|qwen` — unloads **both** engines from VRAM (frees ~9.5 GB for the trainer), then spawns the model's trainer script (`tools/train_lora_voxtral.py` / `tools/train_lora_qwen.py`; voxtral falls back to the legacy `train_lora.py` name for old installs); 202 if started, 409 if already running (one `training_status` — only one training at a time regardless of model; `running` lasts until the engine reload after training is done) or while a `/qwen/switch` is in progress. The shared pairs pool trains both models' LoRAs. Both engines are reloaded automatically when training finishes; ASR requests on either engine error with "Server is training…" until then.
- `GET /training/status` — `{"status":"idle"|"running"|"done"|"error","log":[...]}`

**Dictation review (real dictations as training-pair candidates):** read-aloud calibration
sentences train the LoRA on a different speaking style than free dictation. The client can save
a finished real dictation (audio + model transcript) as a *candidate*; the user later plays it,
optionally re-transcribes it, corrects the text, and accepts it into the training set — or
discards it. Candidates live in `training/review/*.wav` + `training/review.jsonl` and are
invisible to the trainer until accepted.

- `POST /training/review?text=<url-encoded>[&events=<url-encoded-json>]` — body raw f32 LE PCM 16kHz; `text` = model transcript at save time; same ID scheme + `pair_write_lock` as pairs. `events` (optional) is a JSON array of per-final markers the userscript collected during the dictation — `{"type":"text","text":...,"t":ms}` for normal finals and `{"type":"control","action":"delete_word"|"delete_sentence"|"delete_punctuation"|"delete_newline","matched":...,"t":ms}` for each control word that fired (see Control words, below). Stored verbatim as an `events` field in the `review.jsonl` entry when present and parseable; absent for older clients or calibration pairs (`POST /training/pair` never carries it).
- `GET /training/reviews` — `{"reviews":[{"id","text","duration_s"}]}` sorted by id
- `GET /training/review/audio/{id}` — serve candidate WAV (playback / client-side re-transcription)
- `POST /training/review/{id}/accept` — `{"text":"…"}` (corrected transcript) — moves the WAV into `training/audio/` under a fresh pair ID, appends to `pairs.jsonl`, removes the candidate
- `DELETE /training/review/{id}` — discard candidate (WAV + JSONL entry). Like the pair delete, a missing WAV does not block removing the JSONL entry; 404 only when neither existed.

**Edit-log mining:**
**No longer called by the userscript (v0.1.29+):** an overlay edit can't distinguish a
mishearing from a content change (`rechts` → `links` because you misspoke would become a
`rechts=links` pair silently rewriting every future dictation). Word pairs are now added only
deliberately via right-click → correct. The endpoints remain server-side, unused.

- `POST /log/edit` — `{"original","edited","timestamp"}` — appended to `edit_log.jsonl` (formerly sent by the userscript when a commit-mode dictation was edited before insertion). The handler then runs `word_diffs(original, edited)` (LCS word diff; changed runs ≤4 words paired as removed→inserted, punctuation-only changes skipped) and auto-adds each pair as a `wrong=correct` line straight into `custom_words.txt` via the same read-modify-write + corrector-rebuild path as `POST /words` (`apply_words_patch`) — no separate suggestions/review step. An unwanted auto-added pair is removed the same way any custom word is: delete the line in the Eigene Wörter tab; dictating the correction again re-derives it if still needed.
- `GET /edits/report` — `{"entries":N,"suggestions":[{"original","edited","count"},…]}` — aggregates the edit log into the most frequent word-level corrections, same `word_diffs` logic, top 30 by count. Diagnostic only now (no UI reads it) since corrections are auto-added on every edit.

Training data stored in `~/.config/voicetserver/training/audio/*.wav` + `pairs.jsonl` (one
pool — it trains both models' LoRAs). Adapter output is per-model:
`~/.config/voicetserver/lora_adapter/` (voxtral) and `lora_adapter_qwen/` (qwen); set
`lora_adapter` / `lora_adapter_qwen` in config to load at startup.

`tools/import_pairs.py` — one-time importer that appends another server's collected pairs
into the pool, re-IDing with the server's rule (`max(highest id, training/pairs.last_id)+1`)
and advancing `pairs.last_id` past the imported IDs. Run with voicetserver stopped (the `pair_write_lock` is in-process
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

**Unified frontend** (v0.1.16+): one userscript, one server, one URL + API key. A single
"Modell:" dropdown above the tab bar (`renderModelSelect()`) covers every engine/size
combination the server has loaded: "Voxtral" and, when the qwen engine is enabled, "Qwen
`<primary size>`" plus — only when `qwen_model_dir_alt` is configured — "Qwen `<secondary
size>`", labelled from `GET /config`'s `qwen_model_size`/`qwen_model_size_alt`. Its option
values encode both halves of the state as `"voxtral"` or `"qwen:primary"`/`"qwen:secondary"`;
selecting an option sets the client-side engine choice (GM value `active_model` → `?model=`
on the WS URL, instant, no save needed) and, for a `"qwen:"` option, hot-swaps the loaded
checkpoint via `POST /qwen/switch?slot=` when it differs from `qwen_active_slot` (see
Runtime-switchable second size, above) — both blocked while recording.
Storage keys are the dual-backend era's Voxtral
profile keys (`server_url_voxtral`/`api_key_voxtral`, falling back to the pre-profile
`server_url`/`api_key`; `active_backend` seeds `active_model`), so both upgrade and rollback
keep working; the qwen profile keys stay dormant. The Einstellungen tab is a single pane
driven by the union `GET /config`: qwen rows (`context_biasing`, language) are hidden when
the server's `models` list lacks `"qwen"`, `delay`/`german_prime` rows appear only when the
server reports those fields — so the script still degrades gracefully against an old
single-model server.

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

Right-click → eight tabs: **Eigene Wörter** (server-side custom words — added manually or via right-click → correct
on overlay text; edits before insertion are no longer mined, see Edit-log mining above),
**Hotwords** (client-side GM-stored list sent per session
via `?hotwords=`; biasing only on Qwen3), **Steuerwörter** (client-side GM-stored control-word
vocabulary, see Control words below), **Aufnehmen** (record calibration sentences),
**2. Durchgang** (record once-recorded sentences — `pair_ids.length === 1` — a second time),
**Training** (review pairs, delete, LoRA), **Diktate** (real-dictation review, below),
**Einstellungen** (server URL/key, hotkey, runtime params).

**Control words (delete word/sentence/punctuation/newline):** for pause-heavy dictation where
a mistake needs fixing mid-flow without reaching for the mouse. A **whole** pause-bounded
`final` (not a substring) that matches a configured phrase is treated as a command instead of
inserted text — matched via `matchControlWord()` against the GM-stored `control_words` list
(Steuerwörter tab, `phrase=action` per line, defaults in `DEFAULT_CONTROL_WORDS`; purely
client-side, no server round-trip, same storage pattern as Hotwords). Four actions:
`delete_word`, `delete_sentence`, `delete_punctuation`, `delete_newline` — each deletes
backward from the current cursor position (not the field's absolute end), computed once by
`computeDeleteCount()` (word = last whitespace-delimited run; sentence = back to the
second-to-last `[.!?]` terminator; punctuation/newline = one trailing char) and physically
applied differently per target: `deleteInTextarea()` splices `el.value` directly;
`deleteInContentEditable()` reuses the browser's own text model
(`Selection.modify("extend","backward",…)` + `document.execCommand("delete")` — native
word/sentence granularity, precomputed character count for punctuation/newline) so native undo
(Ctrl+Z) and the host page's own input listeners keep working; in commit mode
`deleteFromBuffer()` does plain string surgery on `pendingText`/`overlayTextDiv` (nothing has
hit the host page yet), the same precedent as the right-click word-correction splice. Feedback
via `showToast()`. Every text final and control-word action is pushed to a per-session
`dictationEvents` array (reset in `startRecording()`, attached to `lastDictation.events` in
`snapshotDictation()`) and sent as `POST /training/review`'s optional `events=` param when a
dictation is saved — see the `events` field under `POST /training/review`, above.

**Audio capture (all paths — ASR, Aufnehmen, 2. Durchgang):** `createCaptureContext()` requests
an `AudioContext({ sampleRate: 16000 })` so the **browser** resamples the mic stream with proper
low-pass filtering. The old path decimated the native 48 kHz stream (every 3rd sample), folding
high frequencies into the speech band as aliasing noise. Naive decimation survives only as a
fallback (`captureChunk()`) for browsers that refuse a fixed-rate context. The mic is opened
with `MIC_CONSTRAINTS` (`autoGainControl: false`; echo cancellation / noise suppression left at
browser defaults): browser AGC ramps the gain up during pauses, boosting distant background
speech into the ASR engine, where Qwen turns it into hallucinated sentences. The close-talk
headsets in use (Jabra Evolve 65 TE / Evolve2 65) level the mic themselves.

**Diktate tab (real dictations → training pairs):** every dictation's 16 kHz PCM is kept
client-side (`dictationPcmBuffers` → `lastDictation` snapshot on stop, together with the
transcript — the edited overlay text in commit mode). Nothing is saved automatically: the commit-mode overlay has
plain **↵** (insert only) and **↵💾** (insert + save, with the edited overlay text);
"💾 Letztes Diktat speichern" in the tab saves without inserting. Saving POSTs to
`/training/review`. A **▶** button (overlay toolbar, and "▶ Anhören" in the tab) plays the
last dictation's PCM client-side (`toggleDictationPlayback()`) to check recording quality when
a transcript comes out bad. The snapshot's long pauses are shortened (`compressSilences()`:
100 ms blocks below `silence_threshold`, runs > 1.0 s cut to 0.5 s keeping 0.25 s at each
edge) — for replay and saved candidates only; the live stream to the server is untouched and
`events` timestamps keep the original timeline. The tab lists all
candidates with ▶ playback, ⬇ WAV download (`downloadReviewAudio()`, authFetch + object URL) and ✕ delete; selecting one opens a detail area with an editable
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

Each socket's handlers are bound to their own instance (`if (sock !== ws) return;`), so a
stale socket (replaced by a reconnect, or already finalized) can never deliver a late final or
call `finalizeStop()` on the current session. `startRecording()` ignores toggles while
`starting` (getUserMedia pending — a second stream would leak the mic) and while
`awaitingFinalFlush` (toast "Bitte warten — Diktat wird abgeschlossen"). A server
`{"type":"error"}` frame is shown as a toast and stops the recording without reconnect
retries (with `recording` false, the following onclose just finalizes).

## Silence / final message behaviour

`ChunkOutput::Silence` fires after silence threshold; server calls `take_text_buf()` which
returns and clears the accumulated text. Do NOT call `self.text_buf.clear()` before this —
it would cause every silence-triggered final to send empty text.

Custom word correction is applied to the text returned by `take_text_buf()` before sending.

Arming (both engines): the detector arms after `min_speech` loud ticks (smoothed RMS ≥
`silence_threshold`) counted **cumulatively** since the last reset, not consecutively — a dip
between syllables does not restart the count, so short utterances and control words
("Wort löschen") arm and become their own final. The count restarts after a final and after a
full `silence_flush` run while unarmed, so isolated blips spread over a long pause don't add
up. Default `min_speech` = 4 (was 15 consecutive ≈ 1.2 s Voxtral / 1.5 s Qwen).

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

# Versioning

Semantic versioning in `Cargo.toml`. Increment patch on each push.

# Multi-User Architecture (Exploration Notes)

Not yet implemented — single-user server. Notes for future reference.

## Shared LoRA adapter for two users

Train a single LoRA on both speakers' audio combined. Both have the same vocabulary; the adapter mainly corrects phonetic patterns. Start with a combined adapter — if quality is unacceptable for one user, train separate adapters and use `POST /lora/reload` to switch.

## Concurrent sessions

**Same LoRA adapter:** Qwen: works (per-session streaming state). Voxtral: **one session at a time** — the KV caches live in the shared `ModelInner` and a new session's prefill resets them, so the GPU lock alone does not isolate sessions; a second session is refused (see GPU locks). Concurrent Voxtral sessions would need per-session KV caches in `StreamingState`.

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

# NixOS migration (planned, not started — noted 2026-10-18)

The GPU host is to move to NixOS together with `uplprod` (flake `~/nixos`, step 7 of
`docs/nixos-server-plan.md`). Deferred deliberately: while the server is still being iterated on,
testing on Ubuntu is faster than a pin bump + rebuild. The single-binary architecture stays as is —
only the deployment around it changes.

**Target shape**
- Built from source in the flake (`rustPlatform.buildRustPackage`, `cargoLock.lockFile`), source
  pinned by hand like `pkgs/wwphone` / `pkgs/tabby` there — a pin bump is the deploy. The prebuilt
  binaries in `tools/` are not used on NixOS.
- A NixOS module declares the systemd unit; `install.sh` and its generated
  `voicetserver.service` are not used on NixOS.
- Model weights (~11 GB) and `data_dir` stay on a ZFS dataset (the installer's base-directory
  layout), never in the Nix store. `config.toml` stays mutable on disk — Nix supplies paths and
  environment only, not the config content.
- API key via agenix.

**Known hurdles in the CUDA build**
- `cudaforge` clones CUTLASS from GitHub at build time (see Build); the Nix sandbox has no
  network → CUTLASS must be a pinned fetched source with `CUDAFORGE_HOME` pointed at it.
- CUDA toolkit from `cudaPackages` (unfree), `CUDA_COMPUTE_CAP=89`.
- `libcuda.so.1` is `dlopen`ed at runtime (see CUDA driver loading) — on NixOS it lives in
  `/run/opengl-driver/lib` (`addDriverRunpath`), replacing the current unit's copied
  `LD_LIBRARY_PATH`/`PATH` from the installing shell.
- No binary cache for this build: flash-attn kernels compile on the server on each pin bump.
- Fallback if the source build fights back: build on the dev box as today and have Nix fetch that
  binary with `autoPatchelfHook`.

**Code changes needed before/at migration**
- A foreground mode for systemd: no fork/detach, no watchdog, no PID file, log to stdout (journal)
  instead of `~/.config/voicetserver/logs/`. Check what the binary does with no TTY; add
  `--foreground` if needed. Until then, don't extend the daemon/watchdog code further.
- `restartIfChanged = false` on the unit: a restart drops live dictation sessions and reloads the
  weights, so restarts after a deploy are done deliberately, outside practice hours.
