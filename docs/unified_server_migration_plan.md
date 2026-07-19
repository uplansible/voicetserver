# Unified server migration plan — one server, both models (Voxtral + Qwen3-ASR)

Status: **phases 1–7 implemented** on the `unified` branch (production burn-in +
schmidiscribe retirement pending, see phase 7 step 3). Written 2026-07-18. Baseline commits:
voicetserver `ac9140c` (v0.1.22), schmidiscribe `5d9fbf9` (v0.1.17) — both pushed to `main`.

## Decision: where the merged code lives

**New branch `unified` in the voicetserver repo.** Not the schreibot repo, not a new repo.

Rationale:
- The merged server is ~90% voicetserver code (feature superset: review queue, edit-log
  mining, abbrev expander, watchdog/daemon machinery, installer). Branching keeps history,
  blame, CLAUDE.md and the vendored `candle-fork/` in place.
- `main` stays deployable/stable the whole time; merge `unified` → `main` when proven.
- schreibot (failed whisper test) should be nuked independently — reusing its repo would
  mean an unrelated history under a misleading name. Nuke-and-pave it whenever, separately.
- schmidiscribe repo stays as-is (still deployable standalone); retire/archive it only after
  the unified server is stable in production.

Versioning: continue voicetserver's Cargo.toml versioning on the branch; bump to **0.2.0**
when merging to `main`.

## End state

One binary `voicetserver` serving **both** models on one port (8765), one TLS cert, one API
key, one config file, one data dir:

- Voxtral Mini 4B Realtime (~8 GB VRAM, BF16) — current engine, unchanged.
- Qwen3-ASR-0.6B (~1.5 GB VRAM) via vendored `qwen3-asr-rs` — ported from schmidiscribe.
- Both resident simultaneously on the 16 GB card (headroom left for KV caches).
- WS session picks the model: `ws://…/asr?model=voxtral|qwen` (default configurable,
  fallback `voxtral`). Each engine has its **own** GPU mutex → sessions on different
  models can run concurrently.
- Shared training data: one pairs/review/sentences pool trains **both** models' LoRAs.
- Frontend: single URL + API key; the Voxtral/Qwen3 switcher just changes `?model=`.

## Key compatibility facts (verified 2026-07-18)

- `candle-fork/` is candle **0.9.2**; `qwen3-asr-rs` depends on candle-core/candle-nn
  **0.9.2** from crates.io → repoint its deps to the fork paths so there is exactly one
  candle build / CUDA context. Fork has Voxtral-specific additions but the 0.9.2 API base
  is the same. **Compile-test this first — it is the only real technical risk.**
- Both servers already share the WS protocol (raw f32 LE PCM 16 kHz in; partial/final/error
  JSON out; `"stop"` text frame → drain/finish → final → close) and most of the HTTP API.
- LoRA weight-key formats differ per model:
  - Voxtral: `layers.{i}.attention.{wq,wk,wv,wo}.lora_{a,b}.weight` (src/lora.rs)
  - Qwen3: `q_proj`/`v_proj` attention keys (schmidiscribe tools/train_lora.py)
  → adapters are strictly per-model.
- Trainer scripts differ: voicetserver `tools/train_lora.py` (Voxtral, mistral-common) vs
  schmidiscribe `tools/train_lora.py` (Qwen3, tokenizers/transformers). Both must ship.
- schmidiscribe extras to port: `has_speech()` partial-suppression gate, context biasing
  (`?hotwords=`/`?patient=` → system prompt via `StreamingOptions::with_context()`),
  forced language (`--language German`, `?lang=` override), `context_biasing` toggle.
- voicetserver extras that stay: german_prime, AbbrevExpander, review queue, edit-log
  mining, auto-unload for training, 64 MB body limit.

## Phases

### Phase 1 — vendor + compile (risk retirement)

1. `git checkout -b unified` in voicetserver.
2. Copy `/home/schmidi/schmidiscribe/qwen3-asr-rs/` → `voicetserver/qwen3-asr-rs/`
   (source-vendor like candle-fork; drop its `.git` if any).
3. Edit `qwen3-asr-rs/Cargo.toml`: `candle-core = { path = "../candle-fork/candle-core" }`,
   same for candle-nn.
4. voicetserver `Cargo.toml`: add `qwen3-asr = { path = "qwen3-asr-rs", default-features
   = false }`; extend `cuda` feature with `"qwen3-asr/cuda"`.
5. `cargo check` (CPU) then full CUDA build (see CLAUDE.md build command). Fix any fork/API
   drift now. **Do not proceed until this builds.**

### Phase 2 — engine plumbing

1. New `src/qwen.rs` (or extend main.rs): wrap `qwen3_asr::AsrInference` behind its own
   `tokio::sync::Mutex<Option<…>>` (same unload pattern as `ModelInner`).
2. `AppState`: add the qwen engine alongside `VoxtralModel`.
3. `src/config.rs` + `src/settings.rs`: new startup fields `qwen_model_dir` (Option — if
   unset, qwen engine disabled and `/asr?model=qwen` returns an error frame), `language`
   (default `"German"`); new runtime field `context_biasing` (default true).
4. Startup: load Voxtral as today; load qwen if `qwen_model_dir` set. `--version` and
   startup banner mention both.
5. Qwen model files (from schmidiscribe CLAUDE.md): `model.safetensors` (~1.8 GB),
   `config.json`, `tokenizer.json` (generated via transformers if missing). Currently at
   `/home/schmidi/qwen3-asr-rs/models` on the dev/prod box — reuse that path in config.

### Phase 3 — WS routing + qwen streaming

1. Port schmidiscribe `src/streaming.rs` → `src/qwen_streaming.rs`: wraps
   `qwen3_asr::StreamingState`, SilenceDetector with `has_speech()` gate (suppresses
   hallucinated partials during silence — do not lose this), silence reset carrying last
   200 chars as initial_text.
2. `handle_asr_session`: parse `?model=`; dispatch to the Voxtral path (unchanged) or a new
   qwen path (port schmidiscribe's handler: spawn_blocking inference, context biasing from
   `collect_terms()` = custom_words plain terms + `?hotwords=` + `?patient=`, gated by
   `context_biasing`; `?lang=` override).
3. Both paths run the same finalize pipeline: literal WordsCorrector → AbbrevExpander →
   FuzzyMatcher (`finalize_text()` in main.rs). Qwen gains abbrev expansion for free —
   schmidiscribe never had it.
4. Stop protocol: Voxtral drains `delay + 3` ticks; qwen calls `finish_streaming()`. Same
   client-visible behaviour.

### Phase 4 — API surface

1. `GET /config`: union of both models' fields (`delay`, `german_prime` for voxtral;
   `context_biasing`, `language` for qwen; shared silence/fuzzy params). Report
   `"server":"voicetserver"` **plus** `"models":["voxtral","qwen"]` (or `["voxtral"]` if
   qwen disabled). Per-model LoRA state: `lora_active_voxtral`, `lora_active_qwen`,
   `lora_dir_voxtral`, `lora_dir_qwen`. Keep old `lora_active`/`lora_dir` as voxtral
   aliases during the frontend transition.
2. `PATCH /config`: accept the union; existing validation stays.
3. LoRA endpoints gain a model scope: `POST /lora/reload?model=…`, `DELETE /lora?model=…`
   (default voxtral). Separate adapter dirs: `lora_adapter/` (voxtral, unchanged) and
   `lora_adapter_qwen/`; config keys `lora_adapter` (existing) + `lora_adapter_qwen`.
4. `POST /training/run?model=voxtral|qwen`: picks the trainer script. Ship both as
   `tools/train_lora_voxtral.py` and `tools/train_lora_qwen.py`; update `find_script()`
   and the installer deploy step. **Unload both engines** before spawning either trainer
   (16 GB card; simplest safe rule), reload both after. One `training_state` — only one
   training at a time regardless of model.
5. Everything else (words, sentences, pairs, review, edit-log, health) is already
   model-agnostic — no changes.

### Phase 5 — data migration

1. Unified data dir = voicetserver's (`~/.config/voicetserver/` or configured `data_dir`).
2. `custom_words.txt`: merge schmidiscribe's entries in by hand (they overlap heavily);
   plain terms feed fuzzy + abbrev on both models and prompt biasing on qwen.
3. Training pairs/sentences: voicetserver's pool is canonical. Optionally import
   schmidiscribe's `training/audio/*.wav` + `pairs.jsonl` (re-ID via the normal
   `max(id)+1` append path — small importer script or manual curl loop). **Open
   question for the user: import or start fresh?**
4. Python venv: the unified venv needs the union of trainer deps: torch, safetensors,
   numpy, tqdm, packaging + `mistral-common` (voxtral) + `tokenizers`, `transformers`
   (qwen). One `venv_path` config key as today.

### Phase 6 — frontend (schmidispeech.user.js)

1. Collapse per-backend URL/key profiles into **one** server URL + API key (migration:
   read `server_url_voxtral`/`api_key_voxtral` as the initial value; keep the qwen keys
   ignored/dormant).
2. The Voxtral/Qwen3 switcher now sets a GM value `active_model` and appends
   `?model=` on WS connect (both live sessions and `transcribePcm()` — Diktate
   re-transcribe buttons pass the model instead of a different URL/key).
3. Einstellungen: single settings pane driven by the union `GET /config`; hide the qwen
   rows when `models` lacks `"qwen"`. Keep field-presence gating so the script still works
   against an old single-model server during transition.
4. Training/Diktate/Wörter tabs: no changes (same endpoints, one server). LoRA toggle
   becomes two checkboxes (per model) driven by `lora_active_voxtral`/`_qwen`.
5. Bump `@version`; the userscript no longer needs schmidiscribe compatibility once the
   unified server is deployed, but keep the graceful 404 handling anyway.

### Phase 7 — installer, docs, deprecation

1. `tools/install.sh`: optionally download Qwen3 model files; deploy both trainer scripts;
   config template gains `qwen_model_dir`, `lora_adapter_qwen`.
2. CLAUDE.md: rewrite architecture/API sections for two engines; note qwen is optional.
3. After production burn-in: stop the schmidiscribe service (port 8767), archive the repo
   with a pointer to voicetserver. Nuke schreibot independently.

## Testing checklist (per phase and before merge)

- CUDA build + CPU `cargo check` both pass.
- Offline WAV mode for both models (add `--asr-model qwen` CLI flag or reuse `?model=`
  default) — A/B same WAV.
- Live dictation each model; switch models between sessions without restart; two
  concurrent sessions (one per model).
- `stop` drain: last words not truncated on either model.
- Fuzzy/abbrev/custom-words pipeline fires on both models' finals.
- LoRA: train voxtral (model unload → train → reload, qwen back too), train qwen,
  per-model toggle on/off, `GET /config` state correct.
- Silence handling: no hallucinated partials on qwen during long pauses (has_speech gate).
- Auth: `/health` public, everything else 401 without key; WS `?api_key=`.
- Userscript against unified server AND against old schmidiscribe (graceful degradation).

## Open questions (defaults chosen, user can override)

1. Import schmidiscribe's existing training pairs into the shared pool? *Default: yes,
   via append-path importer.*
2. Default model for sessions without `?model=`? *Default: voxtral (config key
   `default_model`).*
3. Keep qwen optional via missing `qwen_model_dir` (saves ~1.5 GB VRAM when unused)?
   *Default: yes.*
