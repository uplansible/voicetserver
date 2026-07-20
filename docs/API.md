# voicetserver — API Reference

Complete reference for the HTTP + WebSocket API of the unified two-engine ASR server
(Voxtral Mini 4B Realtime + Qwen3-ASR-0.6B).

Source of truth: the `server` module in `src/main.rs`.

---

## Conventions

**Base URL** — `http(s)://<host>:<port>` (default port `8765`). TLS is enabled when both
`tls_cert` and `tls_key` are configured; then use `https://` / `wss://`.

**Authentication** — every endpoint except `GET /health` requires the API key:

| Transport | How to pass the key |
|-----------|---------------------|
| HTTP      | `X-Api-Key: <key>` header |
| WebSocket | `?api_key=<key>` query param (browsers cannot set headers on the WS upgrade) |

The `?api_key=` query param is accepted on **all** endpoints, not just the WebSocket — the
header is preferred for HTTP because query strings leak into access logs and browser history.
Comparison is constant-time (`ct_eq`). Failure → `401 Unauthorized` with a plain-text body.

The key is auto-generated (16 random bytes, hex) on first server start, persisted to
`~/.config/voicetserver/config.toml` (chmod `0600`), and printed in a box at startup when newly
generated.

**CORS** — `Access-Control-Allow-Origin: *`, methods `GET, POST, PATCH, DELETE, OPTIONS`, all
headers allowed. The CORS layer sits *outside* the auth middleware, so preflight `OPTIONS`
requests are answered without a key.

**Body limit** — 64 MB (`DefaultBodyLimit`). Sized for whole-dictation f32 PCM uploads
(64 KB/s → ~16 min).

**Error bodies** — JSON `{"error": "..."}` for the LoRA endpoints, **plain text** everywhere
else. Clients should parse defensively (`try { JSON.parse(raw) } catch {}`).

**Model scoping** — endpoints that act on one engine take `?model=voxtral|qwen`. Absent or empty
selects voxtral; an unrecognised value (e.g. a typo like `?model=qwem`) is rejected — `400` with
a plain-text body on HTTP, an `{"type":"error"}` frame on the WebSocket — rather than silently
falling back to voxtral.

---

## Health

### `GET /health`

**Public — no API key required.** The only unauthenticated endpoint.

```json
{ "status": "ready", "connections": 2 }
```

`connections` is the number of currently open WebSocket ASR sessions (both engines).

`status` is `"ready"` or `"loading"`. It reports `"loading"` during startup model loading and for
the whole LoRA-training window — from the moment the engines are unloaded, through the trainer
subprocess, until both engines are reloaded (10–30 s) — so a monitor or load balancer can tell
when ASR sessions would error out. If the reload after training fails, it stays `"loading"`.
`GET /training/status` still distinguishes *why*.

---

## Configuration

### `GET /config`

Returns the union of both engines' settings: live runtime values (read from atomics) plus the
startup snapshot.

```json
{
  "version": "0.1.26",
  "server": "voicetserver",
  "models": ["voxtral", "qwen"],

  "delay": 6,
  "silence_threshold": 0.006,
  "silence_flush": 20,
  "min_speech": 15,
  "rms_ema": 0.3,
  "fuzzy_hotwords": true,
  "fuzzy_max_ratio": 0.34,
  "german_prime": false,
  "context_biasing": true,

  "model_dir": "/models/Voxtral-Mini-4B-Realtime",
  "qwen_model_dir": "/models/Qwen3-ASR-0.6B",
  "language": "German",
  "device": 0,
  "port": 8765,
  "bind_addr": "0.0.0.0",
  "tls_enabled": true,
  "lora_adapter": null,
  "lora_adapter_qwen": null,
  "venv_path": "/mnt/ssdupl/voicetserver-venv",

  "lora_active": false,
  "lora_dir": "/home/user/.config/voicetserver/lora_adapter",
  "lora_active_voxtral": false,
  "lora_dir_voxtral": "/home/user/.config/voicetserver/lora_adapter",
  "lora_active_qwen": false,
  "lora_dir_qwen": "/home/user/.config/voicetserver/lora_adapter_qwen",

  "_startup_only": ["model_dir", "qwen_model_dir", "language", "device", "port",
                    "bind_addr", "tls_cert", "tls_key", "lora_adapter",
                    "lora_adapter_qwen", "venv_path"],
  "_note": "Changing startup_only fields writes to config file but requires server restart."
}
```

Key fields:

- **`server`** — backend identity (`"voicetserver"`), lets the shared userscript adapt its UI.
- **`models`** — `["voxtral", "qwen"]`, or `["voxtral"]` when `qwen_model_dir` is unset. The
  frontend hides all qwen UI when `"qwen"` is absent.
- **`lora_active_<model>`** — is a LoRA currently applied in memory for that engine.
- **`lora_dir_<model>`** — the path the frontend's "LoRA verwenden" toggle re-applies on enable:
  the active path if loaded, else the configured `lora_adapter`/`lora_adapter_qwen`, else the
  model's default training output dir. This is what keeps the toggle usable after `DELETE /lora`.
- **`lora_active` / `lora_dir`** — unsuffixed voxtral aliases, retained from the frontend
  transition.
- `tls_cert` / `tls_key` are never echoed — only the derived boolean `tls_enabled`.
- The API key is never echoed.

### `PATCH /config`

Body: JSON object with any subset of the fields below. Absent fields are left unchanged;
explicit `null` is treated as absent (fields cannot be unset via the API).

Applies runtime params to the shared atomics immediately, writes **all** supplied fields to
`config.toml`, then returns the same body as `GET /config`.

**Runtime-adjustable** (take effect immediately or on the next session):

| Field | Type | Validation | Notes |
|-------|------|-----------|-------|
| `delay` | int | 1–30 | 80 ms of lookahead each; new sessions only |
| `silence_threshold` | float | 0.0–1.0 | RMS threshold |
| `silence_flush` | int | 1–250 | consecutive silent chunks before a final; `0` is rejected (it would disable silence finals entirely) |
| `min_speech` | int | 1–250 | speech chunks before silence detection arms |
| `rms_ema` | float | 0.0–1.0 | EMA smoothing |
| `fuzzy_hotwords` | bool | — | fuzzy phonetic snapping on/off |
| `fuzzy_max_ratio` | float | 0.0–1.0 | lower = stricter |
| `german_prime` | bool | — | experimental Voxtral prefill priming; next session |
| `context_biasing` | bool | — | qwen prompt biasing; next session |

**Startup-only** (persisted to `config.toml`, **require a restart**): `model_dir`,
`qwen_model_dir`, `language`, `device`, `port`, `bind_addr`, `tls_cert`, `tls_key`,
`lora_adapter`, `lora_adapter_qwen`, `venv_path`.

Responses: `200` with the new config · `422 Unprocessable Entity` (plain text) on a range
violation · `500` if the config file could not be written.

> Writing the config file re-serializes it from the in-memory struct, so **the comments in the
> generated `config.toml` template are lost after the first `PATCH /config`**.

---

## Transcription (WebSocket)

### `ws[s]://<host>:<port>/asr`

Query parameters:

| Param | Applies to | Description |
|-------|-----------|-------------|
| `api_key` | both | required |
| `model` | both | `voxtral` (default) or `qwen`; anything else → voxtral |
| `lang` | qwen only | overrides the configured `language` for this session |
| `hotwords` | qwen only | extra biasing terms, separated by `,` `;` or newline |
| `patient` | qwen only | patient context string for prompt biasing |

`lang` / `hotwords` / `patient` are accepted but **ignored** on Voxtral sessions — Voxtral has
no language token and no biasing mechanism.

`hotwords` and `patient` are merged with the plain terms from `custom_words.txt`, deduplicated
order-preservingly, and injected into the qwen system prompt as
`Eigennamen und Fachbegriffe: <a>, <b>, ….` — only when the runtime `context_biasing` toggle
is on (read once at connection start).

**Client → server**

- **Binary frames** — raw f32 LE PCM, 16 kHz, mono. Byte length must be a multiple of 4;
  a trailing partial sample is discarded. Empty decodes are skipped.
- **Text frame `"stop"`** — graceful stop (matched after trimming, ASCII **case-insensitive**,
  so `"Stop"` / `"STOP"` work too). Any other text frame is ignored.

**Server → client** (all JSON text frames)

```json
{ "type": "partial", "text": "…" }
{ "type": "final",   "text": "…" }
{ "type": "error",   "text": "…" }
```

- `partial` — incremental decoded text. Only `ß→ss` normalization is applied.
- `final` — a completed segment, emitted on silence detection and once at end of session. Runs
  the full correction pipeline (below).
- `error` — session-fatal; the socket closes afterwards.

**Correction pipeline** (finals only, in this order):

1. literal `wrong=correct` replacements from `custom_words.txt` (aho-corasick)
2. abbreviation letter-name expansion (`Em Er I` → `MRI`)
3. fuzzy phonetic snapping onto plain terms (Kölner Phonetik + Levenshtein), if
   `fuzzy_hotwords` is on
4. `ß` → `ss` (Swiss orthography, unconditional)

**Stop protocol** — identical on both engines:

1. client sends the text frame `"stop"`
2. server drains the engine — Voxtral feeds `delay + 3` ticks of silence
   (`StreamingState::drain_sync`) because the decoder lags the audio by `delay` tokens; qwen
   calls `finish_streaming()`
3. server sends the remaining text as a `{"type":"final"}` (omitted if empty) and closes
4. client waits for `ws.onclose` (2 s fallback) before finalizing

A plain client close without `"stop"` triggers the same drain + flush, but the final may be
lost if the client is already gone.

**Error conditions** (delivered as an `error` frame, then close):

| Condition | Message |
|-----------|---------|
| Engines unloaded for LoRA training | `Server is training a new voice model — please try again in a few minutes` |
| `?model=qwen` with `qwen_model_dir` unset | `Qwen engine not enabled — set qwen_model_dir in config.toml` |

Concurrency: each engine has its own GPU lock, so a Voxtral session and a qwen session run
concurrently; two sessions on the same engine serialize.

---

## Custom words

### `GET /words`

```json
{ "words": ["# Kommentar", "Migration=Miktion", "Betmiga", "MRI"] }
```

Returns the raw lines of `{data_dir}/custom_words.txt`.

### `POST /words`

```json
{ "add": ["Betmiga", "PSA=P S A"], "remove": ["Alterwort"] }
```

Both fields optional. Entries are trimmed; empty ones are dropped. Comment lines (`#…`) are
preserved in their original order and written first; entry lines are stored as a **sorted,
deduplicated set**, so the file is normalized on every write.

Rewrites the file, then hot-reloads the aho-corasick corrector, the fuzzy matcher, and the
acronym expander — no restart needed. Returns the new `{"words": [...]}`.

Line formats:

- `wrong=correct` — literal replacement pair
- `PlainTerm` — fuzzy phonetic target (single all-alphabetic words only) and, when 2–6
  uppercase letters, an acronym-expansion target (`MRI`, `TUR-B`)
- `# …` — comment

---

## Training — calibration sentences

### `GET /training/sentences`

Creates a stub `training_sentences.txt` on first call if absent.

```json
{ "sentences": [
    { "text": "Der Patient klagt über Schmerzen.", "recorded": true, "pair_ids": ["0001"] },
    { "text": "Die Untersuchung war unauffällig.", "recorded": false, "pair_ids": [] }
] }
```

`pair_ids` are the IDs in `pairs.jsonl` whose text matches the sentence exactly (both sides
trimmed). The frontend's "2. Durchgang" tab selects sentences with `pair_ids.length === 1`.

### `POST /training/sentence`

`{"text": "…"}` — appends one sentence. `400` on empty text.
Returns `{"added": true}`.

### `PATCH /training/sentence`

`{"old": "…", "new": "…"}` — replaces the **first** matching line. `400` if `new` is empty,
`404` if `old` was not found. Returns `{"updated": true}`.

### `DELETE /training/sentence`

`{"text": "…"}` — removes the **first** matching line. `404` if not found.
Returns `{"deleted": true}`.

All three serialize on the shared `pair_write_lock`.

---

## Training — pairs

A "pair" is one recorded WAV plus its ground-truth transcript. The pool is **shared by both
engines** — the same pairs train the Voxtral LoRA and the qwen LoRA.

Storage: `{data_dir}/training/audio/{id}.wav` (16-bit mono 16 kHz) + one JSONL line per pair in
`{data_dir}/training/pairs.jsonl`.

IDs are zero-padded 4-digit strings assigned as `max(existing_id) + 1` — **never** the line
count, so deletions can't cause collisions.

### `POST /training/pair?text=<url-encoded>`

Body: raw f32 LE PCM, 16 kHz mono (also accepts anything symphonia can decode — OGG/WAV/WebM —
resampled to 16 kHz mono; raw PCM is the fallback when probing fails).

```json
{ "id": "0042", "duration_s": 3.512, "count": 42 }
```

`400` on an empty body. The `text` query param is required (a missing one is a `400` from the
extractor); an *empty* `text` is accepted and stores a pair with an empty transcript.

### `GET /training/pairs`

```json
{ "pairs": [ { "id": "0001", "text": "…", "duration_s": 3.512 } ] }
```

Sorted numerically by `id`.

### `GET /training/audio/{id}`

Serves the WAV as `audio/wav`. `id` must be all ASCII digits (path-traversal guard) → else
`400`; missing file → `404`.

Browsers cannot send `X-Api-Key` on `<audio src>`, so the frontend fetches this with
`authFetch` + `URL.createObjectURL(blob)`.

### `DELETE /training/pair/{id}`

Removes the WAV and rewrites `pairs.jsonl` without that entry. `400` on a non-numeric id.
A missing WAV does **not** abort the delete — the JSONL entry is removed either way, so a lost
recording cannot leave an entry stuck in the pool for the trainer to choke on. `404` only when
neither the WAV nor a JSONL entry existed. Returns `{"deleted": "0042"}`.

### `GET /training`

```json
{ "count": 120, "duration_sec": 512.4 }
```

### `DELETE /training`

Deletes `{data_dir}/training/audio/` and `pairs.jsonl` — the recorded pairs only. Pending
dictation-review candidates (`training/review/` + `review.jsonl`) live under the same directory
but are **kept**: they are a separate, not-yet-accepted pool. Serialised against pair uploads via
`pair_write_lock`. Returns `{"deleted": true}`.

---

## Training — dictation review

Read-aloud calibration sentences train the LoRA on a different speaking style than free
dictation. These endpoints let the client park a finished real dictation (audio + model
transcript) as a *candidate*, review it later (play, re-transcribe, correct), then accept it
into the training set or discard it.

Candidates live in `{data_dir}/training/review/*.wav` + `review.jsonl` and are **invisible to
the trainer until accepted**. Review IDs are a separate ID space from pair IDs.

### `POST /training/review?text=<url-encoded>`

Same body format, response shape, ID scheme and locking as `POST /training/pair`.
`text` is the model transcript at save time.

### `GET /training/reviews`

```json
{ "reviews": [ { "id": "0003", "text": "…", "duration_s": 12.7 } ] }
```

Sorted numerically by `id`.

### `GET /training/review/audio/{id}`

Serves the candidate WAV as `audio/wav`. Same numeric-id guard as the pairs variant. Used both
for playback and for client-side re-transcription (the frontend parses the PCM16 and replays it
through a normal WS session with `?model=voxtral|qwen`).

### `POST /training/review/{id}/accept`

Body `{"text": "…"}` — the corrected transcript.

Moves the WAV into `training/audio/` under a fresh **pair** ID, appends to `pairs.jsonl`, and
removes the candidate from `review.jsonl`. Duration is taken from the review entry, falling
back to the WAV file size.

```json
{ "id": "0043", "duration_s": 12.7 }
```

`400` on a non-numeric id or empty text; `404` if the candidate WAV is gone.

### `DELETE /training/review/{id}`

Discards a candidate (WAV + JSONL entry). Returns `{"deleted": "0003"}`.

---

## Training — LoRA runs

### `POST /training/run?model=voxtral|qwen`

Starts a LoRA training run. Default `voxtral`.

Sequence:

1. Locate the trainer script — `tools/train_lora_qwen.py` or `tools/train_lora_voxtral.py`
   (voxtral also falls back to the legacy `tools/train_lora.py` name for old installs), searched
   in cwd, then the binary's dir, then `~/.config/voicetserver/`.
2. Validate prerequisites (script found, qwen enabled if `?model=qwen`, training dir exists) —
   all *before* the status is set, so a failed validation never leaves it stuck on `running`.
3. Atomically check-and-set the status to `running` (`409` if a run is already in progress —
   only one at a time, **regardless of model**).
4. **Unload both engines** from VRAM (~9.5 GB), `device.synchronize()`, then
   `cuMemPoolTrimTo(pool, 0)` so the separate Python process can actually allocate the freed
   memory.
5. Spawn the trainer with `--data-dir`, `--model-dir`, `--output-dir`; stream its stdout+stderr
   into the status log (capped at the last 200 lines).
6. When the subprocess exits (success *or* failure), reload both engines and re-apply each
   model's recorded LoRA path.

Responses: `202 Accepted` "Training started" · `409 Conflict` "Training already running" ·
`422` with a plain-text reason (script missing / qwen not enabled / no training data yet).

Transcription is unavailable on **both** engines for the duration of training plus reload
(~10–30 s) — WS sessions get the "Server is training…" error frame. A qwen session already in
flight keeps its `Arc` handle and finishes; new sessions are blocked immediately.

### `GET /training/status`

```json
{ "status": "running", "log": ["epoch 1/3 loss 2.41", "…"] }
```

`status` ∈ `idle` · `running` · `done` · `error`. The status stays at `done`/`error` until the
next run — it never returns to `idle`.

---

## LoRA adapters

Adapters are **strictly per-model** — the weight-key formats differ (Voxtral:
`layers.{i}.attention.{wq,wk,wv,wo}.lora_{a,b}.weight`; Qwen: `q_proj`/`v_proj` keys). Both
runtimes apply the delta at inference (`out += scale * B @ A @ x`, `scale = lora_alpha / r`) —
no weight merging.

### `POST /lora/reload?model=voxtral|qwen`

Optional JSON body `{"path": "/abs/path/to/adapter_dir"}`. Omit the body to reload from the
currently recorded path (`400` `{"error":"no lora path configured"}` if there is none).

If the directory does not contain `adapter_model.safetensors`, the LoRA is **cleared** (revert
to base model) rather than erroring.

```json
{ "status": "ok", "action": "applied", "path": "/…/lora_adapter" }
```

`action` ∈ `applied` · `cleared`.

The path is recorded **only** when an adapter was actually applied — recording it on a
`cleared` outcome would make `GET /config` report `lora_active: true` and re-apply a
nonexistent adapter after the next training run.

While an engine is unloaded for training, the call still succeeds and only records the path;
the adapter is applied on the post-training reload.

Errors: `400` (no path configured, unknown `?model=`, or malformed JSON body) · `422`
(`?model=qwen` with the engine disabled) · `500` (load failure), all as `{"error": "..."}`.

The body is read as raw bytes and parsed only when non-empty, so `Content-Type:
application/json` with an empty body is treated as "no body" (it used to be rejected with `400`
by axum's `Option<Json<…>>` extractor, which only yields `None` when the header is absent).

### `DELETE /lora?model=voxtral|qwen`

Unloads the engine's active adapter in memory (revert to base model). Files on disk are left
untouched. Also clears the recorded path, so nothing is re-applied on the next training reload.

Returns `{"status": "cleared"}`. No-op (still `200`) while the engine is unloaded for training.
`422` for `?model=qwen` when the qwen engine is disabled.

This is what makes live A/B testing of an adapter against its base model possible — useful when
an adapter trained on read-aloud audio hurts free-speech accuracy.

---

## Edit-log mining

### `POST /log/edit`

```json
{ "original": "…", "edited": "…", "timestamp": "2026-07-19T10:00:00Z" }
```

Appends one line to `{data_dir}/edit_log.jsonl`. Posted by the userscript when a commit-mode
dictation is edited before insertion. All three fields are required.
Returns `{"status": "ok"}`.

### `GET /edits/report`

Aggregates the edit log into the most frequent word-level corrections — direct candidates for
`custom_words.txt` `wrong=correct` entries.

```json
{ "entries": 84, "suggestions": [ { "original": "Miktion", "edited": "Migration", "count": 7 } ] }
```

Method: LCS word diff per entry over whitespace tokens; changed regions become
removed-run → inserted-run pairs. Pure insertions/deletions are skipped (they make no
replacement pair), as are punctuation-only changes and runs longer than 4 words (those are
rewrites, not corrections). Leading/trailing punctuation is trimmed from both sides. Entries
whose diff table would exceed 250 000 cells are skipped rather than stalling the request. Top
30 by count, ties broken alphabetically.

Surfaced by the userscript's "💡 Vorschläge aus Korrekturen" button.

---

## Quick reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | **no** | liveness + connection count |
| GET | `/config` | yes | full settings union |
| PATCH | `/config` | yes | update settings |
| — | `/asr` (WS) | yes | audio stream → transcription |
| GET | `/words` | yes | list custom words |
| POST | `/words` | yes | add/remove custom words |
| GET | `/training/sentences` | yes | calibration sentences + status |
| POST | `/training/sentence` | yes | add sentence |
| PATCH | `/training/sentence` | yes | edit sentence |
| DELETE | `/training/sentence` | yes | remove sentence |
| POST | `/training/pair` | yes | upload recorded pair |
| GET | `/training/pairs` | yes | list pairs |
| GET | `/training/audio/{id}` | yes | pair WAV |
| DELETE | `/training/pair/{id}` | yes | delete one pair |
| GET | `/training` | yes | pair count + total duration |
| DELETE | `/training` | yes | delete **all** training data |
| POST | `/training/review` | yes | save dictation candidate |
| GET | `/training/reviews` | yes | list candidates |
| GET | `/training/review/audio/{id}` | yes | candidate WAV |
| POST | `/training/review/{id}/accept` | yes | promote candidate → pair |
| DELETE | `/training/review/{id}` | yes | discard candidate |
| POST | `/training/run` | yes | start LoRA training |
| GET | `/training/status` | yes | training status + log |
| POST | `/lora/reload` | yes | apply/swap LoRA adapter |
| DELETE | `/lora` | yes | unload LoRA adapter |
| POST | `/log/edit` | yes | record an edit |
| GET | `/edits/report` | yes | aggregated correction suggestions |

> `DELETE /training/pairs` does **not** exist — the delete-all route is `DELETE /training`.
> `/training/pairs` accepts `GET` only.
