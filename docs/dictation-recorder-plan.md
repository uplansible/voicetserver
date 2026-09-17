# Dictation Recorder — implementation plan

Port of the **dictupl** desktop app (`/home/schmidi/dictupl`, Rust/egui voice
recorder) into the SCHMIDIspeech browser workflow. This plan is self-contained
so it can be picked up in a fresh session. Both halves live in **this repo**
(`/home/schmidi/voicetserver`): the server is Rust under `src/`, the client is
the Tampermonkey userscript `schmidispeech.user.js`.

## Goal

Add a dictation recorder that:
1. Records mic audio, applies Voice Activity Detection (VAD) to drop silence,
2. Lets the user play back the silence-removed audio (preview) before saving,
3. Saves a small **Ogg Opus** file with a `{prefix}-{counter:03}_{timestamp}`
   name to a shared folder that the staff-side **schreibomat** extension plays
   from,
4. Copies `{prefix}-{counter} (Nmin)` to the clipboard on save.

This is the recording counterpart to **schreibomat**
(`/home/schmidi/schreibomat`, the staff-side foot-pedal *player*). schreibomat
plays `mp3/ogg/wav` from a working folder — so `.ogg` output drops straight
into its picker.

## Key architecture decisions (and why)

- **Home = `schmidispeech.user.js`, not schreibomat.** The recorder runs on the
  user's own machines where SCHMIDIspeech already runs; schreibomat is on staff
  machines. schmidispeech already does mic capture and has the tabbed panel +
  GM storage + clipboard helper to reuse.
- **Encode + save on the server (voicetserver), not in the browser.** Chosen
  over browser-side WASM Opus (host-page CSP can block WASM) and browser File
  System Access (per-origin handle, Brave flag). The server already receives
  PCM and writes audio files (training-pairs subsystem), so this is a trodden
  path. Mirrors dictupl's original "encode + write to SMB share" step.

  **Re-affirmed 2026-08-22** after re-examining the browser alternatives:
  - *On-device encode* is feasible via **WebCodecs `AudioEncoder`**
    (`codec:'opus'`, native, faster than realtime, no WASM and so no host-page
    CSP exposure) — but it emits **raw Opus packets**, so the Ogg container
    (OpusHead/OpusTags, page framing, lacing, 48 kHz granule positions, per-page
    CRC32) would be ours to hand-write and own. Rust gets that from the `ogg`
    crate. `MediaRecorder` is not an option: it only encodes a live MediaStream,
    so a VAD-trimmed buffer would have to be replayed in **realtime** (10 min of
    audio = 10 min of encoding), and it emits WebM rather than Ogg.
  - *Browser-side write* via File System Access is likewise workable — schreibomat
    already does exactly this from a **content script**, i.e. on the page origin
    with page-origin IndexedDB (see its `db.js` header), so a Violentmonkey script
    could do the same: `unsafeWindow.showDirectoryPicker({mode:"readwrite"})`,
    handle persisted in page-origin IndexedDB (**never** `GM_setValue` — GM values
    are JSON-serialized and the handle would come back as `{}`), re-granted on a
    user gesture after each browser restart. Costs: the
    `brave://flags/#file-system-access-api` flag, a separate grant per origin, and
    Violentmonkey's sandbox realm as a live risk.

  Both were rejected because **voicetserver is being merged into the main server**,
  which already reaches the dictation share — so the server-side path costs neither
  a hand-rolled muxer nor a per-origin FSA grant. If that merge is ever abandoned,
  the fallback is *server encodes, returns the Ogg bytes in the response body,
  browser writes them via FSA* — which keeps libopus in Rust without the server
  needing the share at all.
- **Reuse the existing 16 kHz mono PCM capture.** schmidispeech already captures
  16 kHz mono for ASR (`SAMPLE_RATE`, `dictationPcmBuffers`, `concatPcm`). Opus
  encodes 16 kHz natively (wideband — clearer than telephone), tiny files,
  intelligible for a transcriptionist. dictupl used 48 kHz but that is overkill
  for speech. (If staff ever need fuller band, add a separate 48 kHz capture.)
- **VAD runs offline (once, at Stop), not in real time.** dictupl does real-time
  VAD with a pre-roll ring buffer because it streams to disk for unbounded
  sessions. Dictations here are 2–10 min, so we buffer the whole PCM in memory
  (~19 MB/10 min as int16) and run VAD in a single pass over the complete
  buffer. This is **simpler** (no ring buffer / pause-resume timing) and gives
  **better onset preservation**: with the full timeline we can include a real
  lead window before each speech onset instead of approximating it. VAD cost is
  ~10–50 ms for 10 min — negligible, so playback is silence-free with no lag.
- **Opus, not MP3/WAV.** Smallest files; schreibomat plays `.ogg`; server-side
  libopus avoids all browser CSP/WASM concerns.

## Format / config summary

| Thing | Value |
|---|---|
| Capture | 16 kHz mono, int16 (reuse existing) |
| VAD | offline, dictupl RMS logic + lead window (defaults below) |
| Encode | Ogg Opus, ~32 kbps mono (server-side, configurable) |
| Output ext | `.ogg` |
| Filename | `{prefix}-{counter:03}_{dd-mm-yy_HH_MM}.ogg` e.g. `spm-001_22-07-26_14_30.ogg` |
| Clipboard | `{prefix}-{counter:03} (Nmin)` e.g. `spm-001 (4min)` |
| Save dir | server config `dictation_dir`, default **TBD — see status note** (configurable) |

### dictupl defaults to carry over (from `/home/schmidi/dictupl/src/config.rs`)

- Prefixes: `spm, ire, awe, ma` (active index 0), per-prefix counters start at 1.
- VAD: `threshold_rms = 0.011`, `lead_ms = 250`, `hangover_ms = 500`,
  `start_min_ms = 60`, `frame_ms = 20`.
- `Nmin` = total recording wall-clock duration (start→stop) rounded to nearest
  minute, min 1. (dictupl's auto-timer ran for the whole session and ignored
  VAD pauses, so use total capture time, not voiced-only time.)

Threshold `0.011` was tuned on normalized RMS of `[-1,1]` samples; schmidispeech
capture is already normalized Float32, so it should carry over — expose it so it
can be retuned.

---

## Status (2026-08-22) — implementation is sequenced AFTER the server merge

**Do not implement yet.** voicetserver is being merged into the main server, and
that merge decides the save path — so `dictation_dir` is deliberately left
unresolved here. Pick it up once the merge has landed and the path is known.

What is known about the path today:
- The fleet's dictation folder is **`/mnt/upl/diktat`** — `//100.99.88.66/daten`
  mounted at `/mnt/upl` by `nixos/modules/smbmounts.nix` on *all* workstation
  roles, bookmarked in `home/upl.nix:127` and `home/ma.nix:194`.
- The earlier `/mnt/ssdupl/daten/diktat` default in this plan does **not** match
  that, and no `/mnt/*` exists on the current dev box — treat it as stale.
- A **third writer** exists and is not going away: `nixos/modules/mountdpm.nix`
  rsyncs the Philips DPM recorder into `/mnt/upl/diktat` on plug-in. The share is
  therefore the source of truth regardless of what the server does.

Also decided 2026-08-22: **schreibomat is not merged into voicetserver.** It stays
a separately deployed extension reading the share directly. Serving dictations over
HTTP would add a second access path to the same directory (the DPM sync cannot use
it), make a clinical workflow depend on the ASR box being up, and require handing
the single shared `api_key` — which also grants `POST /training/run`, `PATCH /config`
and `GET /edits/report` — to every staff workstation. A read-only HTTP endpoint
remains cheap to add later as a *fallback* for machines off the share.

## Part A — Server (voicetserver, Rust)

**New dependencies** (`Cargo.toml`): `audiopus` (libopus bindings) + `ogg`
(Ogg container muxing). Confirm the libopus system dep is available for the
build environment.

**New config field** (`src/config.rs`, `~/.config/voicetserver/config.toml`):
- `dictation_dir: PathBuf`, default TBD (see status note above). Startup or
  runtime-patchable is fine; simplest is a plain config field created on load.
  Works the same whether it is a local directory or a mounted share (CIFS/NFS) —
  to the handler it is just a path.

  **Do NOT `create_dir_all` it.** If the target is a share that is not mounted
  yet, that would silently create a local directory under the mountpoint and
  write dictations into it, where they vanish the moment the mount appears.
  Require the directory to already exist and return a clear error if it does not
  (mount missing / share down is exactly the failure worth surfacing loudly).

**New route** — add to the api-key-protected router group in `src/main.rs`
(around lines 1146–1172, alongside `/training/pair`):
```
.route("/dictation/save", axum::routing::post(dictation_save_handler))
```
It sits behind `api_key_auth`, so the client sends the API key header exactly
like the training endpoints do.

**Handler `dictation_save_handler`:**
- **Body**: raw int16 LE mono 16 kHz PCM. Mirror the existing training-pair
  upload — READ the current `/training/pair` client+server code first
  (`saveTrainingPair` in the userscript ~line 2188; server `training_pair`
  handler in `src/main.rs`) to copy the exact body/metadata convention.
- **Metadata**: filename via query param `?name=...` (or header). **Sanitize**:
  reject path separators / `..`, enforce `.ogg` extension, enforce the
  `{prefix}-{ccc}_{timestamp}` shape loosely.
- **Encode**: int16 PCM → Ogg Opus at ~32 kbps mono, 16 kHz. audiopus encodes
  20 ms frames (320 samples @ 16 kHz); mux packets into an Ogg stream with the
  `ogg` crate (OpusHead + OpusTags headers, then audio pages with correct
  granule positions). Consider a small `src/opus.rs` helper module.
- **Write**: to `dictation_dir/{name}`, **atomically** — write to a temp name in
  the *same* directory, then `rename` into place. schreibomat lists the folder and
  filters on `AUDIO_EXTENSIONS`, so a partially-written `.ogg` would show up in its
  picker as a playable-looking file; the rename makes the file appear complete or
  not at all. (Same-directory rename is atomic on CIFS as well as on a local fs.)
  Verify the directory exists first rather than creating it (see the config note
  above). If the file exists, the client is responsible for counter selection, but
  still guard (return a conflict rather than overwrite, or write and let client
  counter advance — decide during impl; dictupl picks the next free counter by
  scanning the dir, see below).
- **Response**: JSON `{ ok, path, bytes }` or an error status.

**Body limit**: current `DefaultBodyLimit::max(64 MB)` is enough (10 min int16
16 kHz ≈ 19 MB).

**Optional (nice-to-have, matches dictupl)**: a `GET /dictation/next-counter?
prefix=spm` that scans `dictation_dir` for existing `spm-###_` files and returns
the next free counter, so counters survive across machines/reinstalls. If
skipped, counters live only in the client GM storage.

## Part B — Client (`schmidispeech.user.js`)

Add a **new panel tab "Diktat"** next to the existing tabs (`Eigene Wörter`,
`Hotwords`, `Aufnehmen`, `2. Durchgang`, `Training`, `Diktate`,
`Einstellungen`). Note: the existing **"Aufnehmen"** tab is for reading training
sentences — do NOT overload it; add a separate tab. Reuse the panel style
constants (`BTN_PRIMARY`, `BTN_CANCEL`, `INPUT_STYLE`, `LABEL_STYLE`) and the
tab/pane wiring pattern (see panel markup ~lines 415–540 and `switchTab`
handlers ~lines 682+).

**Reuse from the existing script:**
- Mic capture: `getUserMedia` + `ScriptProcessor` (see `setupAudio` ~line 2094),
  the 16 kHz resample (`captureChunk`, `ratio`), `concatPcm` (~line 995).
- Clipboard: `navigator.clipboard.writeText` (~line 2294).
- GM storage (`GM_getValue`/`GM_setValue`) for config.

**State machine:** `IDLE → RECORDING → REVIEW (playback) → (Save) → IDLE`.

**Config (GM keys, `rec_` namespace):**
- `rec_prefixes` (default `spm,ire,awe,ma`), `rec_active_prefix` (index),
  `rec_counter_{prefix}` (per prefix, default 1),
- `rec_vad_threshold` (0.011), `rec_vad_lead_ms` (250),
  `rec_vad_hangover_ms` (500), `rec_vad_start_min_ms` (60),
  `rec_vad_frame_ms` (20), `rec_vad_enabled` (true),
- `rec_seek_step_secs` (2).
Surface these in the Diktat tab and/or the Einstellungen tab.

**Recording:** on Record, capture all 16 kHz PCM into an in-memory array
(reuse the `dictationPcmBuffers` pattern but a dedicated buffer so it doesn't
collide with ASR dictation snapshotting). Show a live level meter (RMS of each
ScriptProcessor block). Record/Stop button.

**Offline VAD (at Stop)** — port of dictupl `recorder.rs`, adapted to run over
the full buffer:
1. Frame size = `frame_ms * 16000 / 1000` (20 ms → 320 samples).
2. Per frame, RMS over normalized samples; frame is *voiced* if
   `rms >= threshold`.
3. Find voiced runs. A run only counts as a speech segment if it reaches
   `start_min_ms` of voiced frames (start-frames filter).
4. Extend each segment **forward** by `hangover_ms` (tail) and **backward** by
   `lead_ms` (the pre-roll/lead — this is what prevents onset chopping; with the
   full buffer we just move the start earlier, clamped to 0 / previous segment).
5. Merge overlapping/adjacent segments.
6. Concatenate the PCM in those ranges → **silence-free Float32 buffer**. Keep
   the raw buffer too, so VAD can be re-run with a different threshold without
   re-recording.
If `rec_vad_enabled` is false, the silence-free buffer == the raw buffer.

**Preview playback:** Web Audio — decode the silence-free buffer into an
`AudioBuffer`, play via `AudioBufferSourceNode`. Controls: play/pause, rewind /
forward by `rec_seek_step_secs`, a seek bar + position/total readout. (Speed
control is schreibomat's playback feature and is codec-independent; not required
here but easy to add later via `playbackRate`/`preservesPitch`.)

**Filename + counter:** active prefix + its counter →
`{prefix}-{counter:03}_{dd-mm-yy_HH_MM}.ogg`. Timestamp is local time,
zero-padded `dd-mm-yy_HH_MM`.

**Save:** convert the silence-free Float32 buffer to int16 LE, `POST` to
`{serverHttpBase}/dictation/save?name={filename}` with the API key header (see
how `saveTrainingPair` builds its request / `getHttpBase`). On success:
- increment `rec_counter_{prefix}` (wrap 1..=999 like dictupl),
- copy `{prefix}-{counter:03} ({Nmin}min)` to the clipboard,
- toast success; reset to IDLE.
On failure: toast the error, keep the recording so the user can retry.

**UI (Diktat tab) minimum:** prefix dropdown, counter display, VAD on/off,
Record/Stop, level meter, then (in REVIEW) play/pause + ⏪/⏩ + seek bar +
duration, and a Save button + status line.

---

## Build / versioning workflow (from CLAUDE.md)

voicetserver is a Rust binary — **bump `Cargo.toml` patch version before
compiling**. The userscript has its own `@version` — bump it too when the client
changes. Then `cargo build --release` (or `cargo build` for dev; note the
`cuda` feature is production-only). Commit to `main` on "ok go".

- Server dev build (CPU): `cargo build`
- Server release: `cargo build --release --features cuda` (GPU box)

## Suggested implementation order

1. **Server first**: add deps, `dictation_dir` config, `src/opus.rs` encoder
   helper, `/dictation/save` route + handler. Test with a curl of a raw int16
   PCM blob → verify a playable `.ogg` lands in `dictation_dir` and plays in
   schreibomat.
2. **Client**: new Diktat tab, capture buffer, offline VAD, preview playback,
   filename/counter, Save → server. Test end to end.
3. Retune VAD threshold on real mic input if needed.

## Reference files

- dictupl VAD + config source of truth: `/home/schmidi/dictupl/src/recorder.rs`,
  `/home/schmidi/dictupl/src/config.rs`, `/home/schmidi/dictupl/src/app.rs`
  (filename/counter/clipboard/save logic).
- schreibomat player (consumes the output): `/home/schmidi/schreibomat/content.js`
  (`AUDIO_EXTENSIONS = ["mp3","ogg","wav"]`), `README.md`.
- Server router + training-pair precedent: `src/main.rs` (~1146–1172 routes;
  training-pair + `/training/audio/{id}` handlers), `src/streaming.rs` (PCM
  handling), `src/config.rs`.
- Client: `schmidispeech.user.js` (`setupAudio` ~2094, `concatPcm` ~995,
  clipboard ~2294, panel/tabs ~415–540 & ~682+).
