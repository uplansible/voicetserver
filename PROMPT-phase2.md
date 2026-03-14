# Phase 2 Implementation Prompt: Streaming Mic Capture + Real-Time Inference

## Context

**Voicet** is a Rust speech-to-text engine running Voxtral Mini 4B locally on GPU (candle + CUDA, BF16). Phase 1 — offline WAV transcription — is complete and tested. Phase 2 adds real-time microphone input: open the mic, stream audio through the model, emit text as you speak.

Read `ARCHITECTURE.md` and `PLAN_voicet-rust-rewrite.md` for full model architecture details, streaming protocol, and project goals. This prompt tells you everything you need to implement Phase 2.

---

## Codebase snapshot

Builds with zero warnings (`cargo build --release`). All source in `src/`:

| File | Lines | What it does |
|---|---|---|
| `main.rs` | ~255 | Entry point. Loads model, reads WAV, runs offline inference loop. |
| `common.rs` | ~115 | Shared types used by encoder + decoder: `RmsNorm`, `RotaryEmbedding`, `KvCache`, `deinterleave_qk`. |
| `encoder.rs` | ~270 | Audio encoder. Conv1d stem (2 layers, GELU) + 32 causal transformer layers. Sliding window attention (750 frames). Processes in chunks of 4 frames with KV cache. |
| `decoder.rs` | ~280 | Text decoder. 26 layers, GQA (32 query / 8 KV heads, head_dim=128), RoPE, SwiGLU, Ada-RMSNorm delay conditioning. Tied embed/lm_head weights. |
| `adapter.rs` | ~51 | 4x downsample. Concatenates 4 encoder frames → Linear(5120→3072) → GELU → Linear(3072→3072). |
| `tokenizer.rs` | ~76 | Tekken BPE tokenizer (decode-only). Parses `tekken.json`. |
| `mel.rs` | ~213 | Log mel spectrogram. N_FFT=400, HOP_LENGTH=160, N_MELS=128. Multi-threaded FFT, reflect-pad, drops last STFT frame. |

**Cargo.toml deps**: candle-core (CUDA), candle-nn, safetensors, anyhow, memmap2, serde_json, base64, hound, num_cpus, clap.

**Model weights**: `Voxtral-Mini-4B-Realtime/consolidated.safetensors` (~9GB BF16), loaded via mmap.

**Verified test output**: `test01_16khz_3.7s.wav` → " Dancing in the" (40 tokens: 3 text, 36 pad, 1 word-boundary, 0 eos, ~31.5 tok/s).

---

## Timing chain

This is the most important thing to internalize. Every number here matters.

```
Audio source:   16kHz mono f32
                ↓
Mel hop:        160 samples = 10ms → 1 mel frame
                (window: N_FFT=400 samples = 25ms, but hops at 10ms)
                ↓
Conv stem:      stride-2 downsample → 2 mel frames = 1 encoder frame = 20ms
                ↓
Encoder:        CHUNK_SIZE=4 frames → 8 mel frames = 1280 samples = 80ms per chunk
                (KV-cached, sliding window 750 frames = 15s)
                ↓
Adapter:        4 encoder frames → 1 adapter frame = 80ms of audio
                (1 encoder chunk of 4 frames = exactly 1 adapter output)
                ↓
Decoder:        1 adapter frame → 1 token decision = 80ms per token
```

**The fundamental tick: every 80ms of audio (1280 samples, 8 mel frames) produces exactly 1 decoder token.**

---

## Streaming protocol

From the HF reference implementation, the model uses a specific startup and streaming sequence:

### Startup (one-time, before real audio)
1. **Silence prefix**: Left-pad audio with 40,960 zero samples (= 32 tokens × 8 mel frames/token × 160 samples/frame). Process through mel → conv stem → encoder → adapter. Produces 32 adapter frames of silence context.
2. **Decoder prefill**: Feed BOS + 38 STREAMING_PAD tokens (32 from silence + 6 delay tokens) through the decoder, each fused (element-wise add) with its corresponding adapter frame. This fills the decoder KV cache.

### Steady-state streaming (repeats for each 80ms of audio)
3. For each new adapter frame:
   - Fuse: `embed(last_generated_token) + adapter_frame`
   - Decoder forward (single token, uses KV cache) → logits → argmax
   - Emit the token

### Token types
| ID | Constant | Meaning | Action |
|---|---|---|---|
| 32 | `STREAMING_PAD` | Model needs more context | Print nothing |
| 33 | `STREAMING_WORD` | Word boundary | Print space |
| 2 | `EOS` | End of sequence | Stop |
| Other | — | Text token | Decode via tokenizer, print immediately |

---

## Current offline pipeline (what exists in main.rs)

For reference, the current code does everything in batch:

1. Load entire WAV → all samples at once
2. Prepend 40,960 silence samples
3. Compute mel spectrogram for entire padded audio
4. `encoder.forward(&mel)` — internally resets KV caches, processes all chunks
5. `adapter.forward(&encoder_output)` — batch 4x downsample
6. Decoder prefill: BOS + 38 PAD tokens, each fused with adapter frame
7. Decode loop: for each remaining adapter frame, fuse with last token, decoder forward, argmax

**Phase 2 transforms this into an incremental pipeline** where audio arrives continuously from the mic and processing happens chunk-by-chunk.

---

## Phase 2 sub-tasks

### 2a — Mic capture + sample accumulation

**Add `cpal = "0.15"` to Cargo.toml.**

1. Open default input device. Request 16kHz, mono, f32. Request **10ms buffer** (160 samples = exactly 1 mel hop). cpal may give a slightly different size; that's fine.
2. In the cpal callback: push samples into an `mpsc::channel`. The callback must never block or allocate. Just `tx.send(chunk)` and return.
3. The inference thread pulls from the channel receiver and accumulates samples into a contiguous buffer.

### 2b — Incremental mel spectrogram

Create an `IncrementalMel` struct in `mel.rs` (keep the existing batch function for offline mode).

**How it works:**
- Holds a sample ring buffer of at least N_FFT (400) samples
- Precomputes Hann window and mel filterbank (same as existing code)
- `push_samples(&mut self, samples: &[f32])` — appends to buffer
- `next_frame(&mut self) -> Option<[f32; N_MELS]>` — when >= N_FFT samples available at the current hop position, compute one mel frame:
  1. Extract 400-sample window, apply Hann
  2. FFT (reuse existing `fft()` function)
  3. Magnitude-squared for bins 0..N_FFT/2+1 (one-sided, no conjugate mirror)
  4. Dot product with mel filterbank → 128 values
  5. Normalize: `max(log10(val), -6.5) / 4.0 + 1.0` (where -6.5 = GLOBAL_LOG_MEL_MAX - 8.0)
  6. Advance hop position by HOP_LENGTH (160)

**Reflect padding**: The offline code reflect-pads 200 samples on each side. For streaming, pre-fill the sample buffer with 200 reflected samples from the first real audio chunk before computing any frames. This matches offline behavior exactly.

### 2c — Streaming encoder

The encoder's `forward()` method resets all KV caches and processes everything. **For streaming, add a `forward_chunk()` method** that:
- Takes a single chunk of post-conv frames (shape `[1, chunk_len, 1280]`)
- Takes the frame offset for RoPE positioning
- Runs through all 32 layers using existing KV caches (does NOT reset them)
- Returns the chunk output

Also expose a `reset_caches()` method (the layer-level one already exists).

**Conv stem handling**: The conv stem is cheap (~0.1% of compute). Two options:
- **Simple**: accumulate all mel frames so far, re-run conv stem each time, take only the new output frames. Wasteful but trivial and correct.
- **Better**: track how many mel frames have been processed, only run new frames through the conv with cached padding state. Conv1 (stride=1, kernel=3) needs 2 samples of left context. Conv2 (stride=2, kernel=3) needs 1 sample of left context.

Start with the simple approach. Optimize later if needed.

### 2d — Streaming adapter

Simple: maintain a `Vec<Tensor>` buffer of encoder output frames. When 4 accumulate, concatenate them and run the adapter projection. Clear the buffer.

In streaming mode, the encoder produces exactly 4 frames per chunk (CHUNK_SIZE=4), so every encoder chunk immediately yields 1 adapter output. No partial buffering needed in practice.

### 2e — Streaming decoder + token emission

For each adapter frame (from 2d):

1. `decoder.embed_tokens(&[last_token_id])` → token embedding `[1, 1, 3072]`
2. Add adapter frame: `token_embed + adapter_frame` (element-wise)
3. `decoder.forward(&fused_embed, &t_cond)` → logits → argmax → next token ID
4. Emit immediately to stdout (no buffering, no word accumulation):
   - PAD → nothing
   - WORD → print `" "`
   - EOS → done
   - Text token → `tokenizer.decode_token(id)` → print bytes
5. `last_token_id = next_token_id` for next iteration

Use `print!` + `stdout().flush()` for immediate output (not `println!`).

### 2f — Startup sequence

Before streaming real audio, bootstrap the model:

1. Generate 40,960 zero samples
2. Compute mel frames for silence (using `IncrementalMel` or batch mel — doesn't matter, it's one-time)
3. Run silence mel through conv stem + encoder (chunked) + adapter → ~32 adapter frames
4. Compute delay conditioning: `sinusoidal_embedding(6.0, &device, dtype)` → `t_cond [1, 3072]`
5. Prefill decoder: BOS + 38 PAD tokens, each fused with corresponding silence adapter frame

**Optimization**: Open the mic BEFORE startup. While the GPU processes the silence warm-up (~2s), the mic is already capturing audio into the channel. When startup finishes, drain the channel and start processing — zero audio is lost.

### 2g — Silence detection + session management

Silence detection aligns to the model's natural processing boundary: **1 adapter token = 80ms of audio = 1280 samples**.

**How it works:**
- Each time enough samples accumulate for one adapter token (1280 samples / 8 mel frames), compute the RMS energy of those 1280 samples
- If RMS is below a threshold, increment a silence counter. Otherwise, reset it to 0.
- When the silence counter reaches `silence_chunks` consecutive silent tokens, trigger a flush: print a newline, optionally reset decoder KV caches for a fresh utterance

**Configuration** (constants or CLI flags):
- `silence_threshold: f32` — RMS energy threshold (default: e.g. 0.01, tune empirically)
- `silence_chunks: usize` — consecutive silent 80ms chunks before flush (default: 8 = 640ms)

These are cheap computations on raw audio samples — no GPU involved.

**Ctrl+C handling**: Register a signal handler (or `ctrlc` crate) that sets an atomic flag. The main loop checks the flag each iteration and exits cleanly, printing any final text.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Main Thread                            │
│                                                              │
│  1. Load model (encoder, adapter, decoder, tokenizer)        │
│  2. Open mic (cpal callback → mpsc::channel)                 │
│  3. Run startup sequence (silence encode + decoder prefill)  │
│     (mic buffers audio into channel during this time)        │
│  4. Processing loop:                                         │
│     a. Pull samples from channel into accumulator            │
│     b. IncrementalMel: samples → mel frames                  │
│     c. Conv stem: mel frames → encoder input frames          │
│     d. When 4 encoder input frames ready:                    │
│        - encoder.forward_chunk() → 4 encoder output frames   │
│        - adapter.forward() → 1 adapter frame                 │
│        - decoder step → 1 token → emit text                  │
│     e. Check silence counter → flush if needed               │
│     f. Check Ctrl+C flag → exit if set                       │
│                                                              │
│  cpal audio thread ──tx.send(samples)──► mpsc channel        │
└──────────────────────────────────────────────────────────────┘
```

Single inference thread. The GPU serializes kernel execution anyway, so multiple threads add complexity without speed. The only separate thread is cpal's audio callback (managed by cpal).

---

## Constraints

1. **Output equivalence**: Feed the test WAV samples through the streaming pipeline. It must produce the same tokens as offline mode. This is the primary correctness check.
2. **Latency target**: 10ms mic buffer + 80ms chunk accumulation + ~40ms inference ≈ ~130ms end-to-end. Text should appear within ~200ms of being spoken.
3. **No blocking in audio callback**: `tx.send()` only. No allocation, no inference, no locks.
4. **BF16 precision**: `set_gemm_reduced_precision_bf16(true)` — already set in main.rs, must stay.
5. **Preserve offline mode**: `voicet <file.wav>` = offline transcription (existing behavior). `voicet` with no args = streaming mic mode. Use CLI dispatch.
6. **Minimize changes to existing modules**: Encoder, decoder, adapter, tokenizer are correct. Add new methods (e.g. `forward_chunk`) rather than rewriting existing ones. The offline `forward()` should keep working.

---

## Files to create/modify

| File | Change |
|---|---|
| `Cargo.toml` | Add `cpal = "0.15"` |
| `src/main.rs` | CLI dispatch: if arg given → offline WAV mode (existing), else → call streaming mode |
| `src/streaming.rs` **(new)** | All streaming orchestration: mic setup, incremental processing loop, token emission, silence detection |
| `src/mel.rs` | Add `IncrementalMel` struct alongside existing batch function |
| `src/encoder.rs` | Add `forward_chunk()` and `reset_caches()` public methods |
| `src/decoder.rs` | No changes expected (already supports single-token forward with KV cache) |
| `src/adapter.rs` | No changes expected (already works on any multiple-of-4 input) |
| `src/tokenizer.rs` | No changes expected |
| `src/common.rs` | No changes expected |

---

## Implementation order

1. **Mic capture**: Add cpal, open mic, print sample counts to verify audio is flowing
2. **Incremental mel**: Build `IncrementalMel`, test it produces the same frames as the batch function on the test WAV
3. **Encoder streaming**: Add `forward_chunk()`, test that chunked-incremental encoding matches batch encoding on test WAV
4. **Wire it all up**: mic → mel → conv → encoder → adapter → decoder → print tokens
5. **Startup sequence**: Silence prefill, decoder warm-up
6. **Validate**: Pipe test WAV samples through the streaming path, verify identical output to offline mode
7. **Silence detection**: RMS energy monitoring at 80ms boundaries, configurable threshold and chunk count
8. **Ctrl+C handling**: Graceful shutdown

---

## Gotchas

- **Encoder `forward()` resets KV caches** on every call. The new `forward_chunk()` must NOT reset caches. Provide a separate `reset_caches()` for explicit control.
- **Conv stem re-processing**: If using the simple approach (re-run on growing buffer), make sure to only take NEW output frames each time. Track the previous output length.
- **Mel reflect-padding at stream start**: The first 200 samples of reflect padding depend on the first real audio chunk. You can't compute initial mel frames until you've received at least the first audio chunk. Buffer it, create reflect padding from it, then process.
- **Mel last-frame drop**: The offline code drops the last STFT frame (`stft_frames - 1`). In streaming mode, this doesn't apply — you just keep producing frames as samples arrive. The final frame issue only matters at end-of-stream.
- **Decoder KV cache offset**: `cache.current_len()` auto-tracks position. No manual offset management needed for streaming.
- **Adapter always gets exactly 4 frames**: In steady-state streaming, each encoder chunk produces 4 frames = 1 adapter input. No partial-frame buffering needed in practice.
- **Token IDs**: BOS=1, EOS=2, STREAMING_PAD=32, STREAMING_WORD=33. These are defined in `tokenizer.rs`.
- **`stdout` buffering**: Use `print!()` + `io::stdout().flush()` for immediate token output. `println!` adds newlines you don't want.
