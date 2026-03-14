# Phase 2 Optimization

## Context

**Voicet** is a Rust speech-to-text engine running Voxtral Mini 4B locally on GPU (candle + CUDA, BF16). Phase 1 (offline WAV transcription) is complete. Phase 2 (real-time mic streaming) is functionally working but has bugs that need fixing before Phase 3.

Read `ARCHITECTURE.md` and `PLAN_voicet-rust-rewrite.md` for model details. Read all source files in `src/` before making changes.

---

## How it works end-to-end

### Timing chain

```
16kHz mono f32 audio
  → Mel spectrogram (N_FFT=400, HOP=160) → 1 frame per 10ms
    → Conv stem (stride-2) → 1 encoder frame per 20ms
      → Encoder (chunks of 4 frames) → 4 frames per 80ms
        → Adapter (4:1 downsample) → 1 adapter frame per 80ms
          → Decoder (fuse token+audio) → 1 token per 80ms
```

**Fundamental tick: 1280 samples = 8 mel frames = 4 encoder frames = 1 adapter frame = 1 decoder token = 80ms.**

### Two modes (CLI dispatch in main.rs)

- `voicet <file.wav>` → offline transcription via `run_offline()` in main.rs
- `voicet` (no args) → live streaming via `streaming::run_streaming()`

### Offline pipeline (main.rs::run_offline)

1. Load WAV → prepend 40,960 silence zeros (32 tokens × 1280 samples)
2. Batch mel spectrogram on entire [silence + audio]
3. `enc.forward()` — resets KV caches, runs conv stem + 32 transformer layers in 4-frame chunks
4. `adapter.forward()` — batch 4x downsample
5. Decoder prefill: BOS + 38 PAD tokens, each fused (element-wise add) with adapter frames 0–38
6. Decode loop: for each remaining adapter frame, fuse with last token embedding, decoder forward, argmax → emit token

### Streaming pipeline (streaming.rs)

**Startup** (`run_startup`):

1. Open mic, buffer 8960 samples (560ms = 7 adapter frames of "delay audio")
2. Concatenate [40960 silence + 8960 delay audio] = 49920 samples
3. Batch mel → `enc.forward()` → `adapter.forward()` → 39 adapter frames
4. Decoder prefill: BOS + 38 PAD fused with adapter frames 0–38 (identical to offline)
5. Decode any remaining adapter frames from delay audio (positions 39+, usually 0)
6. Save mel frames as conv stem prefix for streaming loop
7. Return: last_token, encoder_offset (156), t_cond, mel_frames prefix

**Steady-state loop** (`run_processing_loop`, called from `run_streaming`):

1. Pull raw audio from mpsc channel (cpal callback → channel → inference thread)
2. Resample 48kHz→16kHz by integer decimation (ratio 3, continuous buffer)
3. Feed 16kHz samples to `IncrementalMel` → drain mel frames
4. Append new mel frames to `all_mel_frames` (starts with startup prefix)
5. When 8+ new mel frames available beyond what conv stem has processed:
   a. Build tensor from ALL accumulated mel frames
   b. Re-run `enc.conv_stem()` on entire buffer (simple approach — correct but O(n))
   c. Take new 4 conv output frames at `prev_conv_output_len`
   d. `enc.forward_chunk()` — processes through 32 layers using existing KV cache (no reset)
   e. `adapter.forward()` → 1 adapter frame
   f. Fuse `embed(last_token) + adapter_frame` → `dec.forward()` → argmax → emit token
6. Silence detection: RMS energy on 1280-sample blocks, newline after 8 consecutive silent chunks
7. Ctrl+C → atomic flag → clean exit

### Key files

| File           | What it does                                                                                                                                                               |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.rs`      | Entry point, CLI dispatch, offline transcription, WAV loading                                                                                                              |
| `streaming.rs` | All streaming: mic setup, startup, incremental processing loop, token emission, silence detection                                                                          |
| `encoder.rs`   | Conv1d stem + 32 causal transformer layers, sliding window attention (750 frames), RoPE, KV cache. Public: `forward()`, `forward_chunk()`, `conv_stem()`, `reset_caches()` |
| `decoder.rs`   | 26 layers, GQA (32Q/8KV), Ada-RMSNorm delay conditioning, tied embed/lm_head. Public: `forward()`, `embed_tokens()`, `reset_caches()`                                      |
| `adapter.rs`   | Concat 4 encoder frames → Linear(5120→3072) → GELU → Linear(3072→3072)                                                                                                     |
| `mel.rs`       | Batch `log_mel_spectrogram()` + `IncrementalMel` (streaming frame-by-frame). FFT, mel filterbank, reflect padding, normalization                                           |
| `common.rs`    | `RmsNorm`, `RotaryEmbedding`, `KvCache`, `deinterleave_qk`                                                                                                                 |
| `tokenizer.rs` | Tekken BPE decode-only. BOS=1, EOS=2, PAD=32, WORD=33                                                                                                                      |

### Key constants

- Encoder: hidden=1280, 32 layers, 32 heads, head_dim=64, sliding_window=750, RoPE theta=1M
- Decoder: hidden=3072, 26 layers, 32Q/8KV heads, head_dim=128, vocab=131072, RoPE theta=1M
- Mel: N_FFT=400, HOP=160, N_MELS=128
- Delay: 6 tokens, silence prefix: 32 tokens, prefill: 39 tokens (BOS + 38 PAD)

---

## ## OPTIMIZATION — DONE

Seven optimizations were applied across three phases, reducing per-token latency from ~64ms to ~42ms (offline, synced). The attention pipeline in both encoder and decoder is now handled entirely by FlashAttention v2, and remaining non-attention costs (normalization, lm_head, Ada-RMSNorm conditioning) are minimized via fused kernels and precomputation.

### Active optimizations (current codebase)

| #   | Optimization              | File(s)      | What it does                                                                                                                                                                                                                                                                                                             |
| --- | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Flash Attention (encoder) | `encoder.rs` | `flash_attn_windowed(q, k, v, scale, Some(749), Some(0))` — single fused CUDA kernel per layer handles Q@K, causal sliding window masking, softmax, and @V. No intermediate attention matrix materialized. 32 layers × 1 kernel replaces 32 × (matmul + transpose + mask + softmax + matmul + transpose + 2 contiguous). |
| 2   | Flash Attention (decoder) | `decoder.rs` | `flash_attn_windowed(q, k, v, scale, Some(2047), Some(0))` — same as encoder, plus native GQA support (32Q/8KV heads handled internally, no `repeat_kv()` needed). 26 layers × 1 kernel.                                                                                                                                 |
| 3   | KV cache layout           | `common.rs`  | `KvCache` uses `[batch, seq, heads, dim]` layout (flash attention's expected format). Append/trim/len all operate on dim 1.                                                                                                                                                                                              |
| 4   | RoPE layout               | `common.rs`  | `RotaryEmbedding::apply` broadcasts over dim 2 (heads), matching the `[batch, seq, heads, dim]` tensor layout. Seq_len read from `dim(1)`, cos/sin unsqueeze on dim 2.                                                                                                                                                   |
| 5   | Fused RMSNorm             | `common.rs`  | `candle_nn::ops::rms_norm` — single CUDA kernel replaces 7 separate ops (to_dtype, sqr, mean, add, sqrt, div, mul, to_dtype). Affects both encoder (65 calls) and decoder (53 calls) per forward. Saves ~530 kernel launches/token.                                                                                      |
| 6   | Cached lm_head transpose  | `decoder.rs` | Pre-compute `tok_embeddings.t().contiguous()` at load time as `tok_embeddings_t`. Eliminates transposing + copying 800MB (131072×3072 BF16) every forward call.                                                                                                                                                          |
| 7   | Precomputed Ada-RMSNorm   | `decoder.rs` | `precompute_t_cond()` computes all 26 per-layer Ada-RMSNorm scales once (Linear→GELU→Linear→unsqueeze→add). Stored in `ada_scales: Option<Vec<Tensor>>`. Called once after t_cond is created; forward uses cached values. Saves ~130 kernel launches/token.                                                              |

**Prerequisite fix**: `candle-nn` was missing `features = ["cuda"]` in Cargo.toml, which caused `#[cfg(feature = "cuda")]` blocks to be compiled out. Fused CUDA kernels for `rms_norm` silently fell back to multi-op defaults. Fixed by adding `features = ["cuda"]` to the `candle-nn` dependency.

### Cumulative performance (offline, 3.7s test clip — "Dancing in the")

| Stage            | Baseline | After dec/enc opts | After flash attn   | Total improvement |
| ---------------- | -------- | ------------------ | ------------------ | ----------------- |
| Encoder          | 1.77s    | 1.30s (27% faster) | 1.03s (42% faster) | 42%               |
| Decoder          | 0.80s    | 0.64s (20% faster) | 0.58s (28% faster) | 28%               |
| Tok/s            | 52.3     | 65.9               | 72.9               | +39%              |
| Real-time factor | 0.22x    | 0.17x              | 0.16x              | —                 |

**Note:** First run after build shows inflated times (~1.36s encoder, ~0.80s decoder) due to one-time CUDA JIT compilation of flash attention kernels. Subsequent runs show the true performance.

### Changes by file

- `Cargo.toml`: Added `features = ["cuda"]` to `candle-nn`; added `candle-flash-attn = { git = "..." }` dependency
- `common.rs`:
  - `RmsNorm::forward`: replaced with `candle_nn::ops::rms_norm()` call
  - `KvCache`: uses `[batch, seq, heads, dim]` layout — append cats on dim 1, current_len reads dim 1, trim narrows on dim 1
  - `RotaryEmbedding::apply`: seq_len read from `dim(1)`, cos/sin unsqueeze on dim 2
- `encoder.rs`:
  - `Attention::forward`: replaced entire matmul+mask+softmax+matmul pipeline with `flash_attn_windowed()`; no mask parameter, no transposes
  - Removed `precompute_mask()` function
  - `EncoderLayer::forward`: no mask parameter
  - `AudioEncoder::forward_chunk` / `forward`: no mask computation or passing
- `decoder.rs`:
  - `Attention::forward`: replaced entire matmul+mask+softmax+matmul pipeline with `flash_attn_windowed()`; no GQA head repetition, no transposes
  - Removed `repeat_kv()`, `apply_causal_sliding_window_mask()`, `NUM_GROUPS` constant
  - `TextDecoder` struct: added `tok_embeddings_t`, `ada_scales` fields
  - `TextDecoder::load`: pre-computes `tok_embeddings_t = tok_embeddings.t().contiguous()`
  - `TextDecoder::precompute_t_cond()`: precomputes 26 ada_scale tensors
  - `TextDecoder::forward`: uses precomputed ada_scales and cached `tok_embeddings_t` for lm_head
  - `DecoderLayer::forward`: parameter changed from `t_cond: &Tensor` to `ada_scale: &Tensor`
- `main.rs`: added `dec.precompute_t_cond(&t_cond)?` before first forward call
- `streaming.rs`: added `dec.precompute_t_cond(&t_cond)?` in `run_startup` before first forward call

### Superseded optimizations (no longer in codebase)

Four intermediate optimizations were applied during earlier phases but were fully superseded when flash attention replaced the entire attention pipeline:

1. **Fused softmax** (decoder + encoder) — `softmax_last_dim()` replaced manual 7-op softmax, but flash attention now handles softmax internally.
2. **Skip mask for q_len=1** (decoder) — fast path in `apply_causal_sliding_window_mask()`, but that entire function was removed.
3. **Shared mask across layers** (encoder) — `precompute_mask()` computed mask once per chunk, but that entire function was removed.

These are listed for historical context only. The code and measurements above reflect the current state.

**Verified**: Offline mode produces identical output (" Dancing in the"). Zero warnings, zero errors.



## Constraints

1. **Don't break offline mode**: `voicet <file.wav>` must still work identically.
2. **Don't break streaming correctness**: The startup + decode loop must produce the same tokens.
3. **Minimize changes**: Fix the specific bugs. Don't restructure what's working.
4. **Zero warnings**: `cargo build --release` should compile with no warnings and no errors.
5. **Test**: After fixes, verify offline still works: `cargo run --release -- Voxtral-Mini-4B-Realtime/test01_16khz_3.7s.wav` should produce " Dancing in the".
