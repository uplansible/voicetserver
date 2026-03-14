#### NOTE: ### Step 1: Try gelu_erf fix (fastest test) WE ALREADY TRIED THIS DIDNT WORK

# Phase 1c v4: Fix audio pipeline bug, complete end-to-end transcription

## Symptom

Decoder produces all STREAMING_PAD tokens — zero text. Python (both F32 and BF16) produces correct text (e.g. `" Dancing in the masquerade ""`). The bug was originally suspected in decoder attention, but investigation proved the decoder code is correct.

## What's already done (DO NOT REDO)

- **Mel, encoder, adapter, decoder, tokenizer, generation loop, CLI** — all implemented and compiling. Mel validated to 4 decimal places. Encoder and adapter validated against Python (mean, std, first 5 values match).
- **Debug infrastructure**: `src/decoder.rs` has extensive debug prints (layer stats, attention internals, BF16 vs F32 comparison). `src/main.rs` has debug prints for logit top-3. A `NO_AUDIO=1` env var skips audio fusion for isolated decoder testing.

## Investigation completed — what we know

### 1. Decoder attention code is CORRECT (definitively proven)

Ran the decoder WITHOUT audio (`NO_AUDIO=1`) — only tok_embeddings, no adapter output. Every single intermediate value matches a standalone Python computation to 6 decimal places:

```
                        Rust                Python
attn_norm:       mean=-0.004611, std=0.262722   mean=-0.004611, std=0.262723
q:               mean=-0.029480, std=0.726579   mean=-0.029479, std=0.726582
v:               mean= 0.000194, std=0.173119   mean= 0.000199, std=0.173105
scores pre-mask: mean=-0.547859, std=4.488439   mean=-0.547669, std=4.488911
after wo:        mean= 0.004056, std=0.114685   mean= 0.004053, std=0.114690
after attn+res:  mean= 0.003771, std=0.112482   mean= 0.003768, std=0.112486
```

This confirms: RmsNorm, Linear projections, reshape/transpose, RoPE, repeat_kv, causal mask, softmax, GQA matmul, output projection — ALL correct.

### 2. Weight values match Python exactly

```
Python wq[0,:5]: [0.00519, 0.01086, -0.00729, 0.00946, 0.01575]
Rust   wq[0,:5]: [0.00519, 0.01086, -0.00729, 0.00946, 0.01575]
(wv, wo also match)
```

### 3. BF16 precision is NOT the issue

Added F32 comparison for q and v projections: computed `x_f32 @ w_f32^T` manually and compared with candle's BF16 `Linear::forward`. Results match to 4+ decimal places at every layer. The decoder produces the same (wrong) results in both BF16 and F32.

### 4. Conv stem padding is correct

`left_pad = kernel_size - stride` (2 for conv1, 1 for conv2) matches Python's `VoxtralRealtimeConvolution.left_pad` formula exactly.

## What was ruled out

| Theory                     | How ruled out                                                        |
| -------------------------- | -------------------------------------------------------------------- |
| Decoder attention code bug | No-audio test matches Python perfectly                               |
| Weight loading error       | Printed values match Python                                          |
| BF16 matmul precision      | F32 manual comparison identical                                      |
| Missing biases on decoder  | Safetensors has no decoder bias tensors                              |
| repeat_kv interleaving     | Code matches PyTorch `repeat_interleave`, confirmed by no-audio test |
| Causal mask bug            | Correct by inspection + no-audio test                                |
| RoPE bug                   | Correct by inspection + no-audio test                                |
| Conv stem padding          | Matches Python formula                                               |

## The real bug: upstream of the decoder

Since the decoder is correct in isolation, the bug is in the **input** to the decoder. The input is `tok_embed(token) + adapter_out[position]`. `tok_embed` uses the same weight file and same token IDs → identical. Therefore the **adapter output must be wrong**, or its **alignment with token positions** is wrong.

With audio, layer 0 attention contributes almost nothing (after-wo std=0.019). Without audio, attention contributes significantly (after-wo std=0.115). Same decoder code, different input → the audio features are corrupting the input pattern in a way that makes attention collapse.

## Prime suspect: GELU variant mismatch

Candle `tensor.gelu()` uses the **tanh approximation**: `0.5x(1 + tanh(√(2/π) · x · (1 + 0.044715x²)))`

Python `nn.functional.gelu()` uses the **exact erf** version: `0.5x(1 + erf(x/√2))`

Candle has `tensor.gelu_erf()` for the exact version but the code uses `gelu()` in three places on the audio path:

- `src/encoder.rs:350,352` — conv stem (after conv1 and conv2)
- `src/adapter.rs:47` — adapter projector (between linear1 and linear2)

The max difference between tanh-GELU and erf-GELU is ~0.004 at x≈±1.5. While small per-element, this is a **systematic** (non-random) error that:

1. Affects the conv stem output fed to ALL 32 encoder transformer layers
2. Further affects the adapter projection
3. Could shift the 3072-dim audio feature vector in a direction the decoder wasn't trained to handle

BF16 truncation doesn't cause this issue because it's **random** (cancels out over many operations), while GELU approximation error is **directional** (always biased the same way).

The Ada-RMSNorm in decoder.rs:360 also uses `gelu()`, but this was tested in the no-audio experiment (t_cond is independent of audio) and doesn't cause issues.

## What to do next

### Step 1: Try gelu_erf fix (fastest test)

Change `gelu()` → `gelu_erf()` in encoder.rs (lines 350, 352) and adapter.rs (line 47). Rebuild and run. If output changes significantly, this is the bug.

### Step 2: If Step 1 doesn't fix it — write a Python end-to-end comparison

Write `debug_pipeline.py` that loads the safetensors weights (CPU, BF16→F32) and processes `test01_16khz_3.7s.wav` through the exact same pipeline: mel → conv stem → encoder → adapter → decoder layer 0. Print stats at every stage. Compare with Rust to find where divergence starts.

Key comparison points:

- Conv stem output [0,0,:5] and [0,5,:5] (first two frames)
- Encoder output [0,0,:5] and [0,5,:5]
- Adapter output — print ALL 62 frame norms, not just first 5 elements of frame 0
- Decoder layer 0: q, k, v stats after projection

### Step 3: If the adapter output turns out to be correct — check alignment

Verify that adapter frame `i` in Rust corresponds to the same audio content as adapter frame `i` in Python. Check whether the Python HuggingFace pipeline applies any offset, reordering, or padding to the adapter frames before adding them to token embeddings.

## Files

| File               | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `src/main.rs`      | Generation loop, WAV loading, CLI (has debug prints + NO_AUDIO mode)           |
| `src/mel.rs`       | Mel spectrogram (validated ✅)                                                  |
| `src/encoder.rs`   | Conv stem + 32 transformer layers (validated ✅, but GELU suspect)              |
| `src/adapter.rs`   | 4x downsample + projection (validated ✅, but GELU suspect)                     |
| `src/decoder.rs`   | 26-layer GQA decoder (code proven correct, has debug prints)                   |
| `src/tokenizer.rs` | Tekken decode-only tokenizer                                                   |
| `debug_layer0.py`  | Python standalone layer-0 comparison (no-audio, used to prove decoder correct) |

## Architecture reference

### Decoder

- 26 layers, hidden=3072, intermediate=9216, GQA 32Q/8KV heads, head_dim=128
- RoPE theta=1M, RMSNorm eps=1e-5, SwiGLU MLP, no biases
- Ada-RMSNorm: `Linear(3072→32) → GELU → Linear(32→3072)`, modulates ffn_norm only
- Tied tok_embeddings [131072, 3072] for embed + lm_head

### Generation protocol

- Prefill: BOS(1) + STREAMING_PAD(32) × 38 = 39 tokens
- Audio-text fusion: `tok_embed(token) + adapter_out[position]` at each position
- Delay conditioning: sinusoidal embedding of 6.0, dim=3072, theta=10000
- Stop on EOS(2) or all adapter frames consumed

## Once the bug is fixed

1. Remove all debug prints from `decoder.rs` and `main.rs`
2. Remove `NO_AUDIO` env var hack from `main.rs`
3. Run on `test01_16khz_3.7s.wav` and compare: Python output = `" Dancing in the masquerade ""`
4. Print performance stats (tok/s, real-time factor)
5. Update CLAUDE.md and memory files

**STOP** after offline transcription works. Do not proceed to Phase 2.
