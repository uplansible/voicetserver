# Voicet Debugging Learnings - Session 2

## Fix 1: Conv Embedder Frame Truncation (Step 4)

**Problem**: Rust was truncating the conv embedder output (313 frames) to the nearest multiple of CHUNK_SIZE (4), dropping it to 312 frames. Python processes all 313 frames.

**Root cause**: The chunked encoder loop only handled full chunks. The remainder frame (313 % 4 = 1) was silently dropped.

**Fix** (`src/encoder.rs`): Handle remainder chunks properly — 78 full chunks of 4 + 1 remainder chunk of 1. No truncation.

## Fix 2: Q/K Weight Deinterleaving (Step 5)

**Problem**: Step 5 (encoder RoPE) showed Q and K values completely mismatched between Python and Rust. Max diff was ~2.6 for Q and ~5.2 for K — far beyond BF16 noise.

**Diagnosis approach**:
1. At position 0, cos(0)=1 and sin(0)=0, so RoPE is a no-op. Q values at pos 0 ARE the raw projection output. They still diverged, ruling out a RoPE formula bug.
2. Compared Rust's Q pos-0 values deinterleaved (even indices → first half, odd → second half) against Python's. Max diff dropped to 0.0078 — BF16 noise. This confirmed a dimension ordering issue.
3. Compared the raw `wq.weight` from `consolidated.safetensors` against `q_proj.weight` from `model.safetensors`. Direct diff was 0.186. After deinterleaving: diff was exactly 0.0.

**Root cause**: The project loads weights from `consolidated.safetensors` (Mistral native format). This format stores Q and K weight matrices with head dimensions **interleaved**: `[d0, d_half, d1, d_half+1, ...]` per head block. HF's `model.safetensors` has them in **paired-halves** format: `[d0, d1, ..., d_half-1, d_half, ...]`. The Rust RoPE uses HF's `rotate_half` convention (split first/second half), so the weights must be deinterleaved.

**Fix** (`src/encoder.rs` and `src/decoder.rs`): Added `deinterleave_qk()` function called at load time on wq and wk weight tensors. It reshapes to `[num_heads, head_dim, in_features]`, gathers even-indexed rows as first half and odd-indexed rows as second half, then reshapes back. One-time cost at load, zero runtime overhead. V and O weights don't need deinterleaving.

**Affected weights**:
- Encoder: wq (32 heads, head_dim=64), wk (32 heads, head_dim=64)
- Decoder: wq (32 heads, head_dim=128), wk (8 kv_heads, head_dim=128)

**Result**: Transcription now matches Python ground truth: `" Dancing in the"`. Also tested on a 28-second clip successfully.
