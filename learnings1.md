# Learnings: Fixing Rust Voxtral Pipeline Steps 1-5

## Summary

Four bugs were found and fixed to make the Rust pipeline match Python through step 5.
The Rust pipeline now produces correct output: `" Dancing in the"`.

---

## Bug 1: Mel spectrogram frame count (step 2)

**File**: `src/mel.rs` lines 137-162

**Symptom**: Rust produced 800 mel frames, Python produced 626.

**Root cause**: The mel code was copied from Whisper (via candle) which pads the frame
count to a multiple of 100 then adds another 100. Voxtral doesn't do this — it uses
the natural STFT frame count.

**Fix**: Removed the Whisper-style padding:
```rust
// REMOVED:
let pad = 100;
let n_len = if n_len % pad != 0 { (n_len / pad + 1) * pad } else { n_len };
let n_len = n_len + pad;

// REPLACED WITH:
let stft_frames = 1 + (padded_samples.len() - fft_size) / fft_step;
let n_len = stft_frames - 1; // drop last frame to match HF's stft[..., :-1]
```

**Lesson**: Don't assume one model's audio preprocessing matches another's. Always verify
the exact frame count formula against the reference implementation.

---

## Bug 2: Mel spectrogram center padding mode (step 2)

**File**: `src/mel.rs` lines 137-151

**Symptom**: Mel values diverged from Python (max diff ~1.97) even after fixing the frame count.

**Root cause**: Rust zero-padded both sides for center STFT. PyTorch's `torch.stft(center=True)`
uses **reflect padding** by default. For audio that starts/ends with non-silence, this makes
a real difference at the edges.

**Fix**: Changed from zero-pad to reflect-pad:
```rust
// Reflect pad at the start: samples[center_pad], samples[center_pad-1], ..., samples[1]
for i in (1..=center_pad).rev() {
    padded_samples.push(samples[i.min(n_samples - 1)]);
}
padded_samples.extend_from_slice(samples);
// Reflect pad at the end: samples[n-2], samples[n-3], ...
for i in 0..center_pad {
    let idx = n_samples.saturating_sub(2 + i);
    padded_samples.push(samples[idx]);
}
```

**After fix**: Max diff dropped from 1.97 to 0.0039 (f32 FFT precision noise only).

**Lesson**: "center=True" in torch.stft doesn't just mean "pad both sides" — it means
"reflect-pad both sides". The padding mode matters.

---

## Bug 3: Mel spectrogram last-frame drop (step 2)

**File**: `src/mel.rs` line 156

**Symptom**: Frame count was off by 1 even after fixing the padding formula.

**Root cause**: The HF feature extractor does `stft[..., :-1].abs() ** 2` — it drops the
**last STFT frame** before computing power spectra. Rust didn't do this.

**Fix**: `let n_len = stft_frames - 1;`

**Lesson**: Read the reference code carefully, line by line. A single `[..., :-1]` slice
is easy to miss but changes the output shape.

---

## Bug 4: Encoder frame truncation (step 4)

**File**: `src/encoder.rs` lines 535-539

**Symptom**: Rust produced 312 encoder frames, Python produced 313. The conv2 (stride=2)
on 626 mel frames gives 313 output frames. Rust was truncating to 312 (nearest multiple
of CHUNK_SIZE=4).

**Root cause**: The Rust encoder processed frames in fixed chunks of 4 and discarded the
remainder: `let n_usable = (total_frames / CHUNK_SIZE) * CHUNK_SIZE;`. Python processes
all frames — no truncation.

**Fix**: Handle the remainder chunk:
```rust
let n_full_chunks = total_frames / CHUNK_SIZE;
let remainder = total_frames % CHUNK_SIZE;
let n_chunks = n_full_chunks + if remainder > 0 { 1 } else { 0 };

for chunk_idx in 0..n_chunks {
    let offset = chunk_idx * CHUNK_SIZE;
    let chunk_len = if chunk_idx < n_full_chunks { CHUNK_SIZE } else { remainder };
    let chunk = x.narrow(1, offset, chunk_len)?;
    // ... process chunk
}
```

**Lesson**: When chunking a sequence for KV-cached processing, you must handle the
remainder. Dropping frames silently corrupts the output.

---

## Bug 5: Q/K weight interleaving (step 5)

**File**: `src/encoder.rs` lines 133-150, `src/decoder.rs` lines 119-131

**Symptom**: Q and K values after RoPE were completely wrong — first-5 values didn't
align at all between Python and Rust. Max diff > 2.0.

**Root cause**: The model ships with two weight files:
- `consolidated.safetensors` (Mistral native) — Q/K head dimensions in **interleaved** order:
  `[d0, d_half, d1, d_half+1, d2, d_half+2, ...]`
- `model.safetensors` (HF converted) — Q/K head dimensions in **paired-halves** order:
  `[d0, d1, d2, ..., d_half-1, d_half, d_half+1, ...]`

Rust loads from `consolidated.safetensors` but uses HF's `rotate_half` RoPE implementation
which expects paired-halves format. The RoPE splits the head dimension at the midpoint:
first half gets `cos`, second half gets `-sin * rotate`. If the weights are interleaved,
every other dimension is in the wrong half, causing completely wrong rotations.

**Fix**: Added `deinterleave_qk()` called at weight load time on `wq` and `wk` matrices
(NOT `wv` or `wo` — those don't participate in RoPE):
```rust
fn deinterleave_qk(w: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let in_features = w.dim(1)?;
    let half = head_dim / 2;
    let w = w.reshape((num_heads, head_dim, in_features))?;
    let even_idx: Vec<u32> = (0..head_dim as u32).step_by(2).collect();  // [0,2,4,...]
    let odd_idx: Vec<u32> = (1..head_dim as u32).step_by(2).collect();   // [1,3,5,...]
    let first_half = w.index_select(&even_idx, 1)?;   // gather even → first half
    let second_half = w.index_select(&odd_idx, 1)?;    // gather odd → second half
    Tensor::cat(&[&first_half, &second_half], 1)?.reshape((num_heads * head_dim, in_features))
}
```

**How to verify**: At position 0, RoPE is a no-op (cos=1, sin=0), so Q[pos=0] is the raw
projection output. Compare Rust vs Python Q values at position 0 — they should match
within BF16 tolerance after the fix.

**Lesson**: When loading Mistral-format weights and using HF-style RoPE, you MUST
deinterleave Q and K weight matrices. This is the single most impactful bug — without it,
every attention computation in every layer is wrong.

---

## Debugging methodology that worked

1. **Save intermediate outputs at each step** in both pipelines using a standard format
2. **Compare shapes first** — shape mismatches indicate logic bugs, not numerical noise
3. **Compare values** — max diff > 0.01 means a real bug, < 0.01 is BF16 rounding
4. **Work forward** — fix step N before moving to step N+1, since errors propagate
5. **Use position 0 as a RoPE sanity check** — at pos 0, cos=1 and sin=0, so RoPE is identity
6. **Always compare the FINAL output** of each step, not intermediate states
