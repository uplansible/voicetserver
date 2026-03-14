Pipeline Timing Instrumentation

Per-stage timing was added to `run_processing_loop()` in `streaming.rs`. Each stage of the per-token pipeline is wrapped with `Instant::now()` / `elapsed()` and printed to stderr so it doesn't mix with transcription output on stdout:

```
eprintln!("[tok] total={:.1}ms  mel_build={:.1} conv={:.1} enc={:.1} enc_trim={:.1} adapt={:.1} dec={:.1} argmax={:.1} dec_trim={:.1}", ...);
```

Stages measured (in order):

1. **mel_build** — transpose mel frames into `[N_MELS, process_len]` tensor, upload to GPU, cast to BF16
2. **conv** — `enc.conv_stem()` (2 Conv1d layers with GELU)
3. **enc** — `enc.forward_chunk()` (32 transformer layers with KV cache)
4. **enc_trim** — `enc.trim_caches()` (narrow+contiguous on 32 layers × K,V)
5. **adapt** — `adapter.forward()` (concat 4 frames → 2 linears with GELU)
6. **dec** — `dec.embed_tokens()` + fuse + `dec.forward()` (26 transformer layers with KV cache)
7. **argmax** — `argmax_last()` (BF16→F32 cast + argmax + `to_scalar` GPU→CPU sync)
8. **dec_trim** — `dec.trim_caches()` (narrow+contiguous on 26 layers × K,V)

Note: without explicit `device.synchronize()` calls, CUDA operations are async. The timers measure kernel launch time, not completion. The `to_scalar()` in argmax is the only sync point, so argmax absorbs all pending GPU work.

## Pipeline Timing Results

### Async timings (no CUDA sync — default, used in production)

```
[tok] total=55.5ms  mel_build=0.0 conv=0.2 enc=22.2 enc_trim=0.6 adapt=0.0 dec=15.1 argmax=17.3 dec_trim=0.0
[tok] total=55.0ms  mel_build=0.0 conv=0.2 enc=21.9 enc_trim=0.5 adapt=0.0 dec=15.4 argmax=16.9 dec_trim=0.0
[tok] total=54.5ms  mel_build=0.1 conv=0.2 enc=21.7 enc_trim=0.6 adapt=0.0 dec=15.5 argmax=16.4 dec_trim=0.0
```

Misleading: argmax appears to take ~16ms but is actually absorbing decoder GPU completion time.

### Accurate timings (with `device.synchronize()` after each stage)

```
[tok] total=64.0ms  mel_build=0.1 conv=0.3 enc=22.6 enc_trim=2.3 adapt=0.1 dec=38.4 argmax=0.2 dec_trim=0.0
[tok] total=63.7ms  mel_build=0.1 conv=0.2 enc=22.2 enc_trim=2.3 adapt=0.1 dec=38.7 argmax=0.1 dec_trim=0.0
[tok] total=63.2ms  mel_build=0.1 conv=0.2 enc=21.7 enc_trim=2.3 adapt=0.1 dec=38.7 argmax=0.1 dec_trim=0.0
```

Syncs add ~10ms total overhead (7 sync points × ~1.5ms pipeline stall each). These are removed for production; used only for profiling.

### True per-stage costs

| Stage     | Real time | % of total | Notes                                       |
| --------- | --------- | ---------- | ------------------------------------------- |
| **dec**   | **39ms**  | **60%**    | 26 transformer layers, GQA, 3072-dim hidden |
| **enc**   | **22ms**  | **34%**    | 32 transformer layers, 1280-dim hidden      |
| enc_trim  | 2.3ms     | 4%         | 64 narrow+contiguous ops (32 layers × K,V)  |
| conv      | 0.3ms     | <1%        | 2 Conv1d layers                             |
| mel_build | 0.1ms     | <1%        | CPU transpose + GPU upload                  |
| adapt     | 0.1ms     | <1%        | 2 linear layers                             |
| argmax    | 0.2ms     | <1%        | argmax over 131K vocab                      |
| dec_trim  | 0.0ms     | <1%        | Not yet hitting sliding window              |

**Budget: 80ms per token. Current: ~55ms (async). Headroom: ~25ms.**

The decoder's KV cache hasn't hit the 8192 sliding window yet, so dec_trim is 0.0ms. The dec stage (39ms) may grow as the cache fills. enc_trim is already 2.3ms and stable (already trimming at the 750-frame window).
