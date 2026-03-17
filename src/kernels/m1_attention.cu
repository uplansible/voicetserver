// M=1 GQA Attention kernel for streaming decoder single-token inference.
//
// Design: 1 warp (32 threads) per query head, 4 dims per thread.
// 32 threads x 4 dims = 128 = HEAD_DIM.
//
// Inspired by antirez's voxtral.c Metal kernel:
// - Cooperative Q.K dot product via warp shuffle reduction
// - Online softmax (all threads maintain identical state)
// - Per-thread V accumulation (no cross-thread reduction needed)
//
// Layout assumptions (flash attention format, contiguous):
//   Q: [batch, 1, num_q_heads, HEAD_DIM]      -- single query token
//   K: [batch, kv_len, num_kv_heads, HEAD_DIM] -- KV cache
//   V: [batch, kv_len, num_kv_heads, HEAD_DIM]
//   O: [batch, 1, num_q_heads, HEAD_DIM]

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define HEAD_DIM 128
#define WARP_SIZE 32
#define DIMS_PER_THREAD 4  // HEAD_DIM / WARP_SIZE

extern "C" __global__ void m1_gqa_attn_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    const int kv_len,
    const int num_kv_heads,
    const int gqa_ratio,
    const float scale,
    const int window_size
) {
    const int q_head = blockIdx.x;
    const int kv_head = q_head / gqa_ratio;
    const int tid = threadIdx.x;
    const int d_start = tid * DIMS_PER_THREAD;

    // Load this thread's 4 Q values into registers
    const int q_off = q_head * HEAD_DIM + d_start;
    float q0 = __bfloat162float(q[q_off]);
    float q1 = __bfloat162float(q[q_off + 1]);
    float q2 = __bfloat162float(q[q_off + 2]);
    float q3 = __bfloat162float(q[q_off + 3]);

    // Sliding window: only attend to last window_size entries
    int start = (kv_len > window_size) ? kv_len - window_size : 0;

    // Online softmax state (identical across all 32 threads after each warp reduce)
    float max_score = -CUDART_INF_F;
    float sum_exp = 0.0f;

    // V accumulator (each thread owns its own 4 dims -- no final reduction needed)
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

    for (int pos = start; pos < kv_len; pos++) {
        // Index into K/V: [pos, kv_head, d_start..]
        const int kv_off = (pos * num_kv_heads + kv_head) * HEAD_DIM + d_start;

        // Partial Q.K dot product (4 dims per thread)
        float dot = q0 * __bfloat162float(k[kv_off])
                  + q1 * __bfloat162float(k[kv_off + 1])
                  + q2 * __bfloat162float(k[kv_off + 2])
                  + q3 * __bfloat162float(k[kv_off + 3]);

        // Warp reduce sum: all 32 threads get the full 128-dim dot product
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }
        dot *= scale;

        // Online softmax update (deterministic: all threads compute identical values)
        float old_max = max_score;
        max_score = fmaxf(max_score, dot);
        float exp_diff = __expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + __expf(dot - max_score);

        // V accumulation with rescaling for numerical stability
        float weight = __expf(dot - max_score);
        v0 = v0 * exp_diff + weight * __bfloat162float(v[kv_off]);
        v1 = v1 * exp_diff + weight * __bfloat162float(v[kv_off + 1]);
        v2 = v2 * exp_diff + weight * __bfloat162float(v[kv_off + 2]);
        v3 = v3 * exp_diff + weight * __bfloat162float(v[kv_off + 3]);
    }

    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    const int o_off = q_head * HEAD_DIM + d_start;
    out[o_off]     = __float2bfloat16(v0 * inv_sum);
    out[o_off + 1] = __float2bfloat16(v1 * inv_sum);
    out[o_off + 2] = __float2bfloat16(v2 * inv_sum);
    out[o_off + 3] = __float2bfloat16(v3 * inv_sum);
}
