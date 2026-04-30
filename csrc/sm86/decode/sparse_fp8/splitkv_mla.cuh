#pragma once
// Sparse FP8 decode SM86 — simple CUDA-core kernel
// Handles FP8 dequantization on CUDA cores, then bf16 attention

#include <math_constants.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "defines.h"
#include "utils.h"
#include "params.h"

namespace sm86 {
namespace decode {

// FP8 KV cache layout (V32): 656 bytes per token
//   512 bytes: float8_e4m3 NOPE (4 tiles of 128)
//   16 bytes:  4 x float32 scale factors
//   128 bytes: bfloat16 ROPE (64 elements)
// For MODEL1: 512 bytes NOPE (7 tiles of 64 + 1 padding) with fp8_e8m0 scales

// Dequantize V32-style: fp8 NOPE -> bf16 using per-tile float32 scales
template<int BLK_M, int BLK_N, int HD_K, int HD_NOPE, int HD_ROPE, int NUM_SCALES>
__device__ void dequant_v32_fp8_to_bf16(
    const uint8_t* fp8_kv,  // [BLK_N, 656] packed
    bf16* sK,               // [BLK_N, HD_K] bf16 output in shared memory
    int valid_tokens)
{
    const int tid = threadIdx.x, num_threads = BLK_M > 0 ? 128 : 32;
    // Each thread processes some elements
    for (int tok = 0; tok < valid_tokens; ++tok) {
        const uint8_t* token = fp8_kv + tok * 656;
        // NOPE part: 512 fp8 values -> bf16
        for (int i = tid; i < HD_NOPE; i += num_threads) {
            int tile = i / 128;  // which 128-element tile (0..3)
            float scale = ((const float*)(token + 512))[tile];
                float val = ((const fp8*)(token))[i];
            sK[tok * HD_K + i] = bf16(val * scale);
        }
        // ROPE part: 64 bf16 values (native, no dequant needed)
        for (int i = tid; i < HD_ROPE; i += num_threads) {
            sK[tok * HD_K + HD_NOPE + i] = ((const bf16*)(token + 528))[i];
        }
    }
}

// Sparse MLA decode kernel
template<int BLK_M, int BLK_N, int HD_K, int HD_V, bool HAVE_ATTN_SINK>
__global__ void __launch_bounds__(128)
flash_fwd_sparse_decode_mla_kernel(const SparseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;
    
    DecodingSchedMeta* sched_ptr = params.tile_scheduler_metadata_ptr;
    DecodingSchedMeta smeta = sched_ptr[partition_idx];
    if (smeta.begin_req_idx >= params.b) return;

    for (int bi = smeta.begin_req_idx; bi <= smeta.end_req_idx; ++bi) {
        int n_split = bi == smeta.begin_req_idx ? smeta.begin_split_idx : 0;
        int num_valid = min(params.h_q - m_block_idx * BLK_M, BLK_M);

        // Load Q
        const bf16* q_ptr = params.q + bi * params.stride_q_b + m_block_idx * BLK_M * params.stride_q_h_q;
        int my_row = tid % BLK_M;
        bool valid = my_row < num_valid;

        // Per-row state
        float running_max = -1e30f;
        float running_sum = 0.0f;
        float o_acc[HD_V] = {};

        // Sparse: iterate over indices
        const int* idx_base = params.indices + bi * params.stride_indices_b;
        int topk = params.topk;
        int num_tk_blocks = (topk + BLK_N - 1) / BLK_N;

        for (int tb = 0; tb < num_tk_blocks; ++tb) {
            int tk_start = tb * BLK_N;
            int this_tk = min(BLK_N, topk - tk_start);
            if (this_tk <= 0) break;

            // Compute QK^T and softmax per row
            if (valid) {
                float P[BLK_N];
                float cur_max = -1e33f;
                for (int ni = 0; ni < this_tk; ++ni) {
                    int idx = __ldg(idx_base + bi * params.stride_indices_b / sizeof(int) + tk_start + ni);
                    if (idx < 0) { P[ni] = -1e33f; continue; }
                    int block_id = idx / params.page_block_size;
                    int offset = idx % params.page_block_size;
                    
                    // FP8 dequant on-the-fly
                    const uint8_t* token_fp8 = ((const uint8_t*)params.kv) + block_id * params.stride_kv_block + offset * 656;
                    float dot = 0.0f;
                    // Q from global
                    const bf16* q_row = q_ptr + my_row * params.stride_q_h_q;
                    
                    // NOPE part: dequant fp8->bf16, then dot product
                    for (int k = 0; k < 512; ++k) {
                        int tile = k / 128;
                        float scale = ((const float*)(token_fp8 + 512))[tile];
                        float kv_val = (float)((const fp8*)(token_fp8))[k] * scale;
                        dot += (float)q_row[k] * kv_val;
                    }
                    // ROPE part: native bf16
                    for (int k = 0; k < 64; ++k) {
                        float kv_val = (float)((const bf16*)(token_fp8 + 528))[k];
                        dot += (float)q_row[512 + k] * kv_val;
                    }
                    dot *= params.sm_scale_div_log2;
                    P[ni] = dot;
                    cur_max = max(cur_max, dot);
                }
                for (int ni = this_tk; ni < BLK_N; ++ni) P[ni] = -1e33f;

                // Online softmax
                float old_max = running_max;
                float new_max = max(old_max, cur_max);
                float scale_old = (old_max <= -1e29f) ? 0.0f : exp2f(old_max - new_max);
                running_max = new_max;
                running_sum *= scale_old;
                for (int c = 0; c < HD_V; ++c) o_acc[c] *= scale_old;

                float cur_sum = 0.0f;
                for (int ni = 0; ni < this_tk; ++ni) {
                    float val = P[ni];
                    if (val <= -1e32f) { P[ni] = 0.0f; continue; }
                    val = exp2f(val - new_max); P[ni] = val; cur_sum += val;
                }
                running_sum += cur_sum;

                // PV
                for (int ni = 0; ni < this_tk; ++ni) {
                    float pn = P[ni]; if (pn == 0.0f) continue;
                    int idx = __ldg(idx_base + bi * params.stride_indices_b / sizeof(int) + tk_start + ni);
                    if (idx < 0) continue;
                    int block_id = idx / params.page_block_size;
                    int offset = idx % params.page_block_size;
                    const uint8_t* token_fp8 = ((const uint8_t*)params.kv) + block_id * params.stride_kv_block + offset * 656;
                    
                    // V is first 512 elements of K (NOPE part + first 0 of ROPE... no, just NOPE)
                    for (int c = 0; c < 512; ++c) {
                        int tile = c / 128;
                        float scale = ((const float*)(token_fp8 + 512))[tile];
                        float kv_val = (float)((const fp8*)(token_fp8))[c] * scale;
                        o_acc[c] += pn * kv_val;
                    }
                }
            }
        }

        // Write output
        if (valid) {
            float rcp = (running_sum == 0.0f || running_sum != running_sum) ? 1.0f : (1.0f / running_sum);
            float attn_sink_adj = 1.0f;
            if (HAVE_ATTN_SINK && params.attn_sink) {
                float sink = __ldg(params.attn_sink + m_block_idx * BLK_M + my_row) * CUDART_L2E_F;
                attn_sink_adj = 1.0f / (1.0f + exp2f(sink - running_max));
                if (running_sum == 0.0f) attn_sink_adj = 0.0f;
            }

            bf16* out_row = params.out + bi * params.stride_o_b + (m_block_idx * BLK_M + my_row) * params.stride_o_h_q;
            for (int c = 0; c < HD_V; ++c) out_row[c] = bf16(o_acc[c] * rcp * attn_sink_adj);

            float* lse_row = params.lse + bi * params.stride_lse_b + (m_block_idx * BLK_M + my_row) * params.stride_lse_s_q;
            *lse_row = (running_sum == 0.0f) ? INFINITY : logf(running_sum) + running_max * CUDART_LN2_F;
        }
    }
#else
    asm("trap;");
#endif
}

// Host launch
void run_sparse_decode_fwd_kernel(SparseAttnDecodeParams &params) {
    constexpr int BLK_M = 16, BLK_N = 64, HD_K = 576, HD_V = 512;
    const int num_m_block = (params.h_q + BLK_M - 1) / BLK_M;
    auto kernel = &flash_fwd_sparse_decode_mla_kernel<BLK_M, BLK_N, HD_K, HD_V, true>;
    // Try both with and without attn_sink
    if (params.attn_sink == nullptr) {
        auto kernel_ns = &flash_fwd_sparse_decode_mla_kernel<BLK_M, BLK_N, HD_K, HD_V, false>;
        kernel_ns<<<dim3(num_m_block, 1, params.num_sm_parts), dim3(128), 0, params.stream>>>(params);
    } else {
        kernel<<<dim3(num_m_block, 1, params.num_sm_parts), dim3(128), 0, params.stream>>>(params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace decode
}  // namespace sm86
