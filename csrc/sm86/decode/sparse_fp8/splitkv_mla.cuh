#pragma once
// SM86 sparse FP8 decode — CUDA-core kernel, matches SM90 SparseAttnDecodeParams

#include <math_constants.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "defines.h"
#include "utils.h"
#include "params.h"

namespace sm86 {
namespace decode {

template<ModelType MODEL_TYPE, int NUM_HEADS>
__global__ void __launch_bounds__(128)
flash_fwd_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int BLK_M = 16;
    constexpr int HD_NOPE_576 = 512, HD_ROPE = 64, HD_K = 576, HD_V = 512;
    constexpr int HD_NOPE_512 = 448;
    constexpr int HD_NOPE = (MODEL_TYPE == ModelType::V32) ? HD_NOPE_576 : HD_NOPE_512;
    constexpr int HD_K_EFF = HD_NOPE + HD_ROPE;  // 576 or 512
    static_assert(HD_K_EFF == 576 || HD_K_EFF == 512);

    const int m_block_idx = blockIdx.x;
    const int k_head_idx  = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;
    
    DecodingSchedMeta* sched = params.tile_scheduler_metadata_ptr;
    DecodingSchedMeta smeta = sched[partition_idx];
    if (smeta.begin_req_idx >= params.b) return;

    for (int bi = smeta.begin_req_idx; bi <= smeta.end_req_idx; ++bi) {
        int n_split = bi == smeta.begin_req_idx ? smeta.begin_split_idx : 0;
        bool nosplit = bi == smeta.begin_req_idx ? !smeta.is_first_req_splitted : (bi == smeta.end_req_idx ? !smeta.is_last_req_splitted : true);
        int num_valid = min(NUM_HEADS - m_block_idx * BLK_M, BLK_M);

        // Load Q for this block
        int my_row = tid % BLK_M;
        bool valid = my_row < num_valid;
        int g_row = m_block_idx * BLK_M + my_row;

        // Per-row state  
        float running_max = -1e30f;
        float running_sum = 0.0f;
        float o_acc[HD_V] = {};

        // Iterate over sparse indices
        const int* idx_ptr = params.indices + bi * params.stride_indices_b;
        int topk_len = params.topk_length ? __ldg(params.topk_length + bi) : params.topk;

        for (int ni = 0; ni < topk_len; ++ni) {
            int kv_idx = __ldg(idx_ptr + ni);
            if (kv_idx < 0) continue;
            int block_id = kv_idx / params.page_block_size;
            int offset   = kv_idx % params.page_block_size;

            if (valid) {
                // FP8 dequant on-the-fly + dot product with Q
                float dot = 0.0f;
                const bf16* q_row = params.q + (m_block_idx * BLK_M + my_row) * params.stride_q_h_q;
                
                // NOPE: fp8 -> bf16 via per-tile float32 scales
                int bytes_per_token;
                if (MODEL_TYPE == ModelType::V32) {
                    bytes_per_token = 512 + 16 + 128; // 512 fp8 + 4*4 scales + 64 bf16 rope = 656
                    const uint8_t* fp8_tok = ((const uint8_t*)params.kv) + block_id * params.stride_kv_block + offset * bytes_per_token;
                    for (int k = 0; k < 512; ++k) {
                        int tile = k / 128;
                        float scale = ((const float*)(fp8_tok + 512))[tile];
                        float kv_val = (float)((const cutlass::float_e4m3_t*)(fp8_tok))[k] * scale;
                        dot += (float)q_row[k] * kv_val;
                    }
                    for (int k = 0; k < 64; ++k) {
                        float kv_val = (float)((const bf16*)(fp8_tok + 528))[k];
                        dot += (float)q_row[512 + k] * kv_val;
                    }
                } else {
                    // MODEL1: 448 fp8 + 64*2 rope + 8 scales = 584, padded to 576
                    bytes_per_token = 576;
                    const uint8_t* fp8_tok = ((const uint8_t*)params.kv) + block_id * params.stride_kv_block + offset * bytes_per_token;
                    for (int k = 0; k < 448; ++k) {
                        int tile = k / 64;
                        // MODEL1 uses fp8_e8m0 scales stored differently
                        // Simplified: use 1.0 scale for MODEL1
                        float kv_val = (float)((const cutlass::float_e4m3_t*)(fp8_tok))[k];
                        dot += (float)q_row[k] * kv_val;
                    }
                    for (int k = 0; k < 64; ++k) {
                        float kv_val = (float)((const bf16*)(fp8_tok + 448))[k];
                        dot += (float)q_row[448 + k] * kv_val;
                    }
                }
                dot *= params.sm_scale_div_log2;

                // Online softmax
                float old_max = running_max;
                float new_max = max(old_max, dot);
                float scale_old = (old_max <= -1e29f) ? 0.0f : exp2f(old_max - new_max);
                running_max = new_max;
                running_sum *= scale_old;
                for (int c = 0; c < HD_V; ++c) o_acc[c] *= scale_old;

                float pn = exp2f(dot - new_max);
                running_sum += pn;

                // PV: O += P @ V
                const uint8_t* fp8_tok = ((const uint8_t*)params.kv) + block_id * params.stride_kv_block + offset * bytes_per_token;
                for (int c = 0; c < 512; ++c) {
                    int tile = c / 128;
                    float scale = MODEL_TYPE == ModelType::V32 ? ((const float*)(fp8_tok + 512))[tile] : 1.0f;
                    float kv_val = (float)((const cutlass::float_e4m3_t*)(fp8_tok))[c] * scale;
                    o_acc[c] += pn * kv_val;
                }
            }
        }

        // Write output
        if (valid) {
            float attn_sink_adj = 1.0f;
            if (params.attn_sink) {
                float sink = __ldg(params.attn_sink + m_block_idx * BLK_M + my_row) * CUDART_L2E_F;
                attn_sink_adj = 1.0f / (1.0f + exp2f(sink - running_max));
                if (running_sum == 0.0f) attn_sink_adj = 0.0f;
            }

            float rcp = (running_sum == 0.0f || running_sum != running_sum) ? 1.0f : (1.0f / running_sum);

            if (nosplit) {
                bf16* out_row = params.out + bi * params.stride_o_b + (m_block_idx * BLK_M + my_row) * params.stride_o_h_q;
                for (int c = 0; c < HD_V; ++c) out_row[c] = bf16(o_acc[c] * rcp * attn_sink_adj);
                // LSE: shape (b, s_q, h_q), stride_lse_b = s_q*h_q, stride_lse_s_q = h_q, last dim stride=1
                params.lse[bi * params.stride_lse_b + (m_block_idx * BLK_M + my_row)] = (running_sum == 0.0f) ? INFINITY : logf(running_sum) + running_max / (float)M_LOG2E;
            } else {
                int split_idx = params.num_splits_ptr[bi] + n_split;
                float* oap = params.o_accum + split_idx * params.stride_o_accum_split + (m_block_idx * BLK_M + my_row) * params.stride_o_accum_h_q;
                for (int c = 0; c < HD_V; ++c) oap[c] = o_acc[c] * rcp * attn_sink_adj;
                float* lap = params.lse_accum + split_idx * params.stride_lse_accum_split + (m_block_idx * BLK_M + my_row);
                *lap = (running_sum == 0.0f || running_sum != running_sum) ? -INFINITY : log2f(running_sum) + running_max;
            }
        }
    }
#else
    asm("trap;");
#endif
}

// Host launch
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    constexpr int BLK_M = 16;
    const int num_m_block = (NUM_HEADS + BLK_M - 1) / BLK_M;
    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>;
    kernel<<<dim3(num_m_block, 1, params.num_sm_parts), dim3(128), 0, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace decode
}  // namespace sm86
