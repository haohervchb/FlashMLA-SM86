#pragma once
// Sparse prefill SM86 - simple CUDA-core kernel

#include <math_constants.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "defines.h"
#include "utils.h"
#include "params.h"

namespace sm86 {
namespace prefill {

template<int D_QK, bool HAVE_TOPK_LENGTH>
__global__ void __launch_bounds__(128)
sparse_attn_fwd_kernel(const SparseAttnFwdParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int D_V = 512, B_H = 16;
    const int q_h_idx  = blockIdx.x % (params.h_q / B_H);
    const int s_q_idx  = blockIdx.x / (params.h_q / B_H);
    const int tid = threadIdx.x;
    const int my_row = tid % B_H;
    const int g_row = q_h_idx * B_H + my_row;
    const bool valid = g_row < params.h_q;

    const int topk = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_tk_blocks = (topk + 63) / 64;

    float o_acc[D_V] = {};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    const bf16* q_ptr = params.q + s_q_idx * params.stride_q_s_q;
    const int* idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;

    for (int tb = 0; tb < num_tk_blocks; ++tb) {
        int tk_start = tb * 64;
        int this_tk = min(64, topk - tk_start);
        if (this_tk <= 0) break;

        if (valid) {
            float P[64];
            float cur_max = -1e33f;
            for (int ni = 0; ni < this_tk; ++ni) {
                int tok_idx = tk_start + ni;
                int g_tok = __ldg(idx_ptr + tok_idx);
                if (g_tok < 0 || g_tok >= params.s_kv) { P[ni] = -1e33f; continue; }
                float dot = 0.0f;
                const bf16* q_row = q_ptr + g_row * params.stride_q_h_q;
                const bf16* kv_row = params.kv + g_tok * params.stride_kv_s_kv;
                for (int k = 0; k < D_QK; ++k) {
                    dot += (float)q_row[k] * (float)kv_row[k];
                }
                dot *= params.sm_scale_div_log2;
                P[ni] = dot;
                cur_max = max(cur_max, dot);
            }
            for (int ni = this_tk; ni < 64; ++ni) P[ni] = -1e33f;

            float old_max = running_max;
            float new_max = max(old_max, cur_max);
            float scale_old = (old_max <= -1e29f) ? 0.0f : exp2f(old_max - new_max);
            running_max = new_max;
            running_sum *= scale_old;
            for (int c = 0; c < D_V; ++c) o_acc[c] *= scale_old;

            float cur_sum = 0.0f;
            for (int ni = 0; ni < this_tk; ++ni) {
                float val = P[ni];
                if (val <= -1e32f) { P[ni] = 0.0f; continue; }
                val = exp2f(val - new_max); P[ni] = val; cur_sum += val;
            }
            running_sum += cur_sum;

            for (int ni = 0; ni < this_tk; ++ni) {
                float pn = P[ni]; if (pn == 0.0f) continue;
                int tok_idx = tk_start + ni;
                int g_tok = __ldg(idx_ptr + tok_idx);
                if (g_tok < 0 || g_tok >= params.s_kv) continue;
                const bf16* kv_row = params.kv + g_tok * params.stride_kv_s_kv;
                for (int c = 0; c < D_V; ++c) o_acc[c] += pn * (float)kv_row[c];
            }
        }
    }

    if (valid) {
        float attn_sink = params.attn_sink ? (float)(params.attn_sink[g_row]) * CUDART_L2E_F : -CUDART_INF_F;
        float scale_factor = 1.0f / (running_sum + exp2f(attn_sink - running_max));
        if (running_sum == 0.0f) scale_factor = 0.0f;

        bf16* out_row = params.out + s_q_idx * (params.h_q * D_V) + g_row * D_V;
        for (int c = 0; c < D_V; ++c) out_row[c] = bf16(o_acc[c] * scale_factor);

        float lse_val = (running_sum == 0.0f) ? INFINITY : logf(running_sum) + running_max * CUDART_LN2_F;
        params.lse[s_q_idx * params.h_q + g_row] = lse_val;
        params.max_logits[s_q_idx * params.h_q + g_row] = (running_max <= -1e29f) ? -INFINITY : running_max * CUDART_LN2_F;
    }
#else
    asm("trap;");
#endif
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    const int grid_dim = (params.h_q / 16) * params.s_q;
    auto kernel = &sparse_attn_fwd_kernel<D_QK, HAVE_TOPK_LENGTH>;
    kernel<<<dim3(grid_dim), dim3(128), 0, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace prefill
}  // namespace sm86
