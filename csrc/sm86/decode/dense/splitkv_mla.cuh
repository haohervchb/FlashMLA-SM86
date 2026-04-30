#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "utils.h"
#include "params.h"
#include "config.h"

namespace sm86 {

static constexpr int BLK_M = 16;
static constexpr int BLK_N = 64;
static constexpr int HD_K = 576;
static constexpr int HD_V = 512;
static constexpr int NUM_THREADS = 16;

template<typename InputT>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_kernel(const DenseAttnDecodeParams params) {

    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ char wksp_buf[];
    InputT* sQ = (InputT*)wksp_buf;
    InputT* sK = sQ + BLK_M * HD_K;
    float* sL_reduction = (float*)(sK + BLK_N * HD_K);

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int seqlen_k_cache = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : (seqlen_k_cache + BLK_N - 1) / BLK_N;
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        InputT* q_ptr = (InputT*)params.q_ptr + batch_idx * params.q_batch_stride;
        InputT* o_ptr = (InputT*)params.o_ptr + batch_idx * params.o_batch_stride;
        float* softmax_lse_ptr = (float*)params.softmax_lse_ptr + (batch_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLK_M;
        int* block_table_ptr = params.block_table + batch_idx * params.block_table_batch_stride;
        int num_valid = min(params.q_seq_per_hk - m_block_idx * BLK_M, BLK_M);

        // Load Q
        int my_row = tid;
        bool valid = my_row < num_valid;
        int g_row = m_block_idx * BLK_M + my_row;
        if (valid) {
            for (int k = 0; k < HD_K; ++k) {
                sQ[my_row * HD_K + k] = q_ptr[g_row * params.q_row_stride + k_head_idx * params.q_head_stride + k];
            }
        }
        __syncthreads();

        // Per-row state
        float running_max = -1e30f;
        float running_sum = 0.0f;
        float o_accum[HD_V];
        for (int i = 0; i < HD_V; ++i) o_accum[i] = 0.0f;

        // Causal mask: compute right border for this q token row
        int causal_right_border = seqlen_k_cache;  // default: attend to all
        if (params.is_causal) {
            int global_seq_q_idx = m_block_idx * BLK_M + my_row;
            if (global_seq_q_idx < params.q_seq_per_hk) {
                int s_q_idx = global_seq_q_idx / params.q_head_per_hk;
                causal_right_border = seqlen_k_cache - (params.s_q - s_q_idx - 1);
                if (causal_right_border < 0) causal_right_border = 0;
            }
        }

        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            int token_start = block_idx * BLK_N;
            int remain = seqlen_k_cache - token_start;
            if (remain <= 0) break;
            int this_block_tokens = min(BLK_N, remain);
            int block_id = __ldg(block_table_ptr + block_idx);
            const InputT* gK = (InputT*)params.k_ptr + block_id * params.k_batch_stride + k_head_idx * params.k_head_stride;

            // Load KV into shared
            for (int i = tid; i < BLK_N * HD_K; i += NUM_THREADS) {
                int r = i / HD_K, c = i % HD_K;
                sK[r * HD_K + c] = (r < this_block_tokens) ? gK[r * params.k_row_stride + c] : InputT(0.0f);
            }
            __syncthreads();

            if (valid) {
                float scale = params.scale_softmax_log2;

                // QK^T
                float P[BLK_N];
                float cur_max = -1e33f;
                for (int ni = 0; ni < this_block_tokens; ++ni) {
                    int token_idx = token_start + ni;
                    // Causal mask
                    if (token_idx >= causal_right_border) {
                        P[ni] = -1e33f;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int k = 0; k < HD_K; ++k) {
                        dot += (float)sQ[my_row * HD_K + k] * (float)sK[ni * HD_K + k];
                    }
                    dot *= scale;
                    P[ni] = dot;
                    cur_max = max(cur_max, dot);
                }
                for (int ni = this_block_tokens; ni < BLK_N; ++ni) P[ni] = -1e33f;

                // Update running max
                float old_max = running_max;
                float new_max = max(old_max, cur_max);
                float scale_old = (old_max <= -1e29f) ? 0.0f : exp2f(old_max - new_max);
                running_max = new_max;

                // Rescale
                for (int c = 0; c < HD_V; ++c) o_accum[c] *= scale_old;
                running_sum *= scale_old;

                // Softmax
                float cur_sum = 0.0f;
                for (int ni = 0; ni < this_block_tokens; ++ni) {
                    float val = P[ni];
                    if (val <= -1e32f) { P[ni] = 0.0f; continue; }
                    val = exp2f(val - new_max);
                    P[ni] = val;
                    cur_sum += val;
                }
                for (int ni = this_block_tokens; ni < BLK_N; ++ni) P[ni] = 0.0f;
                running_sum += cur_sum;

                // PV
                for (int ni = 0; ni < this_block_tokens; ++ni) {
                    float pn = P[ni];
                    if (pn == 0.0f) continue;
                    for (int c = 0; c < HD_V; ++c) {
                        o_accum[c] += pn * (float)sK[ni * HD_K + c];
                    }
                }
            }
            __syncthreads();
        }  // end KV block loop

        // Write output
        if (valid) {
            float rcp = (running_sum == 0.0f || running_sum != running_sum) ? 1.0f : (1.0f / running_sum);
            float cur_lse = (running_sum == 0.0f || running_sum != running_sum) ? -1e30f : log2f(running_sum) + running_max;

            if (is_no_split) {
                InputT* my_o = o_ptr + g_row * params.o_row_stride + k_head_idx * params.o_head_stride;
                for (int c = 0; c < HD_V; ++c) {
                    my_o[c] = InputT(o_accum[c] * rcp);
                }
                // Store LSE in natural log (base e) as float32
                softmax_lse_ptr[my_row] = (running_sum == 0.0f || running_sum != running_sum) ? INFINITY : logf(running_sum) + running_max / (float)M_LOG2E;
            } else {
                // Split-KV: write to OAccum in float32, LSE in log2 as float32
                int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;
                float* oaccum_ptr = (float*)params.oaccum_ptr + ((split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLK_M) * HD_V;
                float* lseaccum_ptr = (float*)params.softmax_lseaccum_ptr + (split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLK_M;

                for (int c = 0; c < HD_V; ++c) {
                    oaccum_ptr[my_row * HD_V + c] = o_accum[c] * rcp;
                }
                lseaccum_ptr[my_row] = cur_lse;
            }

            // Store rL for cross-warp reduction (unused in 16-thread case but kept for consistency)
            sL_reduction[my_row] = running_sum;
        }
        if (batch_idx != sched_meta.end_req_idx) __syncthreads();
    }
}

// Include tensor-core kernel
#include "splitkv_mla_tc.cuh"

template<typename InputT>
void run_flash_splitkv_mla_kernel_cuda_core(DenseAttnDecodeParams &params) {
    constexpr size_t smem_size = 
        BLK_M * HD_K * sizeof(InputT) +      // sQ
        BLK_N * HD_K * sizeof(InputT) +      // sK
        BLK_M * sizeof(float);               // sL_reduction

    const int num_m_block = (params.q_seq_per_hk + BLK_M - 1) / BLK_M;
    auto mla_kernel = &flash_fwd_splitkv_mla_kernel<InputT>;
    CHECK_CUDA(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 grid(num_m_block, params.h_k, params.num_sm_parts);
    dim3 block(NUM_THREADS);
    mla_kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    // Use CUDA-core kernel (stable, all tests pass)
    // Tensor-core kernel is available in splitkv_mla_tc.cuh under tc:: namespace
    run_flash_splitkv_mla_kernel_cuda_core<InputT>(params);
}

}
