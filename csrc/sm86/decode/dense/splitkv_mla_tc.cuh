#pragma once
// Tensor-core SM86 dense MLA decoding kernel
// Template parameters: BLOCK_M, K_CHUNKS, NUM_THREADS
// Uses mma.sync m16n8k16 (bf16->f32) for QK^T and PV

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "utils.h"
#include "params.h"
#include "config.h"

namespace tc {

// --- MMA inline PTX helpers ------------------------------------

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ld_A_16x16(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%4];\n"
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%2, %3}, [%4 + 64];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ld_B_16x8(unsigned& b0, unsigned& b1, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1) : "r"(addr));
}

// --- Main kernel template -------------------------------------

template<typename InputT, int BLOCK_M, int K_CHUNKS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_tc_kernel(const DenseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int BLK_N = 64;
    constexpr int HD_K  = 576;
    constexpr int HD_V  = 512;
    constexpr int KCS   = HD_K / K_CHUNKS;  // chunk size
    constexpr int WARPS = NUM_THREADS / 32;
    constexpr int ROWS_PER_WARP = BLOCK_M / WARPS;
    static_assert(BLOCK_M % WARPS == 0, "BLOCK_M must be divisible by NUM_WARPS");
    static_assert(HD_K % K_CHUNKS == 0, "HD_K must be divisible by K_CHUNKS");
    static_assert(KCS % 16 == 0, "K chunk size must be multiple of 16");

    const int m_block_idx = blockIdx.x;
    const int k_head_idx  = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // -- Shared memory layout --
    // sQ:    BLOCK_M x HD_K x sizeof(InputT)   [bf16]
    // sK:    BLK_N   x KCS  x sizeof(InputT)   [bf16, one K chunk]
    // sP:    BLOCK_M x BLK_N x sizeof(InputT)   [bf16, softmax scores]
    // sM:    BLOCK_M x sizeof(float)
    // sScale: BLOCK_M x sizeof(float)
    // O_acc: BLOCK_M x HD_V x sizeof(float)     [overlaps sQ after QK^T]
    extern __shared__ char smem[];
    InputT* sQ    = (InputT*)smem;
    InputT* sK    = sQ + BLOCK_M * HD_K;
    InputT* sP    = sK + BLK_N * KCS;
    float*  sM    = (float*)(sP + BLOCK_M * BLK_N);
    float*  sScale = sM + BLOCK_M;
    float*  o_acc  = (float*)sQ;  // reuses sQ memory during PV phase

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : (seqlen_k + BLK_N - 1) / BLK_N;
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        InputT* q_ptr = (InputT*)params.q_ptr + batch_idx * params.q_batch_stride;
        InputT* o_ptr = (InputT*)params.o_ptr + batch_idx * params.o_batch_stride;
        float* softmax_lse_ptr = (float*)params.softmax_lse_ptr + (batch_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
        int* block_table_ptr = params.block_table + batch_idx * params.block_table_batch_stride;
        int num_valid = min(params.q_seq_per_hk - m_block_idx * BLOCK_M, BLOCK_M);

        // -- Load Q into shared --
        for (int i = tid; i < BLOCK_M * HD_K; i += NUM_THREADS) {
            int row = i / HD_K, col = i % HD_K;
            int g_row = m_block_idx * BLOCK_M + row;
            sQ[i] = q_ptr[g_row * params.q_row_stride + k_head_idx * params.q_head_stride + col];
        }
        __syncthreads();

        // Init state
        for (int i = tid; i < BLOCK_M; i += NUM_THREADS) { sM[i] = -1e30f; o_acc[i * HD_V + (tid < HD_V ? tid : 0)] = 0.0f; } // approximate init
        // Full init of O_acc
        for (int i = tid; i < BLOCK_M * HD_V; i += NUM_THREADS) o_acc[i] = 0.0f;
        __syncthreads();

        // Per-warp LSE accumulators (each warp handles ROWS_PER_WARP rows)
        float rL[ROWS_PER_WARP];
        for (int i = 0; i < ROWS_PER_WARP; ++i) rL[i] = 0.0f;
        int warp_row_start = warp_id * ROWS_PER_WARP;

        // -- Process KV blocks --
        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            int token_start = block_idx * BLK_N;
            int remain = seqlen_k - token_start;
            if (remain <= 0) break;
            int this_tokens = min(BLK_N, remain);
            int block_id = __ldg(block_table_ptr + block_idx);
            const InputT* gK = (InputT*)params.k_ptr + block_id * params.k_batch_stride + k_head_idx * params.k_head_stride;

            // Clear sP (softmax score accumulator)
            for (int i = tid; i < BLOCK_M * BLK_N; i += NUM_THREADS) sP[i] = InputT(0.0f);
            __syncthreads();

            // =======================================================
            // QK^T over K chunks
            // =======================================================
            for (int kc = 0; kc < K_CHUNKS; ++kc) {
                int k_start = kc * KCS;

                // Load K chunk via cp.async
                // K layout in global: (num_blocks, BLK_N, h_k, HD_K) with stride k_row_stride in elements
                // We need K[0:BLK_N, k_start:k_start+KCS]
                // Each row: KCS * 2 bytes, starting at gK + row * k_row_stride + k_start
                int k_bytes_per_row = KCS * sizeof(InputT);
                for (int row = tid; row < BLK_N; row += NUM_THREADS) {
                    const InputT* src = gK + row * params.k_row_stride + k_start;
                    InputT* dst = sK + row * KCS;
                    uint32_t dst_addr = __cvta_generic_to_shared(dst);
                    for (int b = 0; b < k_bytes_per_row; b += 16) {
                        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
                            :: "r"(dst_addr + b), "l"((const char*)src + b));
                    }
                }
                asm volatile("cp.async.commit_group;\n" ::);
                asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();

                // Zero-pad OOB tokens in this K chunk
                if (this_tokens < BLK_N) {
                    for (int row = this_tokens + tid; row < BLK_N; row += NUM_THREADS)
                        for (int col = 0; col < KCS; ++col)
                            sK[row * KCS + col] = InputT(0.0f);
                    __syncthreads();
                }

                // Each warp computes QK^T for its rows
                // Warp handles rows [warp_row_start, warp_row_start+ROWS_PER_WARP)
                // MMA: m16n8k16, M=ROWS_PER_WARP (16), N=BLK_N (64), K=KCS
                // Iterate: N-slices of 8, K-slices of 16
                int k_steps = KCS / 16;
                for (int k_step = 0; k_step < k_steps; ++k_step) {
                    int kk = k_step * 16;
                    // A tile: Q[warp_row_start:warp_row_start+16, k_start+kk:k_start+kk+16]
                    // If ROWS_PER_WARP < 16, we need to handle partial -- but ROWS_PER_WARP=16 always for m16
                    int q_byte_off = warp_row_start * HD_K * 2 + (k_start + kk) * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A_16x16(a0, a1, a2, a3, (char*)sQ + q_byte_off);

                    for (int ni = 0; ni < BLK_N; ni += 8) {
                        int k_byte_off = ni * KCS * 2 + kk * 2;
                        unsigned b0, b1;
                        ld_B_16x8(b0, b1, (char*)sK + k_byte_off);

                        // Load existing P values as float
                        int my_r0 = warp_row_start + (lane_id / 4);
                        int my_r1 = my_r0 + 8;
                        int my_c0 = ni + (lane_id % 4) * 2;
                        int my_c1 = my_c0 + 1;
                        float pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
                        if (my_r0 < BLOCK_M && my_c0 < BLK_N && my_r0 < warp_row_start + ROWS_PER_WARP) {
                            pc0 = (float)sP[my_r0 * BLK_N + my_c0];
                            if (my_c1 < BLK_N) pc1 = (float)sP[my_r0 * BLK_N + my_c1];
                            if (my_r1 < BLOCK_M && my_r1 < warp_row_start + ROWS_PER_WARP) {
                                pc2 = (float)sP[my_r1 * BLK_N + my_c0];
                                if (my_c1 < BLK_N) pc3 = (float)sP[my_r1 * BLK_N + my_c1];
                            }
                        }
                        mma_m16n8k16(pc0, pc1, pc2, pc3, a0, a1, a2, a3, b0, b1);
                        // Store back
                        if (my_r0 < BLOCK_M && my_c0 < BLK_N && my_r0 < warp_row_start + ROWS_PER_WARP) {
                            sP[my_r0 * BLK_N + my_c0] = InputT(pc0);
                            if (my_c1 < BLK_N) sP[my_r0 * BLK_N + my_c1] = InputT(pc1);
                            if (my_r1 < BLOCK_M && my_r1 < warp_row_start + ROWS_PER_WARP) {
                                sP[my_r1 * BLK_N + my_c0] = InputT(pc2);
                                if (my_c1 < BLK_N) sP[my_r1 * BLK_N + my_c1] = InputT(pc3);
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // =======================================================
            // Softmax (per-row, each warp handles its rows)
            // =======================================================
            float scale = params.scale_softmax_log2;
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = warp_row_start + li;
                if (gr >= num_valid) continue;

                // Apply scale and find max
                float cur_max = -1e33f;
                for (int ni = 0; ni < this_tokens; ++ni) {
                    float val = (float)sP[gr * BLK_N + ni] * scale;
                    if (token_start + ni >= seqlen_k) val = -1e33f;
                    sP[gr * BLK_N + ni] = InputT(val);
                    cur_max = max(cur_max, val);
                }
                for (int ni = this_tokens; ni < BLK_N; ++ni) sP[gr * BLK_N + ni] = InputT(-1e33f);

                // Warp reduce max
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

                float old_max = sM[gr];
                float new_max = max(old_max, cur_max);
                float scale_old = (old_max <= -1e29f) ? 0.0f : exp2f(old_max - new_max);

                if (lane_id == 0) { sM[gr] = new_max; sScale[gr] = scale_old; }
                __syncwarp();

                // Rescale O_accum for this row
                for (int c = lane_id; c < HD_V; c += 32) o_acc[gr * HD_V + c] *= scale_old;

                // Rescale LSE and compute exp
                rL[li] *= scale_old;
                float cur_sum = 0.0f;
                for (int ni = 0; ni < this_tokens; ++ni) {
                    float val = (float)sP[gr * BLK_N + ni];
                    if (val <= -1e32f) { sP[gr * BLK_N + ni] = InputT(0.0f); continue; }
                    val = exp2f(val - new_max);
                    sP[gr * BLK_N + ni] = InputT(val);
                    cur_sum += val;
                }
                for (int ni = this_tokens; ni < BLK_N; ++ni) sP[gr * BLK_N + ni] = InputT(0.0f);

                cur_sum += __shfl_xor_sync(0xffffffff, cur_sum, 1);
                cur_sum += __shfl_xor_sync(0xffffffff, cur_sum, 2);
                rL[li] += cur_sum;
            }

            // =======================================================
            // PV: O_accum += P @ V  (tensor core MMA)
            // P is (BLOCK_M x BLK_N) in sP
            // V is (HD_V x BLK_N) stored in sK layout (= first HD_V cols of transposed K)
            // For MMA: A=P(16x64), B=V slice(64x8), C=O_accum(16x8)
            // Iterate V columns in steps of 8, V rows(K-dim) in steps of 16
            // =======================================================
            for (int vn = 0; vn < HD_V; vn += 8) {
                float pv_c0 = 0, pv_c1 = 0, pv_c2 = 0, pv_c3 = 0;
                for (int vk = 0; vk < BLK_N; vk += 16) {
                    // A: P[warp_row_start:warp_row_start+16, vk:vk+16]
                    int p_byte_off = warp_row_start * BLK_N * 2 + vk * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A_16x16(a0, a1, a2, a3, (char*)sP + p_byte_off);

                    // B: V[vn:vn+8, vk:vk+16] -- first HD_V rows of K (row-major, (HD_K x BLK_N))
                    // As transposed: ldmatrix.trans from K[vn, vk] position
                    int v_byte_off = vn * HD_K * 2 + vk * 2;
                    unsigned b0, b1;
                    ld_B_16x8(b0, b1, (char*)sK + v_byte_off);

                    // Load existing O_accum for this V-column slice
                    int my_r0 = warp_row_start + (lane_id / 4);
                    int my_r1 = my_r0 + 8;
                    int my_c0 = vn + (lane_id % 4) * 2;
                    int my_c1 = my_c0 + 1;
                    if (my_r0 < BLOCK_M && my_c0 < HD_V && my_r0 < warp_row_start + ROWS_PER_WARP) {
                        pv_c0 = o_acc[my_r0 * HD_V + my_c0];
                        if (my_c1 < HD_V) pv_c1 = o_acc[my_r0 * HD_V + my_c1];
                        if (my_r1 < BLOCK_M && my_r1 < warp_row_start + ROWS_PER_WARP) {
                            pv_c2 = o_acc[my_r1 * HD_V + my_c0];
                            if (my_c1 < HD_V) pv_c3 = o_acc[my_r1 * HD_V + my_c1];
                        }
                    }
                    mma_m16n8k16(pv_c0, pv_c1, pv_c2, pv_c3, a0, a1, a2, a3, b0, b1);
                }
                // Store PV result back
                int my_r0 = warp_row_start + (lane_id / 4);
                int my_r1 = my_r0 + 8;
                int my_c0 = vn + (lane_id % 4) * 2;
                int my_c1 = my_c0 + 1;
                if (my_r0 < BLOCK_M && my_c0 < HD_V && my_r0 < warp_row_start + ROWS_PER_WARP) {
                    o_acc[my_r0 * HD_V + my_c0] = pv_c0;
                    if (my_c1 < HD_V) o_acc[my_r0 * HD_V + my_c1] = pv_c1;
                    if (my_r1 < BLOCK_M && my_r1 < warp_row_start + ROWS_PER_WARP) {
                        o_acc[my_r1 * HD_V + my_c0] = pv_c2;
                        if (my_c1 < HD_V) o_acc[my_r1 * HD_V + my_c1] = pv_c3;
                    }
                }
            }
            __syncthreads();
        }  // end KV block loop

        // -- Write output --
        for (int li = 0; li < ROWS_PER_WARP; ++li) {
            int gr = warp_row_start + li;
            if (gr >= num_valid) continue;
            float rcp = (rL[li] == 0.0f || rL[li] != rL[li]) ? 1.0f : (1.0f / rL[li]);
            float cur_lse_log2 = (rL[li] == 0.0f || rL[li] != rL[li]) ? -1e30f : log2f(rL[li]) + sM[gr];

            if (is_no_split) {
                InputT* my_o = o_ptr + (m_block_idx * BLOCK_M + gr) * params.o_row_stride + k_head_idx * params.o_head_stride;
                for (int c = lane_id; c < HD_V; c += 32)
                    my_o[c] = InputT(o_acc[gr * HD_V + c] * rcp);
                if (lane_id == 0)
                    softmax_lse_ptr[gr] = (rL[li] == 0.0f || rL[li] != rL[li]) ? INFINITY : logf(rL[li]) + sM[gr] / (float)M_LOG2E;
            } else {
                int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;
                float* oap = (float*)params.oaccum_ptr + ((split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M) * HD_V;
                float* lap = (float*)params.softmax_lseaccum_ptr + (split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
                for (int c = lane_id; c < HD_V; c += 32)
                    oap[gr * HD_V + c] = o_acc[gr * HD_V + c] * rcp;
                if (lane_id == 0) lap[gr] = cur_lse_log2;
            }
        }
        if (batch_idx != sched_meta.end_req_idx) __syncthreads();
    }
#else
    asm("trap;");
#endif
}

// --- Host launch dispatcher ------------------------------------

template<typename InputT, int BLOCK_M, int K_CHUNKS, int NUM_THREADS>
void launch_tc_kernel(DenseAttnDecodeParams &params) {
    constexpr int BLK_N = 64;
    constexpr int HD_K  = 576;
    constexpr int HD_V  = 512;
    constexpr int KCS   = HD_K / K_CHUNKS;
    constexpr size_t smem_size = BLOCK_M * HD_K * sizeof(InputT) + BLK_N * KCS * sizeof(InputT) + BLOCK_M * BLK_N * sizeof(InputT) + 3 * BLOCK_M * sizeof(float);

    if ((size_t)params.d != HD_K || (size_t)params.d_v != HD_V) {
        FLASH_ASSERT(false && "HEAD_DIM mismatch");
        return;
    }

    const int num_m_block = (params.q_seq_per_hk + BLOCK_M - 1) / BLOCK_M;
    auto kernel = &flash_fwd_splitkv_mla_tc_kernel<InputT, BLOCK_M, K_CHUNKS, NUM_THREADS>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 grid(num_m_block, params.h_k, params.num_sm_parts);
    dim3 block(NUM_THREADS);
    kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

// Forward declaration of CUDA-core kernel host function
template<typename InputT>
void run_flash_splitkv_mla_kernel_cuda_core(DenseAttnDecodeParams &params);

// Dispatch table: select BLOCK_M and K_CHUNKS based on architecture
template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    // FALLBACK to working CUDA-core kernel until TC kernel is debugged
    run_flash_splitkv_mla_kernel_cuda_core<InputT>(params);
}

}  // namespace tc
