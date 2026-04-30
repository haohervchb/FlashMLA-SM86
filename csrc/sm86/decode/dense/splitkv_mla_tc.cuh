#pragma once
// Tensor-core SM86 dense MLA decoding kernel
// mma.sync m16n8k16, 32 threads, full K in shared memory (94 KB)
// Compile with -DFLASH_MLA_USE_TENSOR_CORE to enable

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "utils.h"
#include "params.h"
#include "config.h"

namespace sm86 {
namespace tc {

// --- MMA inline PTX helpers ---

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

__device__ __forceinline__ void ld_A(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%4];\n"
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%2, %3}, [%4 + 64];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ld_B(unsigned& b0, unsigned& b1, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1) : "r"(addr));
}

// --- Main tensor-core kernel ---

template<typename InputT, int BLOCK_M, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_tc_kernel(const DenseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int BLK_N = 64, HD_K = 576, HD_V = 512;
    constexpr int WARPS = NUM_THREADS / 32;
    constexpr int ROWS_PER_WARP = BLOCK_M / WARPS;
    static_assert(ROWS_PER_WARP == 16, "Must have 16 rows per warp for MMA");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be multiple of 16");
    // WARPS can be 1,2,4 — each warp handles different N-slices (columns) in parallel

    const int m_block_idx = blockIdx.x, k_head_idx = blockIdx.y, partition_idx = blockIdx.z;
    const int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;

    extern __shared__ char smem[];
    InputT* sQ = (InputT*)smem;
    InputT* sK = sQ + BLOCK_M * HD_K;
    InputT* sP = sK + BLK_N * HD_K;
    float*  sM = (float*)(sP + BLOCK_M * BLK_N);
    float*  sScale = sM + BLOCK_M;

    DecodingSchedMeta smeta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (smeta.begin_req_idx >= params.b) return;

    for (int bi = smeta.begin_req_idx; bi <= smeta.end_req_idx; ++bi) {
        const int n_split = bi == smeta.begin_req_idx ? smeta.begin_split_idx : 0;
        int sk_len = __ldg(params.seqlens_k_ptr + bi);
        const int sblk = bi == smeta.begin_req_idx ? smeta.begin_block_idx : 0;
        int   eblk = bi == smeta.end_req_idx ? smeta.end_block_idx : (sk_len + BLK_N - 1) / BLK_N;
        const bool nosplit = bi == smeta.begin_req_idx ? !smeta.is_first_req_splitted : (bi == smeta.end_req_idx ? !smeta.is_last_req_splitted : true);
        int nv = min(params.q_seq_per_hk - m_block_idx * BLOCK_M, BLOCK_M);

        InputT* qp = (InputT*)params.q_ptr + bi * params.q_batch_stride;
        InputT* op = (InputT*)params.o_ptr + bi * params.o_batch_stride;
        float* slp = (float*)params.softmax_lse_ptr + (bi * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
        int* btp = params.block_table + bi * params.block_table_batch_stride;

        for (int i = tid; i < BLOCK_M * HD_K; i += NUM_THREADS) {
            int r = i / HD_K, c = i % HD_K;
            sQ[i] = qp[(m_block_idx * BLOCK_M + r) * params.q_row_stride + k_head_idx * params.q_head_stride + c];
        }
        __syncthreads();

        for (int i = tid; i < BLOCK_M; i += NUM_THREADS) sM[i] = -1e30f;
        __syncthreads();

        int wr = warp_id * ROWS_PER_WARP;
        float rL[ROWS_PER_WARP] = {};

        for (int blk = sblk; blk < eblk; ++blk) {
            int ts = blk * BLK_N, rem = sk_len - ts;
            if (rem <= 0) break;
            int nt = min(BLK_N, rem);
            int bid = __ldg(btp + blk);
            const InputT* gK = (InputT*)params.k_ptr + bid * params.k_batch_stride + k_head_idx * params.k_head_stride;

            // Load full K
            int b = BLK_N * HD_K * (int)sizeof(InputT);
            for (int i = tid; i < b / 16; i += NUM_THREADS) {
                uint32_t d = __cvta_generic_to_shared((char*)sK + i * 16);
                asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(d), "l"((const char*)gK + i * 16));
            }
            asm("cp.async.commit_group;\n" ::);
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            if (nt < BLK_N)
                for (int i = tid + nt * HD_K; i < BLK_N * HD_K; i += NUM_THREADS) sK[i] = InputT(0.0f);
            __syncthreads();

            for (int i = tid; i < BLOCK_M * BLK_N; i += NUM_THREADS) sP[i] = InputT(0.0f);
            __syncthreads();

            // QK^T via MMA
            for (int kk = 0; kk < HD_K; kk += 16) {
                int qo = wr * HD_K * 2 + kk * 2;
                unsigned a0, a1, a2, a3;
                ld_A(a0, a1, a2, a3, (char*)sQ + qo);
                for (int ni = warp_id * 8; ni < BLK_N; ni += WARPS * 8) {
                    int ko = ni * HD_K * 2 + kk * 2;
                    unsigned b0, b1;
                    ld_B(b0, b1, (char*)sK + ko);
                    int r0 = wr + lane_id / 4, r1 = r0 + 8;
                    int c0 = ni + (lane_id % 4) * 2, c1 = c0 + 1;
                    float pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
                    if (r0 < BLOCK_M && c0 < BLK_N) {
                        pc0 = (float)sP[r0 * BLK_N + c0];
                        if (c1 < BLK_N) pc1 = (float)sP[r0 * BLK_N + c1];
                        if (r1 < BLOCK_M) { pc2 = (float)sP[r1 * BLK_N + c0]; if (c1 < BLK_N) pc3 = (float)sP[r1 * BLK_N + c1]; }
                    }
                    mma_m16n8k16(pc0, pc1, pc2, pc3, a0, a1, a2, a3, b0, b1);
                    if (r0 < BLOCK_M && c0 < BLK_N) {
                        sP[r0 * BLK_N + c0] = InputT(pc0); if (c1 < BLK_N) sP[r0 * BLK_N + c1] = InputT(pc1);
                        if (r1 < BLOCK_M) { sP[r1 * BLK_N + c0] = InputT(pc2); if (c1 < BLK_N) sP[r1 * BLK_N + c1] = InputT(pc3); }
                    }
                }
            }
            __syncthreads();

            // Softmax
            float sm_scale = params.scale_softmax_log2;
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = wr + li;
                if (gr >= nv) continue;
                float cm = -1e33f;
                for (int ni = 0; ni < nt; ++ni) {
                    float v = (float)sP[gr * BLK_N + ni] * sm_scale;
                    if (ts + ni >= sk_len) v = -1e33f;
                    sP[gr * BLK_N + ni] = InputT(v);
                    cm = max(cm, v);
                }
                for (int ni = nt; ni < BLK_N; ++ni) sP[gr * BLK_N + ni] = InputT(-1e33f);
                cm = max(cm, __shfl_xor_sync(0xffffffff, cm, 1));
                cm = max(cm, __shfl_xor_sync(0xffffffff, cm, 2));
                float om = sM[gr], nm = max(om, cm);
                float so = (om <= -1e29f) ? 0.0f : exp2f(om - nm);
                if (lane_id == 0) { sM[gr] = nm; sScale[gr] = so; }
                rL[li] *= so;
                float cs = 0.0f;
                for (int ni = 0; ni < nt; ++ni) {
                    float v = (float)sP[gr * BLK_N + ni];
                    if (v <= -1e32f) { sP[gr * BLK_N + ni] = InputT(0.0f); continue; }
                    v = exp2f(v - nm); sP[gr * BLK_N + ni] = InputT(v); cs += v;
                }
                for (int ni = nt; ni < BLK_N; ++ni) sP[gr * BLK_N + ni] = InputT(0.0f);
                cs += __shfl_xor_sync(0xffffffff, cs, 1);
                cs += __shfl_xor_sync(0xffffffff, cs, 2);
                rL[li] += cs;
            }

            // PV via MMA
            float* oa = (float*)sK;
            for (int i = tid; i < BLOCK_M * HD_V; i += NUM_THREADS) oa[i] = 0.0f;
            __syncthreads();
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = wr + li; if (gr >= nv) continue;
                for (int c = lane_id; c < HD_V; c += 32) oa[gr * HD_V + c] *= sScale[gr];
            }
            for (int vn = warp_id * 8; vn < HD_V; vn += WARPS * 8) {
                float pv0 = 0, pv1 = 0, pv2 = 0, pv3 = 0;
                for (int vk = 0; vk < BLK_N; vk += 16) {
                    int po = wr * BLK_N * 2 + vk * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A(a0, a1, a2, a3, (char*)sP + po);
                    int vo = vn * HD_K * 2 + vk * 2;
                    unsigned b0, b1;
                    ld_B(b0, b1, (char*)sK + vo);
                    int r0 = wr + lane_id / 4, r1 = r0 + 8;
                    int c0 = vn + (lane_id % 4) * 2, c1 = c0 + 1;
                    if (r0 < BLOCK_M && c0 < HD_V) {
                        pv0 = oa[r0 * HD_V + c0]; if (c1 < HD_V) pv1 = oa[r0 * HD_V + c1];
                        if (r1 < BLOCK_M) { pv2 = oa[r1 * HD_V + c0]; if (c1 < HD_V) pv3 = oa[r1 * HD_V + c1]; }
                    }
                    mma_m16n8k16(pv0, pv1, pv2, pv3, a0, a1, a2, a3, b0, b1);
                }
                int r0 = wr + lane_id / 4, r1 = r0 + 8;
                int c0 = vn + (lane_id % 4) * 2, c1 = c0 + 1;
                if (r0 < BLOCK_M && c0 < HD_V) {
                    oa[r0 * HD_V + c0] = pv0; if (c1 < HD_V) oa[r0 * HD_V + c1] = pv1;
                    if (r1 < BLOCK_M) { oa[r1 * HD_V + c0] = pv2; if (c1 < HD_V) oa[r1 * HD_V + c1] = pv3; }
                }
            }
            __syncthreads();
        }

        // Write output
        float* oa = (float*)sK;
        for (int li = 0; li < ROWS_PER_WARP; ++li) {
            int gr = wr + li; if (gr >= nv) continue;
            float rcp = (rL[li] == 0.0f || rL[li] != rL[li]) ? 1.0f : (1.0f / rL[li]);
            float ll2 = (rL[li] == 0.0f || rL[li] != rL[li]) ? -1e30f : log2f(rL[li]) + sM[gr];
            if (nosplit) {
                InputT* mo = op + (m_block_idx * BLOCK_M + gr) * params.o_row_stride + k_head_idx * params.o_head_stride;
                for (int c = lane_id; c < HD_V; c += 32) mo[c] = InputT(oa[gr * HD_V + c] * rcp);
                if (lane_id == 0) slp[gr] = (rL[li] == 0.0f || rL[li] != rL[li]) ? INFINITY : logf(rL[li]) + sM[gr] / (float)M_LOG2E;
            } else {
                int sid = params.num_splits_ptr[bi] + n_split;
                float* oap = (float*)params.oaccum_ptr + ((sid * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M) * HD_V;
                float* lap = (float*)params.softmax_lseaccum_ptr + (sid * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
                for (int c = lane_id; c < HD_V; c += 32) oap[gr * HD_V + c] = oa[gr * HD_V + c] * rcp;
                if (lane_id == 0) lap[gr] = ll2;
            }
        }
        if (bi != smeta.end_req_idx) __syncthreads();
    }
#else
    asm("trap;");
#endif
}

// --- Host launch ---

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    constexpr int BLOCK_M = 16, NUM_T = 128, BLK_N = 64, HD_K = 576;
    constexpr size_t sm = BLOCK_M * HD_K * sizeof(InputT) + BLK_N * HD_K * sizeof(InputT) + BLOCK_M * BLK_N * sizeof(InputT) + 2 * BLOCK_M * sizeof(float);

    const int nmb = (params.q_seq_per_hk + BLOCK_M - 1) / BLOCK_M;
    auto k = &flash_fwd_splitkv_mla_tc_kernel<InputT, BLOCK_M, NUM_T>;
    CHECK_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    dim3 g(nmb, params.h_k, params.num_sm_parts), b(NUM_T);
    k<<<g, b, sm, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace tc
}  // namespace sm86
