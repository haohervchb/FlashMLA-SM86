#pragma once
// Tensor-core SM86 dense MLA decoding kernel
// Tile-pipelined: double-buffered 64x64 K-tiles, overlapping cp.async with MMA
// BLOCK_M=16, 128 threads (4 warps), 94 KB shared memory

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "utils.h"
#include "params.h"
#include "config.h"

namespace sm86 {
namespace tc {

// --- MMA inline PTX ---
__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void ld_A(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%4];\n"
                 "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%2,%3},[%4+64];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
}
__device__ __forceinline__ void ld_B(unsigned& b0, unsigned& b1, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n"
        : "=r"(b0),"=r"(b1) : "r"(addr));
}

// --- Tile-pipelined kernel ---
template<typename InputT, int BLOCK_M, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_tc_kernel(const DenseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int BLK_N = 64, HD_K = 576, HD_V = 512;
    constexpr int WARPS = NUM_THREADS / 32;
    constexpr int K_TILES = HD_K / 64;  // 9 tiles of 64x64
    constexpr int V_TILES = HD_V / 64;  // 8 tiles of 64x64
    static_assert(HD_K % 64 == 0 && HD_V % 64 == 0, "dims must be divisible by 64");
    static_assert(ROWS_PER_WARP == 16, "1 warp handles all 16 rows");

    const int m_block_idx = blockIdx.x, k_head_idx = blockIdx.y, partition_idx = blockIdx.z;
    const int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;
    constexpr int ROWS_PER_WARP = BLOCK_M / WARPS;

    // Shared memory: sQ(18KB) + sK0(8KB) + sK1(8KB) + sP(2KB) + sM+sScale(128B) = 36KB
    extern __shared__ char smem[];
    InputT* sQ     = (InputT*)smem;
    InputT* sK0    = sQ + BLOCK_M * HD_K;
    InputT* sK1    = sK0 + BLK_N * 64;
    InputT* sP     = sK1 + BLK_N * 64;
    float*  sM     = (float*)(sP + BLOCK_M * BLK_N);
    float*  sScale = sM + BLOCK_M;

    DecodingSchedMeta smeta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (smeta.begin_req_idx >= params.b) return;

    for (int bi = smeta.begin_req_idx; bi <= smeta.end_req_idx; ++bi) {
        const int n_split = bi == smeta.begin_req_idx ? smeta.begin_split_idx : 0;
        int sk_len = __ldg(params.seqlens_k_ptr + bi);
        const int sblk = bi == smeta.begin_req_idx ? smeta.begin_block_idx : 0;
        int eblk = bi == smeta.end_req_idx ? smeta.end_block_idx : (sk_len + BLK_N - 1) / BLK_N;
        const bool nosplit = bi == smeta.begin_req_idx ? !smeta.is_first_req_splitted : (bi == smeta.end_req_idx ? !smeta.is_last_req_splitted : true);
        int nv = min(params.q_seq_per_hk - m_block_idx * BLOCK_M, BLOCK_M);

        InputT* qp = (InputT*)params.q_ptr + bi * params.q_batch_stride;
        InputT* op = (InputT*)params.o_ptr + bi * params.o_batch_stride;
        float* slp = (float*)params.softmax_lse_ptr + (bi * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
        int* btp = params.block_table + bi * params.block_table_batch_stride;

        // Load Q once (persists across all KV blocks)
        for (int i = tid; i < BLOCK_M * HD_K; i += NUM_THREADS) {
            int r = i / HD_K, c = i % HD_K;
            sQ[i] = qp[(m_block_idx * BLOCK_M + r) * params.q_row_stride + k_head_idx * params.q_head_stride + c];
        }
        __syncthreads();

        // Init state once
        for (int i = tid; i < BLOCK_M; i += NUM_THREADS) sM[i] = -1e30f;
        __syncthreads();

        // --- Process KV blocks ---
        for (int blk = sblk; blk < eblk; ++blk) {
            int ts = blk * BLK_N, rem = sk_len - ts;
            if (rem <= 0) break;
            int nt = min(BLK_N, rem);
            int bid = __ldg(btp + blk);
            const InputT* gK = (InputT*)params.k_ptr + bid * params.k_batch_stride + k_head_idx * params.k_head_stride;

            // Clear sP
            for (int i = tid; i < BLOCK_M * BLK_N; i += NUM_THREADS) sP[i] = InputT(0.0f);
            __syncthreads();

            // ================================================================
            // QK^T: tile-pipelined — 9 K-tiles of 64x64, double-buffered
            // ================================================================
            // Pre-load tile 0 into sK0
            if (tid < NUM_THREADS) {
                int lines_per_tile = (64 * 64 * 2) / 16; // 512 lines per tile
                for (int i = tid; i < lines_per_tile; i += NUM_THREADS) {
                    int row = i / 32, col_base = (i % 32) * 8; // 32 lines per row, 8 bf16 per line
                    int src_off = row * params.k_row_stride + col_base;
                    uint32_t dst = __cvta_generic_to_shared((char*)sK0 + i * 16);
                    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"((const char*)gK + src_off * 2));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            int cur_buf = 0;  // 0 = sK0, 1 = sK1

            for (int kt = 0; kt < K_TILES; ++kt) {
                int k_col = kt * 64;  // column offset in K matrix
                InputT* cur_sK = (cur_buf == 0) ? sK0 : sK1;
                int next_buf = 1 - cur_buf;

                // Start loading NEXT tile while computing current
                if (kt + 1 < K_TILES) {
                    int next_kc = (kt + 1) * 64;
                    InputT* next_sK = (next_buf == 0) ? sK0 : sK1;
                    if (tid < NUM_THREADS) {
                        int lines = (64 * 64 * 2) / 16;
                        for (int i = tid; i < lines; i += NUM_THREADS) {
                            int row = i / 32, col_base = next_kc + (i % 32) * 8;
                            int src_off = row * params.k_row_stride + col_base;
                            uint32_t dst = __cvta_generic_to_shared((char*)next_sK + i * 16);
                            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"((const char*)gK + src_off * 2));
                        }
                    }
                    asm volatile("cp.async.commit_group;\n" ::);
                }

                // MMA: Q @ K_tile
                int k_steps = 64 / 16;  // 4 k16 steps per K tile
                for (int ks = 0; ks < k_steps; ++ks) {
                    int kk = ks * 16;
                    int q_off = warp_id * ROWS_PER_WARP * HD_K * 2 + (k_col + kk) * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A(a0, a1, a2, a3, (char*)sQ + q_off);

                    for (int ni = warp_id * 8; ni < BLK_N; ni += WARPS * 8) {
                        int k_off = ni * 64 * 2 + kk * 2;  // K tile is 64x64, row-major
                        unsigned b0, b1;
                        ld_B(b0, b1, (char*)cur_sK + k_off);

                        int r0 = warp_id * ROWS_PER_WARP + lane_id / 4, r1 = r0 + 8;
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

                // Wait for next tile
                if (kt + 1 < K_TILES) {
                    asm volatile("cp.async.wait_group 0;\n" ::);
                    __syncthreads();
                    cur_buf = next_buf;
                }
            }

            // Zero OOB tokens in P after all tiles (performed by softmax)

            // ================================================================
            // Softmax (unchanged from sequential version)
            // ================================================================
            float sm_scale = params.scale_softmax_log2;
            float rL[ROWS_PER_WARP] = {};
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = warp_id * ROWS_PER_WARP + li;
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

            // ================================================================
            // PV: Reload V tiles and compute O (O_acc in sK0+sK1 space)
            // After softmax, sK0 and sK1 are free. Use them for O_acc.
            // ================================================================
            float* oa0 = (float*)sK0;  // rows 0-7 (8*512*4=16384 bytes in 8192? NO)
            float* oa1 = (float*)sK1;  // rows 8-15

            // O_acc needs 32KB. sK0+sK1=16KB. Only fits half.
            // Use sQ space too: sQ is 18KB, can hold rows 0-8.
            float* oa = (float*)sQ;  // rows 0-8: 9*512*4=18432 bytes (sQ has 18432) PERFECT
            float* oa9 = (float*)sK0; // rows 9-12: 4*512*4=8192 (sK0 has 8192) PERFECT
            float* oa13= (float*)sK1; // rows 13-15: 3*512*4=6144 (sK1 has 8192) FITS

            // Clear O_acc
            for (int i = tid; i < 9 * HD_V + 4 * HD_V + 3 * HD_V; i += NUM_THREADS) {
                // 9 rows in sQ: 0 to 9*512-1
                if (i < 9 * HD_V) oa[i] = 0.0f;
                else if (i < 13 * HD_V) oa9[(i - 9 * HD_V)] = 0.0f;
            }
            __syncthreads();

            // Rescale existing O from previous KV blocks
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = warp_id * ROWS_PER_WARP + li; if (gr >= nv) continue;
                float sf = sScale[gr];
                for (int c = lane_id; c < HD_V; c += 32) {
                    float* ptr = (gr < 9) ? (oa + gr * HD_V) : (gr < 13) ? (oa9 + (gr - 9) * HD_V) : (oa13 + (gr - 13) * HD_V);
                    ptr[c] *= sf;
                }
            }

            // PV MMA: reload V tiles, same pattern as QK^T
            // Pre-load V tile 0
            if (tid < NUM_THREADS) {
                int vbytes = 64 * 64 * 2;
                for (int i = tid; i < vbytes / 16; i += NUM_THREADS) {
                    int row = (i * 16 / 2) / 64, col = (i * 16 / 2) % 64; // col 0-63 of V
                    int src_off = row * params.k_row_stride + col;
                    uint32_t dst = __cvta_generic_to_shared((char*)sK0 + i * 16);
                    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"((const char*)gK + src_off * 2));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            cur_buf = 0;

            for (int vt = 0; vt < V_TILES; ++vt) {
                int v_col = vt * 64;
                InputT* cur_V = (cur_buf == 0) ? sK0 : sK1;
                int next_buf = 1 - cur_buf;

                // Start loading NEXT V tile
                if (vt + 1 < V_TILES && tid < NUM_THREADS) {
                    int next_vc = (vt + 1) * 64;
                    InputT* next_V = (next_buf == 0) ? sK0 : sK1;
                    int vbytes = 64 * 64 * 2;
                    for (int i = tid; i < vbytes / 16; i += NUM_THREADS) {
                        int row = (i * 16 / 2) / 64, col = next_vc + (i * 16 / 2) % 64;
                        int src_off = row * params.k_row_stride + col;
                        uint32_t dst = __cvta_generic_to_shared((char*)next_V + i * 16);
                        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"((const char*)gK + src_off * 2));
                    }
                    asm volatile("cp.async.commit_group;\n" ::);
                }

                // MMA: P @ V_tile
                int v_steps = 64 / 16;  // 4
                for (int vs = 0; vs < v_steps; ++vs) {
                    int vk = vs * 16;
                    int p_off = warp_id * ROWS_PER_WARP * BLK_N * 2 + vk * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A(a0, a1, a2, a3, (char*)sP + p_off);

                    for (int vn = warp_id * 8; vn < 64; vn += WARPS * 8) {
                        int v_off = vn * 64 * 2 + vk * 2;
                        unsigned b0, b1;
                        ld_B(b0, b1, (char*)cur_V + v_off);

                        int r0 = warp_id * ROWS_PER_WARP + lane_id / 4, r1 = r0 + 8;
                        int c0 = v_col + vn + (lane_id % 4) * 2, c1 = c0 + 1;
                        float pv0 = 0, pv1 = 0, pv2 = 0, pv3 = 0;
                        if (r0 < BLOCK_M && c0 < HD_V) {
                            float* ptr0 = (r0 < 9) ? (oa + r0 * HD_V) : (r0 < 13) ? (oa9 + (r0 - 9) * HD_V) : (oa13 + (r0 - 13) * HD_V);
                            pv0 = ptr0[c0]; if (c1 < HD_V) pv1 = ptr0[c1];
                            if (r1 < BLOCK_M) {
                                float* ptr1 = (r1 < 9) ? (oa + r1 * HD_V) : (r1 < 13) ? (oa9 + (r1 - 9) * HD_V) : (oa13 + (r1 - 13) * HD_V);
                                pv2 = ptr1[c0]; if (c1 < HD_V) pv3 = ptr1[c1];
                            }
                        }
                        mma_m16n8k16(pv0, pv1, pv2, pv3, a0, a1, a2, a3, b0, b1);
                        if (r0 < BLOCK_M && c0 < HD_V) {
                            float* ptr0 = (r0 < 9) ? (oa + r0 * HD_V) : (r0 < 13) ? (oa9 + (r0 - 9) * HD_V) : (oa13 + (r0 - 13) * HD_V);
                            ptr0[c0] = pv0; if (c1 < HD_V) ptr0[c1] = pv1;
                            if (r1 < BLOCK_M) {
                                float* ptr1 = (r1 < 9) ? (oa + r1 * HD_V) : (r1 < 13) ? (oa9 + (r1 - 9) * HD_V) : (oa13 + (r1 - 13) * HD_V);
                                ptr1[c0] = pv2; if (c1 < HD_V) ptr1[c1] = pv3;
                            }
                        }
                    }
                }

                if (vt + 1 < V_TILES) {
                    asm volatile("cp.async.wait_group 0;\n" ::);
                    __syncthreads();
                    cur_buf = next_buf;
                }
            }
            __syncthreads();

            // ================================================================
            // Write output
            // ================================================================
            for (int li = 0; li < ROWS_PER_WARP; ++li) {
                int gr = warp_id * ROWS_PER_WARP + li;
                if (gr >= nv) continue;
                float rcp = (rL[li] == 0.0f || rL[li] != rL[li]) ? 1.0f : (1.0f / rL[li]);
                float ll2 = (rL[li] == 0.0f || rL[li] != rL[li]) ? -1e30f : log2f(rL[li]) + sM[gr];
                float* ptr = (gr < 9) ? (oa + gr * HD_V) : (gr < 13) ? (oa9 + (gr - 9) * HD_V) : (oa13 + (gr - 13) * HD_V);

                if (nosplit) {
                    InputT* mo = op + (m_block_idx * BLOCK_M + gr) * params.o_row_stride + k_head_idx * params.o_head_stride;
                    for (int c = lane_id; c < HD_V; c += 32) mo[c] = InputT(ptr[c] * rcp);
                    if (lane_id == 0) slp[gr] = (rL[li] == 0.0f || rL[li] != rL[li]) ? INFINITY : logf(rL[li]) + sM[gr] / (float)M_LOG2E;
                } else {
                    int sid = params.num_splits_ptr[bi] + n_split;
                    float* oap = (float*)params.oaccum_ptr + ((sid * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M) * HD_V;
                    float* lap = (float*)params.softmax_lseaccum_ptr + (sid * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * BLOCK_M;
                    for (int c = lane_id; c < HD_V; c += 32) oap[gr * HD_V + c] = ptr[c] * rcp;
                    if (lane_id == 0) lap[gr] = ll2;
                }
            }
            __syncthreads();
        }  // KV block loop
    }  // batch loop
#else
    asm("trap;");
#endif
}

// --- Host launch ---
template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    constexpr int BLOCK_M = 16, NUM_T = 128, BLK_N = 64, HD_K = 576;
    constexpr size_t sm = BLOCK_M * HD_K * 2 + 2 * BLK_N * 64 * 2 + BLOCK_M * BLK_N * 2 + 2 * BLOCK_M * 4;
    const int nmb = (params.q_seq_per_hk + BLOCK_M - 1) / BLOCK_M;
    auto k = &flash_fwd_splitkv_mla_tc_kernel<InputT, BLOCK_M, NUM_T>;
    CHECK_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    dim3 g(nmb, params.h_k, params.num_sm_parts), b(NUM_T);
    k<<<g, b, sm, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace tc
}  // namespace sm86
