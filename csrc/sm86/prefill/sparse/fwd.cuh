#pragma once

#include "config.h"
#include "utils.h"

namespace sm86 {
namespace prefill {

// SM86-compatible shared memory address conversion
__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }\n" : "=r"(r) : "l"(p));
    return r;
}

// --- MMA inline PTX helpers ---
__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void ld_A(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const void* smem) {
    uint32_t addr = smem_ptr_to_uint(smem);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%4];\n"
                 "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%2,%3},[%4+64];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ld_B(unsigned& b0, unsigned& b1, const void* smem) {
    uint32_t addr = smem_ptr_to_uint(smem);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n"
        : "=r"(b0),"=r"(b1) : "r"(addr));
}

// --- Main kernel ---
template<int D_QK, bool HAVE_TOPK_LENGTH>
__global__ void __launch_bounds__(KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::NUM_THREADS)
sparse_attn_fwd_kernel(const SparseAttnFwdParams params) {
#if __CUDA_ARCH__ >= 800
    using KT = KernelTemplate<D_QK, HAVE_TOPK_LENGTH>;
    constexpr int B_H   = KT::B_H;
    constexpr int B_TK  = KT::B_TOPK;
    constexpr int D_V   = KT::D_V;
    constexpr int NT    = KT::NUM_THREADS;
    constexpr int WARPS = NT / 32;
    constexpr int K_TILES = D_QK / 64;          // 9 for 576, 8 for 512
    constexpr int V_HALF_TILES = (D_V/2) / 64;  // 4 tiles per half-V

    const int q_h_idx  = blockIdx.x % (params.h_q / B_H);
    const int s_q_idx  = blockIdx.x / (params.h_q / B_H);
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;

    extern __shared__ char smem[];
    bf16*  sQ   = (bf16*)smem;
    bf16*  sK0  = sQ + B_H * D_QK;
    bf16*  sK1  = sK0 + B_TK * 64;
    bf16*  sS   = sK1 + B_TK * 64;
    float* sM    = (float*)(sS + B_H * B_TK);

    // --- Load Q tile ---
    int g_q_row_start = q_h_idx * B_H;
    for (int i = tid; i < B_H * D_QK; i += NT) {
        int r = i / D_QK, c = i % D_QK;
        int g_row = g_q_row_start + r;
        sQ[i] = params.q[s_q_idx * params.stride_q_s_q + g_row * params.stride_q_h_q + c];
    }
    __syncthreads();

    // --- Init state ---
    for (int i = tid; i < B_H; i += NT) sM[i] = KT::MAX_INIT_VAL;
    // Per-warp L sum: each thread handles 2 rows (m16n8k16 MMA)
    float rL[2] = {0.0f, 0.0f};
    float sL_reduce[B_H];  // reduce buffer
    __syncthreads();

    const int topk_len = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_tk_blocks = (topk_len + B_TK - 1) / B_TK;

    // O accumulator: reuse sQ space after QK^T
    float* o_acc = (float*)sQ;
    for (int i = tid; i < B_H * D_V; i += NT) o_acc[i] = 0.0f;
    // Per-row running L sums (will be reduced from rL across threads)
    float l_sums[B_H] = {};
    __syncthreads();

    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;

    // --- Process topk blocks ---
    for (int tb = 0; tb < num_tk_blocks; ++tb) {
        int tk_start = tb * B_TK;
        int this_tk = min(B_TK, topk_len - tk_start);

        // Clear sS (softmax scores for this topk block)
        for (int i = tid; i < B_H * B_TK; i += NT) sS[i] = bf16(0.0f);
        __syncthreads();

        // --- QK^T: iterate K tiles ---
        for (int kt = 0; kt < K_TILES; ++kt) {
            int k_col = kt * 64;

            // Load K tile[kt] via cp.async into sK0
            // K tile shape: B_TK x 64, from global KV at positions given by indices[tk_start:tk_start+this_tk]
            int k_bytes = B_TK * 64 * sizeof(bf16);
            for (int i = tid; i < k_bytes / 16; i += NT) {
                int row = (i * 16 / 2) / 64;   // token row within B_TK
                int col = k_col + ((i * 16 / 2) % 64);  // column within K
                int tok_idx = tk_start + row;
                if (row < this_tk && tok_idx < params.topk) {
                    int g_tok = __ldg(gIndices + tok_idx);
                    if (g_tok >= 0 && g_tok < params.s_kv) {
                        const bf16* src = params.kv + g_tok * params.stride_kv_s_kv + col;
                        uint32_t dst = smem_ptr_to_uint((char*)sK0 + i * 16);
                        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
                    } else {
                        // invalid token: zero-fill
                        uint32_t dst = smem_ptr_to_uint((char*)sK0 + i * 16);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};\n" :: "r"(dst), "r"(0));
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();

            // Zero OOB rows
            if (this_tk < B_TK)
                for (int i = tid + this_tk * 64; i < B_TK * 64; i += NT)
                    sK0[i] = bf16(0.0f);
            __syncthreads();

            // MMA: Q @ K_tile[kt]
            int k_steps = 64 / 16;  // 4 k16 steps
            for (int ks = 0; ks < k_steps; ++ks) {
                int kk = ks * 16;
                int q_off = warp_id * (B_H/WARPS) * D_QK * 2 + (k_col + kk) * 2;
                unsigned a0, a1, a2, a3;
                ld_A(a0, a1, a2, a3, (char*)sQ + q_off);

                for (int ni = warp_id * 8; ni < B_TK; ni += WARPS * 8) {
                    int k_off = ni * 64 * 2 + kk * 2;
                    unsigned b0, b1;
                    ld_B(b0, b1, (char*)sK0 + k_off);

                    int r0 = warp_id * (B_H/WARPS) + lane_id / 4, r1 = r0 + 8;
                    int c0 = ni + (lane_id % 4) * 2, c1 = c0 + 1;
                    float pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
                    if (r0 < B_H && c0 < B_TK) {
                        pc0 = (float)sS[r0 * B_TK + c0];
                        if (c1 < B_TK) pc1 = (float)sS[r0 * B_TK + c1];
                        if (r1 < B_H) { pc2 = (float)sS[r1 * B_TK + c0]; if (c1 < B_TK) pc3 = (float)sS[r1 * B_TK + c1]; }
                    }
                    mma_m16n8k16(pc0, pc1, pc2, pc3, a0, a1, a2, a3, b0, b1);
                    if (r0 < B_H && c0 < B_TK) {
                        sS[r0 * B_TK + c0] = bf16(pc0); if (c1 < B_TK) sS[r0 * B_TK + c1] = bf16(pc1);
                        if (r1 < B_H) { sS[r1 * B_TK + c0] = bf16(pc2); if (c1 < B_TK) sS[r1 * B_TK + c1] = bf16(pc3); }
                    }
                }
            }
        }
        __syncthreads();

        // --- Online softmax ---
        float scale = params.sm_scale_div_log2;
        float new_maxs[B_H];
        float scale_olds[B_H];
        float l_sums[B_H];        // running L sum per row
        for (int r = 0; r < B_H; ++r) { new_maxs[r] = -1e33f; scale_olds[r] = 1.0f; l_sums[r] = 0.0f; }

        for (int r = 0; r < B_H; ++r) {
            if (tid != r % NT) continue;  // only 1 thread per row

            float cm = -1e33f;
            float cs = 0.0f;
            for (int ni = 0; ni < this_tk; ++ni) {
                float v = (float)sS[r * B_TK + ni] * scale;
                cm = max(cm, v);
                sS[r * B_TK + ni] = bf16(v);
            }
            for (int ni = this_tk; ni < B_TK; ++ni) sS[r * B_TK + ni] = bf16(-1e33f);

            float om = sM[r], nm = max(om, cm);
            float so = (om <= -1e29f) ? 0.0f : exp2f(om - nm);
            if ((tid % B_H) == r && (tid / B_H) == 0) { sM[r] = nm; }
            new_maxs[r] = nm;
            scale_olds[r] = so;

            // Rescale L sum
            l_sums[r] *= so;
            // Rescale O_acc for this row
            for (int c = tid; c < D_V; c += NT)
                o_acc[r * D_V + c] *= so;

            // Exp and sum
            cs = 0.0f;
            for (int ni = 0; ni < this_tk; ++ni) {
                float v = (float)sS[r * B_TK + ni];
                if (v <= -1e32f) { sS[r * B_TK + ni] = bf16(0.0f); continue; }
                v = exp2f(v - nm); sS[r * B_TK + ni] = bf16(v); cs += v;
            }
            for (int ni = this_tk; ni < B_TK; ++ni) sS[r * B_TK + ni] = bf16(0.0f);
            l_sums[r] += cs;  // accumulate running L sum
        }
        __syncthreads();

        // --- PV: O += softmax(P) @ V ---
        // V is (B_TK x D_V) stored in K's first D_V columns, row-major
        // Process in two halves: left (0-255) and right (256-511)
        for (int half = 0; half < 2; ++half) {
            int v_start = half * (D_V / 2);
            int v_tiles = V_HALF_TILES;

            for (int vt = 0; vt < v_tiles; ++vt) {
                int v_col = v_start + vt * 64;

                // Load V tile: B_TK x 64 from global (same indices as K, first D_V cols)
                int v_bytes = B_TK * 64 * sizeof(bf16);
                for (int i = tid; i < v_bytes / 16; i += NT) {
                    int row = (i * 16 / 2) / 64;
                    int col = v_col + ((i * 16 / 2) % 64);
                    int tok_idx = tk_start + row;
                    if (row < this_tk && tok_idx < params.topk) {
                        int g_tok = __ldg(gIndices + tok_idx);
                        if (g_tok >= 0 && g_tok < params.s_kv) {
                            const bf16* src = params.kv + g_tok * params.stride_kv_s_kv + col;
                            uint32_t dst = smem_ptr_to_uint((char*)sK0 + i * 16);
                            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
                        }
                    }
                }
                asm volatile("cp.async.commit_group;\n" ::);
                asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                if (this_tk < B_TK)
                    for (int i = tid + this_tk * 64; i < B_TK * 64; i += NT) sK0[i] = bf16(0.0f);
                __syncthreads();

                // MMA: P @ V_tile
                int v_steps = 64 / 16;
                for (int vs = 0; vs < v_steps; ++vs) {
                    int vk = vs * 16;
                    int p_off = warp_id * (B_H/WARPS) * B_TK * 2 + vk * 2;
                    unsigned a0, a1, a2, a3;
                    ld_A(a0, a1, a2, a3, (char*)sS + p_off);

                    for (int vn = warp_id * 8; vn < 64; vn += WARPS * 8) {
                        int v_off = vn * 64 * 2 + vk * 2;
                        unsigned b0, b1;
                        ld_B(b0, b1, (char*)sK0 + v_off);

                        int r0 = warp_id * (B_H/WARPS) + lane_id / 4, r1 = r0 + 8;
                        int c0 = v_col + vn + (lane_id % 4) * 2, c1 = c0 + 1;
                        float pv0 = 0, pv1 = 0, pv2 = 0, pv3 = 0;
                        if (r0 < B_H && c0 < D_V) {
                            pv0 = o_acc[r0 * D_V + c0];
                            if (c1 < D_V) pv1 = o_acc[r0 * D_V + c1];
                            if (r1 < B_H) { pv2 = o_acc[r1 * D_V + c0]; if (c1 < D_V) pv3 = o_acc[r1 * D_V + c1]; }
                        }
                        mma_m16n8k16(pv0, pv1, pv2, pv3, a0, a1, a2, a3, b0, b1);
                        if (r0 < B_H && c0 < D_V) {
                            o_acc[r0 * D_V + c0] = pv0;
                            if (c1 < D_V) o_acc[r0 * D_V + c1] = pv1;
                            if (r1 < B_H) { o_acc[r1 * D_V + c0] = pv2; if (c1 < D_V) o_acc[r1 * D_V + c1] = pv3; }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }  // end topk block loop

    // --- Write output ---
    for (int r = tid; r < B_H; r += NT) {
        int g_row = g_q_row_start + r;
        if (g_row >= params.h_q) continue;

        // Compute L_sum and M from shared memory
        float m_val = sM[r];
        float l_sum = l_sums[r];
        
        // Attention sink (if provided)
        float attn_sink = params.attn_sink ? params.attn_sink[g_row] * CUDART_L2E_F : -CUDART_INF_F;
        float scale_factor = 1.0f / (l_sum + exp2f(attn_sink - m_val));
        if (l_sum == 0.0f) scale_factor = 0.0f;
        for (int c = 0; c < D_V; ++c) {
            params.out[s_q_idx * params.h_q * D_V + g_row * D_V + c] = bf16(o_acc[r * D_V + c] * scale_factor);
        }
        // Max logits and LSE
        params.max_logits[s_q_idx * params.h_q + g_row] = (m_val <= -1e29f) ? -INFINITY : m_val * CUDART_LN2_F;
        params.lse[s_q_idx * params.h_q + g_row] = (l_sum == 0.0f) ? INFINITY : logf(l_sum) + m_val * CUDART_LN2_F;
    }

#else
    asm("trap;");
#endif
}

// --- Host launch ---
template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    using KT = KernelTemplate<D_QK, HAVE_TOPK_LENGTH>;
    constexpr size_t sm = sizeof(typename KT::SharedMemoryPlan);
    const int grid_dim = (params.h_q / KT::B_H) * params.s_q;
    auto kernel = &sparse_attn_fwd_kernel<D_QK, HAVE_TOPK_LENGTH>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    kernel<<<dim3(grid_dim), dim3(KT::NUM_THREADS), sm, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace prefill
}  // namespace sm86
