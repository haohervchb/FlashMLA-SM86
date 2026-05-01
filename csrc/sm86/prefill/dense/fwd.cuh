#pragma once
#include <math_constants.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "defines.h"
#include "utils.h"

namespace sm86 {
namespace prefill {

template<int HD_QK, int HD_V, int BLK_M, int BLK_N>
__global__ void __launch_bounds__(128)
flash_attn_varlen_fwd_k(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q, const int* __restrict__ cu_seqlens_kv,
    int max_seqlen_q, int max_seqlen_kv, int h_q, int h_kv,
    float sm_scale, float sm_scale_log2, bool is_causal, int total_q, int total_kv)
{
    const int tid = threadIdx.x, my_row = tid % BLK_M;
    const int batch_id = blockIdx.y;
    const int q_start = __ldg(cu_seqlens_q + batch_id), q_end = __ldg(cu_seqlens_q + batch_id + 1);
    const int kv_start = __ldg(cu_seqlens_kv + batch_id), kv_end = __ldg(cu_seqlens_kv + batch_id + 1);
    const int slen_q = q_end - q_start, slen_kv = kv_end - kv_start;
    if (slen_q == 0 || slen_kv == 0) return;
    
    const int q_row = blockIdx.x * BLK_M + my_row;
    if (q_row >= slen_q) return;
    
    // Each thread handles its query row for ALL heads independently
    for (int qh = 0; qh < h_q; ++qh) {
        int kv_head = qh % h_kv;
        
        float rmax = -1e30f, rsum = 0.0f;
        float oa[HD_V] = {};
        
        for (int kvb = 0; kvb < slen_kv; kvb += BLK_N) {
            int kve = min(kvb + BLK_N, slen_kv), nv = kve - kvb;
            float P[BLK_N]; float cm = -1e33f;
            
            int qi = (q_start + q_row) * h_q * HD_QK + qh * HD_QK;
            for (int ni = 0; ni < nv; ++ni) {
                int ki = (kv_start + kvb + ni) * h_kv * HD_QK + kv_head * HD_QK;
                float dot = 0; for (int d = 0; d < HD_QK; ++d) dot += (float)q[qi+d] * (float)k[ki+d];
                dot *= sm_scale_log2; P[ni] = dot; cm = max(cm, dot);
            }
            for (int ni = nv; ni < BLK_N; ++ni) P[ni] = -1e33f;
            
            float om = rmax, nm = max(om, cm);
            float so = (om <= -1e29f) ? 0.0f : exp2f(om - nm);
            rmax = nm; rsum *= so;
            for (int c = 0; c < HD_V; ++c) oa[c] *= so;
            
            float cs = 0.0f;
            for (int ni = 0; ni < nv; ++ni) {
                float val = P[ni]; if (val <= -1e32f) { P[ni] = 0.0f; continue; }
                val = exp2f(val - nm); P[ni] = val; cs += val;
            }
            rsum += cs;
            
            for (int ni = 0; ni < nv; ++ni) {
                float pn = P[ni]; if (pn == 0.0f) continue;
                int vi = (kv_start + kvb + ni) * h_kv * HD_V + kv_head * HD_V;
                for (int c = 0; c < HD_V; ++c) oa[c] += pn * (float)v[vi+c];
            }
        }
        
        float rcp = (rsum == 0.0f || rsum != rsum) ? 1.0f : (1.0f / rsum);
        int oi = (q_start + q_row) * h_q * HD_V + qh * HD_V;
        for (int c = 0; c < HD_V; ++c) out[oi + c] = bf16(oa[c] * rcp);
        // LSE: shape (total_q, h_q) with stride (1, total_q) due to .T
        lse[(q_start + q_row) + qh * total_q] = (rsum == 0.0f) ? INFINITY : logf(rsum) + rmax / (float)M_LOG2E;
    }
}

}  // namespace prefill
}  // namespace sm86
