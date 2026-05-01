#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace sm86 {
namespace prefill {

void run_flash_attn_varlen_fwd(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k, const cutlass::bfloat16_t* v,
    cutlass::bfloat16_t* out, float* lse,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    int max_seqlen_q, int max_seqlen_kv, int h_q, int h_kv,
    float sm_scale, bool is_causal, int num_batches, int total_q, int total_kv,
    cudaStream_t stream);

}  // namespace prefill
}  // namespace sm86
