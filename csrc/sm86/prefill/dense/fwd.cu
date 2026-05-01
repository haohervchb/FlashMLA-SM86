#include "fwd.cuh"

namespace sm86 {
namespace prefill {

void run_flash_attn_varlen_fwd(
    const bf16* q, const bf16* k, const bf16* v, bf16* out, float* lse,
    const int* cu_seqlens_q, const int* cu_seqlens_kv,
    int max_seqlen_q, int max_seqlen_kv, int h_q, int h_kv,
    float sm_scale, bool is_causal, int num_batches, int total_q, int total_kv,
    cudaStream_t stream)
{
    dim3 grid((max_seqlen_q + 15) / 16, num_batches, 1);  // one group per KV head
    auto kernel = &flash_attn_varlen_fwd_k<576, 512, 16, 64>;
    kernel<<<grid, dim3(128), 0, stream>>>(
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv, h_q, h_kv,
        sm_scale, sm_scale * (1.0f / (float)M_LN2), is_causal, total_q, total_kv);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace prefill
}  // namespace sm86
