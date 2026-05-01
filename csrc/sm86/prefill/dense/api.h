#pragma once

#include "api/common.h"
#include "sm86/prefill/dense/fwd.h"

static void FMHACutlassSM86FwdRun(
    at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
    at::Tensor o, at::Tensor lse,
    int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen)
{
    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm80(), "SM86 dense prefill requires SM80+");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "Only bf16 supported");
    
    int total_q = q.size(0), total_kv = k.size(0);
    int num_batches = cumulative_seqlen_q.size(0) - 1;
    int h_q = q.size(1), h_kv = k.size(1);
    int head_dim_qk = q.size(2), head_dim_v = v.size(2);
    bool is_causal = mask_mode_code == 1;
    
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    sm86::prefill::run_flash_attn_varlen_fwd(
        (const cutlass::bfloat16_t*)q.data_ptr(), (const cutlass::bfloat16_t*)k.data_ptr(), 
        (const cutlass::bfloat16_t*)v.data_ptr(),
        (cutlass::bfloat16_t*)o.data_ptr(), (float*)lse.data_ptr(),
        cumulative_seqlen_q.data_ptr<int>(), cumulative_seqlen_kv.data_ptr<int>(),
        max_seqlen_q, max_seqlen_kv, h_q, h_kv,
        softmax_scale, is_causal, num_batches, total_q, total_kv,
        at::cuda::getCurrentCUDAStream().stream());
}
