#pragma once

#include "api/common.h"
#include "params.h"

namespace sm86 {
namespace decode {
void run_sparse_decode_fwd_kernel(SparseAttnDecodeParams &params);
}  // namespace decode
}  // namespace sm86

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sm86_sparse_attn_decode_interface(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int head_size_v,
    float softmax_scale
) {
    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm80(), "Sparse decode MLA SM86 requires SM80+");

    KU_CHECK_DEVICE(q); KU_CHECK_DEVICE(kcache); KU_CHECK_DEVICE(indices);
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(head_size_v == 512, "head_size_v must be 512");

    int b = q.size(0), s_q = q.size(1), h_q = q.size(2), d_qk = q.size(3);
    TORCH_CHECK(d_qk == 576 || d_qk == 512, "d_qk must be 576 or 512");

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({b, s_q, h_q, 512}, opts);
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    SparseAttnDecodeParams params = {};
    params.b = b; params.s_q = s_q;
    params.h_q = h_q; params.h_kv = 1;
    params.d_qk = d_qk; params.d_v = 512;
    params.sm_scale = softmax_scale;
    params.sm_scale_div_log2 = softmax_scale / (float)M_LN2;
    params.num_blocks = kcache.size(0);
    params.page_block_size = 64;
    params.topk = indices.size(-1);
    params.model_type = (d_qk == 576) ? ModelType::V32 : ModelType::MODEL1;

    params.q = (cutlass::bfloat16_t*)q.data_ptr();
    params.kv = (cutlass::bfloat16_t*)kcache.data_ptr();
    params.indices = indices.data_ptr<int>();
    params.topk_length = topk_length ? topk_length->data_ptr<int>() : nullptr;
    params.attn_sink = attn_sink ? attn_sink->data_ptr<float>() : nullptr;

    params.lse = lse.data_ptr<float>();
    params.out = (cutlass::bfloat16_t*)out.data_ptr();

    params.stride_q_b = q.stride(0); params.stride_q_s_q = q.stride(1); params.stride_q_h_q = q.stride(2);
    params.stride_kv_block = kcache.stride(0); params.stride_kv_row = kcache.stride(1);
    params.stride_indices_b = indices.stride(0); params.stride_indices_s_q = indices.stride(1);
    params.stride_lse_b = lse.stride(0); params.stride_lse_s_q = lse.stride(1);
    params.stride_o_b = out.stride(0); params.stride_o_s_q = out.stride(1); params.stride_o_h_q = out.stride(2);

    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_sm_parts = tile_scheduler_metadata->size(0);
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    sm86::decode::run_sparse_decode_fwd_kernel(params);

    return {out, lse, tile_scheduler_metadata, num_splits};
}
