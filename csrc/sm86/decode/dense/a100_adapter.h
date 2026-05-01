#pragma once
// Adapter: maps DenseAttnDecodeParams to Flash_fwd_mla_params for A100 port kernel

#include "params.h"
#include "flash_mla.h"

namespace sm86 {
namespace tc {

inline Flash_fwd_mla_params adapt_params(const DenseAttnDecodeParams &p) {
    Flash_fwd_mla_params fp = {};
    fp.b = p.b;
    fp.seqlen_q = p.s_q * p.q_head_per_hk;  // = q_seq_per_hk
    fp.d = p.d;
    fp.d_v = p.d_v;
    fp.h = p.h_k;
    fp.h_h_k_ratio = p.h_q / p.h_k;
    fp.ngroups = p.h_q / p.h_k;
    fp.is_causal = p.is_causal;
    fp.scale_softmax = p.scale_softmax;
    fp.scale_softmax_log2 = p.scale_softmax_log2;
    fp.cu_seqlens_k = p.seqlens_k_ptr;

    fp.q_ptr = p.q_ptr;
    fp.k_ptr = p.k_ptr;
    fp.v_ptr = p.k_ptr;  // K=V in MLA
    fp.o_ptr = p.o_ptr;
    fp.softmax_lse_ptr = p.softmax_lse_ptr;

    fp.q_batch_stride = p.q_batch_stride;
    fp.k_batch_stride = p.k_batch_stride;
    fp.v_batch_stride = p.k_batch_stride;
    fp.o_batch_stride = p.o_batch_stride;
    fp.q_row_stride = p.q_row_stride;
    fp.k_row_stride = p.k_row_stride;
    fp.v_row_stride = p.k_row_stride;
    fp.o_row_stride = p.o_row_stride;
    fp.q_head_stride = p.q_head_stride;
    fp.k_head_stride = p.k_head_stride;
    fp.v_head_stride = p.k_head_stride;
    fp.o_head_stride = p.o_head_stride;

    fp.block_table = p.block_table;
    fp.block_table_batch_stride = p.block_table_batch_stride;
    fp.page_block_size = p.page_block_size;

    fp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(p.tile_scheduler_metadata_ptr);
    fp.num_sm_parts = p.num_sm_parts;
    fp.num_splits_ptr = p.num_splits_ptr;

    fp.softmax_lseaccum_ptr = p.softmax_lseaccum_ptr;
    fp.oaccum_ptr = p.oaccum_ptr;

    // Fix: q_seq_per_hk is total query rows. For the A100 kernel, 
    // seqlen_q should be the total number of query tokens per KV head.
    // Our q_seq_per_hk = s_q * h_q / h_kv. The A100 kernel expects seqlen_q = q_seq_per_hk.
    fp.seqlen_q = p.q_seq_per_hk;

    return fp;
}

// Dispatch to A100 port kernel
template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    Flash_fwd_mla_params fp = adapt_params(params);
    // The A100 port uses: mha_fwd_splitkv_mla<T, 576, false>::run(fp, params.stream)
    mha_fwd_splitkv_mla<InputT, 576, false>::run(fp, params.stream);
}

}  // namespace tc
}  // namespace sm86
