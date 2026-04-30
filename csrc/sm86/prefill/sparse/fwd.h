#pragma once

#include "params.h"

namespace sm86 {
namespace prefill {

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}  // namespace prefill
}  // namespace sm86
