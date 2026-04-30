#pragma once

#include "params.h"

namespace sm86 {
namespace decode {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}  // namespace decode
}  // namespace sm86
