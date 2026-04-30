#pragma once

#include "params.h"

namespace sm86 {

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}
