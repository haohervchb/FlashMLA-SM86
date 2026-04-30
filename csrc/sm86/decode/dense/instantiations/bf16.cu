#include "../splitkv_mla.cuh"
#include "../splitkv_mla.h"

namespace sm86 {

template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(DenseAttnDecodeParams &params);

}
