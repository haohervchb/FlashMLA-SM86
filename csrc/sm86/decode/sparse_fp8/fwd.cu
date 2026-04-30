#include "splitkv_mla.cuh"

namespace sm86 {
namespace decode {

// Explicitly instantiate all 4 combinations
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(const SparseAttnDecodeParams&);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 128>(const SparseAttnDecodeParams&);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams&);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 128>(const SparseAttnDecodeParams&);

}  // namespace decode
}  // namespace sm86
