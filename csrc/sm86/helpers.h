#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace sm86 {

// Global -> shared copy using cp.async
// Each call copies 16 bytes from global to shared
__forceinline__ __device__ void cp_async_commit_and_wait(void* smem_ptr, const void* gmem_ptr, int num_bytes, int idx_in_warpgroup, int num_copy_threads = 32) {
    // Distribute copy work across threads
    int total_lines = num_bytes / 16;
    for (int i = idx_in_warpgroup % num_copy_threads; i < total_lines; i += num_copy_threads) {
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr) + i * 16;
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"((const char*)gmem_ptr + i * 16));
    }
    // Commit and wait for all copies to complete
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}

// Global -> shared copy, pipelined -- commit but don't wait yet
__forceinline__ __device__ void cp_async_commit(void* smem_ptr, const void* gmem_ptr, int num_bytes, int idx_in_warpgroup, int num_copy_threads = 32) {
    int total_lines = num_bytes / 16;
    for (int i = idx_in_warpgroup % num_copy_threads; i < total_lines; i += num_copy_threads) {
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr) + i * 16;
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"((const char*)gmem_ptr + i * 16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int COMMIT_GROUPS_BEHIND = 0>
__forceinline__ __device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(COMMIT_GROUPS_BEHIND));
}

// Shared -> global store (bulk copy helper)
// SM86 doesn't have SM90_BULK_COPY_S2G, so we use manual stores
__forceinline__ __device__ void bulk_copy_s2g(const void* smem_ptr, void* gmem_ptr, int num_bytes) {
    const int* src = (const int*)smem_ptr;
    int* dst = (int*)gmem_ptr;
    int tid = threadIdx.x;
    int total_ints = num_bytes / 4;
    for (int i = tid; i < total_ints; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}
