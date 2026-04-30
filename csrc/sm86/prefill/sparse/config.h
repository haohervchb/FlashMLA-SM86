#pragma once

#include <math_constants.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "defines.h"
#include "params.h"

// Pull in bf16 type
using cutlass::bfloat16_t;

namespace sm86 {
namespace prefill {

// using namespace cute;  // removed to avoid host compiler issues

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate {
public:

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;

static constexpr int B_H = 16;          // BLOCK_M reduced for 99KB smem
static constexpr int B_TOPK = 64;       // topk block size
static constexpr int NUM_THREADS = 128; // 4 warps
    static constexpr float MAX_INIT_VAL = -1e30;

    // Shared memory layout: plain C arrays
    struct SharedMemoryPlan {
        bf16 q[B_H * D_Q];
        bf16 k[2][B_TOPK * 64];     // double-buffered K tiles (64x64 each)
        bf16 s[B_H * B_TOPK];       // softmax scores
        float sM[B_H];              // running max
    };

};

}  // namespace prefill
}  // namespace sm86
