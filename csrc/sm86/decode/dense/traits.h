#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include "config.h"

using namespace cute;

template<typename InputT_>
struct Traits {
    using InputT = InputT_;

    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;

    static constexpr int NUM_THREADS = 16;  // 1 thread per BLK_M row

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    using MMA_Atom_QK = SM80_16x8x16_F32BF16BF16F32_TN;
    using TiledMMA_QK = decltype(make_tiled_mma(MMA_Atom_QK{}, Layout<Shape<_1, _1, _1>>{}));
    using MMA_Atom_PV = SM80_16x8x16_F32BF16BF16F32_TN;
    using TiledMMA_PV = decltype(make_tiled_mma(MMA_Atom_PV{}, Layout<Shape<_1, _1, _1>>{}));

    // Plain row-major shared memory layouts (no CuTe swizzle)
    // to avoid tile_to_shape compatibility issues with BLOCK_M=16
    using SmemLayoutQ = Layout<Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>, _1>>;
    using SmemLayoutV = decltype(composition(
        SmemLayoutK{},
        make_layout(Shape<Int<HEAD_DIM_V>, Int<PAGE_BLOCK_SIZE>>{}, GenRowMajor{})
    ));
    using SmemLayoutP = Layout<Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>, Stride<Int<PAGE_BLOCK_SIZE>, _1>>;

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, BLOCK_SIZE_M * HEAD_DIM_K> smem_sQ;
        cute::array_aligned<InputT, PAGE_BLOCK_SIZE * HEAD_DIM_K> smem_sK;
        cute::array_aligned<InputT, BLOCK_SIZE_M * PAGE_BLOCK_SIZE> smem_sP;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sM;
        cute::array_aligned<float, 2*BLOCK_SIZE_M> sL_reduction_wksp;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale;
    };

};

struct DenseAttnDecodeParams;
