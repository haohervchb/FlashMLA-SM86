#pragma once
// Tensor-core SM86 dense MLA decoding kernel v2
// Architecture: 256 threads (8 warps), BLOCK_M=32, BLOCK_N=64
// 3 inner K-stages (192 each), double-buffered cp.async pipeline
// P->smem for PV GEMM (avoids V reload), cross-warp softmax
// Based on pzhao-eng/FlashMLA A100 port architecture

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>

using namespace cute;

#include "utils.h"
#include "params.h"
#include "config.h"
#include "named_barrier.h"
#include "softmax.h"
#include "static_switch.h"

namespace sm86 {
namespace tc {

//////////////////////////////////////////////////////////////////////
// Kernel traits
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         typename elem_type=cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct KernelTraitsV2 {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int kNWarps = kNWarps_;           // 8 
    static constexpr int kNThreads = kNWarps * 32;     // 256
    static constexpr int kNWarpsM = 2;                  // M-dim warps
    static constexpr int kNWarpsN = kNWarps / kNWarpsM; // N-dim warps (4)
    static_assert(kNWarps == kNWarpsM * kNWarpsN);

    static constexpr int kBlockM = kBlockM_;  // 32
    static constexpr int kBlockN = kBlockN_;  // 64
    static constexpr int kHeadDim = kHeadDim_;  // 576
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;  // 512
    static_assert(kHeadDimV % 32 == 0 && kHeadDimV <= kHeadDim);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int kNumInnerStagesK = 3;
    static_assert(kHeadDim % kNumInnerStagesK == 0);

    // SM80 MMA atom
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMma = TiledMMA<
        MMA_Atom,
        Layout<Shape<Int<kNWarpsM>, Int<kNWarpsN>, _1>>,
        Tile<Int<16 * kNWarpsM>, Int<8 * kNWarpsN>, _16>>;

    // Smem layouts (GMMA swizzle for bank conflict avoidance)
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutSplitQ = decltype(composition(
        SmemLayoutQ{}, make_layout(Shape<Int<kBlockM>,
        Int<kHeadDim/kNumInnerStagesK>, Int<kNumInnerStagesK>>{})));

    static constexpr int kP = 2; // double-buffer pipe count
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim/kNumInnerStagesK>, Int<kP>>{}));
    using SmemLayoutSplitK = decltype(composition(
        SmemLayoutK{}, make_layout(Shape<Int<kBlockN>,
        Int<kHeadDim/kNumInnerStagesK>, Int<kNumInnerStagesK>, Int<kP>>{})));

    // P smem layout
    static constexpr int kBlockKSmemP = kBlockN % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleP = kBlockKSmemP == 32 ? 2 : 3;
    using SmemLayoutAtomP = decltype(
        composition(Swizzle<kSwizzleP, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmemP>>,
                           Stride<Int<kBlockKSmemP>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));

    // O smem layout
    using SmemLayoutAtomO = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    // Copy atoms
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using SmemCopyAtomBTransposed = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    // Global memory copy: cp.async with 128-bit loads
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<Gmem_copy_struct, Element>{},
        GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using GmemLayoutAtomO = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
        GmemLayoutAtomO{}, Layout<Shape<_1, _8>>{}));

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
        Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
        GmemLayoutAtomOaccum{}, Layout<Shape<_1, _4>>{}));
};

//////////////////////////////////////////////////////////////////////
// Shared memory
template<typename Kernel_traits>
struct SharedStorageV2 {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK>> smem_k;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM * (Kernel_traits::kNWarpsN + 1)> smem_reduce;
};

//////////////////////////////////////////////////////////////////////
// Helper: convert layout_acc_rowcol (from A100 port softmax.h)
namespace flash {

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout layout) {
    using T = typename Layout::value_type;
    auto [m, n] = layout.shape();
    auto [s0, s1] = layout.stride();
    return Layout<decltype(make_shape(m, n)), decltype(make_stride(s0, s1))>{};
}

template <typename T0, typename T1>
__forceinline__ __device__ auto convert_type(T1 const &t1) {
    Tensor r = make_tensor_like<T0>(t1);
    #pragma unroll
    for (int i = 0; i < size(r); ++i) r(i) = T0(t1(i));
    return r;
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=false,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy const &tiled_copy,
                                     Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D,
                                     Tensor<Engine2, Layout2> const &identity_MN,
                                     Tensor<Engine3, Layout3> &pred_X,
                                     [[maybe_unused]] int max_MN = 0) {
    using Element = typename TiledCopy::ValType;
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    constexpr int Seq_M = 2, Seq_N = 1, Seq_K = 0;
    if constexpr (Clear_OOB_K) { clear(D); }
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(S); ++k) {
        if constexpr (Is_even_K) {
            pred_X(k) = get<Seq_K>(identity_MN(0, 0, k)) < size<2>(S);
        } else {
            pred_X(k) = true;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(S); ++m) {
            const int m_coord = get<Seq_M>(identity_MN(0, m, 0));
            const bool m_pred = (Clear_OOB_MN && Is_even_MN) ? false : (!Clear_OOB_MN || m_coord < max_MN);
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<0>(S); ++n) {
                if (pred_X(k) && m_pred) {
                    D(n, m, k) = S(n, m, k);
                }
            }
        }
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_8x(
    Tensor0 &acc, Tensor1 const &tCrA, Tensor2 const &tCrB,
    Tensor3 const &tCsA, Tensor3 const &tCsB,
    TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA const &smem_thr_copy_A, ThrCopyB const &smem_thr_copy_B) {
    constexpr int kBlockK = size<2>(tCsA);
    #pragma unroll
    for (int k_block = 0; k_block < kBlockK; ++k_block) {
        auto tCrA_view = tCrA(_, _, k_block);
        auto tCrB_view = tCrB(_, _, k_block);
        copy(smem_tiled_copy_A, tCsA(_, _, k_block), tCrA_view);
        copy(smem_tiled_copy_B, tCsB(_, _, k_block), tCrB_view);
        #pragma unroll
        for (int k = 0; k < size<2>(tCrA_view); ++k) {
            int kk = k_block * size<2>(tCrA_view) + k;
            if (kk == 0) { gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc); tiled_mma.accumulate_ = GMMA::ScaleOut::One; }
            else { gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc); }
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, int kN>
__forceinline__ __device__ void cross_warp_reduce_max(
    Tensor<Engine0, Layout0> &val, float *smem, int n_warp_idx, const int *row_indices, int stride) {
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) smem[row_indices[mi] * stride + n_warp_idx] = val(mi);
    __syncthreads();
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        float result = smem[row_indices[mi] * stride];
        #pragma unroll
        for (int w = 1; w < kN; ++w) result = max(result, smem[row_indices[mi] * stride + w]);
        val(mi) = result;
    }
    __syncthreads();
}

template <typename Engine0, typename Layout0, int kN>
__forceinline__ __device__ void cross_warp_reduce_sum(
    Tensor<Engine0, Layout0> &val, float *smem, int n_warp_idx, const int *row_indices, int stride) {
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) smem[row_indices[mi] * stride + n_warp_idx] = val(mi);
    __syncthreads();
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        float result = 0.f;
        #pragma unroll
        for (int w = 0; w < kN; ++w) result += smem[row_indices[mi] * stride + w];
        val(mi) = result;
    }
    __syncthreads();
}

//////////////////////////////////////////////////////////////////////
// Main kernel function
template<typename Kernel_traits, bool Is_causal, bool Split, typename SharedStorage>
__forceinline__ __device__ void compute_attn_kernel_v2(
    const DenseAttnDecodeParams &params,
    const int bidb, const int bidh, const int m_block,
    const int n_split_idx, const int seqlen_k,
    const int n_block_min, const int n_block_max,
    SharedStorage &shared_storage) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNumInnerStagesK = Kernel_traits::kNumInnerStagesK;
    constexpr int kNWarpsM = Kernel_traits::kNWarpsM;
    constexpr int kNWarpsN = Kernel_traits::kNWarpsN;
    constexpr int kReduceStride = kNWarpsN + 1;
    constexpr int kP = Kernel_traits::kP;
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 32;
    const int n_warp_idx = warp_idx / kNWarpsM;

    // Shared memory tensors
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQ_split = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutSplitQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sK_split = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutSplitK{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    float *smem_reduce = reinterpret_cast<float*>(shared_storage.smem_reduce.data());

    // MMA setup
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ_split(_, _, 0));
    Tensor tSrK = thr_mma.partition_fragment_B(sK_split(_, _, 0, 0));

    // Copy atoms for MMA operands
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ_split);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK_split);

    // P: write C-fragment to smem, then read as A for PV
    auto smem_tiled_copy_PwriteC = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_PwriteC = smem_tiled_copy_PwriteC.get_thread_slice(tidx);
    auto smem_tiled_copy_PreadA = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_PreadA = smem_tiled_copy_PreadA.get_thread_slice(tidx);
    Tensor tSrP = thr_mma.partition_fragment_A(sP);

    // Row indices for softmax reduction
    Tensor cS_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS_id = thr_mma.partition_C(cS_id);
    auto scores_id_rowcol = make_tensor(tScS_id.data(), convert_layout_acc_rowcol(tScS_id.layout()));
    constexpr int kSoftmaxRows = decltype(size<0>(scores_id_rowcol))::value;
    int row_indices[kSoftmaxRows];
    #pragma unroll
    for (int mi = 0; mi < kSoftmaxRows; ++mi) row_indices[mi] = int(get<0>(scores_id_rowcol(mi, 0)));

    // Vt for PV (transposed V from sK buffer)
    using SmemLayoutVtransposed = decltype(composition(
        typename Kernel_traits::SmemLayoutK{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>, Int<kP>>{},
                    Stride<Int<kBlockN>, _1, Int<kHeadDimV * kBlockN>>{})));
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutVtransposed{});
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt(_, _, 0));
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomBTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);

    // Accumulators
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    clear(acc_o);
    Softmax<2 * size<1>(acc_o)> softmax;

    // Global memory copy setup
    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);

    const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int n_block = n_block_max - 1;
    int cur_block_table = __ldg(&block_table[n_block]);

    // Q load
    const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);

    // Using simple manual copy for Q (avoids CuTe complexity for strides)
    if (tidx < kNThreads) {
        for (int i = tidx; i < kBlockM * kHeadDim; i += kNThreads) {
            int r = i / kHeadDim, c = i % kHeadDim;
            int g_row = m_block * kBlockM + r;
            int num_valid = min(params.q_seq_per_hk - m_block * kBlockM, kBlockM);
            if (r < num_valid)
                shared_storage.smem_q.data()[i] = reinterpret_cast<Element*>(params.q_ptr)[bidb * params.q_batch_stride + g_row * params.q_row_stride + bidh * params.q_head_stride + c];
            else
                shared_storage.smem_q.data()[i] = Element(0.0f);
        }
    }
    __syncthreads();

    // K load setup
    const index_t row_offset_k = (bidh / params.h_k) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor tKgK = gmem_thr_copy.partition_S(gK);
    Tensor tKsK = gmem_thr_copy.partition_D(sK);
    auto K_PIPE_MAX = size<3>(tKsK);

    // Identity/predicate tensors for K copy
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Prologue: Load first K block
    int smem_pipe_read = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;
    const index_t offset_k0 = cur_block_table * params.k_batch_stride;
    tKgK.data() = tKgK.data() + offset_k0;
    Tensor tKsK_p0 = tKsK(_, _, _, 0);
    flash::copy</*Is_even_MN*/false, /*Is_even_K*/true, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy, tKgK, tKsK_p0, tKVcKV, tKVpKV, seqlen_k - n_block * kBlockN);
    tKgK.data() = tKgK.data() - offset_k0;
    cute::cp_async_fence();
    flash::cp_async_wait<0>();
    __syncthreads();

    // Masking iterations (at least 1, more for causal)
    constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        // Load next K block while computing current
        if (n_block - 1 >= n_block_min) {
            cur_block_table = __ldg(&block_table[n_block - 1]);
            const index_t offset_k = cur_block_table * params.k_batch_stride;
            tKgK.data() = tKgK.data() + offset_k;
            Tensor tKsK_next = tKsK(_, _, _, smem_pipe_write);
            flash::copy</*Is_even_MN*/true, /*Is_even_K*/true>(
                gmem_tiled_copy, tKgK, tKsK_next, tKVcKV, tKVpKV);
            tKgK.data() = tKgK.data() - offset_k;
            cute::cp_async_fence();
        }

        // QK^T GEMM (3 inner K-stages)
        #pragma unroll
        for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
            Tensor tSsQ_p = tSsQ(_, _, _, slice);
            Tensor tSsK_p = tSsK(_, _, _, slice, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p,
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);
        }

        // Mask
        const bool is_first_masking_step = masking_step == 0;
        if constexpr (Is_causal) {
            Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                int row = int(get<0>(tScS(i)));
                int col_limit = seqlen_k - 1 - n_block * kBlockN - (params.s_q - 1 - (m_block * kBlockM + row) / params.q_head_per_hk);
                if (int(get<1>(tScS(i))) > col_limit) acc_s(i) = -INFINITY;
            }
        } else {
            // Non-causal: just mask OOB
            Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) acc_s(i) = -INFINITY;
            }
        }

        // Softmax
        Tensor scale_o;
        if (is_first_masking_step) {
            Softmax<2*size<1>(acc_o)>::template reduce_max</*zero_init=*/true>(acc_s, softmax.row_max);
            flash::cross_warp_reduce_max<kNWarpsN>(softmax.row_max, smem_reduce, n_warp_idx, row_indices, kReduceStride);
            Softmax<2*size<1>(acc_o)>::template scale_apply_exp2(acc_s, softmax.row_max, params.scale_softmax_log2);
            Softmax<2*size<1>(acc_o)>::template reduce_sum</*zero_init=*/true>(acc_s, softmax.row_sum);
            scale_o = make_fragment_like(softmax.row_max);
            clear(scale_o);
        } else {
            auto scores_max_prev = make_fragment_like(softmax.row_max);
            copy(softmax.row_max, scores_max_prev);
            Softmax<2*size<1>(acc_o)>::template reduce_max</*zero_init=*/false>(acc_s, softmax.row_max);
            flash::cross_warp_reduce_max<kNWarpsN>(softmax.row_max, smem_reduce, n_warp_idx, row_indices, kReduceStride);
            scale_o = make_fragment_like(softmax.row_max);
            #pragma unroll
            for (int mi = 0; mi < size(softmax.row_max); ++mi) {
                float scores_scale = exp2f((scores_max_prev(mi) - softmax.row_max(mi)) * params.scale_softmax_log2);
                scale_o(mi) = scores_scale;
                softmax.row_sum(mi) *= scores_scale;
            }
            Softmax<2*size<1>(acc_o)>::template scale_apply_exp2(acc_s, softmax.row_max, params.scale_softmax_log2);
            Softmax<2*size<1>(acc_o)>::template reduce_sum</*zero_init=*/false>(acc_s, softmax.row_sum);
        }

        // Convert P to bf16, store to smem_p
        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
        Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
        copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);

        // Rescale O
        #pragma unroll
        for (int mi = 0; mi < size(scale_o); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o); ++ni) {
                acc_o(mi, ni) *= scale_o(mi);
            }
        }

        // Wait for K load, swap pipes
        flash::cp_async_wait<1>();
        __syncthreads();

        // PV GEMM: P (smem) @ V (K smem, transposed)
        Tensor tPsP_read = smem_thr_copy_PreadA.partition_S(sP);
        Tensor tOsVt_p = smem_thr_copy_V.partition_S(sVt(_, _, smem_pipe_read));
        flash::gemm_8x<false, false>(acc_o, tSrP, tOrVt, tPsP_read, tOsVt_p,
            tiled_mma, smem_tiled_copy_PreadA, smem_tiled_copy_V,
            smem_thr_copy_PreadA, smem_thr_copy_V);

        // Advance pipes
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;

        if (n_masking_steps > 1 && n_block <= n_block_min) { --n_block; break; }
    }

    // Non-masking iterations
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        if (n_block - 1 >= n_block_min) {
            cur_block_table = __ldg(&block_table[n_block - 1]);
            const index_t offset_k = cur_block_table * params.k_batch_stride;
            tKgK.data() = tKgK.data() + offset_k;
            Tensor tKsK_next = tKsK(_, _, _, smem_pipe_write);
            flash::copy</*Is_even_MN*/true, /*Is_even_K*/true>(
                gmem_tiled_copy, tKgK, tKsK_next, tKVcKV, tKVpKV);
            tKgK.data() = tKgK.data() - offset_k;
            cute::cp_async_fence();
        }

        #pragma unroll
        for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
            Tensor tSsQ_p = tSsQ(_, _, _, slice);
            Tensor tSsK_p = tSsK(_, _, _, slice, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p,
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);
        }

        auto scores_max_prev = make_fragment_like(softmax.row_max);
        copy(softmax.row_max, scores_max_prev);
        Softmax<2*size<1>(acc_o)>::template reduce_max</*zero_init=*/false>(acc_s, softmax.row_max);
        flash::cross_warp_reduce_max<kNWarpsN>(softmax.row_max, smem_reduce, n_warp_idx, row_indices, kReduceStride);

        auto scale_o = make_fragment_like(softmax.row_max);
        #pragma unroll
        for (int mi = 0; mi < size(softmax.row_max); ++mi) {
            float scores_scale = exp2f((scores_max_prev(mi) - softmax.row_max(mi)) * params.scale_softmax_log2);
            scale_o(mi) = scores_scale;
            softmax.row_sum(mi) *= scores_scale;
        }
        Softmax<2*size<1>(acc_o)>::template scale_apply_exp2(acc_s, softmax.row_max, params.scale_softmax_log2);
        Softmax<2*size<1>(acc_o)>::template reduce_sum</*zero_init=*/false>(acc_s, softmax.row_sum);

        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
        Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
        copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);

        #pragma unroll
        for (int mi = 0; mi < size(scale_o); ++mi)
            for (int ni = 0; ni < size<1>(acc_o); ++ni)
                acc_o(mi, ni) *= scale_o(mi);

        flash::cp_async_wait<1>();
        __syncthreads();

        Tensor tPsP_read = smem_thr_copy_PreadA.partition_S(sP);
        Tensor tOsVt_p = smem_thr_copy_V.partition_S(sVt(_, _, smem_pipe_read));
        flash::gemm_8x<false, false>(acc_o, tSrP, tOrVt, tPsP_read, tOsVt_p,
            tiled_mma, smem_tiled_copy_PreadA, smem_tiled_copy_V,
            smem_thr_copy_PreadA, smem_thr_copy_V);

        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
    }

    // --- Epilogue: normalize and write output ---
    // Collect cumulative row_sum across N-warps
    flash::cross_warp_reduce_sum<kNWarpsN>(softmax.row_sum, smem_reduce, n_warp_idx, row_indices, kReduceStride);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::SmemCopyAtomO, typename Kernel_traits::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);

    // Normalize and compute LSE
    Tensor lse = make_fragment_like(softmax.row_sum);
    #pragma unroll
    for (int mi = 0; mi < size(softmax.row_sum); ++mi) {
        float sum = softmax.row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : softmax.row_max(mi) * params.scale_softmax + logf(sum);
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o); ++ni) acc_o(mi, ni) *= inv_sum;
    }

    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor tAccR = smem_thr_copy_O.retile_S(rO);
    Tensor tAccS = smem_thr_copy_O.partition_D(sO);
    copy(smem_tiled_copy_O, tAccR, tAccS);
    __syncthreads();

    // Write to global
    int split_offset = params.num_splits_ptr ? __ldg(params.num_splits_ptr + bidb) : 0;
    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oacc = (((split_offset + n_split_idx) * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM) * kHeadDimV;
    const index_t row_offset_lse = (bidb * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM;
    const index_t row_offset_lseacc = ((split_offset + n_split_idx) * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM;

    ElementO* gO_ptr = reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oacc : row_offset_o);
    float* gLSE_ptr = reinterpret_cast<float*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseacc : row_offset_lse);

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

    Tensor gO = make_tensor(make_gmem_ptr(gO_ptr), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<ElementO>(shape(tOgO));
    copy(gmem_tiled_copy_O, tOsO, tOrO);
    copy(tOrO, tOgO);

    // Write LSE
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = taccOcO(make_coord(0, _), _, 0);
    int num_valid = min(params.q_seq_per_hk - m_block * kBlockM, kBlockM);
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            int row = get<0>(taccOcO_row(mi));
            if (row < num_valid) gLSE_ptr[row] = lse(mi);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Kernel entry point
template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
flash_fwd_splitkv_mla_tc_v2_kernel(const DenseAttnDecodeParams params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    #pragma unroll 1
    for (int batch_id = sched_meta.begin_req_idx; batch_id <= sched_meta.end_req_idx; ++batch_id) {
        const int n_split_idx = batch_id == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_id);
        const int n_block_min = batch_id == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        const int n_block_max = batch_id == sched_meta.end_req_idx ? sched_meta.end_block_idx : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = batch_id == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_id == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        if (batch_id > sched_meta.begin_req_idx) __syncthreads();
        if (!NoSplit)
            flash::compute_attn_kernel_v2<Kernel_traits, Is_causal, true>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
        else
            flash::compute_attn_kernel_v2<Kernel_traits, Is_causal, false>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
    }
}

//////////////////////////////////////////////////////////////////////
// Host launch
template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    constexpr int kBlockM = 32, kBlockN = 64, kHeadDim = 576, kHeadDimV = 512, kNWarps = 8;
    using KernelTraits = KernelTraitsV2<kHeadDim, kBlockM, kBlockN, kNWarps>;
    using SharedStorage = SharedStorageV2<KernelTraits>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    const int num_m_block = (params.q_seq_per_hk + kBlockM - 1) / kBlockM;

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash_fwd_splitkv_mla_tc_v2_kernel<KernelTraits, Is_causal, SharedStorage>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<dim3(num_m_block, params.h_k, params.num_sm_parts), KernelTraits::kNThreads, smem_size, params.stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

} // namespace flash
} // namespace tc
} // namespace sm86
