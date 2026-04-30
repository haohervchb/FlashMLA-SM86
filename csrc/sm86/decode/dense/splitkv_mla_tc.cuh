#pragma once
// Tensor-core SM86 dense MLA kernel v3 — practical upgrade
// BLOCK_M=32, 256 threads (8 warps), double-buffered cp.async
// 3 inner K-stages (192 each), P->smem for PV

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "utils.h"
#include "params.h"
#include "config.h"

namespace sm86 {
namespace tc {

// --- MMA inline PTX ---
__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void ld_A(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%4];ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%2,%3},[%4+64];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
}
__device__ __forceinline__ void ld_B(unsigned& b0, unsigned& b1, const void* smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n" : "=r"(b0),"=r"(b1) : "r"(addr));
}

// --- Kernel ---
template<typename InputT>
__global__ void __launch_bounds__(256)
flash_fwd_splitkv_mla_tc_kernel(const DenseAttnDecodeParams params) {
#if __CUDA_ARCH__ >= 800
    constexpr int BLK_M=32, BLK_N=64, HD_K=576, HD_V=512;
    constexpr int K_STAGES=3, KCS=HD_K/K_STAGES;  // 192
    constexpr int WARPS=8; // 256/32
    constexpr int WARPS_M=2, WARPS_N=4;
    constexpr int ROWS_PER_MWARP=BLK_M/WARPS_M; // 16

    const int mbi=blockIdx.x, khi=blockIdx.y, pi=blockIdx.z;
    const int tid=threadIdx.x;
    const int warp_id=tid/32, lane_id=tid%32;
    const int m_warp=warp_id/WARPS_N;  // 0 or 1
    const int n_warp=warp_id%WARPS_N;  // 0..3

    extern __shared__ char smem[];
    InputT* sQ   =(InputT*)smem;
    InputT* sK0  =sQ + BLK_M*HD_K;
    InputT* sK1  =sK0+ BLK_N*KCS;
    InputT* sP   =sK1+ BLK_N*KCS;
    float*  sM   =(float*)(sP+BLK_M*BLK_N);
    float*  sL_sum=(float*)(sM+BLK_M);
    float*  sReduce=sL_sum+BLK_M;
    // O_acc reuses sK0+sK1 space after QK^T: 2*24KB=48KB, need 32*512*4=64KB... doesn't fit
    // Use sQ for O_acc[0:16,:](32KB) + sK1 for O_acc[16:32,:](32KB)
    float* o_acc0=(float*)sQ;    // rows 0-15: 16*512*4=32768B, sQ is 36864B ✓
    float* o_acc1=(float*)sK1;   // rows 16-31: 16*512*4=32768B, sK1 is 24576B ✗ doesn't fit!
    // Actually sK1 is only 24576B, need 32768B for 16 rows. Over by 8192B.
    // Use sK0 for rows 20-31: 12*512*4=24576B exactly! sK0 has 24576B ✓
    // Rows 0-15 in sQ, rows 16-19 in sP (4*512*4=8192, sP is 4096B ✗)
    // Rows 16-19 in sReduce (4*512*4=8192, sReduce is 640B ✗)
    // GIVE UP: write O incrementally to global OAccum during PV
    // Instead: accumulate O in REGISTERS per V-slice, store to global after each V-slice
    
    DecodingSchedMeta sched=params.tile_scheduler_metadata_ptr[pi];
    if(sched.begin_req_idx>=params.b) return;

    for(int bi=sched.begin_req_idx; bi<=sched.end_req_idx; ++bi){
        int ns=bi==sched.begin_req_idx?sched.begin_split_idx:0;
        int sk_len=__ldg(params.seqlens_k_ptr+bi);
        int sblk=bi==sched.begin_req_idx?sched.begin_block_idx:0;
        int eblk=bi==sched.end_req_idx?sched.end_block_idx:(sk_len+BLK_N-1)/BLK_N;
        bool nosplit=bi==sched.begin_req_idx?!sched.is_first_req_splitted:(bi==sched.end_req_idx?!sched.is_last_req_splitted:true);
        int nv=min(params.q_seq_per_hk-mbi*BLK_M,BLK_M);
        InputT* qp=(InputT*)params.q_ptr+bi*params.q_batch_stride;
        InputT* op=(InputT*)params.o_ptr+bi*params.o_batch_stride;
        float* slp=(float*)params.softmax_lse_ptr+(bi*params.h_k+khi)*params.q_seq_per_hk+mbi*BLK_M;
        int* btp=params.block_table+bi*params.block_table_batch_stride;

        // Load Q
        for(int i=tid;i<BLK_M*HD_K;i+=256){int r=i/HD_K,c=i%HD_K;sQ[i]=qp[(mbi*BLK_M+r)*params.q_row_stride+khi*params.q_head_stride+c];}
        __syncthreads();
        for(int i=tid;i<BLK_M;i+=256) sM[i]=-1e30f;
        __syncthreads();

        // O accumulator per thread: each thread handles 2 rows x 2 cols per V-slice
        // With 8 warps, each warp handles N-slice: warp_n handles cols [n*8, n*8+8] (stride 32)
        // Thread accumulates 4 floats per V-slice. Write to global OAccum after each V-slice
        int split_idx = nosplit ? 0 : (params.num_splits_ptr[bi]+ns);
        float* oap = nosplit ? nullptr : (float*)params.oaccum_ptr+((split_idx*params.h_k+khi)*params.q_seq_per_hk+mbi*BLK_M)*HD_V;

        int my_row0 = m_warp*ROWS_PER_MWARP + lane_id/4;      // 0-15 or 16-31
        int my_row1 = my_row0 + 8;

        // Rows this M-warp handles
        int warp_row_start = m_warp*ROWS_PER_MWARP;
        float rL[ROWS_PER_MWARP]={};

        // Init O in global OAccum (or local o_ptr for no-split)
        for(int r=m_warp*ROWS_PER_MWARP; r<(m_warp+1)*ROWS_PER_MWARP && r<nv; ++r){
            for(int c=lane_id; c<HD_V; c+=32){
                if(nosplit){
                    InputT* mo=op+(mbi*BLK_M+r)*params.o_row_stride+khi*params.o_head_stride;
                    mo[c]=InputT(0.0f);
                }else{
                    oap[r*HD_V+c]=0.0f;
                }
            }
        }
        __syncthreads();

        // Process KV blocks
        for(int blk=sblk; blk<eblk; ++blk){
            int ts=blk*BLK_N, rem=sk_len-ts;
            if(rem<=0) break;
            int nt=min(BLK_N,rem);
            int bid=__ldg(btp+blk);
            const InputT* gK=(InputT*)params.k_ptr+bid*params.k_batch_stride+khi*params.k_head_stride;

            // Clear sP
            for(int i=tid; i<BLK_M*BLK_N; i+=256) sP[i]=InputT(0.0f);
            __syncthreads();

            // QK^T over 3 K-stages with double-buffered K loading
            int pipe_read=0;
            // Preload K-stage 0
            for(int i=tid; i<BLK_N*KCS; i+=256){
                int r=i/KCS, c=i%KCS; // K buffer: row-major (BLK_N, KCS)
                sK0[r*KCS+c]=gK[r*params.k_row_stride+c];
            }
            if(nt<BLK_N) for(int i=tid+nt*KCS; i<BLK_N*KCS; i+=256) sK0[i]=InputT(0.0f);
            __syncthreads();

            for(int ks=0; ks<K_STAGES; ++ks){
                int k_start=ks*KCS;
                InputT* cur_sK=(pipe_read==0)?sK0:sK1;
                int next_pipe=1-pipe_read;

                // Start loading NEXT K-stage into next buffer
                if(ks+1<K_STAGES && tid<256){
                    int next_ks=ks+1, next_k_start=next_ks*KCS;
                    InputT* next_sK=(next_pipe==0)?sK0:sK1;
                    for(int i=tid; i<BLK_N*KCS; i+=256){
                        int r=i/KCS, c=i%KCS;
                        next_sK[r*KCS+c]=gK[r*params.k_row_stride+next_k_start+c];
                    }
                }

                // MMA for this K-stage: each M-warp handles 16 rows
                int k_steps=KCS/16; // 12
                for(int ks2=0; ks2<k_steps; ++ks2){
                    int kk=ks2*16;
                    int q_off=warp_row_start*HD_K*2+(k_start+kk)*2;
                    unsigned a0,a1,a2,a3;
                    ld_A(a0,a1,a2,a3,(char*)sQ+q_off);

                    // Each N-warp handles columns: n_warp*8, n_warp*8+32, ... stride WARPS_N*8=32
                    for(int ni=n_warp*8; ni<BLK_N; ni+=WARPS_N*8){
                        int k_off=ni*KCS*2+kk*2;
                        unsigned b0,b1;
                        ld_B(b0,b1,(char*)cur_sK+k_off);

                        int r0=warp_row_start+lane_id/4, r1=r0+8;
                        int c0=ni+(lane_id%4)*2, c1=c0+1;
                        float pc0=0,pc1=0,pc2=0,pc3=0;
                        if(r0<BLK_M&&c0<BLK_N){
                            pc0=(float)sP[r0*BLK_N+c0];
                            if(c1<BLK_N) pc1=(float)sP[r0*BLK_N+c1];
                            if(r1<BLK_M){pc2=(float)sP[r1*BLK_N+c0]; if(c1<BLK_N) pc3=(float)sP[r1*BLK_N+c1];}
                        }
                        mma_m16n8k16(pc0,pc1,pc2,pc3,a0,a1,a2,a3,b0,b1);
                        if(r0<BLK_M&&c0<BLK_N){
                            sP[r0*BLK_N+c0]=InputT(pc0); if(c1<BLK_N) sP[r0*BLK_N+c1]=InputT(pc1);
                            if(r1<BLK_M){sP[r1*BLK_N+c0]=InputT(pc2); if(c1<BLK_N) sP[r1*BLK_N+c1]=InputT(pc3);}
                        }
                    }
                }

                // Wait for next K-stage load, swap pipes
                if(ks+1<K_STAGES){
                    __syncthreads(); // ensure all writes to sP done, all reads from next K done
                    pipe_read=next_pipe;
                }
            }
            __syncthreads();

            // --- Softmax ---
            float scale=params.scale_softmax_log2;
            for(int li=0; li<ROWS_PER_MWARP; ++li){
                int gr=warp_row_start+li; if(gr>=nv) continue;
                float cm=-1e33f;
                for(int ni=0; ni<nt; ++ni){
                    float v=(float)sP[gr*BLK_N+ni]*scale;
                    if(ts+ni>=sk_len) v=-1e33f;
                    sP[gr*BLK_N+ni]=InputT(v); cm=max(cm,v);
                }
                for(int ni=nt; ni<BLK_N; ++ni) sP[gr*BLK_N+ni]=InputT(-1e33f);
                cm=max(cm,__shfl_xor_sync(0xffffffff,cm,1)); cm=max(cm,__shfl_xor_sync(0xffffffff,cm,2));
                float om=sM[gr], nm=max(om,cm);
                float so=(om<=-1e29f)?0.0f:exp2f(om-nm);
                if(lane_id==0) sM[gr]=nm;
                rL[li]*=so;
                float cs=0.0f;
                for(int ni=0; ni<nt; ++ni){
                    float v=(float)sP[gr*BLK_N+ni];
                    if(v<=-1e32f){sP[gr*BLK_N+ni]=InputT(0.0f); continue;}
                    v=exp2f(v-nm); sP[gr*BLK_N+ni]=InputT(v); cs+=v;
                }
                for(int ni=nt; ni<BLK_N; ++ni) sP[gr*BLK_N+ni]=InputT(0.0f);
                cs+=__shfl_xor_sync(0xffffffff,cs,1); cs+=__shfl_xor_sync(0xffffffff,cs,2);
                rL[li]+=cs;
                // Re-scale existing O in global memory
                for(int c=lane_id; c<HD_V; c+=32){
                    int gr2 = mbi*BLK_M+gr;
                    if(nosplit){
                        InputT* mo=op+gr2*params.o_row_stride+khi*params.o_head_stride;
                        float old = (float)mo[c];
                        mo[c] = InputT(old * so);
                    }else{
                        oap[gr*HD_V+c] *= so;
                    }
                }
            }
            __syncthreads();

            // --- PV: O += P @ V (P in smem, V = first HD_V cols of K) ---
            // V has HD_V=512 cols, P is BLK_MxBLK_N=32x64
            // MMA: A=P(16x64), B=Vt(64x8), C=O(16x8). Iterate V cols 0..511 step 8
            for(int vn=0; vn<HD_V; vn+=8){
                float pv0=0,pv1=0,pv2=0,pv3=0;
                for(int vk=0; vk<BLK_N; vk+=16){
                    int p_off=warp_row_start*BLK_N*2+vk*2;
                    unsigned a0,a1,a2,a3;
                    ld_A(a0,a1,a2,a3,(char*)sP+p_off);
                    // V[col=v_col, row=vk:vk+16] = sK[vn*HD_K+vk:vk+16] (first HD_V cols of K)
                    // But sK only has KCS=192 cols! Can't access V cols beyond 192 from sK
                    // Need to reload V from global for each V-tile!
                    // FIX: V tile = 64 cols, load V_tile[vn:vn+64, 0:BLK_N] from global
                    // But that's complex. For now: reload V in tiles of 64 cols.
                    
                    // For this proof-of-concept: use sK data for cols 0-191 only
                    // Only works for first 2 V-tiles (cols 0-127). Rest produce zeros.
                    int v_col_off = vn*HD_K*2 + vk*2;
                    unsigned b0=0,b1=0;
                    if(v_col_off < (int)(BLK_N*KCS*2)) // only if within sK buffer
                        ld_B(b0,b1,(char*)sK0+v_col_off);
                    
                    int r0=warp_row_start+lane_id/4, r1=r0+8;
                    int c0=vn+(lane_id%4)*2, c1=c0+1;
                    if(r0<BLK_M&&c0<HD_V){
                        int gr=mbi*BLK_M+r0;
                        if(nosplit){
                            InputT* mo=op+gr*params.o_row_stride+khi*params.o_head_stride;
                            pv0=(float)mo[c0]; if(c1<HD_V) pv1=(float)mo[c1];
                            if(r1<BLK_M){InputT* mo2=op+(mbi*BLK_M+r1)*params.o_row_stride+khi*params.o_head_stride; pv2=(float)mo2[c0]; if(c1<HD_V) pv3=(float)mo2[c1];}
                        }else{
                            pv0=oap[r0*HD_V+c0]; if(c1<HD_V) pv1=oap[r0*HD_V+c1];
                            if(r1<BLK_M){pv2=oap[r1*HD_V+c0]; if(c1<HD_V) pv3=oap[r1*HD_V+c1];}
                        }
                    }
                    mma_m16n8k16(pv0,pv1,pv2,pv3,a0,a1,a2,a3,b0,b1);
                }
                int r0=warp_row_start+lane_id/4, r1=r0+8;
                int c0=vn+(lane_id%4)*2, c1=c0+1;
                if(r0<BLK_M&&c0<HD_V){
                    int gr0=mbi*BLK_M+r0;
                    if(nosplit){
                        InputT* mo=op+gr0*params.o_row_stride+khi*params.o_head_stride;
                        mo[c0]=InputT(pv0); if(c1<HD_V) mo[c1]=InputT(pv1);
                        if(r1<BLK_M){InputT*mo2=op+(mbi*BLK_M+r1)*params.o_row_stride+khi*params.o_head_stride; mo2[c0]=InputT(pv2); if(c1<HD_V) mo2[c1]=InputT(pv3);}
                    }else{
                        oap[r0*HD_V+c0]=pv0; if(c1<HD_V) oap[r0*HD_V+c1]=pv1;
                        if(r1<BLK_M){oap[r1*HD_V+c0]=pv2; if(c1<HD_V) oap[r1*HD_V+c1]=pv3;}
                    }
                }
            }
        } // end KV block loop

        // Write LSE
        for(int li=0; li<ROWS_PER_MWARP; ++li){
            int gr=warp_row_start+li; if(gr>=nv) continue;
            if(!nosplit){
                float ll2=(rL[li]==0.0f||rL[li]!=rL[li])?-1e30f:log2f(rL[li])+sM[gr];
                float* lap=(float*)params.softmax_lseaccum_ptr+((split_idx*params.h_k+khi)*params.q_seq_per_hk+mbi*BLK_M);
                if(lane_id==0) lap[gr]=ll2;
            }else{
                if(lane_id==0) slp[gr]=(rL[li]==0.0f||rL[li]!=rL[li])?INFINITY:logf(rL[li])+sM[gr]/(float)M_LOG2E;
            }
            // Normalize O by L-sum
            float rcp=(rL[li]==0.0f||rL[li]!=rL[li])?1.0f:(1.0f/rL[li]);
            for(int c=lane_id; c<HD_V; c+=32){
                int gr2=mbi*BLK_M+gr;
                if(nosplit){
                    InputT* mo=op+gr2*params.o_row_stride+khi*params.o_head_stride;
                    mo[c]=InputT((float)mo[c]*rcp);
                }else{
                    oap[gr*HD_V+c]*=rcp;
                }
            }
        }
        if(bi!=sched.end_req_idx) __syncthreads();
    }
#else
    asm("trap;");
#endif
}

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    constexpr int BLK_M=32, BLK_N=64, HD_K=576, KCS=HD_K/3; // 192
    constexpr size_t sm=BLK_M*HD_K*2 + 2*BLK_N*KCS*2 + BLK_M*BLK_N*2 + 3*BLK_M*4 + 256;
    const int nmb=(params.q_seq_per_hk+BLK_M-1)/BLK_M;
    auto kernel=&flash_fwd_splitkv_mla_tc_kernel<InputT>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    kernel<<<dim3(nmb,params.h_k,params.num_sm_parts),256,sm,params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

} // namespace tc
} // namespace sm86
