///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include "defs.h"
#ifndef TILE_SIZE_M
#define TILE_SIZE_M 128
#endif
#ifndef TILE_SIZE_N
#define TILE_SIZE_N 128
#endif
#ifndef BLOCK_SIZE_N
#define BLOCK_SIZE_N 8
#endif
#ifndef BLOCK_SIZE_M
#define BLOCK_SIZE_M 8
#endif

#ifndef TILE_SIZE_K
#define TILE_SIZE_K 16
#endif 

#ifndef TILE_OFFSET
#define TILE_OFFSET 1
#endif

#ifndef ZORDER
#define ZORDER 0
#endif


#ifndef ATRANS
#define ATRANS 0
#endif

#ifndef BTRANS
#define BTRANS 0
#endif

#ifndef BIAS
#define BIAS 0
#endif

#ifndef GROUPS
#define GROUPS 1
#endif

#define BLOCK_SIZE_NY (BLOCK_SIZE_N*BLOCK_SIZE_M)
#define BLOCKS_IN_TILE_N (TILE_SIZE_N / BLOCK_SIZE_N)
#define BLOCKS_IN_TILE_M (TILE_SIZE_M / BLOCK_SIZE_M)
#define WG_SIZE (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)


#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))

#ifndef CONVGEMM
#define CONVGEMM 0
#endif

#if CONVGEMM == 3 || REDUCE_K > 1
#include "atomic.h"
#if ACTIVATION != ACTIVATION_IDENTITY
# error "Can't use activation with atomic ops"
#endif
#endif

#if CONVGEMM != 0
    #if CONVGEMM == 3
    void add_img_value(__global float *ptr,int matrix_row,int matrix_col,float dV)
    #else
    float get_img_value(__global float const *ptr,int matrix_row,int matrix_col)
    #endif
    {
#if KERN_W == 1 && KERN_H == 1 && STRIDE_H == 1 && STRIDE_W == 1 && DILATE_H == 1 && DILATE_W == 1 \
        && PAD_H == 0 && PAD_W == 0 && GROUPS == 1 && IMG_COLS == SRC_COLS && IMG_ROWS == SRC_ROWS 
        int channel = matrix_col;
        int b  = matrix_row / (IMG_COLS * IMG_ROWS);
        int rc = matrix_row % (IMG_COLS * IMG_ROWS);

        int address = (b * CHANNELS_IN + channel) * (SRC_ROWS * SRC_COLS) + rc;
        #if CONVGEMM != 3
            return ptr[address];
        #else
            #if REDUCE_K > 1
                atomic_addf(ptr+address,dV);
            #else
                ptr[address] += dV;
            #endif
        #endif
#else
        int channel = matrix_col / (KERN_H * KERN_W);
        int k_index = matrix_col % (KERN_H * KERN_W);
        
        int dy = k_index / KERN_W;
        int dx = k_index % KERN_W;
        
        int b  = matrix_row / (IMG_COLS * IMG_ROWS);
        int rc = matrix_row % (IMG_COLS * IMG_ROWS);

        int r  = rc / IMG_COLS;
        int c  = rc % IMG_COLS;

        int y_pos = -PAD_H + r * STRIDE_H;
        int x_pos = -PAD_W + c * STRIDE_W;

        int y = y_pos + dy * DILATE_H;
        int x = x_pos + dx * DILATE_W;

        int address = ((b *  (CHANNELS_IN*GROUPS) + channel) * SRC_ROWS + y) * SRC_COLS + x;

        #if CONVGEMM != 3
            #if PAD_W > 0 || PAD_H > 0
                if(x >= 0 && y >= 0 && x < SRC_COLS && y < SRC_ROWS) {
                    return ptr[address];
                }
                return 0;
            #else
                return ptr[address];
            #endif  
        #else
            #if PAD_W > 0 || PAD_H > 0
                if(x >= 0 && y >= 0 && x < SRC_COLS && y < SRC_ROWS) 
                    atomic_addf(ptr+address,dV);
            #else
                atomic_addf(ptr+address,dV);
            #endif
        #endif
#endif  
    }
#endif



#if CONVGEMM == 0 || CONVGEMM == 3
#  if BTRANS == 0
#    define get_B(r,c) (B[(r)*ldb + (c)])
#  else
#    define get_B(r,c) (B[(c)*ldb + (r)])
#  endif
#else
#  if BTRANS == 0
#    define get_B(r,c) get_img_value(B,r,c)
#  else
#    define get_B(r,c) get_img_value(B,c,r)
#  endif
#endif

#if CONVGEMM  == 0 || CONVGEMM == 1
    #if  ATRANS == 0
        #define get_A(r,c) (A[(r)*lda + (c)])
    #else
        #define get_A(r,c) (A[(c)*lda + (r)])
    #endif
#else
    float get_y_value(int row,int matrix_col,__global float const *A,int ldc,int M)
    {
        int batch = matrix_col / IM2COL_OCHAN;
        int incol = matrix_col % IM2COL_OCHAN;
        int offset = batch * (IM2COL_OCHAN * GROUPS) * M + incol;
        int index =row*IM2COL_OCHAN + offset;
        return A[index];
    }

    #if CONVGEMM == 3
        #define GET_Y_STEP K_src
    #else
        #define GET_Y_STEP M
    #endif
    #if  ATRANS == 0
        #define get_A(r,c) (get_y_value(r,c,A,lda,GET_Y_STEP))
    #else
        #define get_A(r,c) (get_y_value(c,r,A,lda,GET_Y_STEP))
    #endif

#endif

#define lA(x,y) a_tile[(x)][(y) / BLOCK_SIZE_M][(y) % BLOCK_SIZE_M]
#define lB(x,y) b_tile[(x)][(y) / BLOCK_SIZE_N][(y) % BLOCK_SIZE_N]


#if TILE_SIZE_M != TILE_SIZE_N
#error "Unsupported condif"
#endif

#if defined(cl_intel_subgroups)
#define INTEL_PLATFORM 1
#else
#define INTEL_PLATFORM 0
#endif

#define vload1(off,addr) ((addr)[off])
#define vstore1(val,off,addr) ((addr)[off]=(val))

#if BLOCK_SIZE_M == 1
#define vloadM vload1
#define vstoreM vstore1
#define floatM float
#elif BLOCK_SIZE_M == 4
#define vloadM vload4
#define vstoreM vstore4
#define floatM float4
#elif BLOCK_SIZE_M == 8
#define vloadM vload8
#define vstoreM vstore8
#define floatM float8
#elif BLOCK_SIZE_M == 16 
#define vloadM vload16
#define vstoreM vstore16
#define floatM float16
#endif

#if BLOCK_SIZE_N == 1
#define vloadN vload1
#define vstoreN vstore1
#define floatN float
#elif BLOCK_SIZE_N == 4
#define vloadN vload4
#define vstoreN vstore4
#define floatN float4
#elif BLOCK_SIZE_N == 8
#define vloadN vload8
#define vstoreN vstore8
#define floatN float8
#elif BLOCK_SIZE_N == 16 
#define vloadN vload16
#define vstoreN vstore16
#define floatN float16
#endif

#if TILE_SIZE_K == 1
#define vloadK vload1
#define vstoreK vstore1
#define floatK float
#elif TILE_SIZE_K == 4
#define vloadK vload4
#define vstoreK vstore4
#define floatK float4
#elif TILE_SIZE_K == 8
#define floatK float8
#define vloadK vload8
#define vstoreK vstore8
#elif TILE_SIZE_K == 16 
#define vloadK vload16
#define vstoreK vstore16
#define floatK float16
#endif


#ifndef BATCH_GEMM
#define BATCH_GEMM 0
#endif

#ifndef REDUCE_K
#define REDUCE_K 1
#endif

#if GROUPS == 1 && REDUCE_K == 1 && BATCH_GEMM == 0
#define DIM_M 0
#define DIM_N 1
#define DIM_G 2
#define EXTRA_DIM 0
#else
#define DIM_M 1
#define DIM_N 2
#define DIM_G 0
#define EXTRA_DIM 1
#endif

int zorder_a(int x)
{
    return
          ((x & (1<<0)) >> 0 )
        | ((x & (1<<2)) >> 1 )
        | ((x & (1<<4)) >> 2 )
        | ((x & (1<<6)) >> 3 )
        | ((x & (1<<8)) >> 4 )
        | ((x & (1<<10)) >> 5 )
        | ((x & (1<<12)) >> 6 )
        | ((x & (1<<14)) >> 7 )
        | ((x & (1<<16)) >> 8 )
        | ((x & (1<<18)) >> 9 )
        | ((x & (1<<20)) >> 10 )
        | ((x & (1<<22)) >> 11 )
        | ((x & (1<<24)) >> 12 );
}
int zorder_b(int x)
{
    return zorder_a(x>>1);
}


__kernel 
#if INTEL_PLATFORM == 1
__attribute__((intel_reqd_sub_group_size(8)))
#endif
#if EXTRA_DIM == 0
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N, 1)))
#else
__attribute__((reqd_work_group_size(1,BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N)))
#endif
void    sgemm(    
#if BATCH_GEMM == 1
        int batches,
#endif        
        int M,int N,int K,
        __global const float * restrict A,ulong offset_A,
#if BATCH_GEMM == 1
        int batch_stride_a,
#endif                
        int lda,
        __global const float * restrict B,ulong offset_B,
#if BATCH_GEMM == 1
        int batch_stride_b,
#endif                
        int ldb,
        __global float * restrict C,ulong offset_C,
#if BATCH_GEMM == 1
        int batch_stride_c,
#endif                
        int ldc,
        float beta_factor
#if BIAS != 0
        , __global const float * restrict bias,ulong offset_bias
#endif
        )
{
    A += offset_A;
    B += offset_B;
    C += offset_C;
#if BATCH_GEMM == 1 
    int batch_id = get_global_id(DIM_G);
    if(batch_id >= batches)
        return;
    A += batch_stride_a * batch_id;
    B += batch_stride_b * batch_id;
    C += batch_stride_c * batch_id;
#endif    

#if CONVGEMM > 0 && GROUPS > 1
    if(get_global_id(DIM_G) >= REDUCE_K * GROUPS)
        return;
    int group = get_global_id(DIM_G) / REDUCE_K;
    #if CONVGEMM == 1
        A += M*K*group;
        B += SRC_COLS*SRC_ROWS*CHANNELS_IN*group;
        C += (IM2COL_OCHAN) * M *group;
        #if BIAS != 0
        bias += M*group;
        #endif
    #elif CONVGEMM == 2
        // M = channels_out / groups
        int step_g_y = (M*IM2COL_OCHAN);
        int step_g_x = (SRC_COLS*SRC_ROWS) * CHANNELS_IN;
        int step_g_w = M*(CHANNELS_IN*KERN_W*KERN_H);
        
        A += step_g_y * group;
        B += step_g_x * group;
        C += step_g_w * group;
    #elif CONVGEMM == 3
        // K = channels_out / group
        int step_g_y = (K*IM2COL_OCHAN);
        int step_g_x = (SRC_COLS*SRC_ROWS) * CHANNELS_IN;
        int step_g_w = K*(CHANNELS_IN*KERN_W*KERN_H);
        
        A += step_g_y * group;
        B += step_g_w * group;
        C += step_g_x * group;
    #else
    #error "Invalid CONVGEMM Value"
    #endif
#endif   

#if ZORDER == 1
    int gr_m = get_group_id(DIM_M);
    int gr_n = get_group_id(DIM_N);
    int gr_size_m = get_num_groups(DIM_M);
    int gr_size_n = get_num_groups(DIM_N);
    if(gr_size_m == gr_size_n && popcount(gr_size_m) == 1) {
        int grs  = gr_n * gr_size_m + gr_m;
        gr_n = zorder_a(grs);
        gr_m = zorder_b(grs);
    }
#else
    int gr_m = get_group_id(DIM_M);
    int gr_n = get_group_id(DIM_N);
#endif
    int tile_row0 = gr_m*TILE_SIZE_M;
    int tile_col0 = gr_n*TILE_SIZE_N;

#if ZORDER == 1
    if(tile_row0 >= M || tile_col0 >= N)
        return;
#endif        

    int row = tile_row0 + get_local_id(DIM_M) * BLOCK_SIZE_M;
    int col = tile_col0 + get_local_id(DIM_N) * BLOCK_SIZE_N;


    int lid0 = get_local_id(DIM_M);
    int lid1 = get_local_id(DIM_N);
    
    int local_tile_id = lid0 * get_local_size(DIM_N) + lid1;

    #define local_wg_size (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)
    #define load_step (TILE_SIZE_M * TILE_SIZE_K / local_wg_size)

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N] = {{0.0f}};
    
    int K_src = K;

#if INTEL_PLATFORM == 0
    float ap[BLOCK_SIZE_M];
    float bp[BLOCK_SIZE_N];

    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][BLOCKS_IN_TILE_M][BLOCK_SIZE_M+TILE_OFFSET];
    ALIGN_FLOAT4 __local float b_tile[TILE_SIZE_K][BLOCKS_IN_TILE_N][BLOCK_SIZE_N+TILE_OFFSET];
#else
    #if ATRANS == 1
    float a[TILE_SIZE_K][BLOCK_SIZE_M];
    #define pA(ind1,ind2) (a[(ind2)][(ind1)])
    #else
    float a[BLOCK_SIZE_M][TILE_SIZE_K];
    #define pA(ind1,ind2) (a[(ind1)][(ind2)])
    #endif
#endif    

#if REDUCE_K > 1
    int KS = (K + REDUCE_K - 1) / REDUCE_K;
    int sec = get_global_id(DIM_G) % REDUCE_K;
    int k_start=KS * sec;
    K = min(K_src,KS * (sec + 1));
    int k = k_start;
#else
    int k=0;
#endif

#if TILE_SIZE_N == TILE_SIZE_M && TILE_SIZE_K % load_step  == 0 && load_step <= TILE_SIZE_K
#define LOAD_VARIANT 0
#else
#define LOAD_VARIANT 1
#endif

#if INTEL_PLATFORM == 0

    #if LOAD_VARIANT == 1
    int dM[load_step];
    int dN[load_step];
    int dK [load_step];
    __local float *aP[load_step];
    __local float *bP[load_step];

    for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
        int tile_kdir = read_pos / TILE_SIZE_M;
        int tile_tdir = read_pos % TILE_SIZE_M;
        dM[i] = tile_tdir + tile_row0;
        dN[i] = tile_tdir + tile_col0;
        dK[i]  = tile_kdir;
        aP[i] = &lA(tile_kdir,tile_tdir);
        bP[i] = &lB(tile_kdir,tile_tdir);
    }
    #endif


 
    for(;k<K;k+=TILE_SIZE_K) {


        #if LOAD_VARIANT == 0
        {
            int tile_kdir0 = local_tile_id / TILE_SIZE_M;
            int tile_tdir  = local_tile_id % TILE_SIZE_M;
            int a_row = tile_tdir + tile_row0;
            int b_col = tile_tdir + tile_col0;

            if(a_row >= M) {
                #pragma unroll
                for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lA(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_M)) {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        int k_rc  = tile_kdir + k;
                        lA(tile_kdir,tile_tdir) = get_A(a_row,k_rc);
                    }
                }
                else {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        int k_rc  = tile_kdir + k;
                        lA(tile_kdir,tile_tdir) = k_rc < K ? get_A(a_row,k_rc) : 0.0f;
                    }
                }
            }
            if(b_col >= N) {
                #pragma unroll
                for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lB(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_N)) {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        int k_rc  = tile_kdir + k;
                        lB(tile_kdir,tile_tdir) = get_B(k_rc,b_col);
                    }
                }
                else {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        int k_rc  = tile_kdir + k;
                        lB(tile_kdir,tile_tdir) = k_rc < K ? get_B(k_rc,b_col) : 0.0f;
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #else
        {
            if(tile_row0 + TILE_SIZE_M <= M && k + TILE_SIZE_K <= K) {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int a_row = dM[i];
                    int k_rc  = dK[i] + k;
                    *aP[i] =  get_A(a_row,k_rc);
                }
            }
            else {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int a_row = dM[i];
                    int k_rc  = dK[i] + k;
                    *aP[i] = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
                }
            }
            if(tile_col0 + TILE_SIZE_N <= N && k + TILE_SIZE_K <= K) {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int k_rc  = dK[i]  + k;
                    int b_col = dN[i];
                    *bP[i] = get_B(k_rc,b_col);
                }
            }
            else {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int k_rc  = dK[i]  + k;
                    int b_col = dN[i];
                    *bP[i] = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;

                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #endif

        // Mutliplication loop
        #pragma unroll(4)
        for(int dk=0;dk<TILE_SIZE_K;dk++) {
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                ap[dr] = a_tile[dk][lid0][dr];
            }
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                bp[dc] = b_tile[dk][lid1][dc];
            }
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                    c[dr][dc] = mad(ap[dr],bp[dc],c[dr][dc]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
#else // INTEL_PLATFORM == 1
    // for intel we don't use local memory
    // we use optimized loads from global memory for A
    // and intel_sub_group_shuffle for optimal loading B
    for(;k<K;k+=TILE_SIZE_K) {
        if(row + BLOCK_SIZE_M - 1 < M && k + TILE_SIZE_K-1 < K) {
            #if CONVGEMM == 2 || CONVGEMM == 3
                #pragma unroll
                for(int dr=0;dr<BLOCK_SIZE_M;dr++){
                    for(int dk=0;dk < TILE_SIZE_K;dk++) {
                        pA(dr,dk)=get_A(row+dr,k+dk);
                    }
                }
            #else
                #if ATRANS == 0
                    #pragma unroll
                    for(int dr=0;dr<BLOCK_SIZE_M;dr++){
                        floatK v=vloadK(0,&get_A(row+dr,k));
                        vstoreK(v,0,a[dr]);
                    }
                #else // ATRANS
                    #pragma unroll
                    for(int dk=0;dk<TILE_SIZE_K;dk++){
                        floatM v=vloadM(0,&get_A(row,k+dk));
                        vstoreM(v,0,a[dk]);
                    }
                #endif
            #endif
        }
        else {
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++){
                #pragma unroll
                for(int dk=0;dk < TILE_SIZE_K;dk++) {
                    pA(dr,dk) = (row + dr < M && k+dk < K) ? get_A(row+dr,k+dk): 0;
                }
            }
        }

        #pragma unroll(TILE_SIZE_K)
        for(int dk=0;dk<TILE_SIZE_K;dk++) {
            if(k + dk >= K)
                continue;
            #if BLOCK_SIZE_N == 8
                int mycol = col + get_sub_group_local_id();
                float myv = (mycol < N) ? get_B(k+dk,col + get_sub_group_local_id()) : 0;
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_N;dc++){
                    float b_dc = intel_sub_group_shuffle(myv,dc);
                    for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                        c[dr][dc] = mad(pA(dr,dk),b_dc,c[dr][dc]);
                    }
                }
            #else
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_N;dc++){
                    float b_dc = (col + dc < N) ? get_B(k+dk,col+dc) : 0;
                    for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                        c[dr][dc] = mad(pA(dr,dk),b_dc,c[dr][dc]);
                    }
                }
            #endif
        }
    }
#endif // INTEL_PLATFORM = 1


#if BIAS != 0
    bias += offset_bias;
#endif

#if BIAS == 1
    #if REDUCE_K > 1
    if(k_start == 0)
    #endif
    {
        float offset;
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            offset = row + dr < M ? bias[(row+dr)] : 0.0f;
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                c[dr][dc] += offset;
            }
        }
    }
#elif BIAS == 2
    #if REDUCE_K > 1
    if(k_start == 0)
    #endif
    {
        float offset;
        #pragma unroll
        for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
            offset = (col + dc) < N ? bias[(col+dc)] : 0.0f;
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                c[dr][dc] += offset;
            }
        }
    }
#endif    

#if CONVGEMM == 1
    {
        #pragma unroll
        for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
            if(col + dc >= N)
                continue;
            int matrix_col = col + dc;
            int batch = matrix_col / IM2COL_OCHAN;
            int incol = matrix_col % IM2COL_OCHAN;
            int offset = batch * (IM2COL_OCHAN * GROUPS) * M + incol;
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                if(row+dr < M) {
                    int index =(row + dr)*ldc + offset;
                    #if REDUCE_K > 1
                    atomic_addf(C+index,c[dr][dc]);
                    #else
                    if(beta_factor != 0)
                        C[index] = mad(C[index], beta_factor,ACTIVATION_F(c[dr][dc]));
                    else
                        C[index] = ACTIVATION_F(c[dr][dc]);
                    #endif
                }
            }
        }
    }
#else
    {
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                if(row + dr < M && col+dc < N) {
                    #if CONVGEMM == 3
                        add_img_value(C,row+dr,col+dc,c[dr][dc]);
                    #else
                        int index = (row+dr)*ldc+col+dc;
                        #if REDUCE_K > 1
                        atomic_addf(C+index,ACTIVATION_F(c[dr][dc]));
                        #else
                        if(beta_factor != 0)
                            C[index] = mad(C[index], beta_factor,ACTIVATION_F(c[dr][dc]));
                        else
                            C[index] = ACTIVATION_F(c[dr][dc]);
                        #endif
                    #endif
                }
            }
        }
    }
#endif
}


