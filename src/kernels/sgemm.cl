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


#ifndef ATRANS
#define ATRANS 0
#endif

#ifndef BTRANS
#define BTRANS 0
#endif

#ifndef BIAS
#define BIAS 0
#endif


#define BLOCK_SIZE_NY (BLOCK_SIZE_N*BLOCK_SIZE_M)
#define BLOCKS_IN_TILE_N (TILE_SIZE_N / BLOCK_SIZE_N)
#define BLOCKS_IN_TILE_M (TILE_SIZE_M / BLOCK_SIZE_M)
#define WG_SIZE (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)


#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))

#ifndef CONVGEMM
#define CONVGEMM 0
#endif

#if CONVGEMM == 0

#  if BTRANS == 0
#    define get_B(r,c) (B[(r)*ldb + (c)])
#  else
#    define get_B(r,c) (B[(c)*ldb + (r)])
#  endif
#else
float get_img_value(__global float const *ptr,int matrix_row,int matrix_col)
{
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

    if(x >= 0 && y >= 0 && x < SRC_COLS && y < SRC_ROWS) {
        return ptr[((b *  CHANNELS_IN + channel) * SRC_ROWS + y) * SRC_COLS + x];
    }
    return 0;
}

#  if BTRANS == 0
#    define get_B(r,c) get_img_value(B,r,c)
#  else
#    define get_B(r,c) get_img_value(B,c,r)
#  endif
#endif

#if  ATRANS == 0
#define get_A(r,c) (A[(r)*lda + (c)])
#else
#define get_A(r,c) (A[(c)*lda + (r)])
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





__kernel 
#if INTEL_PLATFORM == 1
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N, 1)))
void    sgemm(    int M,int N,int K,
        __global const float * restrict A,int offset_A,int lda,
        __global const float * restrict B,int offset_B,int ldb,
        __global float * restrict C,int offset_C,int ldc
#if BIAS != 0
        , __global const float * restrict bias,int offset_bias
#endif
        )
{
    A += offset_A;
    B += offset_B;
    C += offset_C;

   
    int row = get_global_id(0) * BLOCK_SIZE_M;
    int col = get_global_id(1) * BLOCK_SIZE_N;

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    
    int local_tile_id = lid0 * get_local_size(1) + lid1;

    int tile_row0 = get_group_id(0)*TILE_SIZE_M;
    int tile_col0 = get_group_id(1)*TILE_SIZE_N;

    const int local_wg_size = BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N;
    const int load_step = TILE_SIZE_M * TILE_SIZE_K / local_wg_size;

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N] = {{0.0f}};

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

#if INTEL_PLATFORM == 0
 
    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {

        #ifdef SIM
        if(k==0) {
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_M;
                int tile_tdir = read_pos % TILE_SIZE_M;
                lA(tile_kdir,tile_tdir) = 1.3f;
            }
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_N;
                int tile_tdir = read_pos % TILE_SIZE_N;
                lB(tile_kdir,tile_tdir) = 2.3f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        //#elif 0
        #elif (TILE_SIZE_M == 32  && TILE_SIZE_N == 32  && BLOCK_SIZE_M==4 && BLOCK_SIZE_N == 4 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64)) \
          ||  (TILE_SIZE_M == 64  && TILE_SIZE_N == 64  && BLOCK_SIZE_M==8 && BLOCK_SIZE_N == 8 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64)) \
          ||  (TILE_SIZE_M == 128 && TILE_SIZE_N == 128 && BLOCK_SIZE_M==8 && BLOCK_SIZE_N == 8 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64))
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
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_M;
                int tile_tdir = read_pos % TILE_SIZE_M;
                int a_row = tile_tdir + tile_row0;
                int k_rc  = tile_kdir + k;
                lA(tile_kdir,tile_tdir) = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
            }
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_N;
                int tile_tdir = read_pos % TILE_SIZE_N;
                int k_rc  = tile_kdir + k;
                int b_col = tile_tdir + tile_col0;
                lB(tile_kdir,tile_tdir) = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;
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
    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {
        if(row + BLOCK_SIZE_M - 1 < M && k + TILE_SIZE_K-1 < K) {
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

#if IM2COL_OCHAN > 0
    {
        #pragma unroll
        for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
            if(col + dc >= N)
                continue;
            int matrix_col = col + dc;
            int batch = matrix_col / IM2COL_OCHAN;
            int incol = matrix_col % IM2COL_OCHAN;
            int offset = batch * IM2COL_OCHAN * M + incol;
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                if(row+dr < M) {
                    C[(row + dr)*ldc + offset] = ACTIVATION_F(c[dr][dc]);
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
                    C[(row+dr)*ldc+col+dc] = ACTIVATION_F(c[dr][dc]);
                }
            }
        }
    }
#endif
}


