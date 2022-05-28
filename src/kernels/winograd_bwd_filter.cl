///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "atomic.h"

/***
 * Implementation concept is based on the
 *
 * Yan, Da, Wei Wang, and Xiaowen Chu.
 * "Optimizing batched winograd convolution on GPUs."
 * Proceedings of the 25th ACM SIGPLAN symposium on
 * principles and practice of parallel programming. 2020.
 *
 * @inproceedings{yan2020optimizing,
 *      title={Optimizing batched winograd convolution on GPUs},
 *      author={Yan, Da and Wang, Wei and Chu, Xiaowen},
 *      booktitle={Proceedings of the 25th ACM SIGPLAN symposium on principles and practice of parallel programming},
 *      pages={32--44},
 *      year={2020}
 * }
 *
 *
 * https://www.cse.ust.hk/~weiwa/papers/yan-ppopp20.pdf
 *
 * In comparison to the paper: using 32x8 tiles block instead of 64x8
 * since it generic OpenCL implementation for different GPUs
 * that can't optimize registers as efficienlty as manually written
 * assembly
 */

float16 tile2x2_to_4x4(float4 v)
{
    //[ 1.  0.]
    //[ 1.  1.]
    //[ 1. -1.]
    //[ 0. -1.]

    float y[2][2] = { { v.s0, v.s1 }, {v.s2, v.s3} };
    float Av[4][2];
    #pragma unroll
    for(int dc=0;dc<2;dc++) {
        Av[0][dc] = y[0][dc];
        Av[1][dc] = y[0][dc] + y[1][dc];
        Av[2][dc] = y[0][dc] - y[1][dc];
        Av[3][dc] =          - y[1][dc];
    }
    // A'
    // 1  1  1  0 
    // 0  1 -1 -1 
    float4 AvAT[4];
    #pragma unroll
    for(int dr=0;dr<4;dr++) {
        AvAT[dr].s0 = Av[dr][0];
        AvAT[dr].s1 = Av[dr][0] + Av[dr][1];
        AvAT[dr].s2 = Av[dr][0] - Av[dr][1];
        AvAT[dr].s3 =           - Av[dr][1];
    }
    return (float16)(AvAT[0],AvAT[1],AvAT[2],AvAT[3]);
}

float16 transform_tile(float4 a[4])
{
    float bta[4][4];

    bta[0][0] = a[0].s0 - a[2].s0;
    bta[0][1] = a[0].s1 - a[2].s1;
    bta[0][2] = a[0].s2 - a[2].s2;
    bta[0][3] = a[0].s3 - a[2].s3;

    bta[1][0] = a[1].s0 + a[2].s0;
    bta[1][1] = a[1].s1 + a[2].s1;
    bta[1][2] = a[1].s2 + a[2].s2;
    bta[1][3] = a[1].s3 + a[2].s3;

    bta[2][0] = a[2].s0 - a[1].s0;
    bta[2][1] = a[2].s1 - a[1].s1;
    bta[2][2] = a[2].s2 - a[1].s2;
    bta[2][3] = a[2].s3 - a[1].s3;

    bta[3][0] = a[1].s0 - a[3].s0;
    bta[3][1] = a[1].s1 - a[3].s1;
    bta[3][2] = a[1].s2 - a[3].s2;
    bta[3][3] = a[1].s3 - a[3].s3;

    float4 btab[4];
    #pragma unroll
    for(int i=0;i<4;i++) {
        btab[i].s0 = bta[i][0] - bta[i][2];
        btab[i].s1 = bta[i][1] + bta[i][2];
        btab[i].s2 = bta[i][2] - bta[i][1];
        btab[i].s3 = bta[i][1] - bta[i][3];
    }
    
    return (float16)(btab[0],btab[1],btab[2],btab[3]);
}


float16 load_4x4_tile_and_transform(__global const float * restrict img,int r,int c)
{
    float4 vals[4];
    #pragma unroll
    for(int dr=0;dr<4;dr++,img+=IMG_W) {
        if(r+dr >= 0 && r+dr < IMG_H) { 
            if(c>= 0 && c + 3 < IMG_W)
                vals[dr] = vload4(0,img);
            else {
                vals[dr].s0 = (c+0 >= 0 && c+0 < IMG_W) ? img[0] : 0;
                vals[dr].s1 = (c+1 >= 0 && c+1 < IMG_W) ? img[1] : 0;
                vals[dr].s2 = (c+2 >= 0 && c+2 < IMG_W) ? img[2] : 0;
                vals[dr].s3 = (c+3 >= 0 && c+3 < IMG_W) ? img[3] : 0;
            }
        }
        else {
            vals[dr] = 0;
        }
    }

    

    return transform_tile(vals);
}


float16 load_2x2_tile_and_transform(__global const float * restrict img,int r,int c)
{
    float4 v;
    if(r < IMG_H) {
        v.s0 = c+0 < IMG_W ? img[0] : 0.0;
        v.s1 = c+1 < IMG_W ? img[1] : 0.0;
    }
    else {
        v.lo = 0;
    }
    if(r + 1 < IMG_H) {
        img += IMG_W;
        v.s2 = c+0 < IMG_W ? img[0] : 0.0;
        v.s3 = c+1 < IMG_W ? img[1] : 0.0;
    }
    else {
        v.hi = 0;
    }

    return tile2x2_to_4x4(v);
}


void transform_kernel_bwd(float16 v,float gtkg[9])
{
   //
    // G' = 1   0.5   0.5  0
    //      0   0.5  -0.5  0
    //      0   0.5   0.5  1 
    // (G'*k) * G

    float4 gtk[3];
    float4 k[4]= { v.lo.lo, v.lo.hi, v.hi.lo, v.hi.hi }; 

    gtk[0] = k[0] + (float4)(0.5) * (k[1] + k[2]);
    gtk[1] = (float4)(0.5) * (k[1] - k[2]);
    gtk[2] = (float4)(0.5) * (k[1] + k[2]) + k[3];

    // G = 
    // [[ 1.   0.   0. ]
    //  [ 0.5  0.5  0.5]
    //  [ 0.5 -0.5  0.5]
    //  [ 0.   0.   1. ]]
  
    #pragma unroll
    for(int i=0,p=0;i<3;i++,p+=3) {
        gtkg[p+0] = gtk[i].s0 + 0.5 * (gtk[i].s1 + gtk[i].s2);
        gtkg[p+1] = 0.5 * (gtk[i].s1 - gtk[i].s2);
        gtkg[p+2] = 0.5 * (gtk[i].s1 + gtk[i].s2) + gtk[i].s3;
    }
}


inline void store_local(__local float *l_val,int strd,float16 v)
{
        l_val[ 0*strd] = v.s0;
        l_val[ 1*strd] = v.s1;
        l_val[ 2*strd] = v.s2;
        l_val[ 3*strd] = v.s3;
        l_val[ 4*strd] = v.s4;
        l_val[ 5*strd] = v.s5;
        l_val[ 6*strd] = v.s6;
        l_val[ 7*strd] = v.s7;
        l_val[ 8*strd] = v.s8;
        l_val[ 9*strd] = v.s9;
        l_val[10*strd] = v.sa;
        l_val[11*strd] = v.sb;
        l_val[12*strd] = v.sc;
        l_val[13*strd] = v.sd;
        l_val[14*strd] = v.se;
        l_val[15*strd] = v.sf;
}

#define WG_SIZE 256
#define XTILES_IN_WG 32
#define YTILES_IN_WG 32
#define WG_K 8

#if  WG_K * XTILES_IN_WG  != WG_SIZE || YTILES_IN_WG * WG_K != WG_SIZE
#error "Parameters do not match"
#endif

#define WG_DIM 0

#define PAD_H 1
#define PAD_W 1

#define PATCH_Y 8
#define PATCH_X 8

#if XTILES_IN_WG * YTILES_IN_WG * 16 != PATCH_Y * PATCH_X * WG_SIZE
#error
#endif

#ifndef TR_STRIDE_OFFSET
#define TR_STRIDE_OFFSET 1
#endif

#if STRIDE_OFFSET > 0 || TR_STRIDE_OFFSET > 0
#define PADDING_FACTOR 1
#else
#define PADDING_FACTOR 0
#endif


__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void winconv_3x3_bwd_filter(int B, int N,int C,
                          __global float const * restrict image,ulong image_offset,
                          __global float * restrict kernels,ulong kernels_offset,
                          __global float const *restrict result,ulong result_offset,
                          float beta
                          )
{
    image += image_offset;
    kernels += kernels_offset;
    result += result_offset;

    #define half_W  ((IMG_W+1)/2)
    #define half_H  ((IMG_H+1)/2)
    #define half_WG (half_W * half_H)

    __local float wg_local_memory[(XTILES_IN_WG + YTILES_IN_WG + 16 * PADDING_FACTOR) * WG_K * 16];

#define l_xtile_stride (XTILES_IN_WG * WG_K + STRIDE_OFFSET)
#define l_ytile_stride (YTILES_IN_WG * WG_K + STRIDE_OFFSET)

#define l_xtiles(a,b,c) wg_local_memory[(a)*l_xtile_stride + (b)*XTILES_IN_WG + (c)]
#define l_ytiles(a,b,c) wg_local_memory[(a)*l_ytile_stride + (b)*YTILES_IN_WG + (c) + (XTILES_IN_WG*WG_K*16 + 32 * STRIDE_OFFSET)]


    // Loading data
    int l_xtile_c = get_local_id(WG_DIM) % XTILES_IN_WG;
    int l_xtile_k = get_local_id(WG_DIM) / XTILES_IN_WG;

    int l_ytile_n = get_local_id(WG_DIM) % YTILES_IN_WG;
    int l_ytile_k = get_local_id(WG_DIM) / YTILES_IN_WG;
   
    int wg_ch_in  = get_group_id(0) * XTILES_IN_WG;
    int wg_ch_out = get_global_id(1) * YTILES_IN_WG; 

    #define s_img_tile(k,t,indx) wg_local_memory[(k)*(XTILES_IN_WG/2*16 + TR_STRIDE_OFFSET) + (t/2) * 16 + (indx)]
    
    __local float *l_xtile_ptr = &l_xtiles(0,l_xtile_k,l_xtile_c);
    __local float *l_ytile_ptr = &l_ytiles(0,l_ytile_k,l_ytile_n);

    // GEMM

    int my_gemm_tile_b  = get_local_id(WG_DIM) / 16;
    int my_gemm_tile_tk = get_local_id(WG_DIM) % 16;
    int my_gemm_tile_kr = my_gemm_tile_tk / (XTILES_IN_WG / PATCH_X) * PATCH_Y;
    int my_gemm_tile_tl = my_gemm_tile_tk % (XTILES_IN_WG / PATCH_X) * PATCH_X;

    float p_C[PATCH_Y][PATCH_X]={{0}};

    int k=0;
    int K_limit = B * half_WG;
    bool reduce_k=false;
    if(get_global_size(2) != 1) {
        int K_size = (K_limit + get_global_size(2) - 1) / get_global_size(2);
        k = K_size * get_global_id(2);
        K_limit = min(K_limit,k + K_size);
        reduce_k = true;
    }
    
    image  += (wg_ch_in  + l_xtile_c) * (IMG_H * IMG_W);
    result += (wg_ch_out + l_ytile_n) * (IMG_H * IMG_W);

    for(;k < K_limit;k+=WG_K)
    {
        {
            int my_k = k + l_xtile_k;
            int batch  = my_k / half_WG;
            int rowcol = my_k % half_WG;
            int row    = rowcol / half_W * 2;
            int col    = rowcol % half_W * 2;
            __global float const *x = image + batch * C * IMG_H * IMG_W + (row - 1) * IMG_W + (col - 1);
            // load relevant tile       
            float16 my_xtile = (k + l_xtile_k < K_limit && l_xtile_c + wg_ch_in < C) ? load_4x4_tile_and_transform(x,row-1,col-1) : 0; 
            store_local(l_xtile_ptr,l_xtile_stride,my_xtile);
        }
        {
            int my_k = k + l_ytile_k;
            int batch  = my_k / half_WG;
            int rowcol = my_k % half_WG;
            int row    = rowcol / half_W * 2;
            int col    = rowcol % half_W * 2;
             __global float const *dy = result + batch * N * IMG_H * IMG_W + row * IMG_W + col;
            // load relevant kernel
            float16 my_ytile = (k + l_ytile_k < K_limit && l_ytile_n + wg_ch_out < N) ? load_2x2_tile_and_transform(dy,row,col) : 0;
            store_local(l_ytile_ptr,l_ytile_stride,my_ytile);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // GEMM
        #pragma unroll
        for(int dk=0;dk<WG_K;dk++) {
            float p_y[PATCH_Y];

            #pragma unroll
            for(int dr=0;dr<PATCH_Y;dr++) {
                p_y[dr] = l_ytiles(my_gemm_tile_b,dk,dr + my_gemm_tile_kr);
            }
            float p_x[PATCH_X];
            #pragma unroll
            for(int dc=0;dc<PATCH_X;dc++) {
                p_x[dc] = l_xtiles(my_gemm_tile_b,dk,dc + my_gemm_tile_tl);
            }
            #pragma unroll
            for(int dr=0;dr<PATCH_Y;dr++) {
                #pragma unroll
                for(int dc=0;dc<PATCH_X;dc++) {
                    p_C[dr][dc] += p_y[dr]*p_x[dc];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int tile0 = get_local_id(WG_DIM) * 4;
    int s_col_t0 = tile0 % XTILES_IN_WG;
    int s_row_t = tile0 / XTILES_IN_WG;

    #pragma unroll
    for(int dc_split = 0; dc_split < 2;dc_split++) {
        // transpose part A

        #pragma unroll
        for(int dr=0;dr<PATCH_Y;dr++) {
            int tile_r = dr + my_gemm_tile_kr;
            #pragma unroll
            for(int dc=0;dc<PATCH_X;dc+=2) {
                int tile_c = dc + dc_split + my_gemm_tile_tl;
                s_img_tile(tile_r,tile_c,my_gemm_tile_b) = p_C[dr][dc + dc_split];
             }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int dc = 0;dc < 4;dc+=2) {
            
            int s_col_t = s_col_t0 + dc + dc_split;

            int kern_ch_out = s_row_t + wg_ch_out;
            int kern_ch_in  = s_col_t + wg_ch_in;

            if(kern_ch_out >= N || kern_ch_in >= C)
                continue;

            float16 s_kern16 = vload16(0,&s_img_tile(s_row_t,s_col_t,0));
            float s_kern9[9];
            transform_kernel_bwd(s_kern16,s_kern9);
            __global float *ptr = kernels + 9* (kern_ch_in +  kern_ch_out * C);
            if(reduce_k) {
                #pragma unroll
                for(int i=0;i<9;i++) {
                    atomic_addf(ptr + i, s_kern9[i]);
                }
            }
            else {
                #pragma unroll
                for(int i=0;i<9;i++) {
                    if(beta == 0)
                        ptr[i] = s_kern9[i];
                    else
                        ptr[i] = ptr[i] * beta + s_kern9[i];
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}


