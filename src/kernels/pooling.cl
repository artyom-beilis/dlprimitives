#include "defs.h"

#ifndef POOL_MODE
#define POOL_MODE 0
#endif

#ifndef POOL_H
#define POOL_H 1
#endif
#ifndef POOL_W
#define POOL_W 1
#endif

#ifndef STRIDE_H 
#define STRIDE_H 1
#endif

#ifndef STRIDE_W
#define STRIDE_W 1
#endif

#ifndef PAD_H 
#define PAD_H 0
#endif

#ifndef PAD_W
#define PAD_W 0
#endif

#ifndef COUNT_INCLUDE_PAD
#define COUNT_INCLUDE_PAD 0
#endif

#if POOL_MODE == 0
#define START_VAL -DTYPE_MAX
#define REDUCE(a,b) max((a),(b))
#define NORMALIZE_FULL(x) (x)
#define NORMALIZE_PARTIAL(x,dr,dc) (x)
#define 
#elif POOL_MODE == 1
#define START_VAL 0.0f
#define REDUCE(a,b) ((a) + (b))
#define NORMALIZE_FULL(x) ((x) * (1.0f / (POOL_H * POOL_W)))
#if COUNT_INCLUDE_PAD == 0
#define NORMALIZE_PARTIAL(x,dr,dc) ((x) * (1.0f /((dr)*(dc))))
#else
#define NORMALIZE_PARTIAL(x,dr,dc) NORMALIZE_FULL(x)
#endif

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void pooling(int BC,int inp_H,int inp_W,int out_H,int out_W,
             __global const dtype *src,int src_offset,
             __global dtype *tgt,int tgt_offset)
{
    int bc = get_global_id(0);
    int or = get_global_id(1);
    int oc = get_global_id(2);
    if(bc >= BC || or >= out_H || oc >= O_w)
        return;

    int row0 = or * STEIDE_H - PAD_H;
    int col0 = oc * STRIDE_W - PAD_W;
    int row1 = row0 + POOL_H;
    int col1 = col0 + POOL_W;

    tgt += tgt_offset + bc * out_H * out_W;
    src += src_offset + bc * inp_H * inp_W;

    dtype val = START_VAL;
    
    if(row0 >= 0 && col0 >= 0 && row1 <= inp_H && col1 <= inp_W) {
        src += row0 * inp_W + col0;
        #pragma unroll  
        for(dr=0;dr<POOL_H;dr++) {
            #pragma unroll
            for(dc = 0;dc < POOL_W; dc++) {
                val = REDUCE(val,src[dr * inp_W + dc])
            }
        }
        val = NORMALIZE_FULL(val); 
    }
    else {
        #pragma unroll
        for(int r=row0;r<row1;r++) {
            #pragma unroll
            for(int c=col0;c<col1;c++) {
                dtype loaded_val = (r >= 0 && r<inp_H && c>=0 && c<inp_W) ? src[r*inp_W + c] : START_VAL;
                val = REDUCE(val,loaded_val);
            }
        }
        val = NORMALIZE_PARTIAL(val, min(row1,inp_H)-max(row0,0), min(col1,inp_W) - max(col0,0));
    }
    tgt[or * out_W + oc] = ACTIVATION_F(val);
}


