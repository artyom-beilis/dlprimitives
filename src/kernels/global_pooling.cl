#include "defs.h"
#include "reduce.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void global_pooling(int items,int over,float scale,__global dtype *in,int data_offset,__global dtype *out,int out_offset)
{
    in += data_offset;
    out += out_offset;
    
    int b = get_global_id(0);

    if(b >= items)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * over;
    out += b;
    
    REDUCE_PREPARE(WG_SIZE,dtype);

#if POOL_MODE == 0

    dtype val = -DTYPE_MAX;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val = max(val,in[c + i]);
        }
    }
    
    my_work_group_reduce_max(val);
#else
    dtype val = 0;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val += in[c+i];
        }
    }
    
    my_work_group_reduce_add(val);
    val = val * scale;
#endif

    *out = val;
}

