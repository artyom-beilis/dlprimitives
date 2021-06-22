#include "defs.h"
#include "reduce.h"
#include "atomic.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#ifndef SIZE_2D
#define SIZE_2D 1
#endif


__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void bwd_bias(int batches,int features,__global dtype *dy,int dy_offset,__global dtype *dx,int dx_offset)
{
    dy += dy_offset;
    dx += dx_offset;

    int batch = get_global_id(1);
    if(batch >= batches)
        return;
    
    int feature = get_global_id(2);
    if(feature >= features)
        return;

    int index = get_local_id(0) * ITEMS_PER_WI;
    dy += (batch * features + feature) * SIZE_2D + index;
    
    REDUCE_PREPARE(WG_SIZE,dtype);

    dtype val = 0;
    #pragma unroll
    for(int i=0;i<ITEMS_PER_WI;i++,index++) {
        if(index >=SIZE_2D)
            break;
        val += *dy++;
    }
    
    my_work_group_reduce_add(val);

    if(get_local_id(0) == 0) {
        atomic_addf(dx + feature,val);
    }
}

