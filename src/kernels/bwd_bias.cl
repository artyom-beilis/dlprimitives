#include "defs.h"
#include "reduce.h"

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
void bwd_bias(int features,int over,__global dtype *dy,int dy_offset,__global dtype *dx,int dx_offset,float beta)
{
    dy += dy_offset;
    dx += dx_offset;
    
    int feature = get_global_id(1);
    if(feature >= features)
        return;

    int position   = get_global_id(0) * ITEMS_PER_WI;
    
    REDUCE_PREPARE(WG_SIZE,dtype);

    dtype val = 0;
    int batch_scale = features * SIZE_2D;
    #pragma unroll
    for(int i=0;i<ITEMS_PER_WI;i++) {
        int index = position + i;
        if(index >= over)
            continue;

        int batch = index / SIZE_2D;
        int rcpos = index % SIZE_2D;
        val += dy[batch * batch_scale + feature * SIZE_2D + rcpos];
    }
    
    my_work_group_reduce_add(val);

    if(beta == 0)
        dx[feature] = val;
    else
        dx[feature] = dx[feature] * beta + val;
}

