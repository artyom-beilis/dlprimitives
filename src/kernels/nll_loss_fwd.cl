///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "reduce.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void nll_loss_forward(int batch,int channels,
             __global dtype const *data,ulong  data_offset,
             __global itype const *label,ulong  label_offset,             
             __global dtype *out,ulong  out_offset,
             dtype scale
             )
{
    data += data_offset;
    label += label_offset;
    out += out_offset;
    
    long item = get_local_id(0) * ITEMS_PER_WI;
    #if REDUCE == 1
    dtype sum = 0;
    #endif

    #pragma unroll
    for(int i=0;i<ITEMS_PER_WI;i++,item++) {
        if(item < batch) {
            long index = (long)(label[item]);
            dtype loss_value = 0;
            if(0<= index && index < channels) {
                loss_value = -data[item*channels + index];
            }
            #if REDUCE==0
            out[item]=loss_value * scale;
            #else
            sum += loss_value;
            #endif
        }
    }
#if REDUCE == 1
    REDUCE_PREPARE(WG_SIZE,dtype);
    my_work_group_reduce_add(sum);
    if(get_local_id(0) == 0) {
        out[0] = sum * scale;
    }
#endif
}


