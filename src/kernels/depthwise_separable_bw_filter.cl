///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "reduce.h"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void conv_bw_filter(
          int batch,int height,int width,
          __global float const *input,ulong input_offset,
          __global float *kern,ulong kernel_offset,
          __global float const *output,ulong output_offset
#if SECOND_REDUCE_SIZE == 1
          ,float factor
#endif          
          )
{
    input   += input_offset;
    output  += output_offset;
    kern    += kernel_offset;

    int k = get_global_id(1);
    if( k > KERN * KERN * CHANNELS)
        return;

    int dk = k % (KERN * KERN);
    int d  = k / (KERN * KERN); 

    int dr = dk / KERN;
    int dc = dk % KERN;

    input  += d * (width * height);
    output += d * (width * height);

    int items = batch * (width * height);
    const int wg_size2 = WG_SIZE * SECOND_REDUCE_SIZE;
    int items_per_wg = (items + wg_size2 - 1) / wg_size2;
    int my_start = items_per_wg * get_global_id(0); // it is same as local id for 1stage reduce
    int my_end   = min(my_start + items_per_wg,items);

    float sum = 0;
    int b  = my_start / (width * height);
    int rc = my_start % (width * height);
    int r = rc / width;
    int c = rc % width;

    #pragma unroll(16)
    for(int index = my_start;index <my_end;index ++) {
        int sr = r - KERN/2 + dr;
        int sc = c - KERN/2 + dc;
        if(b < batch && 0<=sr && sr < height && 0 <= sc && sc < width) {
            float y = output[b*(CHANNELS * height * width) + r  * width +c ];
            float x =  input[b*(CHANNELS * height * width) + sr * width +sc];
            sum += x*y;
        }
        c++;
        if(c == width) {
            c = 0;
            r++;
            if(r == height) {
                r = 0;
                b ++;
            }
        }
    }

    REDUCE_PREPARE(WG_SIZE,float);

    my_work_group_reduce_add(sum);

    if(get_local_id(0) == 0) {
        #if SECOND_REDUCE_SIZE == 1
        if(factor == 0)
            kern[k] = sum;
        else
            kern[k] = mad(kern[k],factor,sum);
        #else
            #define STRIDE (KERN * KERN * CHANNELS)
            kern[k + STRIDE * get_group_id(0)] = sum;
        #endif
    }

}

#if SECOND_REDUCE_SIZE > 1
__kernel
__attribute__((reqd_work_group_size(SECOND_REDUCE_SIZE,1,1)))
void reduce(__global float const * restrict partial_values,ulong partial_values_offset,__global float * restrict sums,ulong sums_offset,float factor)
{
    sums += sums_offset;
    partial_values += partial_values_offset;
    int k = get_global_id(1);
    if(k > KERN * KERN * CHANNELS)
        return;
    
    REDUCE_PREPARE(SECOND_REDUCE_SIZE,float);

    float val = partial_values[k + get_local_id(0) * STRIDE];

    my_work_group_reduce_add(val);

    if(get_local_id(0) == 0) {
        if(factor == 0)
            sums[k] = val;
        else
            sums[k] = mad(sums[k],factor,val);
    }    
}

#endif
