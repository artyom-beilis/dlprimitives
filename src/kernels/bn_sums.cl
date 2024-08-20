///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "reduce2.h"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef BACKWARD
#define BACKWARD 0
#endif

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void compute(
          int batch,int channels,int HW,
          __global float const *x,ulong x_offset,
#if BACKWARD == 1
          __global float const *dy,ulong dy_offset,
          __global float *dyx_sum,ulong dyx_sum_offset,
          __global float *dy_sum,ulong dy_sum_offset
#else
    #if SECOND_REDUCE_SIZE == 1
          __global float *x_mean,ulong x_mean_offset,
          __global float *x_var, ulong x_var_offset
    #else
          __global float *x_sum, ulong x_sum_offset,
          __global float *x2_sum,ulong x2_sum_offset
    #endif          
#endif                
          )
{
    int feature = get_global_id(1);
    if(feature >= channels)
        return;

    x   += x_offset;
#if BACKWARD == 1
    dy   += dy_offset;
    dy_sum += dy_sum_offset;
    dyx_sum += dyx_sum_offset;
#else
  #if SECOND_REDUCE_SIZE == 1
    x_mean += x_mean_offset;
    x_var  += x_var_offset;
  #else
    x_sum += x_sum_offset;
    x2_sum += x2_sum_offset;
  #endif
#endif    
    
    x  += feature * HW;
#if BACKWARD == 1
    dy += feature * HW;
#endif    

    int items = batch * HW;
    const int wg_size2 = WG_SIZE * SECOND_REDUCE_SIZE;
    int items_per_wg = (items + wg_size2 - 1) / wg_size2;
    int my_start = items_per_wg * get_global_id(0); // it is same as local id for 1stage reduce
    int my_end   = min(my_start + items_per_wg,items);

    float sum1 = 0;
    float sum2 = 0;
    int b  = my_start / HW;
    int rc = my_start % HW;

    #pragma unroll(16)
    for(int index = my_start;index <my_end;index ++) {
        if(b < batch && rc < HW) {
            int pos = b*(channels * HW) + rc;
            #if BACKWARD == 1
                float xv  =   x[pos];
                float dyv =  dy[pos];
                sum1 += xv*dyv;
                sum2 += dyv;
            #else
                float val =   x[pos];
                sum1 += val;
                sum2 += val*val;
            #endif
        }
        rc++;
        if(rc == HW) {
            rc = 0;
            b ++;
        }
    }

    REDUCE_PREPARE_X2(WG_SIZE,float);

    float2 sums=(float2)(sum1,sum2);

    my_work_group_reduce_add_x2(sums);
    sum1 = sums.s0;
    sum2 = sums.s1;

    if(get_local_id(0) == 0) {
        #if SECOND_REDUCE_SIZE == 1
            int pos = feature;
        #else
            int pos = feature + channels * get_group_id(0);
        #endif
        #if BACKWARD == 1
            dyx_sum[pos] = sum1;
             dy_sum[pos] = sum2;
        #else
            #if SECOND_REDUCE_SIZE == 1
            float mean_val  = sum1 / (batch * HW);
            float mean2_val = sum2 / (batch * HW);
            x_mean[pos] = mean_val;
             x_var[pos] = mean2_val - mean_val*mean_val;
            #else
             x_sum[pos] = sum1;
            x2_sum[pos] = sum2;
            #endif
        #endif
    }
}

#if SECOND_REDUCE_SIZE > 1
__kernel
__attribute__((reqd_work_group_size(SECOND_REDUCE_SIZE,1,1)))
void reduce(int channels,
            __global float const * restrict s1,ulong s1_offset,
            __global float const * restrict s2,ulong s2_offset,
#if BACKWARD == 1
            __global float *dyx_sum,ulong dyx_sum_offset,
            __global float *dy_sum,ulong dy_sum_offset
#else
            __global float *x_mean,ulong x_mean_offset,
            __global float *x_var, ulong x_var_offset,
            float one_div_M
#endif
      )      
{
    s1 += s1_offset;
    s2 += s2_offset;
#if BACKWARD == 1
    dyx_sum += dyx_sum_offset;
    dy_sum  += dy_sum_offset;
#else
    x_mean += x_mean_offset;
    x_var  += x_var_offset;
#endif

    int f = get_global_id(1);
    if(f >= channels)
        return;
    
    REDUCE_PREPARE_X2(SECOND_REDUCE_SIZE,float);

    int read_pos = f + get_local_id(0) * channels;
    float2 sum;
    sum.s0 = s1[read_pos];
    sum.s1 = s2[read_pos];
    
    my_work_group_reduce_add_x2(sum);

    if(get_local_id(0) == 0) {
        #if BACKWARD == 0
        float mean_val  = sum.s0 * one_div_M;
        float mean2_val = sum.s1 * one_div_M;
        x_mean[f] = mean_val;
         x_var[f] = mean2_val - mean_val*mean_val;
        #else
        dyx_sum[f] = sum.s0;
         dy_sum[f] = sum.s1;
        #endif
    }    
}

#endif
