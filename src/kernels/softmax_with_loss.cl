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

#ifndef LOG_SM
#define LOG_SM 0
#endif

#ifndef CALC_LOSS
#define CALC_LOSS 0
#endif

#if CALC_LOSS==1
#include "atomic.h"
#endif



#define LOCAL_ITEMS_LIMIT 32
__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void softmax(int batch,int channels,
             __global dtype *in,ulong  data_offset,
#if CALC_LOSS==2
             __global dtype *in_diff,ulong  in_diff_offset,
#endif
             __global itype *label,ulong  label_offset,             
             __global dtype *out,ulong  out_offset
#if CALC_LOSS==2
            ,dtype factor
#endif            
             )
{
    in += data_offset;
    out += out_offset;
    
    int b = get_global_id(0);

    if(b >= batch)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * channels;
    label += label_offset + b;
    #if CALC_LOSS == 2
    in_diff += b * channels + in_diff_offset;
    #endif
    
    REDUCE_PREPARE(WG_SIZE,dtype);

    dtype val = -DTYPE_MAX;
    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        dtype values[ITEMS_PER_WI];
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                values[i] = in[c+i];
                val = max(val,values[i]);
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                val = max(val,in[c+i]);
            }
        }
    #endif


    my_work_group_reduce_max(val);
    dtype maxv = val;

    dtype sum = 0;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                sum += values[i] = exp(values[i] - maxv);
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                dtype tmp = exp(in[c+i] - maxv);
                sum += tmp;
            }
        }
    #endif
    my_work_group_reduce_add(sum);

#if CALC_LOSS == 1
    if(get_local_id(1) == 0) {
        int index = *label;
        dtype loss = 0;
        if(0 <= index && index < channels)
            loss = -(in[index] - maxv - log(sum));
        atomic_addf(out,loss / batch);
    }
#elif CALC_LOSS == 2
    val = (dtype)1 / sum;
    int index = *label;
    dtype gr;
    dtype loss = *out / batch;
    
    #pragma unroll(8)
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c + i < channels) {
            #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
            dtype sm_val = values[i];
            #else
            dtype sm_val = exp(in[c+i]-maxv);
            #endif
            gr = loss *( sm_val * val - (int)(index ==  c + i));
            if(factor == 0)
                in_diff[c+i] = gr;
            else
                in_diff[c+i] = in_diff[c+i] * factor + gr;
        }
    }
#endif
}

