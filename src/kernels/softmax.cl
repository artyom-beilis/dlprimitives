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
             __global dtype const *in,ulong  data_offset,
             __global dtype *out,ulong  out_offset)
{
    in += data_offset;
    out += out_offset;
    
    int b = get_global_id(0);

    if(b >= batch)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * channels;
    out += b * channels;
    
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
                #if LOG_SM == 1
                sum += exp(values[i] - maxv);
                #else
                sum += values[i] = exp(values[i] - maxv);
                #endif
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                dtype tmp = exp(in[c+i] - maxv);
                #if LOG_SM == 0
                out[c+i] = tmp;
                #endif
                sum += tmp;
            }
        }
    #endif
    my_work_group_reduce_add(sum);

    #if LOG_SM == 0
    val = (dtype)1 / sum;
    #else
    val = -log(sum);
    #endif

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                #if LOG_SM == 1
                out[c+i] = values[i] - maxv + val;
                #else
                out[c+i] = values[i] * val;
                #endif
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                #if LOG_SM == 1
                out[c+i] = in[c+1] - maxv + val;
                #else
                out[c+i] *= val;
                #endif
            }
        }
    #endif
}


__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void softmax_backward(int batch,int channels,
             __global dtype *in,ulong  data_offset,
             __global dtype const *out,ulong  out_offset,
             __global dtype const *out_diff,ulong  out_diff_offset,
             dtype factor
             )
{
    in += data_offset;
    out += out_offset;
    out_diff += out_diff_offset;
    
    int b = get_global_id(0);

    if(b >= batch)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * channels;
    out += b * channels;
    out_diff += b * channels;
    
    dtype sum = 0;
    REDUCE_PREPARE(WG_SIZE,dtype);

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
    #pragma unroll(ITEMS_PER_WI)
    #else
    #pragma unroll(LOCAL_ITEMS_LIMIT)
    #endif
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < channels) {
            #if LOG_SM == 1
            sum += out_diff[c+i];
            #else
            sum += out_diff[c+i] * out[c+i];
            #endif
        }
    }

    my_work_group_reduce_add(sum);

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
    #pragma unroll(ITEMS_PER_WI)
    #else
    #pragma unroll(LOCAL_ITEMS_LIMIT)
    #endif
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c + i < channels) {
            #if LOG_SM == 1
            float dxval = out_diff[c+i] - exp(out[c+i]) * sum;
            #else
            float dxval = (out_diff[c+i] - sum) * out[c+i]; 
            #endif
            if(factor == 0)
                in[c+i] = dxval;
            else
                in[c+i] = factor * in[c+i] + dxval;
        }
    }
}

