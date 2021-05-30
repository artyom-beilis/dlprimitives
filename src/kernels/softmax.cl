#include "defs.h"
#include "reduce.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#define LOCAL_ITEMS_LIMIT 32
__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void softmax(int batch,int channels,__global dtype *in,int data_offset,__global dtype *out,int out_offset)
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

    dtype sum = 0;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                sum += values[i] = exp(values[i] - val);
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                dtype tmp = exp(in[c+i] - val);
                out[c+i] = tmp;
                sum += tmp;
            }
        }
    #endif
    my_work_group_reduce_add(sum);

    val = (dtype)1 / sum;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                out[c+i] = values[i] * val;
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                out[c+i] *= val;
            }
        }
    #endif
}

