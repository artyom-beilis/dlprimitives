#include "defs.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

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


    dtype val = c < channels ? in[c] : -FLT_MIN; 
    #pragma unroll
    for(int i=1;i<ITEMS_PER_WI;i++) {
        dtype tmp = c + i < channels ? in[c+i] : -FLT_MIN; 
        val = max(val,tmp);
    }

    #if __OPENCL_VERSION__ >= 200
    val = work_group_reduce_max(val);
    #else
    __local dtype reduce[WG_SIZE];
    int lid = get_local_id(1);
    reduce[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i=WG_SIZE/2;i>0; i>>= 1) {
        if(lid < i) {
            reduce[lid] = max(reduce[lid],reduce[lid+i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    val = reduce[0];
    #endif

    dtype sum = 0;

    #pragma unrol;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < channels) {
            dtype tmp = exp(in[c+i] - val);
            out[c+i] = tmp;
            sum += tmp;
        }
    }

    #if  __OPENCL_VERSION__ >= 200
    sum = work_group_reduce_add(sum);
    #else
    barrier(CLK_LOCAL_MEM_FENCE);
    reduce[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i=WG_SIZE/2;i>0; i>>= 1) {
        if(lid < i) {
            reduce[lid] = reduce[lid] + reduce[lid+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = reduce[0];
    #endif


    val = 1 / sum;


    #pragma unrol;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c + i < channels) {
            out[c+i] *= val;
        }
    }
}

