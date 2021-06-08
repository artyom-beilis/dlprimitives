#include "defs.h"

__kernel
void conv(int batch,int height,int width,
          __global const float *input, int input_offset,
          __global const float *kern,int kernel_offset,
#if BIAS != 0
          __global const float *bias  ,int bias_offset,
#endif          
          __global float *output,int output_offset)
{
    input += input_offset;
    output += output_offset;
    kern += kernel_offset;
    int b = get_global_id(0) / CHANNELS;
    int d = get_global_id(0) % CHANNELS;
    int r = get_global_id(1);
    int c = get_global_id(2);
    kern += d * KERN * KERN;
    input  += b * CHANNELS * width * height + d * width * height + r * width + c;
    output += b * CHANNELS * width * height + d * width * height + r * width + c;

    if(r >= height || c >= width || b >= batch)
        return;
    float K_vals[KERN][KERN];
    float I_vals[KERN][KERN] = {{0}};
    #pragma unroll
    for(int dr=0;dr < KERN;dr++)
        for(int dc=0;dc<KERN;dc++)
            K_vals[dr][dc] = *kern ++;

    #pragma unroll
    for(int dr=-KERN/2,kr=0;dr<=KERN/2;dr++,kr++) {
        if(r+dr < 0 || r+dr >= height)
            continue;
        #pragma unroll
        for(int dc=-KERN/2,kc=0;dc<=KERN/2;dc++,kc++) {
            if(c+dc < 0 || c+dc >= height)
                continue;
            I_vals[kr][kc]=input[dr*width+dc];
        }
    }

    float sum = 0;
    #pragma unroll
    for(int dr=0;dr < KERN;dr++)
        #pragma unroll
        for(int dc=0;dc<KERN;dc++)
            sum = mad(K_vals[dr][dc],I_vals[dr][dc],sum);
    #if BIAS != 0
    sum += bias[bias_offset + d];
    #endif

    *output = ACTIVATION_F(sum);
    
}
