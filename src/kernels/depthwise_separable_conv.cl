///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "atomic.h"

#define DIM_R 0
#define DIM_C 1
#define DIM_BD 2

#define KERN_PAD ((KERN-1)/2)

#define PATCH_H (PATCH_ROWS + KERN - 1)
#define PATCH_W (PATCH_COLS + KERN - 1)

__kernel
void conv(int batch,int height,int width,
          __global const float *input,ulong input_offset,
          __global const float *kern,ulong kernel_offset,
#if BIAS != 0
          __global const float *bias,ulong bias_offset,
#endif          
          __global float *output,ulong output_offset,
          float scale)
{
    input += input_offset;
    output += output_offset;
    kern += kernel_offset;
    int r = get_global_id(DIM_R) * PATCH_ROWS;
    int c = get_global_id(DIM_C) * PATCH_COLS;
    int b = get_global_id(DIM_BD) / CHANNELS;
    int d = get_global_id(DIM_BD) % CHANNELS;

    if(r >= height || c >= width || b >= batch)
        return;

    kern += d * KERN * KERN;
    
    input  += b * CHANNELS * width * height + d * width * height + (r - KERN_PAD) * width + c - KERN_PAD;
    output += b * CHANNELS * width * height + d * width * height + r * width + c;

    float K_vals[KERN][KERN];
    float I_vals[PATCH_H][PATCH_W];

    #pragma unroll
    for(int dr=0;dr < KERN;dr++)
        #pragma unroll
        for(int dc=0;dc<KERN;dc++)
            K_vals[dr][dc] = *kern ++;

    #if BIAS != 0
        float start_val = bias[bias_offset + d];
    #else
        const float start_val = 0;
    #endif

            

    int y = r-KERN_PAD;
    #pragma unroll
    for(int dr=0;dr<PATCH_H;dr++,y++) {
        if(y < 0 || y >= height) {
            #pragma unroll
            for(int dc=0;dc<PATCH_W;dc++)
                I_vals[dr][dc]=0;
        }
        else {
            int x = c - KERN_PAD;
            #pragma unroll
            for(int dc=0;dc<PATCH_W;dc++,x++) {
                I_vals[dr][dc]=(0 <= x && x < width) ? input[dr*width+dc] : 0;
            }
        }
    }


    #pragma unroll
    for(int dr=0;dr<PATCH_ROWS;dr++) {
        if(r+dr >= height)
            break;
        #pragma unroll
        for(int dc=0;dc<PATCH_COLS;dc++) {
            if(c+dc>=width)
                break;
            float sum = start_val;
            #pragma unroll
            for(int drk=0;drk < KERN;drk++)
                #pragma unroll
                for(int dck=0;dck<KERN;dck++)
                    sum = mad(K_vals[drk][dck],I_vals[dr+drk][dc+dck],sum);

            float value = ACTIVATION_F(sum);
            __global float *optr = output + dr*width+dc;
            if(scale == 0.0)
                *optr = value;
            else
                *optr = scale * *optr + value;
        }
    }

}

__kernel
void backward_data_conv(int batch,int height,int width,
          __global float *input,ulong input_offset,
          __global const float *kern,ulong kernel_offset,
          __global float const *output,ulong output_offset)
{
    input += input_offset;
    output += output_offset;
    kern += kernel_offset;
    int b = get_global_id(DIM_BD) / CHANNELS;
    int d = get_global_id(DIM_BD) % CHANNELS;
    int r = get_global_id(DIM_R) * PATCH_ROWS;
    int c = get_global_id(DIM_C) * PATCH_COLS;

    if(r >= height || c >= width || b >= batch)
        return;

    kern += d * KERN * KERN;
    
    input  += b * CHANNELS * width * height + d * width * height + (r - KERN_PAD) * width + c - KERN_PAD;
    output += b * CHANNELS * width * height + d * width * height + r * width + c;

    float K_vals[KERN][KERN];
    float I_vals[PATCH_H][PATCH_W] = {{0}};

    #pragma unroll
    for(int dr=0;dr < KERN;dr++)
        #pragma unroll
        for(int dc=0;dc<KERN;dc++)
            K_vals[dr][dc] = *kern ++;

            


    #pragma unroll
    for(int dr=0;dr<PATCH_ROWS;dr++) {
        if(r+dr >= height)
            break;
        #pragma unroll
        for(int dc=0;dc<PATCH_COLS;dc++) {
            if(c+dc>=width)
                break;
            float val = output[dr*width+dc];
            #pragma unroll
            for(int drk=0;drk < KERN;drk++)
                #pragma unroll
                for(int dck=0;dck<KERN;dck++)
                    I_vals[dr+drk][dc+dck] = mad(K_vals[drk][dck],val,I_vals[dr+drk][dc+dck]);

        }
    }

    int y = r-KERN_PAD;
    #pragma unroll
    for(int dr=0;dr<PATCH_H;dr++,y++) {
        if(y < 0 || y >= height) {
            #pragma unroll
            for(int dc=0;dc<PATCH_W;dc++)
                I_vals[dr][dc]=0;
        }
        if(!(y < 0 || y >= height)) {
            int x = c - KERN_PAD;
            #pragma unroll
            for(int dc=0;dc<PATCH_W;dc++,x++) {
                if(0 <= x && x < width) {
                    #if KERN == 1
                    input[dr*width+dc] += I_vals[dr][dc];
                    #else
                    atomic_addf(input + (dr*width+dc),I_vals[dr][dc]);
                    #endif
                }
            }
        }
    }
}
