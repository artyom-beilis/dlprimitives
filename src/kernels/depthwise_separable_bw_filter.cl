#include "defs.h"
#include "reduce.h"


__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void conv_bw_filter(
          int batch,int height,int width,
          __global float const *input,int input_offset,
          __global float *kern,int kernel_offset,
          __global float const *output,int output_offset,
          float factor)
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
    int items_per_wg = (items + WG_SIZE - 1) / WG_SIZE;
    int my_start = items_per_wg * get_local_id(0);
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
        if(factor == 0)
            kern[k] = sum;
        else
            kern[k] = mad(kern[k],factor,sum);
    }

}


