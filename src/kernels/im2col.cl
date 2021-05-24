#include "defs.h"

__kernel
void im2col(int channels,
            int src_rows,
            int src_cols,
            int rows,
            int cols,
            __global dtype const *img,int img_offset,
            __global dtype *mat,int mat_offset)
{
    img += img_offset;
    mat += mat_offset;

    int chan = get_global_id(0);
    int r    = get_global_id(1);
    int c    = get_global_id(2);

    if(r >= rows || c >= cols || chan >= channels)
        return;
    int mat_row = r * cols + c;
    int mat_col = chan * (KERN_H * KERN_W);
    mat += mat_row * channels * (KERN_H * KERN_W) + mat_col;
    int y_pos = -PAD_H + r * STRIDE_H;
    int x_pos = -PAD_W + c * STRIDE_W;
    img += src_cols * (chan * src_rows + y_pos) + x_pos;

    #pragma unroll
    for(int dy = 0;dy < KERN_H * DILATE_H ;dy+= DILATE_H, img += src_cols * DILATE_H) {
        int y = y_pos + dy;
        if(y >= 0 && y < src_rows) {
            #pragma unroll
            for(int dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W) {
                int x = x_pos + dx;
                *mat++ = (x >= 0 && x < src_cols) ? img[dx] : 0;
            }
        }
        else {
            #pragma unroll
            for(int dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W) {
                *mat++ = 0;
            }
            
        }
    }
}
