///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
__attribute__((reqd_work_group_size(1, 8, 8)))
void im2col(int batch,
            int src_rows,
            int src_cols,
            int rows,
            int cols,
            __global dtype const *img,ulong  img_offset,
            __global dtype *mat,ulong  mat_offset)
{
    img += img_offset;
    mat += mat_offset;

    int gid = get_global_id(0);
    
    int chan = gid % CHANNELS;
    int b    = gid / CHANNELS;
    int r    = get_global_id(1);
    int c    = get_global_id(2);

    if(r >= rows || c >= cols || chan >= CHANNELS || b >= batch)
        return;
    mat += CHANNELS * (KERN_H * KERN_W) * rows * cols * b;
    img += CHANNELS * src_rows * src_cols * b;
    int mat_row = r * cols + c;
    int mat_col = chan * (KERN_H * KERN_W);
    mat += mat_row * (CHANNELS * KERN_H * KERN_W) + mat_col;
    int y_pos = -PAD_H + r * STRIDE_H;
    int x_pos = -PAD_W + c * STRIDE_W;
    img += src_cols * (chan * src_rows + y_pos) + x_pos;

    #if PAD_H == 0 && PAD_W == 0
    #pragma unroll
    for(int dy = 0;dy < KERN_H * DILATE_H ;dy+= DILATE_H, img += src_cols * DILATE_H) {
        #pragma unroll
        for(int dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W) {
            *mat++ = img[dx];
        }
    }
    #else
    #pragma unroll
    for(int dy = 0;dy < KERN_H * DILATE_H ;dy+= DILATE_H, img += src_cols * DILATE_H) {
        int y = y_pos + dy;
        #pragma unroll
        for(int dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W) {
            int x = x_pos + dx;
            *mat++ = (y>= 0 && y < src_rows && x >= 0 && x < src_cols) ? img[dx] : 0;
        }
    }
    #endif
}
