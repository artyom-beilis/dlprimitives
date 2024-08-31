///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "atomic.h"

int get_src_pos(int pos,float scale,int limit,float offset)
{
    int src_pos = (pos + offset) * scale;
    return min(src_pos,limit-1);
}
int get_tgt_pos(int pos,float scale,int limit,float offset)
{
    int tgt_pos = ceil(pos * scale - offset);
    return min(tgt_pos,limit);
}


__kernel 
void nearest_fwd(
        int N,int items_per_thread,
        int srcH,int srcW,
        int tgtH,int tgtW,
        float scale_y,float scale_x,
        float offset,
        __global const dtype* x,ulong x_offset,
        __global dtype* y,ulong y_offset)
{
    int n0 = get_global_id(2) * items_per_thread;
    int n1 = min(n0 + items_per_thread,N);
    int r = get_global_id(0);
    int c = get_global_id(1);
    if(n0 >= N || r>= tgtH || c>= tgtW)
        return;

    int step_src = srcW*srcH;
    int step_tgt = tgtW*tgtH;
    x += x_offset + n0 * step_src;
    y += y_offset + n0 * step_tgt;

    int src_r = get_src_pos(r,scale_y,srcH,offset);
    int src_c = get_src_pos(c,scale_x,srcW,offset);
    x+= src_r * srcW + src_c;
    y+= r * tgtW + c;

    for(int n=n0;n<n1;n++) {
        *y = *x;
        x += step_src;
        y += step_tgt;
    }
}

__kernel 
void nearest_bwd(
        int N,int items_per_thread,
        int srcH,int srcW,
        int tgtH,int tgtW,
        float scale_y,float scale_x,
        float offset,
        __global dtype* dx,ulong dx_offset,
        __global dtype const * dy,ulong dy_offset,float beta)
{
    int n0 = get_global_id(2) * items_per_thread;
    int n1 = min(n0 + items_per_thread,N);
    int src_r = get_global_id(0);
    int src_c = get_global_id(1);
    if(n0 >= N || src_r>= srcH || src_c>= srcW)
        return;

    int step_src = srcW*srcH;
    int step_tgt = tgtW*tgtH;
    dx += dx_offset + n0 * step_src;
    dy += dy_offset + n0 * step_tgt;

    int tgt_r0 = get_tgt_pos(src_r,  scale_y, tgtH, offset);
    int tgt_r1 = get_tgt_pos(src_r+1,scale_y, tgtH, offset);
    int tgt_c0 = get_tgt_pos(src_c,  scale_x, tgtW, offset);
    int tgt_c1 = get_tgt_pos(src_c+1,scale_x, tgtW, offset);
   
    dx += src_r * srcW + src_c;

    for(int n=n0;n<n1;n++) {
        dtype grad = 0;
        for(int r=tgt_r0;r<tgt_r1;r++) {
            for(int c=tgt_c0;c<tgt_c1;c++) {
                grad += dy[r*tgtW+c];
            }
        }
        if(beta == 0)
            *dx = grad;
        else
            *dx = beta * *dx + grad;
        dx += step_src;
        dy += step_tgt;
    }
}


float calc_lin_pos(int p,float scale,int align_corners)
{
    if(align_corners)
        return p*scale;
    return max(scale * (p+0.5f) - 0.5f,0.0f);
}

__kernel 
void bilinear(
        int fwd,
        int N,int items_per_thread,
        int srcH,int srcW,
        int tgtH,int tgtW,
        float scale_y,float scale_x,
        int align_corners,
        __global dtype* restrict x,ulong x_offset,
        __global dtype* restrict y,ulong y_offset)
{
    int n0 = get_global_id(2) * items_per_thread;
    int n1 = min(n0 + items_per_thread,N);
    int r = get_global_id(0);
    int c = get_global_id(1);
    if(n0 >= N || r>= tgtH || c>= tgtW)
        return;

    int step_src = srcW*srcH;
    int step_tgt = tgtW*tgtH;
    x += x_offset + n0 * step_src;
    y += y_offset + n0 * step_tgt;

    float src_r0f = calc_lin_pos(r,scale_y,align_corners);
    int src_r0 = src_r0f;
    int dr = (src_r0 < srcH - 1) ? 1 : 0;
    int src_r1 = src_r0 + dr;
    dtype w_r1 = src_r0f - src_r0;
    dtype w_r0 = 1 - w_r1;

    float src_c0f = calc_lin_pos(c,scale_x,align_corners);
    int src_c0 = src_c0f;
    int dc = (src_c0 < srcW - 1) ? 1 : 0;
    int src_c1 = src_c0 + dc;
    dtype w_c1 = src_c0f - src_c0;
    dtype w_c0 = 1 - w_c1;

    y+= r * tgtW + c;

    __global dtype* restrict x00 = x + src_r0 * srcW + src_c0; 
    __global dtype* restrict x01 = x + src_r0 * srcW + src_c1; 
    __global dtype* restrict x10 = x + src_r1 * srcW + src_c0; 
    __global dtype* restrict x11 = x + src_r1 * srcW + src_c1; 

    for(int n=n0;n<n1;n++) {
        if(fwd) {
            dtype val = 
                w_r0 * (w_c0 * *x00 + w_c1 * *x01) +
                w_r1 * (w_c0 * *x10 + w_c1 * *x11);
            *y = val;
        }
        else {
            dtype val = *y;
            atomic_addf(x00,w_r0 * w_c0 * val);
            atomic_addf(x01,w_r0 * w_c1 * val);
            atomic_addf(x10,w_r1 * w_c0 * val);
            atomic_addf(x11,w_r1 * w_c1 * val);
        }

        x00 += step_src;
        x01 += step_src;
        x10 += step_src;
        x11 += step_src;
        y += step_tgt;
    }
}


