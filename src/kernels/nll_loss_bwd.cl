///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void nll_loss_backward(int batch,int channel,
        __global dtype *dx,ulong dx_offset,
        __global itype const *label,ulong label_offset,
        __global dtype const *dy,ulong dy_offset,
        dtype scale,
        dtype factor)
{
    dx += dx_offset;
    label += label_offset;
    dy += dy_offset;
    int c = get_global_id(0);
    int b = get_global_id(1);
    if(b>= batch || c >= channel)
        return;
    long index = (long)label[b];
    ulong offset = b*channel + c;
#if REDUCE == 1
    ulong dyoffset = 0;
#else
    ulong dyoffset = b;
#endif    
    dtype dxval = (c == index) ? -scale * dy[dyoffset] : 0;
    if(factor == 0)
        dx[offset] = dxval;
    else
        dx[offset] = dx[offset]*factor + dxval;
}
