///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void activation(ulong size,__global dtype *a,ulong a_offset, __global dtype *c,ulong c_offset)
{
    ulong pos = get_global_id(0);
    if(pos >= size)
        return;
    a+=a_offset;
    c+=c_offset;
    c[pos] = ACTIVATION_F(a[pos]);
}


__kernel
void activation_diff(ulong size,__global dtype *y,ulong y_offset, __global dtype *dy,ulong dy_offset,__global dtype *dx,ulong dx_offset,dtype beta)
{
    ulong pos = get_global_id(0);
    if(pos >= size)
        return;
    y+=y_offset;
    dy+=dy_offset;
    dx+=dx_offset;
    dtype y_val  = y[pos];
    dtype dy_val = dy[pos];
    dtype diff = ACTIVATION_FINV(y_val,dy_val);
    if(beta == 0)
        dx[pos] = diff;
    else
        dx[pos] = mad(dx[pos],beta,diff);
}


