///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
__kernel
void axpby(ulong size,dtype a,__global const dtype *x,ulong x_off,dtype b,__global const dtype *y,ulong y_off,__global dtype *z,ulong z_off )
{
    ulong pos = get_global_id(0);
    if(pos >= size)
        return;
    x+=x_off;
    y+=y_off;
    z+=z_off;
    z[pos] = a*x[pos] + b*y[pos];
}
