///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
__kernel
void sscal(ulong size,float scale,__global float *p,ulong p_off)
{
    ulong pos = get_global_id(0);
    if(pos >= size)
        return;
    p+=p_off;
    if(scale == 0)
        p[pos] = 0;
    else
        p[pos] = p[pos] * scale;
}
