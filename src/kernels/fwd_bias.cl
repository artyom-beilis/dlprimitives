///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
__kernel void fwd_bias(int B,int F,int RC,
                       __global dtype *tensor,ulong tensor_offset,
                       __global dtype const *bias,ulong bias_offset)
{
    int rc = get_global_id(0);
    int f  = get_global_id(1);
    int b  = get_global_id(2);
    if(rc >= RC || f >= F || b >= B)
        return;
    tensor += tensor_offset;
    bias   += bias_offset;
    tensor[b * (F*RC) + f * RC + rc] += bias[f];
}
