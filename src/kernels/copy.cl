///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel void copy(ulong slice,
                   ulong dim0,
                   ulong dim1_tgt,ulong dim1_tgt_offset,
                   ulong dim1_src,ulong dim1_src_offset,
                   ulong dim2,
                   __global dtype *target, ulong target_offset,
                   __global dtype const *source, ulong source_offset,
                   dtype scale)
{
    ulong p0 = get_global_id(0);
    ulong p1 = get_global_id(1);
    ulong p2 = get_global_id(2);
    if(p0 >= dim2 || p1 >= slice || p2 >= dim0)
        return;
    source += source_offset;
    target += target_offset;
    target += p0 + (p1 + dim1_tgt_offset) * dim2 + p2 * (dim1_tgt * dim2);
    source += p0 + (p1 + dim1_src_offset) * dim2 + p2 * (dim1_src * dim2);
    if(scale == 0.0)
        *target = *source;
    else
        *target = scale * *target + *source;
}
