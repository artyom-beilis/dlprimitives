///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

#include "broadcast_dims.h"

inline ulong get_offset(Shape s,Shape strides,ulong offset)
{
    ulong r = offset;
    #pragma unroll
    for(int i=0;i<DIMS;i++) {
        r+= s.s[i]*strides.s[i];
    }
    return r;
}

inline ulong get_direct_offset(Shape s,Shape sizes,ulong offset)
{
    ulong index = 0;
    #pragma unroll
    for(int i=0;i<DIMS-1;i++) {
        index += s.s[i];
        index *= sizes.s[i+1];
    }
    index += s.s[DIMS-1] + offset;
    return index;
}

#define get_pos(limits) get_pos_broadcast(limits)

inline bool valid_pos(Shape pos,Shape limits)
{
    #pragma unroll
    for(int i=0;i<DIMS;i++)
        if(pos.s[i] >= limits.s[i])
            return 0;
    return 1;

}

__kernel void exec(Shape limit   PARAMS)
{
    Shape index = get_pos(limit);
    if(!valid_pos(index,limit)) {
        return;
    }
    LOADS
    CALC
    SAVES
}


