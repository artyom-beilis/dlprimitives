#include "defs.h"
__kernel void fill(ulong total,__global dtype *p,ulong p_offset,dtype value)
{
    ulong pos = get_global_id(0);
    if(pos >= total)
        return;
    p[p_offset + pos ] = value;
}
