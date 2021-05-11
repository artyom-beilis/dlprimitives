#include "defs.h"

#ifndef ELTOP
#define ELTOP 0
#endif

#if ELTOP == 0
#define EOP(x,y) ((x) + (y))
#elif ELTOP == 1
#define EOP(x,y) ((x) * (y))
#elif ELTOP == 2
#define EOP(x,y) max((x),(y))
#else
#error "Invaid operation"
#endif

__kernel
void eltwise(int size,__global dtype *a,int a_offset, __global dtype *b,int b_offset,__global dtype *c,int c_offset,dtype c1,dtype c2)
{
    int pos = get_global_id(0);
    if(pos >= size)
        return;
    a+=a_offset;
    b+=b_offset;
    c+=c_offset;
    dtype value = EOP(a[pos]*c1,b[pos]*c2);
    c[pos] = ACTIVATION_F(value);
}


