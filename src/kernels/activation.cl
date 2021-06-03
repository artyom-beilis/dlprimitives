#include "defs.h"

__kernel
void activation(int size,__global dtype *a,int a_offset, __global dtype *c,int c_offset)
{
    int pos = get_global_id(0);
    if(pos >= size)
        return;
    a+=a_offset;
    c+=c_offset;
    c[pos] = ACTIVATION_F(a[pos]);
}


