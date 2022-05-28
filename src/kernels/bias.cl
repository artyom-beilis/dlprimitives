///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
__kernel 
void ip_bias(int batch,int N,
        __global dtype *data,ulong data_offset,
        __global const dtype *bias,ulong bias_offset)
{
    data+=data_offset;
    bias+=bias_offset;
    int r = get_global_id(0);
    int c = get_global_id(1);
    if(r >= batch || c >= N)
        return;
    data += r*N+c;
    bias+=c;
    float v = *data + *bias;
    v=ACTIVATION_F(v);
    *data = v;
}


__kernel 
void activation_inplace(int tensor_size,__global dtype *data,ulong data_offset)
{
    data+=data_offset;
    if(get_global_id(0) >= tensor_size)
        return;
    data += get_global_id(0);
    *data = ACTIVATION_F(*data);
}

       
