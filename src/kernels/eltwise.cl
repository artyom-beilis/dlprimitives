///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
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
void eltwise(int size,__global const dtype *a,ulong  a_offset, __global const dtype *b,ulong  b_offset,__global dtype *c,ulong  c_offset,dtype c1,dtype c2)
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

__kernel
void eltwise_bwd(  int size,
                   int da_db_select,
                    __global const dtype *a,ulong  a_offset,
                    __global dtype *da,ulong  da_offset,
                    __global const dtype *b,ulong  b_offset,
                    __global dtype *db,ulong  db_offset,
                    __global const dtype *c, ulong  c_offset,
                    __global const dtype *dc,ulong  dc_offset,
                    dtype c1,dtype c2,
                    dtype factor_a,dtype factor_b)
{
    int pos = get_global_id(0);
    if(pos >= size)
        return;
    a+=a_offset;
    b+=b_offset;
    c+=c_offset;
    da+=da_offset;
    db+=db_offset;
    dc+=dc_offset;
    dtype dy = ACTIVATION_FINV(c[pos],dc[pos]);
    if(da_db_select & 1) { // da+a
        dtype da_val;
        #if ELTOP == 0
        da_val = dy * c1;
        #elif ELTOP == 1
        da_val = dy * c1 * c2 * b[pos];
        #elif ELTOP == 2
        if(c1*a[pos] >= c2*b[pos]) 
            da_val = c1 * dy;
        else
            da_val = 0;
        #endif
        if(factor_a == 0)
            da[pos] = da_val;
        else
            da[pos] = da[pos] * factor_a + da_val;
    }
    if(da_db_select & 2) { // db+b
        dtype db_val;
        #if ELTOP == 0
        db_val = dy * c2;
        #elif ELTOP == 1
        db_val = dy * c1 * c2 * a[pos];
        #elif ELTOP == 2
        if(c1*a[pos] >= c2*b[pos])
            db_val = 0;
        else
            db_val = c2 * dy;
        #endif
        if(factor_b == 0)
            db[pos] = db_val;
        else
            db[pos] = db[pos] * factor_b + db_val;
    }
}

