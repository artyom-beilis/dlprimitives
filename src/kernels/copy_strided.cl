///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel void copy(
                ulong d0,ulong s0,ulong t0,
#if DIMS >= 2
                ulong d1,ulong s1,ulong t1,
#endif
#if DIMS >= 3
                ulong d2,ulong s2,ulong t2,
#endif          
#if DIMS >= 4
                ulong d3,ulong s3,ulong t3,
#endif          
#if DIMS >= 5
                ulong d4,ulong s4,ulong t4,
#endif
#if DIMS >= 6
                ulong d5,ulong s5,ulong t5,
#endif
#if DIMS >= 7
                ulong d6,ulong s6,ulong t6,
#endif
#if DIMS >= 8
                ulong d7,ulong s7,ulong t7,
#endif
                __global dtype_src const *src,ulong src_offset,
                __global dtype_tgt *tgt,ulong tgt_offset)
{
        src+=src_offset;
        tgt+=tgt_offset;
        #if DIMS == 1
            ulong i0 = get_global_id(0);
            if(i0 >= d0)
                return;
            tgt[i0*t0] = src[i0*s0];
        #elif DIMS == 2
            ulong i1 = get_global_id(0);
            ulong i0 = get_global_id(1);
            if(i0 >= d0)
                return;
            if(i1 >= d1)
                return;
            tgt[i0*t0 + i1*t1] = src[i0*s0 + i1*s1];
        #elif DIMS == 3            
            ulong i2 = get_global_id(0);
            ulong i1 = get_global_id(1);
            ulong i0 = get_global_id(2);
            if(i0 >= d0)
                return;
            if(i1 >= d1)
                return;
            if(i2 >= d2)
                return;
            tgt[i0*t0 + i1*t1 + i2*t2] = src[i0*s0 + i1*s1 + i2*s2];
        #elif DIMS == 4
            ulong ic = get_global_id(0);
            ulong i1 = get_global_id(1);
            ulong i0 = get_global_id(2);
            if(i0 >= d0)
                return;
            if(i1 >= d1)
                return;
            if(ic >= d2*d3)
                return;
            ulong i2 = ic / d3;
            ulong i3 = ic % d3;
            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3] = src[i0*s0 + i1*s1 + i2*s2 + i3*s3];
        #elif DIMS == 5
            ulong i34 = get_global_id(0);
            ulong i12 = get_global_id(1);
            ulong i0 = get_global_id(2);
            if(i0 >= d0)
                return;
            if(i12 >= d1*d2)
                return;
            if(i34 >= d3*d4)
                return;
            ulong i1  = i12 / d2;
            ulong i2  = i12 % d2;
            ulong i3  = i34 / d4;
            ulong i4  = i34 % d4;
            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4] = src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4];
        #elif DIMS == 6
            ulong i45 = get_global_id(0);
            ulong i23 = get_global_id(1);
            ulong i01 = get_global_id(2);
            if(i01 >= d0*d1)
                return;
            if(i23 >= d2*d3)
                return;
            if(i45 >= d4*d5)
                return;
            ulong i0  = i01 / d1;
            ulong i1  = i01 % d1;
            ulong i2  = i23 / d3;
            ulong i3  = i23 % d3;
            ulong i4  = i45 / d5;
            ulong i5  = i45 % d5;

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5] = src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5];
        #elif DIMS == 7
            ulong i456 = get_global_id(0);
            ulong i23  = get_global_id(1);
            ulong i01  = get_global_id(2);
            if(i01 >= d0*d1)
                return;
            if(i23 >= d2*d3)
                return;
            if(i456 >= d4*d5*d6)
                return;
            ulong i0  = i01 / d1;
            ulong i1  = i01 % d1;
            ulong i2  = i23 / d3;
            ulong i3  = i23 % d3;
            ulong i4  = i456 / (d5*d6);
            ulong i56 = i456 % (d5*d6);
            ulong i5  = i56 / d6;
            ulong i6  = i56 % d6;

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6] = src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6];
        #elif DIMS == 8
            ulong i567 = get_global_id(0);
            ulong i234 = get_global_id(1);
            ulong i01  = get_global_id(2);
            if(i01 >= d0*d1)
                return;
            if(i234 >= d2*d3*d4)
                return;
            if(i567 >= d5*d6*d7)
                return;
            ulong i0  = i01 / d1;
            ulong i1  = i01 % d1;

            ulong i2  = i234 / (d3*d4);
            ulong i34 = i234 % (d3*d4);
            ulong i3  = i34 / d4;
            ulong i4  = i34 % d4;

            ulong i5  = i567 / (d6*d7);
            ulong i67 = i567 % (d6*d7);
            ulong i6  = i67 / d7;
            ulong i7  = i67 % d7;

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6 + i7*t7] = src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6 + i7*s7];
        #else
        #error "Unsupported dims"
        #endif
}


