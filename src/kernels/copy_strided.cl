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
                __global dtype const *src,ulong src_offset,
                __global dtype *tgt,ulong tgt_offset)
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
        #else
        #error "Unsupported dims"
        #endif
}


