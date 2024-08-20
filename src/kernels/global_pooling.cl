///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "reduce.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#ifndef ENABLE_BWD
#define ENABLE_BWD 0
#endif



__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void global_pooling(int items,int over,float scale,__global dtype *in,ulong  data_offset,__global dtype *out,ulong  out_offset)
{
    in += data_offset;
    out += out_offset;
    
    int b = get_global_id(0);

    if(b >= items)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * over;
    out += b;
    
    REDUCE_PREPARE(WG_SIZE,dtype);

#if POOL_MODE == 0

    dtype val = -DTYPE_MAX;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val = max(val,in[c + i]);
        }
    }
    
    my_work_group_reduce_max(val);
#else
    dtype val = 0;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val += in[c+i];
        }
    }
    
    my_work_group_reduce_add(val);
    val = val * scale;
#endif

    if(get_local_id(1) == 0)
        *out = val;
}

#if ENABLE_BWD == 1

__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void global_pooling_bwd(int items,int over,float scale,
                        #if POOL_MODE == 0
                        __global const dtype *x, ulong  x_offset,
                        #endif
                        __global dtype *dx,ulong  dx_offset,__global const dtype *out,ulong  out_offset,
                        dtype factor)
{
    #if POOL_MODE == 0
    x += x_offset;
    #endif
    dx += dx_offset;
    out += out_offset;
    
    int b = get_global_id(0);

    if(b >= items)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    #if POOL_MODE == 0
    x  += b * over;
    #endif
    dx += b * over;
    out += b;


    
    REDUCE_PREPARE(WG_SIZE,dtype);

#if POOL_MODE == 0

    dtype val = -DTYPE_MAX;
    int index = -1;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            dtype tmp = x[c+i];
            if(tmp > val) {
                index = c+i;
                val = tmp;
            }
        }
    }

    __local int reduce_indx[WG_SIZE];
    __local dtype reduce_vals[WG_SIZE];

    int lid = my_get_local_wg_id();
    reduce_indx[lid] = index;
    reduce_vals[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE); 
    for(int i=WG_SIZE / 2; i > 0 ; i>>=1) {
        if(lid < i) {
            dtype val;
            int ind;
            dtype vl = reduce_vals[lid];
            int   il = reduce_indx[lid];
            dtype vr = reduce_vals[lid+i];
            int   ir = reduce_indx[lid+i];
            if(vl > vr) {
                ind=il;
                val=vl;
            }
            else if(vr > vl) {
                ind=ir;
                val=vr;
            }
            else { // vr == vl
                if(il < ir) {
                    ind=il;
                    val=vl;
                }
                else {
                    ind=ir;
                    val=vr;
                }
            }
            reduce_vals[lid] = val;
            reduce_indx[lid] = ind;
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }

    int target_index = reduce_indx[0];
    dtype store = *out;
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            dtype val = (c + i == target_index) ? store : 0;
            if(factor == 0)
                dx[c+i] = val;
            else
                dx[c+i] = dx[c+i] * factor + val;
        }
    }
#else
    dtype store = *out * scale;
    
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            if(factor == 0)
                dx[c+i] = store;
            else
                dx[c+i] = dx[c+i] * factor + store;
        }
    }
    
#endif

}
#endif
