///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#ifndef CUSTOM_REDUCE
#define CUSTOM_REDUCE 0
#endif

#define my_get_local_wg_id() ((get_local_id(2) * get_local_size(1) * get_local_size(0)) + (get_local_id(1) * get_local_size(0)) + get_local_id(0))
#if __OPENCL_VERSION__ >= 200 && !CUSTOM_REDUCE
#define REDUCE_PREPARE(WG_SIZE,dtype) do {} while(0)
#define my_work_group_reduce_add(val) do { val = work_group_reduce_add(val); } while(0)
#define my_work_group_reduce_max(val) do { val = work_group_reduce_max(val); } while(0)
#else

#define REDUCE_PREPARE(WG_SIZE,dtype) __local dtype my_reduce[WG_SIZE];
#define REDUCE_USING_OP(myval,reduce_op) \
    do { \
        int lid = my_get_local_wg_id(); \
        my_reduce[lid] = myval; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        const int WGS = sizeof(my_reduce)/sizeof(my_reduce[0]); \
        for(int i=WGS / 2;i>0; i>>= 1) { \
            if(lid < i) { \
                my_reduce[lid] = reduce_op(my_reduce[lid],my_reduce[lid+i]); \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        myval = my_reduce[0]; \
    } while(0)

#define REDUCE_OP_ADD(x,y) ((x) + (y))
#define REDUCE_OP_MAX(x,y) max((x),(y))

#define my_work_group_reduce_add(val) REDUCE_USING_OP(val,REDUCE_OP_ADD)
#define my_work_group_reduce_max(val) REDUCE_USING_OP(val,REDUCE_OP_MAX)

#endif
