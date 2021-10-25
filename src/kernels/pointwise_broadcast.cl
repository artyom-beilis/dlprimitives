#include "defs.h"

typedef struct __attribute__ ((packed)) Shape {
    ulong s[DIMS];
} Shape;

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

inline Shape get_pos(Shape limits)
{
    Shape r;
#if DIMS <= 1
    r.s[0] = get_global_id(0);
#elif DIMS == 2
    r.s[0] = get_global_id(1);      
    r.s[1] = get_global_id(0);      
#elif DIMS == 3    
    r.s[0] = get_global_id(2);      
    r.s[1] = get_global_id(1);      
    r.s[2] = get_global_id(0);      
#elif DIMS == 4    
    r.s[0] = get_global_id(2);      
    r.s[1] = get_global_id(1);      
    r.s[2] = get_global_id(0) / limits.s[3];      
    r.s[3] = get_global_id(0) % limits.s[3];      
#elif DIMS == 5    
    r.s[0] = get_global_id(2);      
    r.s[1] = get_global_id(1) / limits.s[2];      
    r.s[2] = get_global_id(1) % limits.s[2];      
    r.s[3] = get_global_id(0) / limits.s[4];      
    r.s[4] = get_global_id(0) % limits.s[4];      
#else
#error "Unsupported dim"
#endif
    return r;
}

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


