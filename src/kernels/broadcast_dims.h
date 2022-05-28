///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
typedef struct Shape {
    ulong s[DIMS];
} Shape;


inline Shape get_pos_broadcast(Shape limits)
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

