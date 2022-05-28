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
#elif DIMS == 6    
    r.s[0] = get_global_id(2) / limits.s[1];      
    r.s[1] = get_global_id(2) % limits.s[1];      
    r.s[2] = get_global_id(1) / limits.s[3];      
    r.s[3] = get_global_id(1) % limits.s[3];      
    r.s[4] = get_global_id(0) / limits.s[5];      
    r.s[5] = get_global_id(0) % limits.s[5];      
#elif DIMS == 7
    r.s[0] = get_global_id(2) / limits.s[1];      
    r.s[1] = get_global_id(2) % limits.s[1];      
    r.s[2] = get_global_id(1) / limits.s[3];      
    r.s[3] = get_global_id(1) % limits.s[3];      
    r.s[4] = get_global_id(0) / (limits.s[5]*limits.s[6]);      
    ulong s56 = get_global_id(0) % (limits.s[5]*limits.s[6]);
    r.s[5] = s56 / limits.s[6];      
    r.s[6] = s56 % limits.s[6];      
#elif DIMS == 8
    r.s[0] = get_global_id(2) / limits.s[1];      
    r.s[1] = get_global_id(2) % limits.s[1];      

    r.s[2] = get_global_id(1) / (limits.s[3]*limits.s[4]);      
    ulong s34 = get_global_id(1) % (limits.s[3]*limits.s[4]);
    r.s[3] = s34 / limits.s[4];      
    r.s[4] = s34 % limits.s[4];      

    r.s[5] = get_global_id(0) / (limits.s[6]*limits.s[7]);      
    ulong s67 = get_global_id(0) % (limits.s[6]*limits.s[7]);
    r.s[6] = s67 / limits.s[7];      
    r.s[7] = s67 % limits.s[7];      
#else
#error "Unsupported dim"
#endif
    return r;
}

