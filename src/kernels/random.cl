///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
inline uint mulhi(uint a,uint b)
{
    ulong v=a;
    v*=b;
    return v>>32;
}

inline uint mullo(uint a,uint b)
{
    ulong v=a;
    v*=b;
    return v;
}


typedef struct state {
    uint l0,r0,l1,r1;
    uint k0,k1;
} state;

inline state single_round(state s)
{
    state next;
    next.l1 = mullo(s.r1, 0xD2511F53);
    next.r1 = mulhi(s.r0, 0xCD9E8D57) ^ s.k0 ^ s.l0;
    next.l0 = mullo(s.r0, 0xCD9E8D57);
    next.r0 = mullo(s.r1, 0xD2511F53) ^ s.k1 ^ s.l1;
    next.k0 = s.k0 + 0xBB67AE85;
    next.k1 = s.k1 + 0x9E3779B9;
    return next;
}

state make_initial_state(ulong seed,ulong sequence)
{
    state s;
    s.l1 = sequence >> 32;
    s.r1 = sequence;
    s.l0 = 0;
    s.r0 = 0;
    s.k0 = seed;
    s.k1 = seed >> 32;
    return s;
}

inline uint4 calculate(state s)
{
    #pragma unroll
    for(int i=0;i<10;i++)
        s=single_round(s);
    uint4 r;
    r.s0 = s.l0;
    r.s1 = s.r0;
    r.s2 = s.l1;
    r.s3 = s.r1;
    return r;
}

float4 calculate_float(state s)
{
    uint4 r = calculate(s);
    float4 f;
    /// make sure float does not become 1 after rounding
    /// 24 - for float/bfloat16
    /// 16 - for half
    /// 32 - for double
    const int accuracy_shift = 24;
    const int drop_bits = 32 - accuracy_shift;
    const float factor = 1.0f / ((ulong)(1) << accuracy_shift);
    f.s0 = (r.s0 >> drop_bits) * factor;
    f.s1 = (r.s1 >> drop_bits) * factor;
    f.s2 = (r.s2 >> drop_bits) * factor;
    f.s3 = (r.s3 >> drop_bits) * factor;
    return f;
}

#if IS_NORMAL == 1
float2 normal_pair(float2 v)
{
    float scale = sqrt(-2.0f*log(1.0f - v.s0));
    float angle = (2.0f*3.1415926535f)*v.s1;
    return (float2)(scale*cos(angle),scale*sin(angle));
}
#endif

__kernel void fill(ulong total,__global float *p,ulong p_offset,ulong seed,ulong seq,float v1,float v2)
{
    ulong pos = get_global_id(0);
    if(pos * 4 >= total)
        return;
    p+=p_offset;
    seq += pos;
    state s = make_initial_state(seed,seq);
    float4 r = calculate_float(s);
#if IS_UNIFORM == 1
    r = r * (float4)(v2-v1) + (float4)(v1);
#endif
#if IS_BERNOULLI == 1
    r.s0 = r.s0 < v1 ? 1:0;
    r.s1 = r.s1 < v1 ? 1:0;
    r.s2 = r.s2 < v1 ? 1:0;
    r.s3 = r.s3 < v1 ? 1:0;
#endif    
#if IS_NORMAL == 1
    r.lo = normal_pair(r.lo);
    r.hi = normal_pair(r.hi);
    r = r*(float4)(v2) + (float4)(v1);
#endif    
    ulong index = pos * 4;
    if(index < total) {
        vstore4(r,0,p + index);
    }
    else {
        if(index + 0 < total) p[index + 0]=r.s0;
        if(index + 1 < total) p[index + 1]=r.s1;
        if(index + 2 < total) p[index + 2]=r.s2;
        if(index + 3 < total) p[index + 3]=r.s3;
    }
}
