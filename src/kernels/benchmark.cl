///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#ifndef USE_HALF
#define USE_HALF 0
#endif
#if USE_HALF == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half vec1;
typedef half2 vec2;
typedef half4 vec4;
typedef half8 vec8;
typedef half16 vec16;
#else
typedef float vec1;
typedef float2 vec2;
typedef float4 vec4;
typedef float8 vec8;
typedef float16 vec16;
#endif

__kernel
void flops_v1(__global float *out)
{
    int i=get_global_id(0);
    vec1 x=(vec1)(0.23f);
    vec1 y=(vec1)(0.5);
    vec1 z=(vec1)(0.0000023);
    vec1 cs = (vec1)(sin(0.56*i));
    vec1 sn = (vec1)(cos(0.56*i));

    #pragma unroll(100)
    for(int i=0;i<10000;i++) {
        vec1 xx = x*cs - (y*sn+z);
        vec1 yy = x*sn + (y*cs-z);
        x=xx;
        y=yy;
    }
    out[get_global_id(0)]=x*y;
}


__kernel
void flops_v2(__global float *out)
{
    int i=get_global_id(0);
    vec2 x=(vec2)(0.23f,0.21);
    vec2 y=(vec2)(0,0.5);
    vec2 z=(vec2)(0.00001,0.0000000023);
    vec2 cs = (vec2)(sin(0.56*i),sin(0.52*i));
    vec2 sn = (vec2)(cos(0.56*i),cos(0.52*i));

    #pragma unroll(100)
    for(int i=0;i<10000;i++) {
        vec2 xx = x*cs - (y*sn+z);
        vec2 yy = x*sn + (y*cs-z);
        x=xx;
        y=yy;
    }
    out[get_global_id(0)]=dot(x,y);
}

__kernel
void flops_v4(__global float *out)
{
    int i=get_global_id(0);
    vec4 x=(vec4)(0.23f,0.21,0.2,0.7);
    vec4 y=(vec4)(0,0.5,1.0,2.0);
    vec4 z=(vec4)(0.00001,0.000002,0.00012,0.0005);
    vec4 cs = (vec4)(sin(0.56*i),sin(0.52*i),sin(0.5*i),sin(0.1*i));
    vec4 sn = (vec4)(cos(0.56*i),cos(0.52*i),cos(0.5*i),cos(0.1*i));

    #pragma unroll(100)
    for(int i=0;i<10000;i++) {
        vec4 xx = x*cs - (y*sn+z);
        vec4 yy = x*sn + (y*cs-z);
        x=xx;
        y=yy;
    }
    out[get_global_id(0)]=dot(x,y);
}

__kernel
void flops_v8(__global float *out)
{
    int i=get_global_id(0);
    vec8 x=(vec8)(0.23f,0.21,0.2,0.7,   0.1,0.11,0.12,0.13);
    vec8 y=(vec8)(0,0.5,1.0 ,2.0,       0.21,0.22,0.23,0.25);
    vec8 z=(vec8)(0.00001,0.000002,0.00012,0.0005, 0.00001,01.0000021,0.000121,0.00051);
    vec8 cs = (vec8)(sin(0.56*i),sin(0.52*i),sin(0.5*i),sin(0.1*i),sin(0.561*i),sin(0.521*i),sin(0.51*i),sin(0.11*i));
    vec8 sn = (vec8)(cos(0.56*i),cos(0.52*i),cos(0.5*i),cos(0.1*i),cos(0.561*i),cos(0.521*i),cos(0.51*i),cos(0.11*i));

    #pragma unroll(100)
    for(int i=0;i<10000;i++) {
        vec8 xx = x*cs - (y*sn+z);
        vec8 yy = x*sn + (y*cs-z);
        x=xx;
        y=yy;
    }
    out[get_global_id(0)]=dot(x.lo,y.lo) + dot(x.hi,y.hi);
}

__kernel
void flops_v16(__global float *out)
{
    int i=get_global_id(0);
    vec16 x=(vec16)(0.23f,0.21,0.2,0.7,   0.1,0.11,0.12,0.13,0,0.5,1.0 ,2.0,       0.21,0.22,0.23,0.25);
    vec16 y=(vec16)(0,0.5,1.0 ,2.0,       0.21,0.22,0.23,0.25,0.23f,0.21,0.2,0.7,   0.1,0.11,0.12,0.13);
    vec16 z=(vec16)(0.00001,0.000002,0.00012,0.0005, 0.00001,01.0000021,0.000121,0.00051,
                    0.000001,0.0000002,0.000012,0.00005, 0.000001,01.0000021,0.0000121,0.000051);
    vec16 cs = (vec16)(sin(0.56*i),sin(0.52*i),sin(0.5*i),sin(0.1*i),sin(0.561*i),sin(0.521*i),sin(0.51*i),sin(0.11*i),
                       sin(1.56*i),sin(1.52*i),sin(1.5*i),sin(1.1*i),sin(1.561*i),sin(1.521*i),sin(1.51*i),sin(1.11*i) );
    vec16 sn = (vec16)(cos(0.56*i),cos(0.52*i),cos(0.5*i),cos(0.1*i),cos(0.561*i),cos(0.521*i),cos(0.51*i),cos(0.11*i),
                       cos(1.56*i),cos(1.52*i),cos(1.5*i),cos(1.1*i),cos(1.561*i),cos(1.521*i),cos(1.51*i),cos(1.11*i) );

    #pragma unroll(100)
    for(int i=0;i<10000;i++) {
        vec16 xx = x*cs - (y*sn+z);
        vec16 yy = x*sn + (y*cs-z);
        x=xx;
        y=yy;
    }
    out[get_global_id(0)]=dot(x.lo.lo,y.lo.lo) + dot(x.lo.hi,y.lo.hi) + dot(x.hi.lo,y.hi.lo) + dot(x.hi.hi,y.hi.hi);

}

__kernel
void memspeed_v1(__global int *p)
{
    p += get_global_id(0);
    *p += 1;
}

__kernel
void memspeed_v2(__global int2 *p)
{
    p += get_global_id(0);
    *p += (int2)(1);
}

__kernel
void memspeed_v4(__global int4 *p)
{
    p += get_global_id(0);
    *p += (int4)(1);
}

__kernel
void memspeed_v8(__global int8 *p)
{
    p += get_global_id(0);
    *p += (int8)(1);
}

__kernel
void memspeed_v16(__global int16 *p)
{
    p += get_global_id(0);
    *p += (int16)(1);
}


