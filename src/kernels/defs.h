///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU     1
#define ACTIVATION_TANH     2
#define ACTIVATION_SIGMOID  3
#define ACTIVATION_RELU6    4

#ifndef dtype
#define dtype float
#define dtype2 float2
#define dtype4 float4
#define DTYPE_MAX FLT_MAX
#define DTYPE_MIN FLT_MIN
#endif


#ifndef ACTIVATION
#define ACTIVATION ACTIVATION_IDENTITY
#endif

#if ACTIVATION == ACTIVATION_IDENTITY
#   define ACTIVATION_F(x) (x)
#   define ACTIVATION_FINV(y,dy) (dy)
#   define ACTIVATION_NAME identity
#elif ACTIVATION == ACTIVATION_RELU
#   define ACTIVATION_F(x) (max((x),(dtype)(0)))
#   define ACTIVATION_FINV(y,dy)  ((y>0)?dy:0)
#   define ACTIVATION_NAME relu
#elif ACTIVATION == ACTIVATION_TANH
#   define ACTIVATION_F(x) (tanh((x)))
#   define ACTIVATION_FINV(y,dy) ((1-(y)*(y))*(dy))
#   define ACTIVATION_NAME tanh 
#elif ACTIVATION == ACTIVATION_SIGMOID
#   define ACTIVATION_F(x) ((dtype)(1) / ((dtype)(1) + exp(-(x))))
#   define ACTIVATION_FINV(y,dy) ((y)*(1-(y))*(dy))
#   define ACTIVATION_NAME sigmoid
#elif ACTIVATION == ACTIVATION_RELU6
#   define ACTIVATION_F(x) (min(max((x),(dtype)(0)),(dtype)(6)))
#   define ACTIVATION_FINV(y,dy)  ((0<y && y<6)?dy:0)
#   define ACTIVATION_NAME relu6
#else
#   error "Unknown activation"
#endif 


