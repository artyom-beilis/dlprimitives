#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU     1
#define ACTIVATION_TANH     2
#define ACTIVATION_SIGMOID  3

#ifndef dtype
#define dtype float
#define DTYPE_MAX FLT_MAX
#define DTYPE_MIN FLT_MIN
#endif


#ifndef ACTIVATION
#define ACTIVATION ACTIVATION_IDENTITY
#endif

#if ACTIVATION == ACTIVATION_IDENTITY
#   define ACTIVATION_F(x) (x)
#   define ACTIVATION_NAME identity
#elif ACTIVATION == ACTIVATION_RELU
#   define ACTIVATION_F(x) (max((x),(dtype)(0)))
#   define ACTIVATION_NAME relu
#elif ACTIVATION == ACTIVATION_TANH
#   define ACTIVATION_F(x) (tanh((x)))
#   define ACTIVATION_NAME tanh 
#elif ACTIVATION == ACTIVATION_SIGMOID
#   define ACTIVATION_F(x) ((dtype)(1) / ((dtype)(1) + exp(-(x))))
#   define ACTIVATION_NAME sigmoid
#else
#   error "Unknown activation"
#endif 


