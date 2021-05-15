#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU     1

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
#else
#   error "Unknown activation"
#endif 


