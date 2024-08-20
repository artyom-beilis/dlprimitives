# Implementation Details of GPU Kernels

## Performance Critical Kernels

DLPrimitives implements 3 variants of convolution algorithms

1. GEMM based - GEMM merged with im2col and col2im into a single kernel
2. Wingorad - Winograd convolution for 3x3, pad=1, stride=1 kernels for AMD and NVidia GPUs
3. Depthwise Separable - direct convolution


### GEMM

GEMM has following optimization. Matrix split into tiles each handled by a separate work group.
Typically each work group of 256 threads group loads 2 tiles of 128x16 into local memory and than
each thread typically calculates a 8x8 matrix. 

#### Convolution

When GEMM based algorithm is used GEMM applied as batched convolution and matrix rows/columns are translated to image position for load and store

For backpropogation atomic operations are used for "col2im" part. So working atomics are required for backpropogation.

#### Intel

For Intel GPU no local memory is used and if possible subgroup sharing is used. Each thread operates on its own small tile due to performance problems with local memory

#### AMD lda/ldb % 1024 == 0

There is a special case when lda or ldb is multiple of 1024 some cache issues occur. For such cases tiles are reordered in z-order curve to improve cache locality

#### Small M/N large K

For such cases additional reduction using atomic operations over K is used.

### Wingorad Convolution

It is currently applied for AMD and NVidia GPUs.

It is based on: 
Yan, Da, Wei Wang, and Xiaowen Chu.
"Optimizing batched winograd convolution on GPUs."
Proceedings of the 25th ACM SIGPLAN symposium on
principles and practice of parallel programming. 2020.

<https://www.cse.ust.hk/~weiwa/papers/yan-ppopp20.pdf>

In comparison to the paper: dlprim uses 32x8 tiles block instead of 64x8
since it generic OpenCL implementation for different GPUs
that can't optimize registers as efficiently as manually written
assembly.


