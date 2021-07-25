## Details

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
that can't optimize registers as efficienlty as manually written
assembly.

## Benchmark Methodology

Original networks are taken from pytorch and converted in train mode to ONNX. ONNX is converted to dlprimitives model or to caffe model.
dlprimitives model (json) is also used to generate keras/plaidml model. Times for both training and testing (inference) are compared.
Warmup of 5 iterations is always used followed by 20 real measurement iterations.

Since Caffe and Keras/Plaidml do not support ReLU6, ReLU is used in benchmarks as substitution for mobilenet\_v2.

- PyTorch 1.8.1/cuda 10.2 is used for GTX 1080 and RTX 2060S
- PyTorch 1.7/cuda 10.2 is used for GTX 960
- PyTorch 1.7/rocm 3.7 is used for AMD Rx 560 (16cu/4GB)
- PlaidML 0.7 with Keras 2.2 is used for PlaidML backend
- OpenCL-Caffe at <https://github.com/artyom-beilis/caffe> a7a5ffb88e2dd is used (since BVLC caffe had degraded in performance since my fork)
- ROCm 3.7 is used for OpenCL AMD Drivers
- Intel OpenCL Neo 18.35.0 used for Intel HD530 GPU/Intel i5-6600 CPU.

Benchmarks published for master 8559e6aae8f682e7fdb71379f49aef9b9db4d6fc commit. Scripts in use

Tool and useful instruments

- `tools/validate_network.py` - pytorch benchmarks and export to ONNX
- `tools/onnx2dp.py` - convert ONNX model to dlprimitives js/h5 model format
- `tools/keras_benchmark.py` - loads dlprimitives js network description and runs benchmark for plaidml
- <https://github.com/MTLab/onnx2caffe> - ONNX to Caffe model converter
- `benchmark` - centeral dlprim benchmarking tool
- `docs/nets_for_benchmark` - networks converted to dlprimitives/caffe format
