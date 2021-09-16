# Benchmark Methodology

Original networks are taken from pytorch and converted in train mode to ONNX. ONNX is converted to dlprimitives model or to caffe model.
dlprimitives model (json) is also used to generate keras/plaidml model. Times for both training and testing (inference) are compared.
Warmup of 5 iterations is always used followed by 20 real measurement iterations. Measurement units are milliseconds per batch.
Input image size is standard ImageNet size 224x224.

Since Caffe and Keras/Plaidml do not support ReLU6, ReLU is used in benchmarks as substitution for mobilenet\_v2.

- PyTorch 1.8.1/cuda 10.2 is used for GTX 1080 and RTX 2060S
- PyTorch 1.7/cuda 10.2 is used for GTX 960
- PyTorch 1.7/rocm 3.7 is used for AMD Rx 560 (16cu/4GB)
- PlaidML 0.7 with Keras 2.2 is used for PlaidML backend
- OpenCL-Caffe at <https://github.com/artyom-beilis/caffe> a7a5ffb88e2dd is used (since BVLC caffe had degraded in performance since my fork)
- ROCm 3.7 is used for OpenCL AMD Drivers
- Intel OpenCL Neo 18.35.0 used for Intel HD530 GPU/Intel i5-6600 CPU.

Benchmarks published for master 8559e6aae8f682e7fdb71379f49aef9b9db4d6fc commit.

Tool and useful instruments

- `tools/validate_network.py` - pytorch benchmarks and export to ONNX
- `tools/onnx2dp.py` - convert ONNX model to dlprimitives js/h5 model format
- `tools/keras_benchmark.py` - loads dlprimitives js network description and runs benchmark for plaidml
- <https://github.com/MTLab/onnx2caffe> - ONNX to Caffe model converter
- `benchmark` - centeral dlprim benchmarking tool
- `docs/nets_for_benchmark` - networks converted to dlprimitives/caffe format
