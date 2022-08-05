# DLPrimitives

This project aims to provide cross platform OpenCL tools for deep learning and inference.

Today, most of deep learning training is done on NVidia GPUs using closed source CUDA and CUDNN libraries.
It is either challenging or virtually impossible to use AMD or Intel GPUs.
For example: AMD provides ROCm platform, but there is no support of RDNA platforms yet (more than a year since a release),
there is no support of APUs and no support 
of any operating systems other than Linux.

## Goals

- Create an open source, cross platform deep learning primitives library similar to cuDNN or MIOpen that supports
multiple GPU architectures.
- Create an inference library with minimal dependencies for efficient inference on any modern GPU, similar to TensorRT or MIGraphX.
- Create minimalistic deep-learning framework as POC of capabilities and performance.
- Integrate to existing large scale deep learning projects like PyTorch, TF, MXNet such that vendor independent open-source OpenCL API will be first class citizen for deep learning.

Please note this is only work in progress - first and preliminary stages.

## Initial Framework Integration

Integration with existing frameworks:

-   Pytorch, (almost) out-of-tree OpenCL backend project:

    <https://github.com/artyom-beilis/pytorch_dlprim>
    
-   Caffe-OpenCL, performance improvements by using dlprimitives: 
    
    <https://github.com/artyom-beilis/caffe/tree/opencl_dlprim>

## Integration With ONNX

ONNX Model loading and inference tested on following imagenet networks:

- Pytorch, opsets 9, 11, 13: `alexnet`, `vgg16`, `resnet18`, `resnext50_32x4d`, `wide_resnet50_2`, `efficientnet_b0`, `efficientnet_b4`, `regnet_y_400mf`, `squeezenet1_0`, `mobilenet_v2`, `densenet121`
- MXNet: `vgg11_bn`, `alexnet`, `mobilenetv2_0.25`, `mobilenet0.25`, `densenet121`, `resnet18_v1`, `squeezenet1.0`
- Tensorflow, limited initial support, channel first: `resnet50`, `densenet121`

## Documentation 

Is published under <http://dlprimitives.org/docs/>


## Features Matrix

|Operator               |Features                               | Comment    |
|-----------------------|---------------------------------------|------------|
|Softmax                | Softmax, LogSoftmax                   |            |
|NLLLoss                |                                       |            |
|MSELoss                |                                       |            |
|SoftmaxWithLoss        |                                       |            |
|Elementwise            | ax+by, max(ax,by), ax\*y, broadcasting|            |
|Concat                 |                                       |            |
|Slice                  |                                       |            |
|Pooling2D              | max, average                          |            |
|GlobalPooling          | max, average                          | 2D only    |
|GlobalAvgPool2d        |                                       |            |
|InnerProduct           |                                       |            |
|BatchNorm              |                                       |            | 
|Reshape                |                                       |            |
|Squeeze                |                                       |            |                                
|Flatten                |                                       |            | 
|Threshold              |                                       |            | 
|Hardtanh               |                                       |            | 
|Abs                    |                                       |            | 
|Parameter              |                                       |ֹUtility     | 
|Reduction              | Sum, Mean, Sum Squares, L1            |            |
|Convolution2D          | GEMM, Winograd, Depthwise Separable   |            |
|TransposedConvolution2D| GEMM, Winograd, Depthwise Separable   |            |
|Activation             | relu, sigmoid, tanh, relu6            |            |

Solvers: SGD, Adam

## Tested GPUs

| Device    | Vendor    |   Notes                       |
|-----------|-----------|-------------------------------|
|RX 6600XT  | AMD       | ROCr                          | 
|RX 560     | AMD       | 16cu model, ROCm, PAL, Clover | 
|HD 530     | Intel     | i5-6600, NEO driver           |
|GTX 960    | NVidia    |                               |
|GTX 1080   | NVidia    |                               |
|RTX 2060S  | NVidia    |                               |
|MaliG52 MC2| ARM       | performance not optimised yet |
|M1 Max     | Apple     | 32-core model                 |

Devices Tested on Windows: AMD RX 560, NVidia GTX 960.

Devices Tested on macOS: Apple M1 Max.

## Other features

- Network object for inference
- ONNX to DLPrimitives model converter
