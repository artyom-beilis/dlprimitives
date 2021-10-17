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
- Integrate to existing large scale deep learing projects like PyTorch, TF, MXNet such that vendor independent open-source OpenCL API will be first class citizen for deep learning.

Please note this is only work in progress - first and preliminary stages.

## Documentation 

Is published under <http://dlprimitives.org/docs/>


## Features Matrix

|Operator               |Features                               | Computation       |
|-----------------------|---------------------------------------|-------------------|
|Softmax, LogSoftmax    |                                       | Fwd               |
|NLLLoss                |                                       | Fwd,Bwd           |
|SoftmaxWithLoss        |                                       | Fwd,Bwd           |
|Elementwise            | ax+by, max(ax,by), ax\*y              | Fwd,Bwd           |
|Concat                 |                                       | Fwd,Bwd           |
|Slice                  |                                       | Fwd,Bwd           |
|MaxPool2d              |                                       | Fwd,Bwd           |
|AvgPool2d              |                                       | Fwd,Bwd           |
|GlobalMaxPool2d        |                                       | Fwd,Bwd           |
|GlobalAvgPool2d        |                                       | Fwd,Bwd           |
|Inner Product          |                                       | Fwd,Bwd           |
|BatchNorm              |                                       | Fwd,Bwd           | 
|Conv2d                 | GEMM, Winograd, Depthwise Separable   | Fwd,Bwd           |
|TransposedConv2d       | GEMM, Winograd, Depthwise Separable   | Fwd,Bwd           |
|Activation             | relu, sigmoid, tanh, relu6            | Fwd,Bwd           |

Solvers: SGD, Adam

## Initial Framework Integration

Pytorch: <https://github.com/artyom-beilis/pytorch_dlprim>

## Validated Networks

| Network       | Source of model       | Operation     |
|---------------|-----------------------|---------------|
| AlexNet       | torchvision.models    | Inference     |
| VGG16         | torchvision.models    | Inference     |
| ResNet50      | torchvision.models    | Inference     |
| ResNet18      | torchvision.models    | Inference     |
| MobileNet v2  | torchvision.models    | Inference     |

The networks were exported from pytorch to ONNX and imported for DLPrimitives.
Results compared with sample images. Note currently only inference validated,
backpropogation is convered by per-layer regression.

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

Devices Tested on Windows: AMD RX 560, NVidia GTX 960.

## Other features

- Network object for inference
- ONNX to DlPrimitives model converter
