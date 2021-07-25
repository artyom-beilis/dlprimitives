# DLPrimitives

This project aims to provide cross platform OpenCL tools for deep learning and inference.

Today, most of deep learning training is done on NVidia GPUs using closed source CUDA and CUDNN libraries.
It is either challenging or virtually impossible to use AMD or Intel GPUs.
For example: AMD provides ROCm platform, but there is no support of RDNA platforms yet (more than a year since a release),
there is no support of APUs and no support 
of any operating systems other than Linux.

This project aims to provide open source deep learning primitives similar to cuDNN and MIOpen that would support
multiple GPU architectures. Another goal is to provide tools for inference of pre-trained models similar to TensorRT or MIGraphX.

Please note this is only work in progress - first and preliminary stages.

## Performance Comparison

Comparison of DLPrimitives vs PyTorch with cudnn/rocm and vs existing OpenCL implementation - plaidml and caffe/opencl.
It is based on average difference over 5 networks: alexnet, resnet18, resnet50, vgg16 and mobilenet\_v2.

|             GPU|Batch|Train, Cuda/HIP|Test, Cuda/HIP|Train, Plaidml/Caffe-OCL|Test,  Plaidml/Caffe-OCL|
|----------------|-----|---------------|--------------|---------------|--------------|
|  Nvidia GTX 960|   16|            51%|        60.73%|           171%|       167.33%|
|  Nvidia GTX 960|    8|            59%|        72.03%|           187%|       155.25%|
| Nvidia GTX 1080|   16|            42%|        41.34%|           207%|       137.52%|
|Nvidia RTX 2060S|   16|            49%|        57.53%|           211%|       149.48%|
|      AMD Rx 560|   16|            53%|        56.82%|           153%|       115.63%|
|      AMD Rx 560|    8|            55%|        54.19%|           172%|       122.64%|
|    Intel HD 530|    8|               |              |           109%|        66.12%|



## Features Matrix

|Operator               |Features                               | Computation       |
|-----------------------|---------------------------------------|-------------------|
|Softmax                |                                       | Fwd               |
|SoftmaxWithLoss        |                                       | Fwd,Bwd           |
|Elementwise            | ax+by, max(ax,by), ax\*y              | Fwd,Bwd           |
|MaxPool2d              |                                       | Fwd,Bwd           |
|AvgPool2d              |                                       | Fwd,Bwd           |
|GlobalMaxPool2d        |                                       | Fwd,Bwd           |
|GlobalAvgPool2d        |                                       | Fwd,Bwd           |
|Inner Product          |                                       | Fwd,Bwd           |
|BatchNorm2D            |                                       | Fwd,Bwd           | 
|Conv2d                 | GEMM, Winograd, Depthwise Separable   | Fwd,Bwd           |
|Activation             | relu, sigmoid, tanh, relu6            | Fwd,Bwd           |

Solvers: SGD, Adam

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
|RX 560     | AMD       | 16cu model, ROCm & Clover     | 
|HD 530     | Intel     | i5-6600, NEO driver           |
|GTX 960    | NVidia    |                               |
|GTX 1080   | NVidia    |                               |
|RTX 2060S  | NVidia    |                               |
|MaliG52 MC2| ARM       | performance not optimised yet |

Devices Tested on Windows: AMD RX 560, NVidia GTX 960.

## Other features

- Network object for inference
- ONNX to DlPrimitives model converter
