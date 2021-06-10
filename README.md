# DLPrimitives

This project that aims to provide cross platform OpenCL tools for deep learning and inference.

Today most of deep learning training is done on NVidia GPUs using closed source CUDA and CUDNN libraries.
Using AMD or Intel GPUs either challenging or virtually impossible.
For example: also AMD provided ROCm platform, there is no support of RDNA platforms yet (more than a year since a release),
there is no support of APUs and of course no support 
of operating systems other than Linux.

So this project aims to provide deep learning primitives similar to cuDNN or MIOpen that would support
multiple GPUs and provide tools that can perform efficient inference of pre-trained models similar to TensorRT or MIGraphX.

Please note this is only work in progress - first and preliminary stages.

## Operators Features Matrix

|Operator               |Features                       | Computation       |
|-----------------------|-------------------------------|-------------------|
|Softmax                |                               | Fwd               |
|Elementwise            | ax+by, max(ax,by), ax\*y      | Fwd               |
|MaxPool2d              |                               | Fwd               |
|AvgPool2d              |                               | Fwd               |
|GlobalMaxPool2d        |                               | Fwd               |
|GlobalAvgPool2d        |                               | Fwd               |
|Inner Product          |                               | Fwd,Bwd           |
|Conv2d                 |                               | Fwd,Bwd           |
|Activation             | relu, sigmoid, tanh, relu6    | Fwd,Bwd           |

## Validated Networks

| Network       | Source of model       | Operation     |
|---------------|-----------------------|---------------|
| AlexNet       | torchvision.models    | Inference     |
| VGG16         | torchvision.models    | Inference     |
| ResNet50      | torchvision.models    | Inference     |
| MobileNet v2  | torchvision.models    | Inference     |

The networks were exported from pytorch to ONNX and imported for DLPrimitives.
Results compared with sample images. Note currently only inference tested,
BN layers are merged into conv layers by torch optimizer.

## Tested GPUs

| Device    | Vendor    |   Notes                       |
|-----------|-----------|-------------------------------|
|RX 560     | AMD       | 16cu model, ROCm & Clover     | 
|HD 530     | Intel     | i5-6600, NEO driver           |
|GTX 960    | NVidia    |                               |
|GTX 1080   | NVidia    |                               |
|RTX 2060S  | NVidia    |                               |

## Other features

- Network object for inference
- ONNX to DlPrimitives model converter
