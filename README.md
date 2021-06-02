# DLPrimitives

This project that aims to provide cross platform OpenCL tools for deep learning and inference.

Today most of deep learning training is done on NVidia GPUs using closed source CUDA and CUDNN libraries.
Using AMD or Intel GPUs either challanging or virtually impossible.
For example: also AMD provided ROCm platform, there is no support of RDNA platforms yet (more than a year since a release),
there is no support of APUs and of course no support 
of operating systems other than Linux.

So this project aims to provide deep learning primitives similar to cuDNN or MIOpen that would support
multiple GPUs and provide tools that can perform efficient inference of pretrained models similar to TensorRT or MIGraphX.

Please note this is only work in progress - first and preliminary stages.

## Operators Features Matrix

|Operator|Forward, Float32|
|--------|----------------|
|Softmax | Yes            |
|ax+by   | Yes            |
|ax*y     | Yes            |
|max(ax,by)    | Yes  |
|MaxPool2d    |Yes|
|AvgPool2d    |Yes|
|GlobalMaxPool2d    |Yes|
|GlobalAvgPool2d    |Yes|
|Fuly Connected|yes|
|Conv2d|yes|
|ReLU|yes|

## Other features

- Network object for inference
- ONNX to DlPrimitives model converted
