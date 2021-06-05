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

|Operator               |Features                   | Computation       |
|-----------------------|---------------------------|-------------------|
|Softmax                |                           | Fwd               |
|Elementwise            | ax+by, max(ax,by), ax\*y  | Fwd               |
|MaxPool2d              |                           | Fwd               |
|AvgPool2d              |                           | Fwd               |
|GlobalMaxPool2d        |                           | Fwd               |
|GlobalAvgPool2d        |                           | Fwd               |
|Inner Product          |                           | Fwd               |
|Conv2d                 |                           | Fwd               |
|Activation             | relu, sigmoid, tanh       | Fwd,Bwd           |

## Other features

- Network object for inference
- ONNX to DlPrimitives model converter
