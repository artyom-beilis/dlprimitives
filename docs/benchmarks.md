# Performance Benchmarks

## Summary

Summary of performance comparison of DLPrmitives to Native Pytorch (cuda+cudnn or hip+miopen) and best of existing OpenCL
solution - Caffe OpenCL or Kerals with PlaidML. Measured performance difference average over 5 networks:
alexnet, resnet18, resnet50, vgg16 and mobilenet\_v2.

|             GPU|Batch|Train, Cuda/HIP|Test, Cuda/HIP|Train, Plaidml/Caffe-OCL|Test,  Plaidml/Caffe-OCL|
|----------------|-----|---------------|--------------|---------------|--------------|
|  Nvidia GTX 960|   16|            51%|        60.73%|           171%|       167.33%|
|  Nvidia GTX 960|    8|            59%|        72.03%|           187%|       155.25%|
| Nvidia GTX 1080|   16|            42%|        41.34%|           207%|       137.52%|
|Nvidia RTX 2060S|   16|            49%|        57.53%|           211%|       149.48%|
|      AMD Rx 560|   16|            53%|        56.82%|           153%|       115.63%|
|      AMD Rx 560|    8|            55%|        54.19%|           172%|       122.64%|
|    Intel HD 530|    8|               |              |           109%|        66.12%|

## DLPrimitives vs Other Frameworks

Tested using ResNet18, batch size is 16 224x224 images. Units: milliseconds per batch.  

|Nvidia GTX 960|train|test|train|test|
|--------------|-----|----|-----|----|
|        dlprim|196.6|50.7|     |    |
|   cudnn/caffe|211.8|65.5| 108%|129%|
|   cudnn/keras|183.9|69.9|  94%|138%|
| cudnn/pytorch|110.2|35.1|  56%| 69%|

|  AMD RX 560|train|test|train|test|
|------------|-----|----|-----|----|
|      dlprim|  318|77.5|     |    |
|  rocm/caffe|  274|79.8|  86%|103%|
|  rocm/keras|240.7|82.8|  76%|107%|
|rocm/pytorch|167.4|39.2|  53%| 51%|

## Methodology

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

## Benchmarks AMD Rx 560

### Train

||          gpu|Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-----|-------|--------|--------|-------|-------------|-------|----|
|       PyTorch|rx560|     16|  74.763| 167.852| 539.06|      1056.31|133.747|    |
|Keras/Plaidml |rx560|     16| 700.167| 944.828|      -|            -|882.795|    |
|  Caffe/OpenCL|rx560|     16|  115.57| 442.022|       |      3239.57|1206.16|    |
|        dlprim|rx560|     16| 117.991| 315.032|867.819|     1820.853|452.457|    |
|     vs opencl|     |       |     98%|    140%|       |         178%|   195%|153%|
|     vs native|     |       |     63%|     53%|    62%|          58%|    30%| 53%|

||          gpu|Batch|alexnet|resnet18|resnet50|   vgg16|mobilenet\_v2|Average|
|--------------|-----|-------|--------|--------|--------|-------------|-------|----|
|       PyTorch|rx560|      8|  48.218|  86.292| 288.387|      556.417| 76.692|    |
|Keras/Plaidml |rx560|      8| 602.172| 517.257|1493.288|            -|500.769|    |
|  Caffe/OpenCL|rx560|      8| 75.2738| 245.099| 1186.12|      1053.39|591.253|    |
|        dlprim|rx560|      8|  80.037| 166.648| 429.012|      955.522|214.052|    |
|     vs opencl|     |       |     94%|    147%|    276%|         110%|   234%|172%|
|     vs native|     |       |     60%|     52%|     67%|          58%|    36%| 55%|


### Test

||          gpu|Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-----|-------|--------|--------|-------|-------------|-------|-------|
|       PyTorch|rx560|     16|  18.774|   39.48|159.765|      224.491| 45.517|       |
|Keras/Plaidml |rx560|     16|  53.715| 114.061|      -|            -|  50.63|       |
|  Caffe/OpenCL|rx560|     16|  40.087| 141.598|       |      645.706|269.348|       |
|        dlprim|rx560|     16|   35.11|  76.409|204.144|      455.075| 88.577|       |
|     vs opencl|     |       |    114%|    149%|       |         142%|    57%|115.63%|
|     vs native|     |       |     53%|     52%|    78%|          49%|    51%| 56.82%|

||          gpu|Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-----|-------|--------|--------|-------|-------------|-------|-------|
|       PyTorch|rx560|      8|  11.687|  19.955| 85.969|      120.146| 22.316|       |
|Keras/Plaidml |rx560|      8|   30.76|  59.429|167.008|            -| 29.689|       |
|  Caffe/OpenCL|rx560|      8|  28.769| 78.3648|243.105|      335.993|144.967|       |
|        dlprim|rx560|      8|  22.976|  44.121|111.317|      240.059| 46.898|       |
|     vs opencl|     |       |    125%|    135%|   150%|         140%|    63%|122.64%|
|     vs native|     |       |     51%|     45%|    77%|          50%|    48%| 54.19%|

## Benchmarks RX 6600xt

Note: rocm does not support 6600 XT yet, so no comparison to pytorch, is given

### Train

|              |      gpu|  Batch| alexnet|resnet18|resnet50|  vgg16     |mobilenet\_v2|
|--------------|---------|-------|--------|--------|-------|-------------|-------|
|        dlprim|RX 6600xt|     16|  30.180|  61.733|190.461|       290.98| 98.854|
|Keras/Plaidml |RX 6600xt|     16|177.546|415.727|977.615 |    3094.2|  355.140 |
|  Caffe/OpenCL|RX 6600xt|     16|64.119  |144.032 | 780.264 |  490.80 | 349.254|

### Test


|               |     gpu|   Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|
|--------------|-------- |-------|--------|--------|-------|-------------|-------|
|        dlprim|RX 6600xt|     16|10.816|  17.696|      48.823|      70.773|  27.083|
|Keras/Plaidml |RX 6600xt|     16|89.684|  190.738|     273.087|     1524.9|  33.210|
|  Caffe/OpenCL|RX 6600xt|     16|14.337|  39.371|     138.089|     159.98|  92.9304|



## Benchmarks For Nvidia GTX 960

### Train

||           gpu|  Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-------|-------|--------|--------|-------|------------|-------|----|
|       PyTorch|Gtx 960|     16|  41.496| 109.986| 350.57|     510.312| 154.39|    |
|Keras/Plaidml |Gtx 960|     16| 220.158| 506.364|      -|           -|570.401|    |
|  Caffe/OpenCL|Gtx 960|     16| 119.161| 410.655|       |            |1007.95|    |
|        dlprim|Gtx 960|     16|  84.693| 197.737|599.814|    1074.196|344.073|    |
|     vs opencl|       |       |    141%|    208%|       |            |   166%|171%|
|     vs native|       |       |     49%|     56%|    58%|         48%|    45%| 51%|

||           gpu|  Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-------|-------|--------|--------|-------|------------|-------|----|
|       PyTorch|Gtx 960|      8|  33.346|  67.673|196.468|     347.423| 82.467|    |
|Keras/Plaidml |Gtx 960|      8| 148.257| 264.462|736.946|           -|296.477|    |
|  Caffe/OpenCL|Gtx 960|      8| 78.0372| 216.396|      -|     1030.12|532.378|    |
|        dlprim|Gtx 960|      8|  56.805| 105.087| 311.24|     571.368| 171.29|    |
|     vs opencl|       |       |    137%|    206%|   237%|        180%|   173%|187%|
|     vs native|       |       |     59%|     64%|    63%|         61%|    48%| 59%|


### Test

||           gpu|  Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|-------|-------|--------|--------|-------|------------|-------|-------|
|       PyTorch|Gtx 960|     16|  11.622|  34.905|110.619|     165.524| 42.399|       |
|Keras/Plaidml |Gtx 960|     16|   42.45|  91.004|      -|           -| 44.615|       |
|  Caffe/OpenCL|Gtx 960|     16| 42.0916| 127.107|      -|     630.991|222.616|       |
|        dlprim|Gtx 960|     16|  22.932|  51.205|163.296|     247.068| 84.704|       |
|     vs opencl|       |       |    184%|    178%|       |        255%|    53%|167.33%|
|     vs native|       |       |     51%|     68%|    68%|         67%|    50%| 60.73%|

||           gpu|  Batch|alexnet|resnet18|resnet50| vgg16|mobilenet\_v2|Average|
|--------------|-------|-------|--------|--------|------|------------|-------|-------|
|       PyTorch|Gtx 960|      8|   8.975|  22.426| 60.15|     122.928| 22.007|       |
|Keras/Plaidml |Gtx 960|      8|  23.366|  48.312|102.89|           -| 25.916|       |
|  Caffe/OpenCL|Gtx 960|      8| 28.5095| 68.6208|199.54|     331.147|119.853|       |
|        dlprim|Gtx 960|      8|  14.109|  27.543|86.552|     128.643| 43.979|       |
|     vs opencl|       |       |    166%|    175%|  119%|        257%|    59%|155.25%|
|     vs native|       |       |     64%|     81%|   69%|         96%|    50%| 72.03%|
## Benchmarks GTX 1080

### Train

||           gpu|   Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|--------|-------|--------|--------|-------|-------------|-------|----|
|       PyTorch|GTX 1080|     16|  15.763|  38.359|125.902|      183.008| 59.379|    |
|Keras/Plaidml |GTX 1080|     16|  92.172| 235.163|702.904|      972.166|330.331|    |
|  Caffe/OpenCL|GTX 1080|     16|  40.536| 165.449|    nan|      662.584|396.664|    |
|        dlprim|GTX 1080|     16|  31.183|  70.496|260.437|      363.556|137.757|    |
|     vs opencl|        |       |    130%|    235%|   270%|         182%|   240%|211%|
|     vs native|        |       |     51%|     54%|    48%|          50%|    43%| 49%|

### Test


||           gpu|   Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|--------|-------|--------|--------|-------|-------------|-------|-------|
|       PyTorch|GTX 1080|     16|   4.915|  11.734| 37.329|       61.542| 15.849|       |
|Keras/Plaidml |GTX 1080|     16|  19.157|  36.124| 74.961|      186.724|  22.21|       |
|  Caffe/OpenCL|GTX 1080|     16|  14.203|  47.517|140.202|      207.394| 84.103|       |
|        dlprim|GTX 1080|     16|   8.128|  18.404| 81.524|       84.023| 35.723|       |
|     vs opencl|        |       |    175%|    196%|    92%|         222%|    62%|149.48%|
|     vs native|        |       |     60%|     64%|    46%|          73%|    44%| 57.53%|

## Benchmarks RTX 2060S

### Train

||           gpu|    Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|---------|-------|--------|--------|-------|------------|-------|----|
|       PyTorch|RTX 2060S|     16|  11.078|  30.969|100.094|     148.916| 37.829|    |
|Keras/Plaidml |RTX 2060S|     16|  68.945|  199.84|541.685|     926.916|243.272|    |
|  Caffe/OpenCL|RTX 2060S|     16|  39.007| 136.877|      -|     623.156|315.503|    |
|        dlprim|RTX 2060S|     16|  32.707|  66.728|181.003|     432.442| 90.459|    |
|     vs opencl|         |       |    119%|    205%|   299%|        144%|   269%|207%|
|     vs native|         |       |     34%|     46%|    55%|         34%|    42%| 42%|


### Test

||           gpu|    Batch|alexnet|resnet18|resnet50|  vgg16|mobilenet\_v2|Average|
|--------------|---------|-------|--------|--------|-------|------------|-------|-------|
|       PyTorch|RTX 2060S|     16|   3.668|   9.772| 30.698|      45.483| 10.234|       |
|Keras/Plaidml |RTX 2060S|     16|  21.884|  38.825| 73.711|     200.792| 21.007|       |
|  Caffe/OpenCL|RTX 2060S|     16|  16.123|  41.711|117.588|     196.366|  70.61|       |
|        dlprim|RTX 2060S|     16|  12.972|   21.64| 56.827|     112.533| 26.355|       |
|     vs opencl|         |       |    124%|    179%|   130%|        174%|    80%|137.52%|
|     vs native|         |       |     28%|     45%|    54%|         40%|    39%| 41.34%|

## Benchmarks Intel HD 530

### Train

||           gpu|       Batch|alexnet|resnet18|resnet50|   vgg16|mobilenet\_v2|Average|
|--------------|------------|-------|--------|--------|--------|-------------|--------|----|
|Keras/Plaidml |Intel HD 530|      8|4804.843|  961.62|3009.687|    18789.277|2367.907|    |
|  Caffe/OpenCL|Intel HD 530|      8| 878.947| 2277.14| 5657.17|        15522| 5002.81|    |
|        dlprim|Intel HD 530|      8| 691.178|1549.528|3601.626|    11487.195|1717.316|    |
|    Vs plaidml|            |       |    695%|     62%|     84%|         164%|    138%|228%|
|      vs caffe|            |       |    127%|    147%|    157%|         135%|    291%|172%|


### Test


||           gpu|       Batch|alexnet|resnet18|resnet50|   vgg16|mobilenet\_v2|Average|
|--------------|------------|-------|--------|--------|-------|-------------|-------|-------|
|Keras/Plaidml |Intel HD 530|      8|  63.433| 116.864|254.115|     2258.903| 199.77|       |
|  Caffe/OpenCL|Intel HD 530|      8| 388.199| 914.962|2106.76|      5781.26|966.915|       |
|        dlprim|Intel HD 530|      8| 169.902| 258.331|642.287|     1834.309|234.213|       |
|    Vs plaidml|            |       |     37%|     45%|    40%|         123%|    85%| 66.12%|
|      vs caffe|            |       |    228%|    354%|   328%|         315%|   413%|327.74%|

