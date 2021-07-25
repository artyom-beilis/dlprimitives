## Summary of Benchmarks

Compare DLPrmitives to Native Pytorch (cuda+cudnn or hip+miopen) and best existing OpenCL
solution - Caffe OpenCL or Kerals with PlaidML. Testing done over 5 networks:
alexnet, resnet18, resnet50, vgg16 and mobilenet\_v2.

|             GPU|Batch|Train vs Native|Test vs Native|Train vs OpenCL|Test vs OpenCL|
|----------------|-----|---------------|--------------|---------------|--------------|
|  Nvidia GTX 960|   16|            51%|        60.73%|           171%|       167.33%|
|  Nvidia GTX 960|    8|            59%|        72.03%|           187%|       155.25%|
| Nvidia GTX 1080|   16|            42%|        41.34%|           207%|       137.52%|
|Nvidia RTX 2060S|   16|            49%|        57.53%|           211%|       149.48%|
|      AMD Rx 560|   16|            53%|        56.82%|           153%|       115.63%|
|      AMD Rx 560|    8|            55%|        54.19%|           172%|       122.64%|
|    Intel HD 530|    8|               |              |           109%|        66.12%|

