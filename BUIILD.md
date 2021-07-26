## Dependencies

Mandatory:

- OpenCL SDK - `CL/cl2.hpp`
- Python for building
- cmake

Recommended:

- HDF5 C++ bindings for model loading/saving
- OpenBLAS for CPU only inference/training 


## Building

Use following:

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

## Running Unit Test

    make test

or
    ctest

## Benchmarking

Running benchmakrs on opencl platform:device 0:0

    # Inference
    ./benchmark 0:0 ../docs/nets_for_benchmark/resnet18-b16.json
    # Treain
    ./benchmark -b 0:0 ../docs/nets_for_benchmark/resnet18-b16.json


