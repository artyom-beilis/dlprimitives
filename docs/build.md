# Build Instructions

## Dependencies

Mandatory:

- OpenCL SDK
- Python3 for building
- cmake

Recommended:

- OpenBLAS for CPU only inference/training 
- Sqlite3 for caching of compiled kernels for non-NVidia GPUs to improve startup times

Optional:

- boost python and boost numpy - for python bindings
- Google protobuf for ONNX support
- HDF5 C++ bindings for model loading/saving in HDF5 format instead of internal


## Building

Use following:

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

## Using CL/cl.hpp

Some OpenCL programs use older header `CL/cl.hpp`, in order to make dlprimitives compatible and use the old header pass a parameter to cmake:

    cmake -DUSE_CL_HPP=ON ..

## Running Unit Test

    make test

or
    ctest

By default ctest used `0:0` GPU - 0 platform, 0 device if you want to change it to another one, lets say platform 1 and device 0, pass parameter to cmake: `-DTEST_DEV="1:0"`


## Installation

Provide `-DCMAKE_INSTALL_PREFIX=/path/to/installation/location` to cmake, for example

    cmake -DCMAKE_INSTALL_PREFIX=/opt/dlprim ..

In order to use python bindings  `export PYTHONPATH=/opt/dlprim/python` - according to your installed location
    

## Benchmarking

Running benchmakrs on opencl platform:device 0:0

    # Inference
    ./dlprim_benchmark 0:0 ../docs/nets_for_benchmark/resnet18-b16.js
    # Train
    ./dlprim_benchmark -b 0:0 ../docs/nets_for_benchmark/resnet18-b16.js


## Windows Notes

- MinGW compiler has troubles with cl2.hpp, so it is recommended to set an option `-DUSE_CL_HPP=ON`
- Under windows the library is build as static library
