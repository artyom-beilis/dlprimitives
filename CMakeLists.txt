cmake_minimum_required(VERSION 3.0)

if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2 -Wall")
endif()

find_path(OCL_PATH CL/cl2.hpp)
find_library(OCL_LIB OpenCL) 

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/my_blas") 
include_directories("${OCL_PATH}")

find_program(PYTHON NAMES python3 python python2)
if(NOT PYTHON)
	message(FATAL_ERROR "Can't build without Python interpreter")
endif(NOT PYTHON)

find_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
else()
    add_definitions(-DDISABLE_HDF5)
endif()

option(USE_INTERNAL_BLAS "Use Internal Blas" OFF)

if(NOT USE_INTERNAL_BLAS)
    find_path(CBLAS_PATH cblas.h)
    find_library(CBLAS_LIB openblas blas) 
endif()



file(GLOB CL_SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cl)
file(GLOB CL_HEADERS  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.h)
add_custom_command(
                  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp 
                  DEPENDS
                  ${CL_SOURCES} ${CL_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/build_source.py
                  COMMAND 
                  ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/build_source.py 
                    -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
                    ${CL_SOURCES}
                  )
if(NOT CBLAS_LIB OR NOT CBLAS_PATH) 
    set(EXTRA_SRC src/my_blas/my_cblas.cpp)
    set(BLAS_LIB )
    set(EXTRA_DEFS -DUSE_INTERNAL_BLAS)
else()
    set(EXTRA_SRC )
    set(BLAS_LIB ${CBLAS_LIB})
    include_directories("${CBLAS_PATH}")
    set(EXTRA_DEFS )
endif()

add_definitions(${EXTRA_DEFS})

set(DLPRIM_SRC
        src/context.cpp
        src/tensor.cpp
        src/program_cache.cpp
        src/operators_factory.cpp
        src/json.cpp
        src/activation.cpp
        src/net.cpp
        src/ops/inner_product.cpp
        src/ops/conv2d.cpp
        src/ops/activation.cpp
        src/ops/softmax.cpp
        src/ops/elementwise.cpp
        src/ops/pooling.cpp
        src/ops/bwd_bias.cpp
        src/ops/axpby.cpp
        src/ops/scal.cpp
        src/gemm.cpp
        ${EXTRA_SRC}
        ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp
        )


if(WIN32)
	add_library(dlprim STATIC ${DLPRIM_SRC})
else()
	add_library(dlprim SHARED ${DLPRIM_SRC})
endif()
		
		
target_link_libraries(dlprim ${OCL_LIB} ${BLAS_LIB})
if(HDF5_FOUND)
    target_link_libraries(dlprim ${HDF5_LIBRARIES} hdf5_cpp) 
endif()

add_executable(test_context tests/test_context.cpp)
target_link_libraries(test_context dlprim)

add_executable(test_from_template tests/test_from_template.cpp)
add_executable(test_json tests/json_test.cpp)
add_executable(benchmark tools/benchmark.cpp)
add_executable(image_predict tools/image_predict.cpp)
add_executable(mnist tests/mnist.cpp)
add_executable(train_mnist tests/train_mnist.cpp)

target_link_libraries(test_from_template dlprim)
target_link_libraries(mnist dlprim)
target_link_libraries(train_mnist dlprim)
target_link_libraries(benchmark dlprim)
target_link_libraries(image_predict dlprim)
target_link_libraries(test_json dlprim)
