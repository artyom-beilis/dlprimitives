project(dlprimitives)
cmake_minimum_required(VERSION 3.0)

enable_testing()

if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2 -Wall")
endif()

option(USE_CL_HPP "Use older CL/cl.hpp instead of CL/cl2.hpp" OFF)
option(USE_INTERNAL_BLAS "Use Internal Blas" OFF)
option(USE_HDF "Compile with HDF support" ON)
option(USE_SQLITE3 "Compile with sqlite3 support for kernel caching" ON)
option(USE_ONNX "Compile with ONNX support" ON)

if(USE_CL_HPP)
    find_path(OCL_PATH CL/cl.hpp)
    set(DLPRIM_USE_CL1_HPP 1)
else()
    find_path(OCL_PATH CL/cl2.hpp)
    set(DLPRIM_USE_CL1_HPP 0)
endif()

find_library(OCL_LIB OpenCL) 

set(WITH_SQLITE3 OFF)
if (USE_SQLITE3)
    find_path(SQLITE3_PATH sqlite3.h)
    find_library(SQLITE3_LIB sqlite3)
    if(SQLITE3_PATH AND SQLITE3_LIB)
        include_directories(${SQLITE3_PATH})
        set(WITH_SQLITE3 ON)
    endif()
endif()

if(USE_ONNX)
    find_package(Protobuf)
    if(Protobuf_FOUND)
        set(WITH_ONNX ON)
    else()
        set(WITH_ONNX OFF)
    endif()
else()
    set(WITH_ONNX OFF)
endif()


configure_file(config.cmake.hpp dlprim/config.hpp)
set(DLPRIM_PUBLIC_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/dlprim/config.hpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 
include_directories("${CMAKE_CURRENT_BINARY_DIR}") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/my_blas") 
include_directories("${OCL_PATH}")

find_program(PYTHON NAMES python3 python python2)
if(NOT PYTHON)
	message(FATAL_ERROR "Can't build without Python interpreter")
endif(NOT PYTHON)

if(USE_HDF)
    find_package(HDF5)
endif()

if(USE_HDF AND HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
else()
    add_definitions(-DDISABLE_HDF5)
endif()

if(NOT LIBDIR)
	set(LIBDIR lib CACHE STRING "Library installation directory" FORCE)
endif()


SET(TEST_DEV "0:0" CACHE STRING "OpenCL platform:device, default 0:0")

if(NOT USE_INTERNAL_BLAS)
    find_path(CBLAS_PATH cblas.h)
    find_library(CBLAS_LIB openblas blas) 
endif()



file(GLOB CL_SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cl)
file(GLOB CL_HEADERS  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.h)

find_package(PythonLibs 3)
if(PYTHONLIBS_FOUND)
    string(REPLACE "." ";" PLIB_VER_LIST ${PYTHONLIBS_VERSION_STRING})
    list(GET PLIB_VER_LIST 0 PLIB_VER_MAJOR)
    list(GET PLIB_VER_LIST 1 PLIB_VER_MINOR)
    set(PLIB_VER "${PLIB_VER_MAJOR}${PLIB_VER_MINOR}")
    find_package(Boost COMPONENTS python${PLIB_VER} numpy${PLIB_VER} python3 numpy3 python numpy)
else()
    set(PLIB_VER "00")
endif()

if(PYTHONLIBS_FOUND AND Boost_NUMPY${PLIB_VER}_FOUND AND Boost_PYTHON${PLIB_VER}_FOUND)
    set(BPYTHON_LIB "${Boost_PYTHON${PLIB_VER}_LIBRARY}")
    set(BNUMPY_LIB "${Boost_NUMPY${PLIB_VER}_LIBRARY}")
    set(BUILD_PYDLPRIM TRUE)
elseif(PYTHONLIBS_FOUND AND Boost_NUMPY3_FOUND AND Boost_PYTHON3_FOUND)
    set(BPYTHON_LIB "${Boost_PYTHON3_LIBRARY}")
    set(BNUMPY_LIB "${Boost_NUMPY3_LIBRARY}")
    set(BUILD_PYDLPRIM TRUE)
elseif(PYTHONLIBS_FOUND AND Boost_NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(BPYTHON_LIB "${Boost_PYTHON_LIBRARY}")
    set(BNUMPY_LIB "${Boost_NUMPY_LIBRARY}")
    set(BUILD_PYDLPRIM TRUE)
else()
    set(BUILD_PYDLPRIM FALSE)
endif()


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



message("=== Status ===")
message("  OpenCL: include ${OCL_PATH}")
message("          lib     ${OCL_LIB}")
message("  Python: ${PYTHON}")
if(BLAS_LIB)
    message("  BLAS: include ${CBLAS_PATH}")
    message("        lib ${CBLAS_LIB}")
else()
    message("  BLAS: None")
endif()

if(HDF5_FOUND)
    message("  HDF5: include ${HDF5_INCLUDE_DIR}")
    message("        lib  ${HDF5_LIBRARIES} hdf5_cpp")
else()
    message("  HDF5: None")
endif()

if(WITH_SQLITE3)
    message("  Sqlite3: include ${SQLITE3_PATH}")
    message("           lib ${SQLITE3_LIB}")
else()
    message("  Sqlite3: disabled")
endif()

if(WITH_ONNX)
    message("  Protobuf: ${Protobuf_VERSION}")
else()
    message("  Protobuf (onnx): disabled")
endif()

if(BUILD_PYDLPRIM)
    message("  Python dlprim: enabled")
    message("  Python version: ${PLIB_VER}")
    message("  Python: lib ${PYTHON_LIBRARIES}")
    message("          include ${PYTHON_INCLUDE_DIRS}")
    message("  Boost: include ${Boost_INCLUDE_DIRS}")
    message("     boost_numpy ${BNUMPY_LIB}")
    message("     boost_python ${BPYTHON_LIB}")
else()
    message("  Python dlprim: disabled")
endif()

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
set(DLPRIM_CORE_SRC
        src/context.cpp
        src/tensor.cpp
        src/program_cache.cpp
        src/gemm.cpp
        src/random.cpp
        src/shape.cpp
        src/core/ip.cpp
        src/core/init.cpp
        src/core/conv.cpp
        src/core/bn.cpp
        src/core/slice.cpp
        src/core/pool.cpp
        src/core/pointwise.cpp
        src/core/loss.cpp
        src/core/activation.cpp
        src/core/util.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp)

set(DLPRIM_SRC 
        src/operators_factory.cpp
        src/json.cpp
        src/activation.cpp
        src/net.cpp
        src/ops/inner_product.cpp
        src/ops/batch_normalization.cpp
        src/ops/conv2d.cpp
        src/ops/activation.cpp
        src/ops/softmax.cpp
        src/ops/elementwise.cpp
        src/ops/pooling.cpp
        src/ops/bwd_bias.cpp
        src/ops/axpby.cpp
        src/ops/scal.cpp
        src/ops/concat.cpp
        src/ops/nll_loss.cpp
        src/ops/mse_loss.cpp
        src/ops/initialization.cpp
        ${EXTRA_SRC}
        )

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIBDIR}")

if(WIN32)
	add_library(dlprim_core STATIC ${DLPRIM_CORE_SRC})
	add_library(dlprim STATIC ${DLPRIM_SRC})
else()
	add_library(dlprim_core SHARED ${DLPRIM_CORE_SRC})
	add_library(dlprim SHARED ${DLPRIM_SRC})
endif()
		
target_link_libraries(dlprim_core ${OCL_LIB})	
target_link_libraries(dlprim dlprim_core ${OCL_LIB} ${BLAS_LIB})
if(HDF5_FOUND)
    target_link_libraries(dlprim ${HDF5_LIBRARIES} hdf5_cpp) 
endif()
if(WITH_SQLITE3)
    target_link_libraries(dlprim_core ${SQLITE3_LIB})
endif()

set_target_properties(dlprim PROPERTIES 
			PUBLIC_HEADER "${DLPRIM_PUBLIC_HEADERS}")

add_executable(test_context tests/test_context.cpp)
add_executable(test_broadcast_reduce tests/test_broadcast_reduce.cpp)
add_executable(test_util tests/test_util.cpp)

target_link_libraries(test_context dlprim_core)
target_link_libraries(test_util dlprim_core)
target_link_libraries(test_broadcast_reduce dlprim_core)

add_executable(test_from_template tests/test_from_template.cpp)
add_executable(test_net tests/test_net.cpp)
add_executable(test_json tests/json_test.cpp)
add_executable(dlprim_benchmark tools/benchmark.cpp)
add_executable(image_predict examples/cpp/image_predict.cpp)
add_executable(mnist tests/mnist.cpp)
add_executable(train_mnist examples/cpp/train_mnist.cpp)
add_executable(dlprim_flops tools/flops.cpp)
add_executable(test_random tests/test_random.cpp)

target_link_libraries(test_from_template dlprim)
target_link_libraries(mnist dlprim)
target_link_libraries(train_mnist dlprim)
target_link_libraries(dlprim_benchmark dlprim)
target_link_libraries(image_predict dlprim)
target_link_libraries(test_json dlprim)
target_link_libraries(dlprim_flops dlprim)
target_link_libraries(test_net dlprim)
target_link_libraries(test_random dlprim)


if(BUILD_PYDLPRIM)
    add_library(pydlprim SHARED python/python_interface.cpp)
    set_target_properties(pydlprim PROPERTIES PREFIX "" OUTPUT_NAME "_pydlprim")
    set_target_properties(pydlprim PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/dlprim")
    foreach(PYNAME __init__.py netconfig.py)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/dlprim/${PYNAME} "${CMAKE_BINARY_DIR}/python/dlprim/${PYNAME}" COPYONLY)
    endforeach()
    include_directories(${Boost_INCLUDE_DIRS})
    include_directories(${PYTHON_INCLUDE_DIRS})
    target_link_libraries(pydlprim dlprim ${BNUMPY_LIB} ${BPYTHON_LIB} ${PYTHON_LIBRARIES})
    file(GLOB PY_SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/python/dlprim/*.py)
    install(FILES ${PY_SOURCES} DESTINATION python/dlprim)
    install(TARGETS pydlprim
	    LIBRARY DESTINATION python/dlprim)
 
endif()

if(WITH_ONNX)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/onnx.proto)
    add_library(dlprim_onnx SHARED src/onnx/onnx.cpp ${PROTO_SRCS} ${PROTO_HDRS}) 
    target_link_libraries(dlprim_onnx dlprim ${Protobuf_LITE_LIBRARIES})
    add_executable(onnx_predict examples/cpp/onnx_predict.cpp)
    target_link_libraries(onnx_predict dlprim dlprim_onnx)
endif()


file(GLOB TEST_TEMPLATES  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_case_*.json)

foreach(TEMPLATE ${TEST_TEMPLATES}) 
    get_filename_component(TEST_NAME ${TEMPLATE} NAME_WE)
    add_test("test_${TEST_NAME}" test_from_template ${TEST_DEV} ${TEMPLATE})
endforeach()
add_test(test_net test_net ${TEST_DEV} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_net.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weights.json)
add_test(test_json test_json)
add_test(test_random test_random ${TEST_DEV})
add_test(test_context test_context ${TEST_DEV})
add_test(test_util test_util ${TEST_DEV})
add_test(test_broadcast_reduce test_broadcast_reduce ${TEST_DEV})


#################
# Installation
#################


install(TARGETS 
    dlprim_core dlprim dlprim_benchmark dlprim_flops
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION ${LIBDIR}
	ARCHIVE DESTINATION ${LIBDIR}
	PUBLIC_HEADER DESTINATION include/dlprim)

install(DIRECTORY include/dlprim DESTINATION include)

