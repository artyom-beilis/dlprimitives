#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    void copy_strided(  Shape shape,
                        cl::Buffer const &src,cl_ulong src_offset,Shape src_strides,
                        cl::Buffer const &dst,cl_ulong dst_offset,Shape dst_strides,
                        DataType dtype_src,
                        DataType dtype_tgt,
                        ExecutionContext const &q);
} // core
} // dlprim

