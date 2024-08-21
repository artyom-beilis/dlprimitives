///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
namespace dlprim {
namespace core {
    SliceCopy::SliceCopy(Context &ctx,DataType dtype) :dtype_(dtype)
    {
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"copy",
                                "dtype",data_type_to_opencl_type(dtype_)
                                );
        cl::Kernel k(prog,"copy");
        kernel_ =  k;
    }
    SliceCopy::~SliceCopy()
    {
    }
    void SliceCopy::tensor_slice_copy(int dim,size_t slice,
                           Tensor &target,size_t target_offset,
                           Tensor &source,size_t source_offset,
                           float scale,ExecutionContext const &q)
    {
        Shape t = target.shape().split_and_merge_over_axis(dim);
        Shape s = source.shape().split_and_merge_over_axis(dim);
        DLPRIM_CHECK(target.dtype() == dtype_);
        DLPRIM_CHECK(source.dtype() == dtype_);
        DLPRIM_CHECK(s[0] == t[0]);
        DLPRIM_CHECK(source_offset + slice <= s[1]);
        DLPRIM_CHECK(target_offset + slice <= t[1]);
        DLPRIM_CHECK(s[2] == t[2]);
        int p = 0;
        kernel_.setArg(p++,cl_ulong(slice));
        kernel_.setArg(p++,cl_ulong(s[0]));
        kernel_.setArg(p++,cl_ulong(t[1]));
        kernel_.setArg(p++,cl_ulong(target_offset));
        kernel_.setArg(p++,cl_ulong(s[1]));
        kernel_.setArg(p++,cl_ulong(source_offset));
        kernel_.setArg(p++,cl_ulong(s[2]));
        target.set_arg(kernel_,p);
        source.set_arg(kernel_,p);
        bind_as_dtype(kernel_,p,scale,dtype_);

        q.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,cl::NDRange(s[2],slice,s[0]),cl::NullRange,q.events(),q.event("slice_copy"));
    }
}
}

