#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>

namespace dlprim {
namespace core {
    void copy_strided(  Shape shape,
                        cl::Buffer const &src,cl_ulong src_offset,Shape src_strides,
                        cl::Buffer const &dst,cl_ulong dst_offset,Shape dst_strides,
                        DataType dtype,
                        ExecutionContext const &q)
    {
        DLPRIM_CHECK(shape.size() == src_strides.size());
        DLPRIM_CHECK(shape.size() == dst_strides.size());
        int dims = shape.size();
        Context ctx(q);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"copy_strided",
                                "dtype",data_type_to_opencl_type(dtype,true),
                                "DIMS",dims);
        cl::NDRange range;
        switch(dims) {
        case 1: range = cl::NDRange(shape[0]); break;
        case 2: range = cl::NDRange(shape[1],shape[0]); break;
        case 3: range = cl::NDRange(shape[2],shape[1],shape[0]); break;
        case 4: range = cl::NDRange(shape[3]*shape[2],shape[1],shape[0]); break;
        case 5: range = cl::NDRange(shape[4]*shape[3],shape[2]*shape[1],shape[0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for strided copy " + std::to_string(dims));
        }
        cl::Kernel k(prog,"copy");
        int p=0;
        for(int i=0;i<dims;i++) {
            k.setArg(p++,cl_ulong(shape[i]));
            k.setArg(p++,cl_ulong(src_strides[i]));
            k.setArg(p++,cl_ulong(dst_strides[i]));
        }
        k.setArg(p++,src);
        k.setArg(p++,src_offset);
        k.setArg(p++,dst);
        k.setArg(p++,dst_offset);
        q.queue().enqueueNDRangeKernel(k,cl::NullRange,range,cl::NullRange,q.events(),q.event("copy_strided"));

    }
}
}

