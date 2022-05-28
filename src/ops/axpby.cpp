///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/axpby.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>
#include <my_cblas.hpp>

namespace dlprim {

AXPBY::AXPBY(Context &ctx,DataType dt) : ctx_(ctx)
{
    DLPRIM_CHECK(dt == float_data);
    if(ctx_.is_cpu_context())
        return;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"axpby");
    kernel_ = cl::Kernel(prog,"axpby");
}
AXPBY::~AXPBY()
{
}

void AXPBY::apply(float a,Tensor &x,float b,Tensor &y,Tensor &z,ExecutionContext const &e)
{
    DLPRIM_CHECK(x.shape().total_size() == y.shape().total_size());
    DLPRIM_CHECK(z.shape().total_size() == y.shape().total_size());
    size_t total = x.shape().total_size();
    e.queue().finish();
    if(ctx_.is_cpu_context()) {
        float *xp = x.data<float>();
        float *yp = y.data<float>();
        float *zp = z.data<float>();
        memmove(zp,xp,total * sizeof(float));
        cblas_sscal(total,a,zp,1);
        cblas_saxpy(total,b,yp,1,zp,1);
    }
    else {
        cl::NDRange l;
        if(total >= 256)
            l=cl::NDRange(256);
        else if(total >= 128)
            l=cl::NDRange(128);
        else
            l=cl::NDRange(64);

        cl::NDRange g=gpu::round_range(total,l);
       
        int p=0;
        kernel_.setArg(p++,cl_ulong(total));
        kernel_.setArg(p++,a);
        x.set_arg(kernel_,p);
        kernel_.setArg(p++,b);
        y.set_arg(kernel_,p);
        z.set_arg(kernel_,p);
        e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,e.events(),e.event("axpby"));
    }
}




} // dlprim
