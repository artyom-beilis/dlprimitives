///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/scal.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/tensor.hpp>
#include <my_cblas.hpp>
namespace dlprim {
    Scal::Scal(Context &ctx,DataType dt) : ctx_(ctx)
    {
        DLPRIM_CHECK(dt==float_data);
        if(ctx_.is_cpu_context())
            return;
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"scal");
        k_ = cl::Kernel(prog,"sscal");
    }
    Scal::~Scal(){}
    
    void Scal::scale(float s,Tensor &t,ExecutionContext const &ec)
    {
        if(ctx_.is_cpu_context()) {
            float *p=t.data<float>();
            if(s == 0)
                memset(p,0,t.shape().total_size()*sizeof(float));
            else
                cblas_sscal(t.shape().total_size(),s,p,1);
        }
        else {
            int p = 0;
            size_t size = t.shape().total_size();
            int wg;
            if(size >= 1024)
                wg = 256;
            else
                wg = 64;
            k_.setArg(p++,cl_ulong(size));
            k_.setArg(p++,s);
            t.set_arg(k_,p);
            cl::NDRange l(wg);
            cl::NDRange g=gpu::round_range(size,l);
            ec.queue().enqueueNDRangeKernel(k_,cl::NullRange,g,l,ec.events(),ec.event("sscal"));
        }
    }

}
