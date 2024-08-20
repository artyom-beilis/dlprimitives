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

#include <iostream>

namespace dlprim {
namespace core {
    Scale::Scale(Context &ctx,DataType dt)
    {
        DLPRIM_CHECK(dt==float_data);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"scal");
        k_ = cl::Kernel(prog,"sscal");
    }
    void Scale::enqueue(float s,Tensor &t,ExecutionContext const &ec)
    {
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
    
    void scale_tensor(float s,Tensor &t,ExecutionContext const &ec)
    {
        Context ctx(ec);
        Scale sc(ctx,t.dtype());
        sc.enqueue(s,t,ec);
    }

    ///
    /// Set to zero tensor - OpenCL only
    ///
    void fill_tensor(Tensor &t,double value,ExecutionContext const &e)
    {
        Context ctx(e);
        pointwise_operation({},{t},{value},"y0=w0;",e);
    }


    void fill_random(Tensor &t,cl_ulong philox_seed,cl_ulong philox_seq,RandomDistribution dist,float p1,float p2,ExecutionContext const &e)
    {
        Context ctx(e);
        DLPRIM_CHECK(t.dtype() == float_data);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"random",
                                        "IS_UNIFORM",int(dist==rnd_uniform),
                                        "IS_NORMAL",int(dist==rnd_normal),
                                        "IS_BERNOULLI",int(dist==rnd_bernoulli)
                                        );
        cl::Kernel k(prog,"fill");
        cl_ulong total = t.shape().total_size();
        int p=0;
        k.setArg(p++,total);
        t.set_arg(k,p);
        k.setArg(p++,philox_seed);
        k.setArg(p++,philox_seq);
        k.setArg(p++,p1);
        k.setArg(p++,p2);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange((total+3)/4),cl::NullRange,e.events(),e.event("fill"));
    }

} // core
} // dlprim
