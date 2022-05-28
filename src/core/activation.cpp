///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    void activation_forward(Tensor &x,Tensor &y,StandardActivations activation, ExecutionContext const &ec)
    {
        Context ctx(ec);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        cl::Kernel k(prog,"activation");

        int p=0;
        cl_ulong size = x.shape().total_size();
        k.setArg(p++,size);
        x.set_arg(k,p);
        y.set_arg(k,p);

        cl::NDRange wg(256);
        cl::NDRange gr=gpu::round_range(size,wg);
        ec.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,wg,ec.events(),ec.event("activation"));
    }
    void activation_backward(Tensor &dx,Tensor &dy,Tensor &y,StandardActivations activation,float beta,ExecutionContext const &ec)
    {
        Context ctx(ec);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        cl::Kernel k(prog,"activation_diff");
        
        
        int p=0;
        cl_ulong size = y.shape().total_size();
        k.setArg(p++,size);
        y.set_arg(k,p);
        dy.set_arg(k,p);
        dx.set_arg(k,p);
        k.setArg(p++,beta);

        cl::NDRange wg(256);
        cl::NDRange gr=gpu::round_range(size,wg);
        ec.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,wg,ec.events(),ec.event("activation_diff"));
    }

} // core
} // dlprim
