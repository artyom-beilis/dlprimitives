#include <dlprim/ops/axpby.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <cblas.h>

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
        kernel_.setArg(p++,int(total));
        kernel_.setArg(p++,a);
        kernel_.setArg(p++,x.device_buffer());
        kernel_.setArg(p++,int(x.device_offset()));
        kernel_.setArg(p++,b);
        kernel_.setArg(p++,y.device_buffer());
        kernel_.setArg(p++,int(y.device_offset()));
        kernel_.setArg(p++,z.device_buffer());
        kernel_.setArg(p++,int(z.device_offset()));
        e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,e.events(),e.event("axpby"));
    }
}




} // dlprim
