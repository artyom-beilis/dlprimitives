#include <dlprim/ops/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>
#include <cblas.h>

namespace dlprim {
ActivationConfig ActivationConfig::from_json(json::value const &v)
{
    ActivationConfig cfg;
    cfg.activation = utils::activation_from_json(v);
    return cfg;
}

Activation::Activation(Context &ctx,ActivationConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
}

Activation::~Activation()
{
}

void Activation::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    out.assign({in[0]});
    p.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"activation",
                                        "ACTIVATION",int(config_.activation));
    kernel_ = cl::Kernel(prog,"activation");
}

void Activation::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==1);
    out.assign({in[0]});
}

void Activation::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(output[0].shape() == input[0].shape());
    
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],output[0]);
    }
    else {
        forward_gpu(input[0],output[0],e);
    }
}

void Activation::forward_cpu(Tensor &in,Tensor &out)
{
    size_t size = in.shape().total_size();
    float *a=in.data<float>();
    float *b=out.data<float>();
    if(a!=b) {
        memmove(b,a,size*sizeof(float));
    }
    cpu::apply_activation(b,size,config_.activation);
}

void Activation::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx)
{
    int p=0;
    int size = in.shape().total_size();
    kernel_.setArg(p++,size);
    kernel_.setArg(p++,in.device_buffer());
    kernel_.setArg(p++,int(in.device_offset()));
    kernel_.setArg(p++,out.device_buffer());
    kernel_.setArg(p++,int(out.device_offset()));
    
    cl::NDRange wg(256);
    cl::NDRange gr=gpu::round_range(size,wg);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("activation"));
    
}

}
