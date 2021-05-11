#include <dlprim/functions.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <math.h>
#include <cblas.h>

namespace dlprim {
SoftMax::~SoftMax() {}

SoftMax::SoftMax(Context &ctx,DataType d) : 
    Operator(ctx),
    dtype_(d),wg_size_(0),items_per_wi_(0),sm_range_(-1)
{
}

void SoftMax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    out = in;
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0].shape()[1]);
}

void SoftMax::setup_kernel(int sm_range)
{
    if(sm_range_ == sm_range)
        return;
    if(sm_range <= 64)
        wg_size_ = 64;
    else if(sm_range <= 128)
        wg_size_ = 128;
    else 
        wg_size_ = 256;
    items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;

    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"softmax","WG_SIZE",wg_size_,"ITEMS_PER_WI",items_per_wi_);
    kernel_ = cl::Kernel(prog,"softmax");
    sm_range_ = sm_range;
    int mpl = wg_size_ * items_per_wi_;
    nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
}


void SoftMax::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 2);
    out = in;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0][1]);
}

void SoftMax::forward_cpu(Tensor &input,Tensor &output)
{
    Shape in_shape = input.shape();
    float *in  = input.data<float>();
    float *out = output.data<float>();
    for(int i=0;i<in_shape[0];i++) {
        float maxv = in[0];
        for(int j=1;j<in_shape[1];j++)
            maxv = std::max(in[j],maxv);
        float sum = 0.0f;
        for(int j=0;j<in_shape[1];j++) 
            sum += out[j] = expf(in[j] - maxv);
        float factor = 1.0f/sum;
        for(int j=0;j<in_shape[1];j++) 
            out[j] *= factor;
        in += in_shape[1];
        out+= in_shape[1];
    }

}

void SoftMax::forward_gpu(Tensor &input, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(in_shape[1] == sm_range_);
    kernel_.setArg(0,in_shape[0]);
    kernel_.setArg(1,sm_range_);
    kernel_.setArg(2,input.device_buffer());
    kernel_.setArg(3,int(input.device_offset()));
    kernel_.setArg(4,output.device_buffer());
    kernel_.setArg(5,int(output.device_offset()));
    
    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event());
}

void SoftMax::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2);
    DLPRIM_CHECK(input[0].shape() == output[0].shape());
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],output[0]);
    }
    else {
        forward_gpu(input[0],output[0],ctx);
    }
}

void SoftMax::backward_data(std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            ExecutionContext const &)
{
    throw NotImplementedError("softmax::backward_data not implemented");
}

Elementwise::Elementwise(Context &ctx,ElementwiseConfig config,DataType dtype) :
    Operator(ctx),
    config_(config),
    dtype_(dtype)
{
    DLPRIM_CHECK(dtype_ == float_data);
}

Elementwise::~Elementwise()
{
}

void Elementwise::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    DLPRIM_CHECK(in[1].dtype() == dtype_);
    DLPRIM_CHECK(in[0].shape() == in[1].shape());
    out.assign({in[0]});
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"eltwise",
                                        "ACTIVATION",int(config_.activation),
                                        "ELTOP",int(config_.op));
    kernel_ = cl::Kernel(prog,"eltwise");
}

void Elementwise::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0] == in[1]);
    out.assign({in[0]});
}

void Elementwise::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(input[0].shape() == input[1].shape());
    DLPRIM_CHECK(output[0].shape() == input[0].shape());
    
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(input[1].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],input[1],output[0]);
    }
    else {
        forward_gpu(input[0],input[1],output[0],e);
    }
}

void Elementwise::forward_cpu(Tensor &a,Tensor &b,Tensor &c)
{
    size_t size = a.shape().total_size();
    float *ap=a.data<float>();
    float *bp=b.data<float>();
    float *cp=c.data<float>();
    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        memcpy(cp,bp,sizeof(float)*size);
        cblas_saxpby(size,config_.coeff[0],ap,1,config_.coeff[1],cp,1); 
        break;
    case ElementwiseConfig::elementwise_prod:
        {
            float c = config_.coeff[0] * config_.coeff[1];
            for(size_t i=0;i<size;i++) 
                cp[i] = c * ap[i] * bp[i];
        }
        break;
    case ElementwiseConfig::elementwise_max:
        {
            float c1 = config_.coeff[0];
            float c2 = config_.coeff[1];
            for(size_t i=0;i<size;i++) 
                cp[i] = std::max(c1 * ap[i], c2 * bp[i]);
        }
        break;
    }
    cpu::apply_activation(cp,size,config_.activation);
}

void Elementwise::forward_gpu(Tensor &a,Tensor &b,Tensor &c,ExecutionContext const &ctx)
{
    int p=0;
    int size = a.shape().total_size();
    kernel_.setArg(p++,size);
    kernel_.setArg(p++,a.device_buffer());
    kernel_.setArg(p++,int(a.device_offset()));
    kernel_.setArg(p++,b.device_buffer());
    kernel_.setArg(p++,int(b.device_offset()));
    kernel_.setArg(p++,c.device_buffer());
    kernel_.setArg(p++,int(c.device_offset()));
    kernel_.setArg(p++,config_.coeff[0]);
    kernel_.setArg(p++,config_.coeff[1]);
    
    cl::NDRange gr((size + 255) / 256 * 256);
    cl::NDRange wg(256);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event());
    
}

void Elementwise::backward_data(std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                ExecutionContext const &)
{
    throw NotImplementedError("Elementwise::backward_data not implemented");
}


} // dlprim


