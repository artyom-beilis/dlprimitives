#include <dlprim/ops/softmax.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <cblas.h>

namespace dlprim {
SoftMax::~SoftMax() {}

SoftMax::SoftMax(Context &ctx,SoftMaxConfig const &) : 
    Operator(ctx),
    dtype_(float_data),wg_size_(0),items_per_wi_(0),sm_range_(-1)
{
}

void SoftMax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    par.clear();
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
    for(int i=0;i<int(in_shape[0]);i++) {
        float maxv = in[0];
        for(int j=1;j<int(in_shape[1]);j++)
            maxv = std::max(in[j],maxv);
        float sum = 0.0f;
        for(int j=0;j<int(in_shape[1]);j++) 
            sum += out[j] = expf(in[j] - maxv);
        float factor = 1.0f/sum;
        for(int j=0;j<int(in_shape[1]);j++) 
            out[j] *= factor;
        in += in_shape[1];
        out+= in_shape[1];
    }

}

void SoftMax::forward_gpu(Tensor &input, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    kernel_.setArg(0,int(in_shape[0]));
    kernel_.setArg(1,sm_range_);
    kernel_.setArg(2,input.device_buffer());
    kernel_.setArg(3,int(input.device_offset()));
    kernel_.setArg(4,output.device_buffer());
    kernel_.setArg(5,int(output.device_offset()));
    
    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("softmax"));
}

void SoftMax::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
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

} // namespace

