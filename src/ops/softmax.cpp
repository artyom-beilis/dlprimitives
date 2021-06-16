#include <dlprim/ops/softmax.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gpu_ops.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <cblas.h>

namespace dlprim {

SoftMaxConfig SoftMaxConfig::from_json(json::value const &v) 
{ 
    SoftMaxConfig cfg;
    cfg.loss = v.get("loss",cfg.loss);
    return cfg;
}


SoftMax::~SoftMax() {}

SoftMax::SoftMax(Context &ctx,SoftMaxConfig const &cfg) : 
    Operator(ctx),
    dtype_(float_data),
    config_(cfg),
    wg_size_(0),items_per_wi_(0),sm_range_(-1)
{
}

void SoftMax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==(1u + unsigned(config_.loss)));
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    if(config_.loss) {
        DLPRIM_CHECK(in[1].shape().total_size() == in[0].shape()[0]);
        DLPRIM_CHECK(in[1].shape()[0] == in[0].shape()[0]);
        DLPRIM_CHECK(in[1].dtype() == int32_data || in[1].dtype() == float_data);
        out = {TensorSpecs(Shape(1),dtype_)};
        if(in[1].dtype() == int32_data)
            itype_ = "int";
        else
            itype_ = "float";
    }
    else {
        out = in;
    }
    par.clear();
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

    sm_range_ = sm_range;
    int mpl = wg_size_ * items_per_wi_;
    nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
    
    if(config_.loss) {
        cl::Program const &prog_fwd = gpu::Cache::instance().get_program(ctx_,"softmax",
                                                                "WG_SIZE",wg_size_,
                                                                "ITEMS_PER_WI",items_per_wi_,
                                                                "itype",itype_,
                                                                "CALC_LOSS",1);
        kernel_ = cl::Kernel(prog_fwd,"softmax");
        cl::Program const &prog_bwd = gpu::Cache::instance().get_program(ctx_,"softmax","WG_SIZE",wg_size_,
                                                    "ITEMS_PER_WI",items_per_wi_,"CALC_LOSS",2,
                                                    "itype",itype_);
        kernel_bwd_ = cl::Kernel(prog_bwd,"softmax");
        scal_.reset(new gpu::Scal(ctx_,dtype_));
    }
    else {
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"softmax","WG_SIZE",wg_size_,"ITEMS_PER_WI",items_per_wi_);
        kernel_ = cl::Kernel(prog,"softmax");
    }
}


void SoftMax::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    if(config_.loss) {
        DLPRIM_CHECK(in.size()==2);
        DLPRIM_CHECK(in[0].size() == 2);
        DLPRIM_CHECK(in[1].total_size() == in[0][0]);
        out = {Shape(1)};
    }
    else {
        DLPRIM_CHECK(in.size()==1);
        DLPRIM_CHECK(in[0].size() == 2);
        out = in;
    }
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0][1]);
}

template<typename IndexType>
void SoftMax::forward_cpu_loss(Tensor &input,Tensor &label,Tensor &loss)
{
    Shape in_shape = input.shape();
    float *in  = input.data<float>();
    IndexType *lbl = label.data<IndexType>();
    float loss_value = 0;
    for(int i=0;i<int(in_shape[0]);i++) {
        float maxv = in[0];
        for(int j=1;j<int(in_shape[1]);j++)
            maxv = std::max(in[j],maxv);
        float sum = 0.0f;
        for(int j=0;j<int(in_shape[1]);j++) 
            sum += expf(in[j] - maxv);
        
        unsigned index = lbl[i];
        DLPRIM_CHECK(index < in_shape[1]);
        loss_value -= in[index]-maxv - logf(sum);
        in += in_shape[1];
    }
    loss_value /= in_shape[0];
    loss.data<float>()[0] = loss_value;
}

template<typename IndexType>
void SoftMax::backward_cpu_loss(Tensor &x,Tensor &dx,Tensor &label,Tensor &loss,float factor)
{
    Shape in_shape = x.shape();
    float *in  = x.data<float>();
    float *grad = dx.data<float>();
    float loss_value = loss.data<float>()[0] / in_shape[0];
    IndexType *lbl = label.data<IndexType>();
    int classes = in_shape[1];
    for(int i=0;i<int(in_shape[0]);i++) {
        float maxv = in[0];
        for(int j=1;j<classes;j++)
            maxv = std::max(in[j],maxv);
        float sum = 0.0f;
        for(int j=0;j<classes;j++) 
            sum += expf(in[j] - maxv);
        float f = 1.0f/sum;
        unsigned index = lbl[i];
        for(int j=0;j<classes;j++) {
            float sm = expf(in[j] - maxv) * f;
            float gr = loss_value * (sm - (int(index) == j));
            if(factor == 0)
                grad[j] = gr;
            else
                grad[j] = grad[j] * factor + gr;
        }
        in += classes;
        grad+= classes;
    }
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

void SoftMax::forward_gpu_loss(Tensor &input,Tensor &label, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_.setArg(p++,int(in_shape[0]));
    kernel_.setArg(p++,sm_range_);
    kernel_.setArg(p++,input.device_buffer());
    kernel_.setArg(p++,int(input.device_offset()));
    kernel_.setArg(p++,label.device_buffer());
    kernel_.setArg(p++,int(label.device_offset()));
    kernel_.setArg(p++,output.device_buffer());
    kernel_.setArg(p++,int(output.device_offset()));

    scal_->scale(0,output,ctx.generate_series_context(0,2));
    
    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    auto ec = ctx.generate_series_context(1,2);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ec.events(),ec.event("softmax_with_loss"));
}

void SoftMax::backward_gpu_loss(Tensor &input,Tensor &diff, Tensor &label,Tensor &output,float factor, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_bwd_.setArg(p++,int(in_shape[0]));
    kernel_bwd_.setArg(p++,sm_range_);
    kernel_bwd_.setArg(p++,input.device_buffer());
    kernel_bwd_.setArg(p++,int(input.device_offset()));
    kernel_bwd_.setArg(p++,diff.device_buffer());
    kernel_bwd_.setArg(p++,int(diff.device_offset()));
    kernel_bwd_.setArg(p++,label.device_buffer());
    kernel_bwd_.setArg(p++,int(label.device_offset()));
    kernel_bwd_.setArg(p++,output.device_buffer());
    kernel_bwd_.setArg(p++,int(output.device_offset()));
    kernel_bwd_.setArg(p++,factor);

    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    ctx.queue().enqueueNDRangeKernel(kernel_bwd_,cl::NullRange,gr,wg,ctx.events(),ctx.event("softmax_with_loss_bwd"));
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
    if(!config_.loss) {
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
    else {
        DLPRIM_CHECK(input.size()==2);
        DLPRIM_CHECK(output.size()==1); 
        DLPRIM_CHECK(input[0].shape().size()==2);
        DLPRIM_CHECK(input[0].dtype() == dtype_);
        DLPRIM_CHECK(input[1].dtype() == dtype_ || input[1].dtype() == int32_data);
        DLPRIM_CHECK(output[0].shape().total_size() == 1);
        if(ctx_.is_cpu_context()) {
            if(output[0].dtype() == float_data)
                forward_cpu_loss<float>(input[0],input[1],output[0]);
            else if(output[0].dtype() == int32_data)
                forward_cpu_loss<int>(input[0],input[1],output[0]);
            else
                throw ValidationError("Invalid data type " + std::to_string(output[0].dtype()));
        }
        else {
            forward_gpu_loss(input[0],input[1],output[0],ctx);
        }
    }
}

void SoftMax::backward( std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &ec)
{
    DLPRIM_CHECK(config_.loss || !input[0].requires_gradient);
    if(!input[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[1].requires_gradient == false);
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    float accum = input[0].accumulate_gradient;
    if(ctx_.is_cpu_context()) {
        if(input[1].data.dtype() == int32_data)
            backward_cpu_loss<int>(input[0].data,input[0].diff,input[1].data,output[0].diff,accum);
        else
            backward_cpu_loss<float>(input[0].data,input[0].diff,input[1].data,output[0].diff,accum);
    }
    else {
        backward_gpu_loss(input[0].data,input[0].diff,input[1].data,output[0].diff,accum,ec);
    }
}


} // namespace

