///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/softmax.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {


bool SoftmaxBase::setup_kernel_params(int sm_range)
{
    if(sm_range_ == sm_range)
        return false;
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
    return true;
}


SoftmaxConfig SoftmaxConfig::from_json(json::value const &v) 
{ 
    SoftmaxConfig cfg;
    cfg.log = v.get<bool>("log",cfg.log);
    return cfg;
}


Softmax::~Softmax() {}
SoftmaxWithLoss::~SoftmaxWithLoss() {}

Softmax::Softmax(Context &ctx,SoftmaxConfig const &cfg) : 
    Operator(ctx),
    cfg_(cfg),
    dtype_(float_data)
{
}

SoftmaxWithLoss::SoftmaxWithLoss(Context &ctx,SoftmaxConfig const &) : 
    Operator(ctx),
    dtype_(float_data)
{
}

void Softmax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].shape().size() == 2 || in[0].shape().size() == 3);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    out = in;
    par.clear();
    ws = 0;
}
void SoftmaxWithLoss::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in[1].shape().total_size() == in[0].shape()[0]);
    DLPRIM_CHECK(in[1].shape()[0] == in[0].shape()[0]);
    DLPRIM_CHECK(in[1].dtype() == int32_data || in[1].dtype() == float_data);
    out = {TensorSpecs(Shape(1),dtype_)};
    if(in[1].dtype() == int32_data)
        itype_ = "int";
    else
        itype_ = "float";
    par.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0].shape()[1]);
}

void SoftmaxWithLoss::setup_kernel(int sm_range)
{
    if(!setup_kernel_params(sm_range))
        return;
    cl::Program const &prog_fwd = gpu::Cache::instance().get_program(ctx_,"softmax_with_loss",
                                                            "WG_SIZE",wg_size_,
                                                            "ITEMS_PER_WI",items_per_wi_,
                                                            "itype",itype_,
                                                            "CALC_LOSS",1);
    kernel_ = cl::Kernel(prog_fwd,"softmax");
    cl::Program const &prog_bwd = gpu::Cache::instance().get_program(ctx_,"softmax_with_loss",
                                                "WG_SIZE",wg_size_,
                                                "ITEMS_PER_WI",items_per_wi_,
                                                "itype",itype_,
                                                "CALC_LOSS",2);

    kernel_bwd_ = cl::Kernel(prog_bwd,"softmax");
    scal_.reset(new Scal(ctx_,dtype_));
}

void Softmax::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 2 || in[0].size() == 3);
    out = in;
    ws = 0;
}


void SoftmaxWithLoss::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].size() == 2);
    DLPRIM_CHECK(in[1].total_size() == in[0][0]);
    out = {Shape(1)};
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0][1]);
}

template<typename IndexType>
void SoftmaxWithLoss::forward_cpu_loss(Tensor &input,Tensor &label,Tensor &loss)
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
void SoftmaxWithLoss::backward_cpu_loss(Tensor &x,Tensor &dx,Tensor &label,Tensor &loss,float factor)
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


void Softmax::forward_cpu(Tensor &input,Tensor &output)
{
    Shape in_shape = input.shape();
    if(in_shape.size() == 2)
        in_shape = Shape(in_shape[0],in_shape[1],1);
    float *in0 = input.data<float>();
    float *out0 = output.data<float>();
    int step = in_shape[2];
    for(int i=0;i<int(in_shape[0]);i++) {
        for(int k=0;k<int(in_shape[2]);k++) {
            int offset = i*in_shape[1]*in_shape[2] + k;
            float *in = in0   + offset;
            float *out = out0 + offset;
            float maxv = in[0];
            for(int j=1;j<int(in_shape[1]);j++)
                maxv = std::max(in[j*step],maxv);
            float sum = 0.0f;
            if(cfg_.log) {
                for(int j=0;j<int(in_shape[1]);j++) 
                    sum += expf(in[j*step] - maxv);
                float factor = -logf(sum);
                for(int j=0;j<int(in_shape[1]);j++) 
                    out[j*step] = in[j*step] - maxv + factor;
            }
            else {
                for(int j=0;j<int(in_shape[1]);j++) 
                    sum += out[j*step] = expf(in[j*step] - maxv);
                float factor = 1.0f/sum;
                for(int j=0;j<int(in_shape[1]);j++) 
                    out[j*step] *= factor;
            }
        }
    }
}

void SoftmaxWithLoss::forward_gpu_loss(Tensor &input,Tensor &label, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_.setArg(p++,int(in_shape[0]));
    kernel_.setArg(p++,sm_range_);
    input.set_arg(kernel_,p);
    label.set_arg(kernel_,p);
    output.set_arg(kernel_,p);

    scal_->scale(0,output,ctx.generate_series_context(0,2));
    
    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    auto ec = ctx.generate_series_context(1,2);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ec.events(),ec.event("softmax_with_loss"));
}

void SoftmaxWithLoss::backward_gpu_loss(Tensor &input,Tensor &diff, Tensor &label,Tensor &output,float factor, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_bwd_.setArg(p++,int(in_shape[0]));
    kernel_bwd_.setArg(p++,sm_range_);
    input.set_arg(kernel_bwd_,p);
    diff.set_arg(kernel_bwd_,p);
    label.set_arg(kernel_bwd_,p);
    output.set_arg(kernel_bwd_,p);
    kernel_bwd_.setArg(p++,factor);

    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    ctx.queue().enqueueNDRangeKernel(kernel_bwd_,cl::NullRange,gr,wg,ctx.events(),ctx.event("softmax_with_loss_bwd"));
}

void Softmax::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2 || input[0].shape().size()==3);
    DLPRIM_CHECK(input[0].shape() == output[0].shape());
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],output[0]);
    }
    else {
        core::softmax_forward(input[0],output[0],cfg_.log,ctx);
    }
}

void Softmax::backward_cpu(Tensor &tdx,Tensor &ty,Tensor &tdy,float accum)
{
    int batch = tdx.shape()[0];
    int chan = tdx.shape()[1];
    int step = 1;
    if(tdx.shape().size()==3)
        step = tdx.shape()[2];
    float *dx = tdx.data<float>();
    float const *dy =tdy.data<float>();
    float const *y = ty.data<float>();
    if(accum == 0)
        memset(dx,0,tdx.memory_size());
    else
        cblas_sscal(tdx.shape().total_size(),accum,dx,1);

    if(cfg_.log) {
        for(int b=0;b<batch;b++) {
            for(int b2=0;b2<step;b2++) {
                float sum_dy = 0;
                for(int c=0;c<chan;c++) {
                    int pos = (b*chan+c)*step + b2;
                    sum_dy += dy[pos];
                }
                for(int c=0;c<chan;c++) {
                    int pos = (b*chan+c)*step + b2;
                    dx[pos] += dy[pos] - expf(y[pos]) * sum_dy;
                }
            }
        }
    }
    else {
        for(int b=0;b<batch;b++) {
            for(int b2=0;b2<step;b2++) {
                float sum_ydy = 0;
                for(int c=0;c<chan;c++) {
                    int pos = (b*chan+c)*step + b2;
                    sum_ydy += y[pos] * dy[pos];
                }
                for(int c=0;c<chan;c++) {
                    int pos = (b*chan+c)*step + b2;
                    dx[pos] += (dy[pos] - sum_ydy) * y[pos];
                }
            }
        }
    }
}

void Softmax::backward( std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &e)
{
    if(!input.at(0).requires_gradient)
        return;
    Tensor dx = input[0].diff;
    float accum = input[0].accumulate_gradient;
    Tensor dy = output.at(0).diff;
    Tensor y  = output.at(0).data;

    if(ctx_.is_cpu_context()) {
        backward_cpu(dx,y,dy,accum);
    }
    else {
        core::softmax_backward(dx,y,dy,cfg_.log,accum,e);
    }
}


void SoftmaxWithLoss::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2);
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(input[1].dtype() == dtype_ || input[1].dtype() == int32_data);
    DLPRIM_CHECK(output[0].shape().total_size() == 1);
    if(ctx_.is_cpu_context()) {
        if(input[1].dtype() == float_data)
            forward_cpu_loss<float>(input[0],input[1],output[0]);
        else if(input[1].dtype() == int32_data)
            forward_cpu_loss<int>(input[0],input[1],output[0]);
        else
            throw ValidationError("Invalid data type " + std::to_string(output[0].dtype()));
    }
    else {
        forward_gpu_loss(input[0],input[1],output[0],ctx);
    }
}

void SoftmaxWithLoss::backward( std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &ec)
{
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

