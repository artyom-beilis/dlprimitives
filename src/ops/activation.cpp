///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/activation.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>
#include <my_cblas.hpp>

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
}

void Activation::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    out.assign({in[0]});
    ws=0;
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
        core::activation_forward(input[0],output[0],config_.activation,e);
    }
}

void Activation::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!input[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[0].diff.shape() == output[0].diff.shape());
    DLPRIM_CHECK(input[0].diff.shape() == output[0].data.shape());
    float accum = input[0].accumulate_gradient;
    if(ctx_.is_cpu_context()) {
        backward_cpu(output[0].data,output[0].diff,input[0].diff,accum);
    }
    else {
        core::activation_backward(input[0].diff,output[0].diff,output[0].data,config_.activation,accum,e);
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

void Activation::backward_cpu(Tensor &y,Tensor &dy,Tensor &dx,float beta)
{
    size_t size = y.shape().total_size();
    float *p_y =y.data<float>();
    float *p_dy=dy.data<float>();
    float *p_dx=dx.data<float>();
    cpu::apply_activation_diff(size,p_y,p_dy,p_dx,beta,config_.activation);
}


std::unique_ptr<Activation> Activation::get_bwd_op(Context &ctx,StandardActivations act,TensorSpecs spec)
{
    ActivationConfig cfg;
    cfg.activation = act;
    std::unique_ptr<Activation> p(new Activation(ctx,cfg));
    p->mode(CalculationsMode::train);
    std::vector<TensorSpecs> in={spec},out,par;
    size_t ws = 0;
    p->setup(in,out,par,ws);
    DLPRIM_CHECK(ws==0);
    DLPRIM_CHECK(par.empty());
    return p;
}

} // dlprim
