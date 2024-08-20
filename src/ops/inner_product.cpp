///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/inner_product.hpp>
#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/ops/activation.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/ops/initialization.hpp>
#include <dlprim/shared_resource.hpp>
#include <dlprim/json.hpp>
#include <my_cblas.hpp>

namespace dlprim {
   
    InnerProductConfig InnerProductConfig::from_json(json::value const &v)
    {
        InnerProductConfig cfg;
        cfg.inputs = v.get("inputs",cfg.inputs);
        cfg.outputs = v.get<int>("outputs");
        cfg.bias = v.get("bias",cfg.bias);
        cfg.activation = utils::activation_from_json(v); 
        return cfg;
    }
    InnerProduct::~InnerProduct()
    {
    }
    
    void InnerProduct::initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e)
    {
        float range = 1.0f / std::sqrt(1.0f * config_.inputs);
        set_to_urandom(parameters.at(0),shared_resource().rng_state(),-range,range,e);
        if(config_.bias)
            set_to_urandom(parameters.at(1),shared_resource().rng_state(),-range,range,e);
    }
    
    InnerProduct::InnerProduct(Context &ctx,InnerProductConfig const &cfg) :
        Operator(ctx),
        config_(cfg),
        dtype_(float_data)
    {
        DLPRIM_CHECK(config_.outputs > 0);
        DLPRIM_CHECK(dtype_==float_data);
    }
    void InnerProduct::setup(std::vector<TensorSpecs> const &in,
                             std::vector<TensorSpecs> &out,
                             std::vector<TensorSpecs> &params,
                             size_t &workspace)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].shape().size() >= 2);
        if(config_.inputs == -1) {
            config_.inputs = in[0].shape().size_no_batch();
        }
        else {
            DLPRIM_CHECK(config_.inputs == int(in[0].shape().size_no_batch()));
        }
        params.push_back(TensorSpecs(Shape(config_.outputs,config_.inputs),dtype_));
        if(config_.bias) 
            params.push_back(TensorSpecs(Shape(config_.outputs),dtype_));

        int batch = in[0].shape()[0];

        out.assign({TensorSpecs(Shape(batch,config_.outputs),in[0].dtype())});
        workspace = 0;

        if(mode_ != CalculationsMode::predict) { 
            if(config_.bias) {
                bwd_bias_.reset(new BWBias(ctx_,out[0].shape(),dtype_));
                workspace = bwd_bias_->workspace();
            }
            
            if(config_.activation != StandardActivations::identity) {
                ActivationConfig cfg;
                cfg.activation = config_.activation;
                activation_.reset(new Activation(ctx_,cfg));
                activation_->mode(mode_);
                std::vector<TensorSpecs> o,p;
                size_t ws=0;
                activation_->setup(out,o,p,ws);
                workspace = std::max(workspace,ws);
            }
        }
 

        if(ctx_.is_cpu_context()) {
            return;
        }
        
        core::IPSettings cfg;
        cfg.inputs = config_.inputs;
        cfg.outputs = config_.outputs;
        cfg.dtype = dtype_;
        cfg.optimal_batch_size = batch;

        ip_ = std::move(core::IPForward::create(ctx_,cfg,config_.bias,config_.activation));
       
        if(mode_ == CalculationsMode::predict)
            return;

        bwd_ip_ = std::move(core::IPBackwardData::create(ctx_,cfg));
        bwd_weights_ip_ = std::move(core::IPBackwardFilter::create(ctx_,cfg));
    }

    void InnerProduct::reshape(std::vector<Shape> const &in,
                               std::vector<Shape> &out,
                               size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].size() >= 2);
        DLPRIM_CHECK(int(in[0].size_no_batch()) == config_.inputs);
        out.assign({Shape(in[0][0],config_.outputs)});
        if(mode_ != CalculationsMode::predict && config_.bias) {
            bwd_bias_.reset(new BWBias(ctx_,out[0],dtype_));
        }
        ws = 0;
        if(bwd_bias_) {
            ws = std::max(ws,bwd_bias_->workspace());
        }
        if(activation_) {
            std::vector<Shape> same;
            size_t act_ws = 0;
            activation_->reshape(out,same,act_ws);
            DLPRIM_CHECK(out == same);
            ws = std::max(act_ws,ws);
        }
    }

    void InnerProduct::forward(std::vector<Tensor> &in,std::vector<Tensor> &out,std::vector<Tensor> &parameters,Tensor &,
            ExecutionContext const &ectx)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(out.size() == 1);
        DLPRIM_CHECK(in[0].shape()[0] == out[0].shape()[0]);
        DLPRIM_CHECK(int(in[0].shape().size_no_batch()) == config_.inputs);
        DLPRIM_CHECK(int(out[0].shape()[1]) == config_.outputs);
        DLPRIM_CHECK(out[0].shape().size() == 2);
        DLPRIM_CHECK(parameters.size()==(1u+unsigned(config_.bias)));
        DLPRIM_CHECK(parameters[0].shape() == Shape(config_.outputs,config_.inputs));
        Tensor &M = parameters[0];
        Tensor *bias = nullptr; 
        if(config_.bias)  {
            DLPRIM_CHECK(parameters[1].shape() == Shape(config_.outputs)); 
            bias = &parameters[1];
        }

        if(ctx_.is_cpu_context())
            forward_cpu(in[0],out[0],M,bias);
        else
            ip_->enqueue(in[0],M,bias,out[0],ectx);
    }

    void InnerProduct::forward_cpu(Tensor &in,Tensor &out,Tensor &mat,Tensor *bias)
    {
        int batch = in.shape()[0];
        float *a = in.data<float>();
        float *b = out.data<float>();
        float *M = mat.data<float>();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    batch,config_.outputs,config_.inputs,
                    1.0f,
                    a,config_.inputs,
                    M,config_.inputs,
                    0.0f,
                    b,config_.outputs);
        if(config_.bias) {
            DLPRIM_CHECK(bias);
            float *bptr = bias->data<float>();
            for(int i=0;i<batch;i++) {
                cblas_saxpy(config_.outputs,1.0f,
                                bptr,1,
                                b + i * config_.outputs,1);
            }
        }
        cpu::apply_activation(b,batch*config_.outputs,config_.activation);
    }

    void InnerProduct::backward(std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &parameters,
                                Tensor &ws,
                                ExecutionContext const &e)
    {
        int steps = 0,step=0;
        if(config_.activation != StandardActivations::identity)
            steps++;
        if(config_.bias && parameters[1].requires_gradient)
            steps++;
        if(parameters[0].requires_gradient)
            steps++;
        if(input[0].requires_gradient)
            steps++;
        if(config_.activation != StandardActivations::identity) {
            std::vector<TensorAndGradient> tmp({output[0]}),empty;
            tmp[0].requires_gradient = true;
            tmp[0].accumulate_gradient = 0.0;
            activation_->backward(tmp,tmp,empty,ws,e.generate_series_context(step++,steps));
        }
        if(config_.bias && parameters[1].requires_gradient) {
            bwd_bias_->backward(output[0].diff,
                                parameters[1].diff,
                                ws,
                                parameters[1].accumulate_gradient,
                                e.generate_series_context(step++,steps));
        }
        if(parameters[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                bwd_weights_ip_->enqueue(input[0].data,parameters[0].diff,output[0].diff,parameters[0].accumulate_gradient,ec);
            }
            else {
                backward_filter_cpu(output[0].diff,input[0].data,parameters[0].diff,
                                    parameters[0].accumulate_gradient);
            }
        }
        if(input[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                bwd_ip_->enqueue(input[0].diff,parameters[0].data,output[0].diff,input[0].accumulate_gradient,ec);
            }
            else {
                backward_data_cpu(output[0].diff,input[0].diff,parameters[0].data,
                                    input[0].accumulate_gradient);
            }
        }

    }

    void InnerProduct::backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dM,float factor)
    {
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                    config_.outputs,config_.inputs,dy.shape()[0],
                    1.0f,
                    dy.data<float>(),
                    config_.outputs,
                    x.data<float>(),
                    config_.inputs,
                    factor,
                    dM.data<float>(),
                    dM.shape()[1]);
    }
    void InnerProduct::backward_data_cpu(Tensor &dy,Tensor &dx,Tensor &M,float factor)
    {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    dy.shape()[0],config_.inputs,config_.outputs,
                    1.0f,
                    dy.data<float>(),
                    config_.outputs,
                    M.data<float>(),
                    M.shape()[1],
                    factor,
                    dx.data<float>(),
                    config_.inputs);
    }


} // dlprim
