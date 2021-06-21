#include <dlprim/ops/inner_product.hpp>
#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/ops/activation.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
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
            if(config_.bias)
                bwd_bias_.reset(new BWBias(ctx_,in[0].shape()[0],1,dtype_));
            
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
        
        gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
            ctx_,dtype_,false,true,
            batch,config_.outputs,config_.inputs,
            (config_.bias ? gpu::GEMM::bias_N : gpu::GEMM::no_bias),
            config_.activation            
        ));
       
        if(mode_ == CalculationsMode::predict)
            return;

        bwd_gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
            ctx_,dtype_,false,false,
            batch,config_.inputs,config_.outputs,
            gpu::GEMM::no_bias,
            StandardActivations::identity            
        ));

        bwd_weights_gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
            ctx_,dtype_,true,false,
            config_.outputs,config_.inputs,batch,
            gpu::GEMM::no_bias,
            StandardActivations::identity            
        ));

    }

    void InnerProduct::reshape(std::vector<Shape> const &in,
                               std::vector<Shape> &out)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].size() >= 2);
        DLPRIM_CHECK(int(in[0].size_no_batch()) == config_.inputs);
        out.assign({Shape(in[0][0],config_.outputs)});
        if(mode_ != CalculationsMode::predict && config_.bias && size_t(bwd_bias_->batch()) < in[0][0]) {
            bwd_bias_.reset(new BWBias(ctx_,in[0][0],1,dtype_));
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
            forward_gpu(in[0],out[0],M,bias,ectx);
    }

    void InnerProduct::forward_gpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,ExecutionContext const &ctx)
    {
        int batch = in.shape()[0];

        int bias_offset = 0;
        cl::Buffer *bias_buffer = nullptr;
        
        if(config_.bias) {
            DLPRIM_CHECK(bias);
            bias_buffer = &bias->device_buffer();
            bias_offset = bias->device_offset();
        }
        
        gemm_->gemm(batch,config_.outputs,config_.inputs,
                    in.device_buffer(),in.device_offset(),config_.inputs,
                    M.device_buffer(),M.device_offset(),config_.inputs,
                    out.device_buffer(),out.device_offset(),config_.outputs,
                    bias_buffer,bias_offset,0.0f,
                    out.shape().total_size(),
                    ctx);
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
                                parameters[1].accumulate_gradient,
                                e.generate_series_context(step++,steps));
        }
        if(parameters[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                backward_filter_gpu(output[0].diff,input[0].data,parameters[0].diff,
                                    parameters[0].accumulate_gradient,ec);
            }
            else {
                backward_filter_cpu(output[0].diff,input[0].data,parameters[0].diff,
                                    parameters[0].accumulate_gradient);
            }
        }
        if(input[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                backward_data_gpu(output[0].diff,input[0].diff,parameters[0].data,
                                    input[0].accumulate_gradient,ec);
            }
            else {
                backward_data_cpu(output[0].diff,input[0].diff,parameters[0].data,
                                    input[0].accumulate_gradient);
            }
        }

    }

    void InnerProduct::backward_filter_gpu(Tensor &dy,Tensor &x,Tensor &dM,float factor,ExecutionContext const &ec)
    {
        bwd_weights_gemm_->gemm(config_.outputs,config_.inputs,dy.shape()[0],
                                dy.device_buffer(),
                                dy.device_offset(),
                                config_.outputs,
                                x.device_buffer(),
                                x.device_offset(),
                                config_.inputs,
                                dM.device_buffer(),
                                dM.device_offset(),
                                dM.shape()[1],
                                nullptr,0,
                                factor,
                                dM.shape().total_size(),
                                ec);

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
    void InnerProduct::backward_data_gpu(Tensor &dy,Tensor &dx,Tensor &M,float factor,ExecutionContext const &ec)
    {
        bwd_gemm_->gemm(dy.shape()[0],config_.inputs,config_.outputs,
                        dy.device_buffer(),
                        dy.device_offset(),
                        config_.outputs,
                        M.device_buffer(),
                        M.device_offset(),
                        M.shape()[1],
                        dx.device_buffer(),
                        dx.device_offset(),
                        config_.inputs,
                        nullptr,0,
                        factor,
                        dx.shape().total_size(),
                        ec);

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
