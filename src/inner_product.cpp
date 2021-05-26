#include <dlprim/operators.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
#include <cblas.h>

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
    
    InnerProduct::InnerProduct(Context &ctx,InnerProductConfig const &cfg,CalculationsMode mode) :
        OperatorWithParameters(ctx,mode),
        config_(cfg),
        dtype_(float_data)
    {
        DLPRIM_CHECK(config_.outputs > 0);
        DLPRIM_CHECK(dtype_==float_data);
    }
    void InnerProduct::setup(std::vector<TensorSpecs> const &in,
                             std::vector<TensorSpecs> &out,
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
        std::vector<TensorSpecs> params;
        params.push_back(TensorSpecs(Shape(config_.outputs,config_.inputs),dtype_));
        if(config_.bias) 
            params.push_back(TensorSpecs(Shape(config_.outputs),dtype_));

        int batch = in[0].shape()[0];

        out.assign({TensorSpecs(Shape(batch,config_.outputs),in[0].dtype())});
        workspace = 0;
        setup_parameters(std::move(params));

        if(ctx_.is_cpu_context())
            return;
        
        gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
            ctx_,dtype_,false,true,
            batch,config_.outputs,config_.inputs,
            (config_.bias ? gpu::GEMM::bias_N : gpu::GEMM::no_bias),
            config_.activation            
        ));
    }

    void InnerProduct::reshape(std::vector<Shape> const &in,
                               std::vector<Shape> &out)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].size() >= 2);
        DLPRIM_CHECK(int(in[0].size_no_batch()) == config_.inputs);
        out.assign({Shape(in[0][0],config_.outputs)});
    }

    void InnerProduct::forward(std::vector<Tensor> &in,std::vector<Tensor> &out,
            ExecutionContext const &ectx)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(out.size() == 1);
        DLPRIM_CHECK(in[0].shape()[0] == out[0].shape()[0]);
        DLPRIM_CHECK(int(in[0].shape().size_no_batch()) == config_.inputs);
        DLPRIM_CHECK(out[0].shape()[1] == config_.outputs);
        DLPRIM_CHECK(out[0].shape().size() == 2);
        DLPRIM_CHECK(parameters().size()==(1u+unsigned(config_.bias)));
        DLPRIM_CHECK(parameters()[0].shape() == Shape(config_.outputs,config_.inputs));
        if(config_.bias)
            DLPRIM_CHECK(parameters()[1].shape() == Shape(config_.outputs)); 

        if(ctx_.is_cpu_context())
            forward_cpu(in[0],out[0]);
        else
            forward_gpu(in[0],out[0],ectx);
    }

    void InnerProduct::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx)
    {
        int batch = in.shape()[0];
        Tensor &M = parameters()[0];

        int bias_offset = 0;
        cl::Buffer *bias_buffer = nullptr;
        
        if(config_.bias) {
            Tensor &bias = parameters()[1];
            bias_buffer = &bias.device_buffer();
        }
        
        gemm_->gemm(batch,config_.outputs,config_.inputs,
                    in.device_buffer(),in.device_offset(),config_.inputs,
                    M.device_buffer(),M.device_offset(),config_.inputs,
                    out.device_buffer(),out.device_offset(),config_.outputs,
                    bias_buffer,bias_offset,
                    ctx.queue(),ctx.events(),ctx.event("ip_gemm"));
    }

    void InnerProduct::forward_cpu(Tensor &in,Tensor &out)
    {
        int batch = in.shape()[0];
        float *a = in.data<float>();
        float *b = out.data<float>();
        float *M = parameters()[0].data<float>();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    batch,config_.outputs,config_.inputs,
                    1.0f,
                    a,config_.inputs,
                    M,config_.inputs,
                    0.0f,
                    b,config_.outputs);
        if(config_.bias) {
            float *bias = parameters()[1].data<float>();
            for(int i=0;i<batch;i++) {
                cblas_saxpy(config_.outputs,1.0f,
                                bias,1,
                                b + i * config_.outputs,1);
            }
        }
        cpu::apply_activation(b,batch*config_.outputs,config_.activation);
    }
    void InnerProduct::backward_data(std::vector<Tensor> &,
                                   std::vector<Tensor> &,
                                   std::vector<Tensor> &,
                                   std::vector<Tensor> &,
                                   ExecutionContext const &)
    {
        throw NotImplementedError("InnerProduct::backward_data");
    }
        
    void InnerProduct::backward_param(std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                ExecutionContext const &)
    {
        throw NotImplementedError("InnerProduct::backward_param");
    }
} // dlprim
