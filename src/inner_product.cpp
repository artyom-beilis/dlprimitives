#include <dlprim/operators.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
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

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"sgemm",
                                        "BIAS", (config_.bias ? 2 : 0),
                                        "BTRANS",1,
                                        "ACTIVATION",int(config_.activation));
        kernel_ = cl::Kernel(prog,"sgemm");

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
        constexpr int tile_size = 128;
        constexpr int block_size = 8;
        int ls = tile_size / block_size; // blocksize/tile-size
        int gs0 = (batch           + tile_size - 1) / tile_size * tile_size / block_size;
        int gs1 = (config_.outputs + tile_size - 1) / tile_size * tile_size / block_size;

        Tensor &M = parameters()[0];

        int ind=0;
        kernel_.setArg(ind++,batch);
        kernel_.setArg(ind++,config_.outputs);
        kernel_.setArg(ind++,config_.inputs);

        kernel_.setArg(ind++,in.device_buffer());
        kernel_.setArg(ind++,int(in.device_offset()));
        kernel_.setArg(ind++,config_.inputs);

        kernel_.setArg(ind++,M.device_buffer());
        kernel_.setArg(ind++,int(M.device_offset()));
        kernel_.setArg(ind++,config_.inputs);

        kernel_.setArg(ind++,out.device_buffer());
        kernel_.setArg(ind++,int(out.device_offset()));
        kernel_.setArg(ind++,config_.outputs);

        if(config_.bias) {
            Tensor &bias = parameters()[1];
            kernel_.setArg(ind++,bias.device_buffer());
            kernel_.setArg(ind++,int(bias.device_offset()));
        }
        cl::NDRange global(gs0,gs1);
        cl::NDRange local(ls,ls);
        ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,global,local,ctx.events(),ctx.event("ip_gemm"));
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
