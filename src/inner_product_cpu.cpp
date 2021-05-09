#include <dlprim/operators.hpp>
#include <dlprim/cpu/cpu_ops.hpp>

namespace dlprim {

    class InnerProductImpl {
    public:
        InnerProductImpl(InnerProduct *s) : self(s) {}
        virtual ~InnerProductImpl() {}
		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             cl::CommandQueue &q,cl::Event *even = nullptr) = 0;
    protected:
        InnerProduct *self;
    };

    void InnerProduct::setup(std::vector<TensorSpecs> const &in,
                             std::vector<TensorSpecs> &out,
                             std::vector<TensorSpecs> &params,
                             size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].dtype() == dtype_);

        auto in_shape = in[0].shape();
        DLPRIM_CHECK(in_shape.size() > 1);
        int bs = in_shape[0];
        int in_features = in_shape.size_no_batch();
        if(config_.inputs == -1)
            config_.inputs = in_features;
        else
            DLPRIM_CHECK(config_.inputs == in_features);

        Shape out_shape = Shape(bs,config_.outputs);
        Shape params_shape = Shape(config_.outputs,config_.inputs);

        out.assign({ TensorSpecs(out_shape,dtype_) });
        params.assign({TensorSpecs(params_shape,dtype_) })
        if(config_.bias)
            params.push_back(TensorSpecs(Shape(config_.outputs),dtype_));
        ws  = 0;
    }
    
    void InnerProduct::reshape(std::vector<Shape> const &in,
                               std::vector<Shape> &out)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(out.size() == 1);
        DLPRIM_CHECK(in[0].dtype() == dtype_);

        auto in_shape = in[0].shape();
        DLPRIM_CHECK(in_shape.size() > 1);
        int bs = in_shape[0];
        int in_features = in_shape.size_no_batch();
        DLPRIM_CHECK(config_.inputs == in_features);
        Shape out_shape = Shape(bs,config_.outputs);

        out[0] = out_shape;
    }

    template<typename T> 
    class InnerProductCPU;

    template<typename T> 
    class InnerProductGPU;

    template<>
    class InnerProductCPU<float> : public InnerProductImpl {
    public:
        InnerProductCPU(InnerProduct *self) : InnerProduct(self)
        {
            
        }
        virtual void forward(std::vector<Tensor> &input,
                                 std::vector<Tensor> &output,
                                 cl::CommandQueue &,cl::Event *)
        {
            DLPRIM_CHECK(in.size() == 1);
            DLPRIM_CHECK(out.size() == 1);
            DLPRIM_CHECK(in[0].shape()[1] == config_.inputs); 
            DLPRIM_CHECK(out[0].shape()[1] == config_.outputs); 

            int batch = in[0].shape()[0];

            float *in_data = in[0].data<float>();
            float *out_data = out[0].data<float>();
            float *param = self->parameters()[0].data<float>();

            cblas_sgemm(CblasRowMajor,
                CblasNoTrans,CblasTrans,
                config_.inputs,config.outputs_,batch,
                1.0f,
                in_data,config_.inputs,
                param_,config_.inputs_,
                0.0f,
                out_data,config_.outputs);

            if(config_.bias) {
                float *bias = parameters_[1].data<float>();
                for(int i=0;i<batch;i++) {
                    cblas_saxpy(config_.outputs,1.0f,bias,1,in_data + config_.inputs * i,1)
                }
            }
            cpu::apply_activation(out_data,out[0].shape().total_size(),config_.activation);
        }
    };

    InnerProduct::InnerProduct(Context &ctx,InnerProductConfig const &cfg,DataType dtype):
        Operator(ctx),
        config_(cfg),
        dtype_(dtype)
    {
        DLPRIM_CHECK(dtype_ == float_data);
        if(ctx_.is_cpu_context())
            impl_.reset(new InnerProductCPU<float>(this));
        else
            impl_.reset(new InnerProductGPU<float>(this));
    }

    template<>
    class InnerProductGPU<float> : public InnerProductImpl {
    public:
        InnerProductGPU(InnerProduct *self) : InnerProduct(self)
        {
            char const *act_src = gpu::activation_function(self->dtype_,self->config().activation);
            char const *kernel_bias = R"xxx(
            __kernel ip_fwd_bias_float(int batch,int N,
                        __global float *data,int data_offset,
                        __global const float *bias,int bias_offset)
            {
                data+=data_offset;
                bias+=bias_offset;
                int r = get_global_id(0);
                int c = get_global_id(1);
                if(r >= batch || c >= N)
                    return;
                data += r*N+c;
                bias+=c;
                float v = *data + *bias;
                v=ACTIVATION(v);
                *data = v;
            }
            )xxx";

        }
        virtual void forward(std::vector<Tensor> &input,
                                 std::vector<Tensor> &output,
                                 cl::CommandQueue &q,cl::Event *event)
        {
            DLPRIM_CHECK(in.size() == 1);
            DLPRIM_CHECK(out.size() == 1);
            DLPRIM_CHECK(in[0].shape()[1] == config_.inputs); 
            DLPRIM_CHECK(out[0].shape()[1] == config_.outputs); 

            int batch = in[0].shape()[0];

            Tensor &in = in[0];
            Tensor &out = out[0];
            Tensor &param = self->parameters()[0];

            cblast::Gemm(
                clblast::Layout::kRowMajor,
                clblast::Transpose::kNo,clblast::Transpose::kYes,
                config_.inputs,config.outputs_,batch,
                1.0f,
                in.device_buffer()(),in.offset(),config_.inputs,
                param.device_buffer()(),param.offset(),config_.inputs_,
                0.0f,
                out.device_buffer()(),out.offset(),config_.outputs,
                q(),event);

            if(config_.bias) {
                float *bias = self->parameters()[1].data<float>();
                for(int i=0;i<batch;i++) {
                    cblas_saxpy(config_.outputs,1.0f,bias,1,in_data + config_.inputs * i,1)
                }
            }
            cpu::apply_activation(out_data,out[0].shape().total_size(),config_.activation);
        }
    };


}
