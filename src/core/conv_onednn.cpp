#include <dlprim/core/conv.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include "conv_onednn.hpp"

#include <iostream>


namespace dlprim {
namespace core {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;


    class Conv2DForwardOneDNN : public Conv2DForward {
    public:

        virtual ~Conv2DForwardOneDNN() {}
        Conv2DForwardOneDNN(Context &ctx,
                            Conv2DSettings const &config,
                            bool bias,
                            StandardActivations activation = StandardActivations::identity):
                config_(config),
                activation_(activation)
        {
            engine_ = dnnl::ocl_interop::make_engine(ctx.device()(),ctx.context()());

            dnnl::memory::dims strides(config_.stride,config_.stride+2);
            dnnl::memory::dims dilate(config_.dilate,config_.dilate+2);
            dilate[0]-=1;
            dilate[1]-=1;
            dnnl::memory::dims pad(config_.pad,config_.pad+2);

            Shape y_shape = Conv2DBase::get_output_shape(config_,config_.shape);
            if(config_.groups == 1) {
                w_shape_ = Shape(config_.channels_out,config_.channels_in,config_.kernel[0],config_.kernel[1]);
                w_tag_ = tag::oihw;
            }
            else {
                w_shape_ = Shape(config_.groups,
                                 config_.channels_out/config_.groups,
                                 config_.channels_in/config_.groups,
                                 config_.kernel[0],config_.kernel[1]);
                w_tag_ = tag::goihw;
            }

            TensorSpecs x(config_.shape,config_.dtype);
            TensorSpecs y(y_shape,config_.dtype);

            TensorSpecs w(w_shape_,config_.dtype);
            TensorSpecs b(Shape(config_.channels_out),config_.dtype);

            #if 0
            auto conv_desc = dnnl::convolution_forward::desc(
                                dnnl::prop_kind::forward_training,
                                dnnl::algorithm::convolution_auto,
                                mem_desc(x,tag::nchw),
                                mem_desc(w,w_tag_),
                                (bias ? mem_desc(b,tag::a) : dnnl::memory::desc()),
                                mem_desc(y,tag::nchw),
                                strides,pad,pad);
            
            auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc,engine_);
            #endif
            auto conv_pd = dnnl::convolution_forward::primitive_desc(
                            engine_,
                            dnnl::prop_kind::forward_training,
                            dnnl::algorithm::convolution_auto,
                            mem_desc(x,tag::nchw),
                            mem_desc(w,w_tag_),
                            (bias ? mem_desc(b,tag::a) : dnnl::memory::desc()),
                            mem_desc(y,tag::nchw),
                            strides,
                            dilate,
                            pad,pad);

            conv_prim_ = dnnl::convolution_forward(conv_pd);
        }

        virtual char const *algo() const 
        {
            return "intel_onednn";
        }

        static dnnl::memory::desc mem_desc(TensorSpecs const &t,dnnl::memory::format_tag tag)
        {
            dnnl::memory::dims d(t.shape().begin(),t.shape().end());
            // FIXME offset, FIXME f32, FIXME nchw
            dnnl::memory::desc ds(d,dt::f32, tag);
            return ds;
        }
        static dnnl::memory mem_from_tensor(Tensor &t,dnnl::engine &e,dnnl::memory::format_tag tag)
        {
            return dnnl::ocl_interop::make_memory(mem_desc(t.specs(),tag),e,t.device_buffer()());
        }
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y,Tensor &ws,float factor,ExecutionContext const &e)
        {
            dnnl::stream stream = dnnl::ocl_interop::make_stream(engine_,e.queue()());

            auto reshaped_w = w.sub_tensor(0,w_shape_,w.dtype());
            auto X = mem_from_tensor(x,engine_,tag::nchw);
            auto W = mem_from_tensor(reshaped_w,engine_,w_tag_);
            auto Y = mem_from_tensor(y,engine_,tag::nchw);
            dnnl::memory B;

            if(bias) {
                B = mem_from_tensor(*bias,engine_,tag::a);
            }

            std::unordered_map<int, dnnl::memory> conv_args;
            conv_args[DNNL_ARG_SRC]=X;
            conv_args[DNNL_ARG_WEIGHTS]=W;
            conv_args[DNNL_ARG_BIAS]=B;
            conv_args[DNNL_ARG_DST]=Y;

            conv_prim_.execute(stream, conv_args);
        }

        Conv2DSettings config_;
        StandardActivations activation_;
        Shape w_shape_;
        tag w_tag_;
        dnnl::engine engine_;
        dnnl::convolution_forward conv_prim_;
    };

    bool is_fwd_onednn_compatible(Context &ctx,Conv2DSettings const &config,StandardActivations activation)
    {
        if(!ctx.is_intel())
            return false;
        /*if(config.dilate[0] != 1 || config.dilate[1] != 1)
            return false;*/
        if(activation != StandardActivations::identity)
            return false;
        return true;
    }

    std::unique_ptr<Conv2DForward> fwd_onednn_create(Context &ctx,Conv2DSettings const &config,bool bias,StandardActivations activation)
    {
        std::unique_ptr<Conv2DForward> r(new Conv2DForwardOneDNN(ctx,config,bias,activation));
        return r;
    }



} // core
} // dlprim
