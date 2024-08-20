///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {    
    
    namespace gpu { class GEMM; }
    namespace core { class Conv2DForward; class Conv2DBackwardData; class Conv2DBackwardFilter; }
    class BWBias;
    class Scal;

    struct Convolution2DConfig : public Convolution2DConfigBase {
        bool bias = true;
        
        std::string fwd_algo="auto";
        std::string bwd_data_algo="auto";
        std::string bwd_filter_algo="auto";

        StandardActivations activation=StandardActivations::identity;
        static Convolution2DConfig from_json(json::value const &v);
    };

    class Convolution2DBase {
    protected:
        template<typename Op,typename DType>
        static void im2col(Shape const &in,Shape const &outs,DType *img_in,DType *mat_in,Convolution2DConfig const &config);
        static void fwd_bwd_cpu(GemmOpMode mode,Tensor &in,Tensor &out,Tensor &W,Tensor *bias_tensor,void *ws,Convolution2DConfig const &config,float fwd_beta=0.0f);
        static void scale_cpu(Tensor &t,float v);
    };

    class Convolution2D : public Operator, public Convolution2DBase {
    public:

        Convolution2D(Context &ctx,Convolution2DConfig const &cfg);
        virtual ~Convolution2D();

        virtual char const *operator_type() const
        {
            return "Convolution2D";
        }
        
        void initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e);

        virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

        virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);


        Shape get_output_shape(Shape const &in);
    protected:

        size_t calc_workspace(Shape const &in);
        void setup_algo(Shape const &in);
        int get_im2col_width();
        void forward_gpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,ExecutionContext const &ctx);
        void forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,void *ws);

        void backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dK,Tensor &ws,float factor);
        void backward_data_cpu(Tensor &dy,Tensor &K,Tensor &dx,Tensor &ws,float factor);



                
        Convolution2DConfig config_;
        DataType dtype_;
        std::unique_ptr<core::Conv2DForward> conv_;
        std::unique_ptr<core::Conv2DBackwardData> conv_bwd_data_;
        std::unique_ptr<core::Conv2DBackwardFilter> conv_bwd_filter_;
        
        std::unique_ptr<Operator>  activation_;
        std::unique_ptr<BWBias> bwd_bias_;
        size_t ws_size_;
        size_t out_h_,out_w_;
        size_t in_h_,in_w_;
        size_t bs_;

    };

    struct TransposedConvolution2DConfig : public Convolution2DConfig {
        int output_pad[2] = {0,0}; // Output padding - due to possible ambiguity of input/output size calcuulations

        static TransposedConvolution2DConfig from_json(json::value const &v);
    };

    class TransposedConvolution2D : public Operator, public Convolution2DBase {
    public:

        TransposedConvolution2D(Context &ctx,TransposedConvolution2DConfig const &cfg);
        virtual ~TransposedConvolution2D();

        virtual char const *operator_type() const
        {
            return "TransposedConvolution2D";
        }
        
        void initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e);

        virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

        virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);


        Shape get_output_shape(Shape const &in);
    protected:

        size_t calc_workspace(Shape const &in);
        void setup_algo(Shape const &in);
        int get_im2col_width();
        void forward_gpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,ExecutionContext const &ctx);
        void forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,void *ws);

        void backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dK,Tensor &ws,float factor);
        void backward_data_cpu(Tensor &dy,Tensor &K,Tensor &dx,Tensor &ws,float factor);


        TransposedConvolution2DConfig config_;
        // in/out channels switched
        Convolution2DConfig conv_config_;


        DataType dtype_;
        std::unique_ptr<core::Conv2DForward> conv_bwd_data_;
        std::unique_ptr<core::Conv2DBackwardData> conv_fwd_;
        std::unique_ptr<core::Conv2DBackwardFilter> conv_bwd_filter_;
        
        std::unique_ptr<Operator>  activation_;
        std::unique_ptr<BWBias> bwd_bias_;
        size_t ws_size_;
        size_t out_h_,out_w_;
        size_t in_h_,in_w_;
        size_t bs_;

    };



} // namespace

