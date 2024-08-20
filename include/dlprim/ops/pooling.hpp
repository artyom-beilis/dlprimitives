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
    namespace json { class value; }
    namespace core { 
        class Pooling2DBackwardBase;
        class Pooling2DForward;
    }
    struct PoolingBase {
        enum Mode {
            max = 0,
            avg = 1
        };

        Mode mode = max;

        static PoolingBase from_json(json::value const &v);
    };

    struct Pooling2DConfig : public PoolingBase {
        int kernel[2] = {1,1};
        int pad[2] = {0,0};
        int stride[2]={1,1};
        bool ceil_mode=false;
        bool count_include_pad = false;
        static Pooling2DConfig from_json(json::value const &v);
    };

    class Pooling2D : public Operator {
    public:
        Pooling2D(Context &ctx,Pooling2DConfig config = Pooling2DConfig());
        virtual ~Pooling2D();

        virtual char const *operator_type() const
        {
            return "Pooling2D";
        }
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

    private:
        Shape calc_shape(Shape ins);
        int calc_output_size(int in_size,int dim);
   		void forward_gpu(Tensor &in,Tensor &output,ExecutionContext const &ctx);
   		void backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ctx);
        
        template<typename Dtype,typename ReduceOpts>
        void forward_cpu(Tensor &in,Tensor &output,ReduceOpts rop);

        void backward_cpu_max(Tensor &x,Tensor &dx,Tensor &dy,float factor);
        template<typename ReduceOpts>
        void backward_cpu_ave(Tensor &dx,Tensor &dy,float factor,ReduceOpts rop);
        
        template<typename T>
        struct MaxRedcue;
        template<typename T>
        struct AveReduceFull;
        template<typename T>
        struct AveReduceValid;
        
        
        struct ReduceMax;
        struct ReduceAve;
        struct NormIdent;
        struct NormPartIdent;
        struct NormAve;
        struct NormPartAve;
        
        Pooling2DConfig config_;
        DataType dtype_;

        std::unique_ptr<core::Pooling2DForward> fwd_;
        std::unique_ptr<core::Pooling2DBackwardBase> bwd_;
    };

    struct GlobalPoolingConfig : public PoolingBase {
        static GlobalPoolingConfig from_json(json::value const &v)
        {
            GlobalPoolingConfig cfg;
            static_cast<PoolingBase &>(cfg) = PoolingBase::from_json(v);
            return cfg;
        }
    };

    class GlobalPooling : public Operator {
    public:
        GlobalPooling(Context &ctx,GlobalPoolingConfig const &config = GlobalPoolingConfig());
        virtual ~GlobalPooling();
        
        virtual char const *operator_type() const
        {
            return "GlobalPooling";
        }
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

    private:
   		void forward_gpu(Tensor &input,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &input,Tensor &output);
        void backward_cpu(Tensor &x,Tensor &dx,Tensor &dy,float factor);
        void backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ctx);
        size_t setup_kernel(Shape const &sp);

        GlobalPoolingConfig cfg_;
        DataType dtype_;
        std::unique_ptr<core::Pooling2DForward> fwd_;
        std::unique_ptr<core::Pooling2DBackwardBase> bwd_;
    };
    
} // namespace
