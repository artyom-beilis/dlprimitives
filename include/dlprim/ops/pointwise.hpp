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
  
    class PointwiseBase : public Operator {
    public:
        
        PointwiseBase(Context &ctx);
        virtual ~PointwiseBase();
        
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
                             Tensor &ws,
                             ExecutionContext const &q);

        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);
    protected:
        virtual void forward_cpu(Tensor &x,Tensor &y); 
        virtual void backward_cpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta);

        virtual void forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q) = 0;
        virtual void backward_gpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta,ExecutionContext const &q) = 0;
        virtual void forward_cpu_float(size_t n,float const *x,float *y) = 0;
        virtual void backward_cpu_float(size_t n,float const *x,float *dx,float const *y,float const *dy,float beta) = 0;
    };

    struct ThresholdConfig {
        float threshold = 0.0f;
        static ThresholdConfig from_json(json::value const &v);
    };

    class Threshold : public PointwiseBase {
    public:
        Threshold(Context &ctx,ThresholdConfig const &cfg = ThresholdConfig()) : 
            PointwiseBase(ctx),
            cfg_(cfg)
        {
        }
        virtual char const *operator_type() const { return "Threshold"; }
        virtual void forward_cpu_float(size_t n,float const *x,float *y);
        virtual void backward_cpu_float(size_t n,float const *x,float *dx,float const *y,float const *dy,float beta);
        virtual void forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q);
        virtual void backward_gpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta,ExecutionContext const &q);
    private:
        ThresholdConfig cfg_;
    };

    struct HardtanhConfig {
        float min_val = -1.0f;
        float max_val =  1.0f;
        static HardtanhConfig from_json(json::value const &v);
    };

    class Hardtanh : public PointwiseBase {
    public:
        Hardtanh(Context &ctx,HardtanhConfig const &cfg = HardtanhConfig()) : 
            PointwiseBase(ctx),
            cfg_(cfg)
        {
        }
        virtual char const *operator_type() const { return "Hardtanh"; }
        virtual void forward_cpu_float(size_t n,float const *x,float *y);
        virtual void backward_cpu_float(size_t n,float const *,float *,float const *,float const *,float beta);
        virtual void forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q);
        virtual void backward_gpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta,ExecutionContext const &q);
    private:
        HardtanhConfig cfg_;
    };

    struct AbsConfig {
        static AbsConfig from_json(json::value const &) { return AbsConfig(); }
    };

    class Abs : public PointwiseBase {
    public:
        Abs(Context &ctx,AbsConfig const & = AbsConfig()) : 
            PointwiseBase(ctx)
        {
        }
        virtual char const *operator_type() const { return "Abs"; }
        virtual void forward_cpu_float(size_t n,float const *x,float *y);
        virtual void backward_cpu_float(size_t n,float const *,float *,float const *,float const *,float beta);
        virtual void forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q);
        virtual void backward_gpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta,ExecutionContext const &q);
    };


} // namespace
 
