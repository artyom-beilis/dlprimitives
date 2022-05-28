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
    namespace core { class BatchNormFwdBwd; }

    struct BatchNormConfig {
        int features = -1;
        float eps = 1e-5f;
        float momentum = 0.1;
        bool affine = true;
        bool use_global_stats = false;
        static BatchNormConfig from_json(json::value const &v);
    };


    class BatchNorm : public Operator {
    public:
        
        BatchNorm(Context &ctx,BatchNormConfig const &config = BatchNormConfig(),DataType dtype=float_data);
        virtual ~BatchNorm();
        
        virtual char const *operator_type() const
        {
            return "BatchNorm";
        }
        
        virtual void initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e);
        virtual void mode(CalculationsMode m);
        virtual CalculationsMode mode() { return Operator::mode(); }

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
        void backward_cpu(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &parameters,
                          Tensor &workspace);
	    void forward_cpu(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace);
        void cpu_backward_data(Tensor &x,Tensor &dx,Tensor &dy,float *mean,float *var,float *dy_sum,float *dyx_sum,float *gamma_in);
        void cpu_forward_data(Tensor &x,Tensor &y,Tensor &scale,Tensor &offset);
        void get_batch_stats(Tensor &x,Tensor &mean,Tensor &var);
        void update_sums(int M,Tensor &cm,Tensor &cv,Tensor &sm,Tensor &sv);
        void compute_conv_parameters(Tensor &mean,Tensor &var,Tensor *at,Tensor *bt);
        template<bool CalcDX>
        void cpu_backward(Tensor &xt,Tensor *dxt,Tensor &dyt,Tensor &scale,Tensor &dscale,Tensor &dbias,float dx_factor);
        static int plane_size(Shape const &s);
        
        Tensor current_mean_,current_var_;
        Tensor combined_scale_,combined_bias_;
        BatchNormConfig config_;
        DataType dtype_;
        Shape setup_shape_;

        std::unique_ptr<core::BatchNormFwdBwd> bn_gpu_;
    };
} // 
