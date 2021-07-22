#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    namespace core { class BatchNorm2DFwdBwd; }

    struct BatchNorm2DConfig {
        int features = -1;
        float eps = 1e-5f;
        float momentum = 0.1;
        bool affine = true;
        bool use_global_stats = false;
        static BatchNorm2DConfig from_json(json::value const &v);
    };


    class BatchNorm2D : public Operator {
    public:
        
        BatchNorm2D(Context &ctx,BatchNorm2DConfig const &config = BatchNorm2DConfig(),DataType dtype=float_data);
        virtual ~BatchNorm2D();
        
        virtual char const *operator_type() const
        {
            return "BatchNorm2D";
        }
        
        virtual void mode(CalculationsMode m);
        virtual CalculationsMode mode() { return Operator::mode(); }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

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
        
        Tensor current_mean_,current_var_;
        Tensor combined_scale_,combined_bias_;
        BatchNorm2DConfig config_;
        DataType dtype_;
        Shape setup_shape_;

        std::unique_ptr<core::BatchNorm2DFwdBwd> bn_gpu_;
    };
} // 
