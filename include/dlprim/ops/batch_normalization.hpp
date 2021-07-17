#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    struct BatchNorm2DConfig {
        int features = -1;
        float eps = 1e-5f;
        float momentum = 0.1;
        bool affine = true;
        bool use_global_stats = false;
        static BatchNorm2DConfig from_json(json::value const &v);
    };

    class Convolution2D;

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
        void cpu_backward_data(Tensor &x,Tensor &dx,Tensor &dy,float *mean,float *var,float *dy_sum,float *dyx_sum,float *gamma_in);
        void get_batch_stats(Tensor &x,Tensor &mean,Tensor &var,ExecutionContext const &e);
        void update_sums(int M,Tensor &cm,Tensor &cv,Tensor &sm,Tensor &sv,ExecutionContext const &e);
        void compute_conv_parameters(Tensor &mean,Tensor &var,Tensor *at,Tensor *bt,ExecutionContext const &e);
        std::unique_ptr<Convolution2D> conv_;
        Tensor current_mean_,current_var_;
        Tensor combined_scale_,combined_bias_;
        size_t conv_ws_size_;
        BatchNorm2DConfig config_;
        DataType dtype_;
    };
} // 
