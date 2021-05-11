#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
    class SoftMax : public Operator {
    public:
        SoftMax(Context &ctx,DataType dtype=float_data);
        virtual ~SoftMax();

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           size_t &workspace) ;

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             ExecutionContext const &ctx);

        virtual void backward_data(std::vector<Tensor> &output,
                                   std::vector<Tensor> &input,
                                   std::vector<Tensor> &output_diff,
                                   std::vector<Tensor> &intput_diff,
                                   ExecutionContext const &ctx);
        
    private:
   		void forward_gpu(Tensor &input,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &input,Tensor &output);
        void setup_kernel(int sm_range);
        DataType dtype_;
        cl::Kernel kernel_;
        int wg_size_;
        int items_per_wi_;
        int sm_range_;
        int nd_range_;
    };


    struct ElementwiseConfig {
        enum Operation {
            elementwise_sum,
            elementwise_prod,
            elementwise_max
        };
        
        Operation op = elementwise_sum;
        float coeff[2] = {1.0f,1.0f};
        StandardActivations activation = StandardActivations::identity;
    };
    
    class Elementwise : public Operator {
    public:
        
        Elementwise(Context &ctx,ElementwiseConfig config = ElementwiseConfig(),
                                 DataType dtype=float_data);
        virtual ~Elementwise();

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             ExecutionContext const &ctx);

        virtual void backward_data(std::vector<Tensor> &output,
                                   std::vector<Tensor> &input,
                                   std::vector<Tensor> &output_diff,
                                   std::vector<Tensor> &intput_diff,
                                   ExecutionContext const &ctx);
        
    private:
   		void forward_gpu(Tensor &a,Tensor &b,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &a,Tensor &b,Tensor &output);
        ElementwiseConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
    };
    
    
}
