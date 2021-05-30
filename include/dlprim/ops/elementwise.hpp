#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    struct ElementwiseConfig {
        enum Operation {
            elementwise_sum,
            elementwise_prod,
            elementwise_max
        };
        
        Operation op = elementwise_sum;
        float coeff[2] = {1.0f,1.0f};
        StandardActivations activation = StandardActivations::identity;

        static ElementwiseConfig from_json(json::value const &v);
    };
   
    class Elementwise : public Operator {
    public:
        
        Elementwise(Context &ctx,ElementwiseConfig config = ElementwiseConfig());
        virtual ~Elementwise();

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


        
    private:
   		void forward_gpu(Tensor &a,Tensor &b,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &a,Tensor &b,Tensor &output);
        ElementwiseConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
    };
} // namespace
 
