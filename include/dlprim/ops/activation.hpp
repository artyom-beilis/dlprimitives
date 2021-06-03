#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    struct ActivationConfig {
        StandardActivations activation = StandardActivations::identity;
        static ActivationConfig from_json(json::value const &v);
    };
   
    class Activation : public Operator {
    public:
        
        Activation(Context &ctx,ActivationConfig config = ActivationConfig());
        virtual ~Activation();
        
        virtual char const *operator_type() const
        {
            return "Activation";
        }

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
   		void forward_gpu(Tensor &a,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &a,Tensor &output);
        ActivationConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
    };
} // namespace
 
