#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
    namespace json { class value; }

    struct SoftMaxConfig {
        static SoftMaxConfig from_json(json::value const &) { return SoftMaxConfig(); }
    };
    
    class SoftMax : public Operator {
    public:
        SoftMax(Context &ctx,SoftMaxConfig const &cfg=SoftMaxConfig());
        virtual ~SoftMax();
        
        virtual char const *operator_type() const
        {
            return "SoftMax";
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
}
