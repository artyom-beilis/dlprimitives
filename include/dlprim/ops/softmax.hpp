#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
    namespace json { class value; }
    namespace gpu { class Scal; }

    struct SoftMaxConfig {
        bool loss=false; // operate as loss rather that prob output
        static SoftMaxConfig from_json(json::value const &v);
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

        virtual void backward(  std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &,
                                Tensor &,
                                ExecutionContext const &e);

    private:
   		void forward_gpu(Tensor &input,Tensor &output,ExecutionContext const &ctx);
        template<typename IndexType>
        void forward_cpu_loss(Tensor &input,Tensor &label,Tensor &loss);
        void forward_cpu(Tensor &input,Tensor &output);
        template<typename IndexType>
        void backward_cpu_loss(Tensor &x,Tensor &dx,Tensor &label,Tensor &loss,float factor);
        void backward_gpu_loss(Tensor &input,Tensor &diff, Tensor &label,Tensor &output,float factor, ExecutionContext const &ctx);
        void forward_gpu_loss(Tensor &input,Tensor &label,Tensor &output,ExecutionContext const &ctx);
        void setup_kernel(int sm_range);
        DataType dtype_;
        std::string itype_;
        SoftMaxConfig config_;
        cl::Kernel kernel_,kernel_bwd_;
        std::unique_ptr<gpu::Scal> scal_;
        
        int wg_size_;
        int items_per_wi_;
        int sm_range_;
        int nd_range_;
    };
}
