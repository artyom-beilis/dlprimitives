#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {
    namespace gpu { class GEMM; }
    namespace json { class value; }

	struct InnerProductConfig {
        int inputs = -1;
		int outputs = -1;
		bool bias = true;
		StandardActivations activation = StandardActivations::identity;
        static InnerProductConfig from_json(json::value const &v);
	};


	class InnerProduct : public Operator {
	public:

        InnerProduct(Context &ctx,InnerProductConfig const &cfg);
        virtual ~InnerProduct();

        virtual char const *operator_type() const
        {
            return "InnerProduct";
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

	protected:
        void forward_gpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,ExecutionContext const &ctx);
        void forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias);
		InnerProductConfig config_;
        DataType dtype_;
        std::unique_ptr<gpu::GEMM> gemm_;
	};
	
}
