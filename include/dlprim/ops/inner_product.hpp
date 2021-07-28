#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {
    namespace gpu { class GEMM; }
    namespace json { class value; }
    namespace core { class IPForward; class IPBackwardData; class IPBackwardFilter; }
    class BWBias;

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


	protected:
        void forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias);
        void backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dM,float factor);
        void backward_data_cpu(Tensor &dy,Tensor &dx,Tensor &M,float factor);


		InnerProductConfig config_;
        DataType dtype_;
        std::unique_ptr<core::IPForward> ip_;
        std::unique_ptr<core::IPBackwardData> bwd_ip_;
        std::unique_ptr<core::IPBackwardFilter> bwd_weights_ip_;
        std::unique_ptr<Operator>  activation_;
        std::unique_ptr<BWBias> bwd_bias_;
	};
	
}
