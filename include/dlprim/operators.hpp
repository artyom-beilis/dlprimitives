#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
	struct InnerProductConfig {
        int inputs = -1;
		int outputs = -1;
		bool bias = true;
		StandardActivations activation = StandardActivations::identity;
        static InnerProductConfig from_json(json::value const &v);
	};


	class InnerProduct : public OperatorWithParameters {
	public:

        InnerProduct(Context &ctx,InnerProductConfig const &cfg,CalculationsMode mode = CalculationsMode::predict);
        virtual ~InnerProduct(){}

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
        
        virtual void backward_param(std::vector<Tensor> &output,
                                std::vector<Tensor> &input,
                                std::vector<Tensor> &output_diff,
                                std::vector<Tensor> &intput_diff,
                                ExecutionContext const &ctx);


	protected:
        void forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx);
        void forward_cpu(Tensor &in,Tensor &out);
		InnerProductConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
	};
	
	struct Convolition2DConfig {
		int channels;
		int features;
		int kernel[2];
		int stride[2];
		int dilate[2];
		int pad[2];
		int groups;
		bool bias;
		StandardActivations activation;
	};

}
