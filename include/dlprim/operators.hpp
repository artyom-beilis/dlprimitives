#pragma once
#include <dlprim/definitions.hpp>
#include <dlprim/tensor.hpp>

namespace dlprim {

	class Operator {
	public:
        Operator(Context const &ctx,CalculationsMode mode = predict) : 
            ctx_(ctx),
            mode_(mode)
        {
        }
        Operator(Operator const &) = delete;
        void operator=(Operator const &) = delete;
        Operator(Operator &&) = delete;
        void operator=(Operator &&) = delete;

        virtual ~Operator() {}
        virtual void mode(CalculationsMode mode)
        {
            mode_ = mode;
        }
        virtual CalculationsMode mode()
        {
            return mode_;
        }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &params,
                           size_t &workspace) = 0;

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out) = 0;

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             cl::CommandQueue &q,cl::Event *even = nullptr) = 0;

		virtual void backward(ExecutionContext &ctx,
                              std::vector<Tensor> &input,
					          std::vector<Tensor> &output,
					          std::vector<Tensor> &output_diff,
					          std::vector<Tensor> &input_diff,
                              cl::CommandQueue &q,cl::Event *even = nullptr) 
		{
			throw dlprim::NotImplemented();
		}

        std::vector<Tensor> &parameters()
        {
            return parameters_;
        }
        std::vector<Tensor> &parameters_diff()
        {
            return parameters_diff_;
        }

    protected:
        Context ctx_;
        CalculationsMode mode_;
        std::vector<Tensor> parameters_;
        std::vector<Tensor> parameters_diff_;
	};
	
	struct InnerProductConfig {
        InnerProductConfig(int out,bool b,StandardActivations a=StandardActivations.identity) :
            inputs(-1),
            outputs(out),
            bias(b),
            activation(a)
        {
        }
        InnerProductConfig(int in,int out,bool b,StandardActivations a=StandardActivations.identity) :
            inputs(in),
            outputs(out),
            bias(b),
            activation(a)
        {
        }
        int inputs;
		int outputs;
		bool bias;
		StandardActivations activation;
	};


    class SoftMax {
    public:
        SoftMax(Context &ctx,DataType dtype=float_data);
   		virtual void reshape(std::vector<Shape> const &in,std::vector<Shape> &out,std::vector<Shape> &param_shapes,size_t &workspace)
		{
			DLPRIM_CHECK(in.size() == 1);
            out.assign({in[0]})
            param_shapes.clear();
			workspace = 0;
		}
        virtual void forward(std::vector<Tensor> &input,
                     std::vector<Tensor> &output,
                     cl::CommandQueue &q,cl::Event *even = nullptr);
    private:
        class SoftMaxImpl;
        std::unque_ptr<SoftMaxImpl> pimpl_;
    };

	class InnerProduct : public Operator {
	public:

        InnerProduct(Context &ctx,InnerProductConfig const &cfg,DataType dtype=float_data);
        virtual ~InnerProduct() {}

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &params,
                           size_t &workspace) = 0;

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out) = 0;

        virtual void forward(std::vector<Tensor> &input,
                     std::vector<Tensor> &output,
                     cl::CommandQueue &q,cl::Event *even = nullptr);

        InnerProductConfig &config()
        {
            return config_;
        }
        Context &context()
        {
            return ctx_;
        }
	protected:
        class InnerProductImpl;
        Context ctx_;
		InnerProductConfig config_;
        std::unique_ptr<InnerProductImpl> impl_;
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
