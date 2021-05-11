#pragma once
#include <dlprim/definitions.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>

namespace dlprim {

    class Operator {
    public:
        Operator(Context const &ctx) : ctx_(ctx)
        {
        }

        virtual ~Operator() 
        {
        }

        Operator(Operator const &) = delete;
        void operator=(Operator const &) = delete;
        Operator(Operator &&) = delete;
        void operator=(Operator &&) = delete;

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           size_t &workspace) = 0;

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out) = 0;

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             ExecutionContext const &ctx) = 0;

        virtual void backward_data(std::vector<Tensor> &output,
                                   std::vector<Tensor> &input,
                                   std::vector<Tensor> &output_diff,
                                   std::vector<Tensor> &intput_diff,
                                   ExecutionContext const &ctx) = 0;
    protected:
        Context ctx_;
    };


	class OperatorWithParameters : public Operator {
	public:
        OperatorWithParameters(Context const &ctx,CalculationsMode mode = CalculationsMode::predict) : 
            Operator(ctx),
            mode_(mode)
        {
        }
        virtual void backward_param(std::vector<Tensor> &output,
                                std::vector<Tensor> &input,
                                std::vector<Tensor> &output_diff,
                                std::vector<Tensor> &intput_diff,
                                ExecutionContext const &ctx) = 0;


        virtual void mode(CalculationsMode mode)
        {
            mode_ = mode;
        }

        virtual CalculationsMode mode()
        {
            return mode_;
        }
        
        void setup_parameters(std::vector<TensorSpecs> &&parameters)
        {
            parameters_specs_ = std::move(parameters);
        }

        std::vector<TensorSpecs> &parameters_specs()
        {
            return parameters_specs_;
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
        CalculationsMode mode_;
        std::vector<TensorSpecs> parameters_specs_;
        std::vector<Tensor> parameters_;
        std::vector<Tensor> parameters_diff_;
	};
} // dlprim
