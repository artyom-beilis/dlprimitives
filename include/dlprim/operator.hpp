#pragma once
#include <dlprim/definitions.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>

namespace dlprim {
    namespace json { class value; }

    class Operator {
    public:
        Operator(Context const &ctx) : ctx_(ctx)
        {
        }

        virtual ~Operator() 
        {
        }

        virtual void mode(CalculationsMode mode)
        {
            mode_ = mode;
        }

        virtual CalculationsMode mode()
        {
            return mode_;
        }

        Operator(Operator const &) = delete;
        void operator=(Operator const &) = delete;
        Operator(Operator &&) = delete;
        void operator=(Operator &&) = delete;

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace) = 0;

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out) = 0;

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx) = 0;

    protected:
        CalculationsMode mode_ = CalculationsMode::predict;
        Context ctx_;
    };
    
    std::unique_ptr<Operator> create_by_name(Context &ctx,
                                             std::string const &name,
                                             json::value const &parameters);


} // dlprim
