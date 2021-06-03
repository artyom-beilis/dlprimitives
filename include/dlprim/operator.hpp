#pragma once
#include <dlprim/definitions.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>

namespace dlprim {
    namespace json { class value; }

    struct TensorAndGradient {
        bool requires_gradient=true;
        Tensor data;
        Tensor grad;
    };

    class Operator {
    public:
        Operator(Context const &ctx) : 
            ctx_(ctx),
            mode_(CalculationsMode::predict)
        {
        }

        virtual ~Operator() 
        {
        }

        virtual char const *operator_type() const = 0;
        
        ///
        /// Can be called with both train and predict before setup() is called.
        /// afterwards if original mode was train - it can be switched to predict and back
        /// but if original mode was predict it can't be switched to train.
        ///
        /// Default is predict
        ///
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
		
        virtual void backward(std::vector<TensorAndGradient> & /*input*/,
                              std::vector<TensorAndGradient> & /*output*/,
                              std::vector<TensorAndGradient> & /*parameters*/,
                              Tensor &/*workspace*/,
                              ExecutionContext const &/*ctx*/)
        {
            throw NotImplementedError("backpropogation is not implemented for " + std::string(operator_type()));
        }

    protected:
        Context ctx_;
        CalculationsMode mode_;
    };
    
    std::unique_ptr<Operator> create_by_name(Context &ctx,
                                             std::string const &name,
                                             json::value const &parameters);


} // dlprim
