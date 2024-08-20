///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/definitions.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>

namespace dlprim {
    namespace json { class value; }

    class SharedResource;


    ///
    /// Base class for backward/forward propogation calculations for internal network
    ///
    class Operator {
    public:
        ///
        /// Create operator for specific context (device/platform)
        ///
        Operator(Context const &ctx) : 
            ctx_(ctx),
            mode_(CalculationsMode::predict)
        {
        }

        ///
        /// Getter for object that is shared between operators accross the net, for example
        /// random numbers generator
        ///
        SharedResource &shared_resource()
        {
            DLPRIM_CHECK(shared_resource_);
            return *shared_resource_;
        }

        /// Setter of the shared resource
        void shared_resource(std::shared_ptr<SharedResource> r)
        {
            shared_resource_ = r;
        }

        virtual ~Operator() 
        {
        }
        
        /// name of the operator type
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

        /// get current mode
        virtual CalculationsMode mode()
        {
            return mode_;
        }

        ///
        /// operator non-copyable/moveable
        ///
        /// manage it with smart pointers
        ///
        Operator(Operator const &) = delete;
        void operator=(Operator const &) = delete;
        Operator(Operator &&) = delete;
        void operator=(Operator &&) = delete;


        ///
        /// returns true of the operator is alias - generation - it only changes the shape of tensor but not its content
        /// it actually does not perform any operation but only changes semantics, in this casse input and output are aliases
        /// of each other
        ///
        virtual bool alias_generator() 
        {
            return false;
        }

        ///
        /// Set default parameters iniitalization
        ///
        virtual void initialize_params(std::vector<Tensor> &/*parameters*/,ExecutionContext const &/*e*/)
        {
        }

        ///
        /// Convigure operator
        ///
        /// \param in - a list of expected input tensors
        /// \param out - a list of output tensots - that opertor calculates in fwd propogation
        /// \param parameters - a list of parameters need for computation. If parameter should not participate in gradient
        ///   desend it should be marked as not trainable
        /// \param workspace size needed for computations in bytes - not preserved between different calls
        ///
        virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace) = 0;

        ///
        /// Reshape layer according to new input size
        ///
        /// \param in - new input tensor sizes
        /// \param out - new output tesor sizes
        /// \param workspace - new workspace size needed
        ///
        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &workspace) = 0;

        ///
        /// Enqueue forward propogation computations 
        ///
        /// \param input - input tesnosrs (X)
        /// \param output - output tesnors (Y)
        /// \param parameters - parameters for computation
        /// \param workspace - workspace as required
        /// \param ctx - execution context
        ///
		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx) = 0;
		
        ///
        /// Enqueue backward propogation computations
        ///
        /// \param input - inputs and their gradients marked for computatins
        /// \param output - outputs and their gradients
        /// \param parameters - parameters parameters and their gradients
        /// \param workspace - workspace as required
        /// \param ctx - execution context
        ///
        /// Note: actual inputs are
        /// 
        ///  - inputs[index].data
        ///  - outputs[index].data
        ///  - outputs[index].diff
        ///  - parameters[index].data
        ///
        /// And it computes in backpropogation
        /// 
        ///  - inputs[index].diff
        ///  - parameters[index].diff
        ///
        /// If computation is not needed TensorAndGradient::requires_gradient need to be set to false
        ///
        virtual void backward(std::vector<TensorAndGradient> & /*input*/,
                              std::vector<TensorAndGradient> & /*output*/,
                              std::vector<TensorAndGradient> & /*parameters*/,
                              Tensor &/*workspace*/,
                              ExecutionContext const &/*ctx*/)
        {
            throw NotImplementedError("backpropogation is not implemented for " + std::string(operator_type()));
        }

    protected:
        Context ctx_; ///< OpenCL/CPU Context to work with
        CalculationsMode mode_; ///< computaions mode
        std::shared_ptr<SharedResource> shared_resource_;
    };
   
    ///
    /// Factory - generate operator by its name (type) with parameters needed
    /// 
    std::unique_ptr<Operator> create_by_name(Context &ctx,
                                             std::string const &name,
                                             json::value const &parameters);


} // dlprim
