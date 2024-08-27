///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    ///
    /// Bind a parameter to kernet casting it to apropriate opencl type dt
    ///
    void bind_as_dtype(cl::Kernel &k,int &p,double value,DataType dt);
    ///
    /// per form operations function(xs,ws)->yw such that
    /// each tensor in xs and ys has same shape, ws are constant parameters
    ///
    /// code should perform assignment to variables y0 to yN and use x0..xM as values, and w0...wK as parameters
    /// for example:
    ///   `pointwise_operation({a,b},{c},{w},"y0 = x0*w0 + x1;",q);`
    ///
    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<double>  ws,
                             std::string const &code,
                             ExecutionContext const &ec);
    ///
    /// Similar to pointwise_operation but xs are broadcasted numpy style. ys must much broadcasted shape, weights are considered
    /// of ys[0].dtype()
    ///
    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double>  weights,
                                        std::string const &code,
                                        ExecutionContext const &e);

    ///
    /// Similar to pointwise_operation but xs are broadcasted numpy style. ys must much broadcasted shape
    ///
    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double>  weights,
                                        std::vector<DataType> weights_types,
                                        std::string const &code,
                                        ExecutionContext const &e,
                                        bool shrink_dims=true);
    ///
    /// Perform pointwise operation with both boradcasting and reduction
    ///
    /// Calculation is performed over a shape that xs and ys tensors are boradcaasted to.
    ///
    /// For example xs have shapes: (64,10,5) and (64,10,1) and ys has shape (10,1) they all
    /// broadcast to 64,10,5 and reduction is performed over dimentsions 0 and 2
    ///
    /// All ys tensors need to have same shape and be boradcastable to total shape
    ///
    /// Optional parameters can be provided that avalible in code as w0... wN, Final ys are computed as `ys[i] = alpha[i] * reduced_result + beta[i] * ys[i]`
    ///
    class PointwiseOperationBroadcastReduce {
    public:
        
        virtual ~PointwiseOperationBroadcastReduce() {}
        ///
        /// Get size of workspace in bytes needed
        ///
        virtual size_t workspace() = 0;
        ///
        /// Perform coputations
        ///
        /// \param xs - vector of input tensor
        /// \param ys - vector of output tenors
        //  \param parameters - the weight paramerters, size should match weights_count
        /// \param alpha - scale for ys, must match size of ys
        /// \param beta - scale for summation of previous ys, must match size of ys
        ///
        ///
        virtual void enqueue(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             Tensor &workspace,
                             std::vector<double> parameters,
                             std::vector<double> alpha,
                             std::vector<double> beta,
                             ExecutionContext const &e) = 0;

        ///
        /// Create objects:
        ///
        /// \param xs - vector of input tensor specs - such tensors are expected to be given to enqueue
        /// \param ys - vector of output tenorr specs - such tensors are expectred to be give to enqueue
        //  \param weights_count - size of parameters vector in enqueue
        /// \param weights_type - type of weights parameters as provided
        ///
        /// \param compute_code - OpenCL code to compute values. You can use x0, x1, ... xN as input values for each x for xs
        /// y0,.., yN for each output and w0,...,wN for each weight. For example "y0 = x0 + w0 * x1;"
        ///
        /// \param reduce_init - initalization of reduction variables `reduce_yN` for example "reduce_y0 = 0;" or "reduce_y0=-FLT_MAX;"
        /// \param reduce - code for sum reduction "reduce_y0 += y0" or max reduction "reduce_y0 = max(reduce_y0,y0)"
        ///
        static std::unique_ptr<PointwiseOperationBroadcastReduce> create(
                        Context &ctx,
                        std::vector<TensorSpecs> xs,
                        std::vector<TensorSpecs> ys,
                        int weights_count,DataType weights_type,
                        std::string const &compute_code,
                        std::string const &reduce_init,
                        std::string const &reduce);

    };

} // core
} // dlprim
