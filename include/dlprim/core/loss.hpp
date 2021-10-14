#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {
    
    ///
    /// Compute softmax output of x-> to y. if log_softmax true compute log of the output value
    ///
    void softmax_forward(Tensor &x,Tensor &y,bool log_softmax,ExecutionContext const &e);
    
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_forward(Tensor &x,Tensor &label,Tensor &y,bool reduce,float scale,ExecutionContext const &e);
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_backward(Tensor &dx,Tensor &label,Tensor &dy,bool reduce,float scale,float factor,ExecutionContext const &e);

} // core
} // dlprim