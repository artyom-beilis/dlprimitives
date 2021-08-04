#pragma once
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
namespace dlprim {
    class RandomState;

    ///
    /// Set value of t to zero
    ///
    void set_to_zero(Context &ctx,ExecutionContext const &e,Tensor &t);
    ///
    /// Set to constant, value is casted to t.dtype()
    ///
    void set_to_constant(Context &ctx,ExecutionContext const &e,Tensor &t,double value);
    ///
    /// set t values to uniform random values in range [minv,maxv), seed is updated
    ///

    void set_to_urandom(Context &ctx,ExecutionContext const &e,Tensor &t,RandomState &state,float minv=0.0f,float maxv=1.0f);

    ///
    /// set t values to normal distribution with mean and sigma), seed is updated
    ///
    void set_to_normal(Context &ctx,ExecutionContext const &e,Tensor &t,RandomState &state,float mean=0.0f,float sigma=1.0f);
    
}
