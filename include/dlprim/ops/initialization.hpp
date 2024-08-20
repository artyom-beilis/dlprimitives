///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
namespace dlprim {
    class RandomState;

    ///
    /// Set value of t to zero
    ///
    void set_to_zero(Tensor &t,ExecutionContext const &e);
    ///
    /// Set to constant, value is casted to t.dtype()
    ///
    void set_to_constant(Tensor &t,double value,ExecutionContext const &e);
    ///
    /// set t values to uniform random values in range [minv,maxv), seed is updated
    ///

    void set_to_urandom(Tensor &t,RandomState &state,float minv,float maxv,ExecutionContext const &e);

    ///
    /// set t values to normal distribution with mean and sigma), seed is updated
    ///
    void set_to_normal(Tensor &t,RandomState &state,float mean,float sigma,ExecutionContext const &e);
    ///
    /// set t values to bernully distribution with mean and sigma), seed is updated
    ///
    void set_to_bernoulli(Tensor &t,RandomState &state,float p,ExecutionContext const &e);
    
}
