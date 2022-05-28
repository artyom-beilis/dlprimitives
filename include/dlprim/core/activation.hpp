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
    /// Appliy activation on X save to Y, Y can be same as X
    /// 
    void activation_forward(Tensor &x,Tensor &y,StandardActivations activation, ExecutionContext const &ec);

    ///
    /// Backward aclivation computed `dx = dx * factor + backward(y,dy)`
    ///
    void activation_backward(Tensor &dx,Tensor &dy,Tensor &y,StandardActivations activation, float factor, ExecutionContext const &ec);

} // core
} // dlprim
