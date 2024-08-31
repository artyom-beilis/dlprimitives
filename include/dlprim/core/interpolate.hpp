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
    /// Interpolate forward y size should be floor(src_size * scale), scale_x,scale_y can be -1 to calculate automatically
    ///
    void interpolate2d(Tensor &x,Tensor &y,double scale_y,double scale_x,InterpolateType method,bool align_corners,ExecutionContext const &e);
    ///
    /// Interpolate forward y size should be floor(src_size * scale), scale_x,scale_y can be -1 to calculate automatically
    ///
    void interpolate2d_backward(Tensor &dx,Tensor &dy,double scale_y,double scale_x,InterpolateType method,bool align_corners,float factor,ExecutionContext const &e);
}
}
