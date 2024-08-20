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
    /// Calculate filter
    ///
    class BiasBackwardFilter {
    public:
        virtual ~BiasBackwardFilter() {}
        ///
        /// return required workspace size in bytes
        ///
        virtual size_t workspace() = 0;
        ///
        /// make sure you provide worksapce of size workspace() for operations
        ///
        /// if workspace() == 0 you can provide non initialized tensor
        ///
        virtual void enqueue(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e) = 0;
        ///
        /// Create operator for backward bias calculation. dy_shape is the shape of output tesnor
        /// for IP it should be (B,feaures_out), for Conv2d (B,feaures_out,H,W)
        /// features is number of output features - size of bias tensor
        ///
        static std::unique_ptr<BiasBackwardFilter> create(Context &ctx,Shape const &dy_shape,DataType dt=float_data);
    };

    ///
    /// Add bias to t over dimentsion 1: t[:,i,:,:] = b[i]
    ///
    void add_bias(Tensor &t,Tensor &b,ExecutionContext const &e);
} // core
} // dlprim
