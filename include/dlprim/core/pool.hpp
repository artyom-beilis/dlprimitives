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
    /// Compute output size of the tensor after pooling in specific dimentions.
    ///
    inline int calc_pooling_output_size(int in_size,int kernel,int pad,int stride,bool ceil_mode)
    {
        int padded_size = in_size + pad*2;
        DLPRIM_CHECK(padded_size >= kernel);
        int offset = ceil_mode ? (stride-1) : 0;
        int size = (padded_size - kernel + offset) / stride + 1;
        if((size - 1) * stride >= in_size + pad) {
            size--;
        }
        return size;
    }


    ///
    /// 2d pooling
    ///
    class Pooling2DForward {
    public:
        virtual ~Pooling2DForward() {}
        // get workspace size
        virtual size_t workspace() = 0;

        ///
        /// when used with kernel based pooling (not global)
        /// X and Y dimensions should match at batch and channels and for H/W  the dimention for Y should be Y_dim = op((X_dim + 2 * pad[dim] - kernel[dim]) / stride[dim]) + 1
        /// where op is either ceil or floor
        ///
        virtual void enqueue(Tensor &X,Tensor &Y,ExecutionContext const &e) = 0;

        ///
        /// Create max pooling for kernel, pad, stride
        ///
        static std::unique_ptr<Pooling2DForward> create_max_pooling(
                            Context &ctx,
                            int kernel[2],int pad[2],int stride[2],
                            DataType dt=float_data);

        ///
        /// Create max pooling for kernel, pad, stride
        ///
        /// if count_include_pad == true than average is normalized by sizeof kernel otherwise by actual
        /// amount of pixel participated
        ///
        static std::unique_ptr<Pooling2DForward> create_avg_pooling(
                    Context &ctx,
                    int kernel[2],int pad[2],int stride[2],bool count_include_pad=false,
                    DataType dt=float_data);
        
        /// Max global pooling
        static std::unique_ptr<Pooling2DForward> create_global_max_pooling(
                    Context &ctx,Shape const &in_shape,DataType dt=float_data);

        /// Avergage global pooling
        static std::unique_ptr<Pooling2DForward> create_global_avg_pooling(
                    Context &ctx,Shape const &in_shape,DataType dt=float_data);
    };

    ///
    /// Backward pooling computation
    ///
    class Pooling2DBackwardBase {
    public:
        /// get workspace
        virtual size_t workspace() = 0;
        virtual ~Pooling2DBackwardBase() {}
        /// X for max pooling to detect the neuron giving max signal
        ///
        /// when used with kernel based pooling (not global)
        /// X/dX and dY dimensions should match at batch and channels and for H/W  the dimention for Y should be Y_dim = op((X_dim + 2 * pad[dim] - kernel[dim]) / stride[dim]) + 1
        /// where op is either ceil or floor
        ///
        virtual void enqueue(Tensor &X,Tensor &dX,Tensor &dY,float factor,ExecutionContext const &e) = 0;
    };

    ///
    /// Backward computation for max  pooling
    ///
    class MaxPooling2DBackward : public Pooling2DBackwardBase {
    public:
        virtual ~MaxPooling2DBackward() {}
        /// Create pooling backward computation, See Pooling2DForward for
        /// details 
        static std::unique_ptr<MaxPooling2DBackward> create(
                    Context &ctx,
                    int kernel[2],int pad[2],int stride[2],
                    DataType dt=float_data);

        /// Create global pooling
        static std::unique_ptr<MaxPooling2DBackward> create_global(
                    Context &ctx,Shape const &in_shape,DataType dt=float_data);
    };

    class AvgPooling2DBackward : public Pooling2DBackwardBase {
    public:
        virtual ~AvgPooling2DBackward() {}

        /// for Avg pooling we don't need X so you can call directrly enqueue(dX,dY,factor,e)
        ///
        /// when used with kernel based pooling (not global)
        /// dX and dY dimensions should match at batch and channels and for H/W  the dimention for Y should be Y_dim = op((X_dim + 2 * pad[dim] - kernel[dim]) / stride[dim]) + 1
        /// where op is either ceil or floor
        ///
        virtual void enqueue(Tensor &/*X*/,Tensor &dX,Tensor &dY,float factor,ExecutionContext const &e) 
        {
            enqueue(dX,dY,factor,e);
        }
        /// actual computation, no need X for backward propogation
        ///
        /// when used with kernel based pooling (not global)
        /// dX and dY dimensions should match at batch and channels and for H/W  the dimention for Y should be Y_dim = op((X_dim + 2 * pad[dim] - kernel[dim]) / stride[dim]) + 1
        /// where op is either ceil or floor
        ///
        virtual void enqueue(Tensor &dX,Tensor &dY,float factor,ExecutionContext const &e)  = 0;
        
        /// Create average pooling with kernel
        static std::unique_ptr<AvgPooling2DBackward> create(
                    Context &ctx,
                    int kernel[2],int pad[2],int stride[2],bool count_include_pad=false,
                    DataType dt=float_data);

        /// Create global average pooling
        static std::unique_ptr<AvgPooling2DBackward> create_global(
                    Context &ctx,Shape const &in_shape,DataType dt=float_data);
    };
} // core
} //dlprim
