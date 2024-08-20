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
    /// Configuration of Convoltion
    ///
    struct Conv2DSettings : public Convolution2DConfigBase {
        Conv2DSettings(Conv2DSettings const &) = default;
        Conv2DSettings(Convolution2DConfigBase const &v,Shape s,DataType dt) :
            Convolution2DConfigBase(v),
            shape(s),
            dtype(dt)
        {
        }

        Shape shape; // input shape size, note batch is hint rather than requirement
        DataType dtype=float_data;
    };
   
    class Conv2DBase  {
    public:
        virtual ~Conv2DBase() {};
        virtual char const *algo() const = 0;
        virtual size_t workspace() { return 0; }
        static Shape get_output_shape(Convolution2DConfigBase const &config,Shape const &in);
        static Shape get_output_shape_transposed(Convolution2DConfigBase const &config,Shape const &in,int output_pad[2]);
    };
    ///
    /// Perform InnerProduct/FullyConnected/Dense forward calulations, allow fusing bias and activation
    /// into same GPU kernel
    /// 
    class Conv2DForward : public Conv2DBase {
    public:
        virtual ~Conv2DForward() {}
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y,Tensor &ws,float factor,ExecutionContext const &e) = 0;
        /// Create optimal object for conv2d
        /// 
        /// algo is one of 
        ///  "" or "auto" - automatic selection,
        ///   "gemm" - use fused GEMM based algo
        ///   "winograd" - use Winograd convoltion - suitable for non strided, non dilated, non grouped 3x3 with pad=1 conv
        ///   "depthwise_separable" 
        static std::unique_ptr<Conv2DForward> create(Context &ctx,
                                                 Conv2DSettings const &config,
                                                 bool bias,
                                                 StandardActivations activation = StandardActivations::identity,
                                                 std::string const &algo = std::string());
    };

    ///
    /// Perform InnerProduct/FullyConnected/Dense backward data calculations
    /// 
    class Conv2DBackwardData: public Conv2DBase  {
    public:
        virtual ~Conv2DBackwardData() {}
        virtual void enqueue(Tensor &dx,Tensor &w,Tensor &dy,Tensor &ws,float factor,ExecutionContext const &e) = 0;
        static std::unique_ptr<Conv2DBackwardData> create(Context &ctx,Conv2DSettings const &config,std::string const &algo = std::string());
    };

    ///
    /// Perform Conv2D backward filter calcilations
    ///
    class Conv2DBackwardFilter: public Conv2DBase  {
    public:
        virtual ~Conv2DBackwardFilter() {}
        virtual void enqueue(Tensor &x,Tensor &dw,Tensor &dy,Tensor &ws,float factor,ExecutionContext const &e) = 0;
        static std::unique_ptr<Conv2DBackwardFilter> create(Context &ctx,Conv2DSettings const &config,std::string const &algo = std::string());
    };

} // core
} // dlprim
