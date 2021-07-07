#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
///
/// All Basic Operations on GPU
///
namespace core {
    ///
    /// Configuration of InnerProduct layer 
    ///
    struct IPSettings {
        int inputs = -1;   /// number of input features 
        int outputs = -1;  /// output features
        int optimal_batch_size = -1;  /// Expected batch size the network is used with
        DataType dtype=float_data;
    };
    
    ///
    /// Perform InnerProduct/FullyConnected/Dense forward calulations, allow fusing bias and activation
    /// into same GPU kernel
    /// 
    class IPForward {
    public:
        virtual ~IPForward() {}
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y,ExecutionContext const &e) = 0;
        ///
        /// Create optimal object for innter product calculation
        ///
        /// config - IP Settings,
        /// bias - apply bias
        /// activation - apply activation
        ///
        static std::unique_ptr<IPForward> create(Context &ctx,
                                                 IPSettings const &config,
                                                 bool bias,
                                                 StandardActivations activation = StandardActivations::identity);
    };

    ///
    /// Perform InnerProduct/FullyConnected/Dense backward data calculations
    /// 
    class IPBackwardData {
    public:
        virtual ~IPBackwardData() {}
        virtual void enqueue(Tensor &dx,Tensor &w,Tensor &dy,float factor,ExecutionContext const &e) = 0;
        ///
        /// Create optimal object for innter product calculation
        ///
        /// config - IP Settings,
        static std::unique_ptr<IPBackwardData> create(Context &ctx,IPSettings const &config);
    };

    ///
    /// Perform InnerProduct/FullyConnected/Dense backward filter calcilations
    ///
    class IPBackwardFilter {
    public:
        virtual ~IPBackwardFilter() {}
        virtual void enqueue(Tensor &x,Tensor &dw,Tensor &dy,float factor,ExecutionContext const &e) = 0;
        ///
        /// Create optimal object for innter product calculation
        ///
        /// config - IP Settings,
        static std::unique_ptr<IPBackwardFilter> create(Context &ctx,IPSettings const &config);
    };


    ///
    /// Configuration of InnerProduct layer 
    ///
    struct Conv2DSettings : public Convolution2DConfigBase {
        Shape shape; // input shape size, note batch is hint rather than requirement
        DataType dtype=float_data;
    };
    
    ///
    /// Perform InnerProduct/FullyConnected/Dense forward calulations, allow fusing bias and activation
    /// into same GPU kernel
    /// 
    class Conv2DForward {
    public:
        virtual ~Conv2DForward() {}
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y,Tensor &ws,ExecutionContext const &e) = 0;
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
    class Conv2DBackwardData {
    public:
        virtual ~Conv2DBackwardData() {}
        virtual void enqueue(Tensor &dx,Tensor &w,Tensor &dy,Tensor &ws,float factor,ExecutionContext const &e) = 0;
        static std::unique_ptr<Conv2DBackwardData> create(Context &ctx,Conv2DSettings const &config,std::string const &algo = std::string());
    };

    ///
    /// Perform Conv2D backward filter calcilations
    ///
    class Conv2DBackwardFilter {
    public:
        virtual ~Conv2DBackwardFilter() {}
        virtual void enqueue(Tensor &x,Tensor &dw,Tensor &dy,Tensor &ws,float factor,ExecutionContext const &e) = 0;
        static std::unique_ptr<Conv2DBackwardFilter> create(Context &ctx,Conv2DSettings const &config,std::string const &algo = std::string());
    };



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

} // core
} // dprim
