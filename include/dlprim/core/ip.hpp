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

} // core
} // dlprim
