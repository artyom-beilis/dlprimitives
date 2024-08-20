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
    /// Performs batch normalization computations over channel #1 (when #0 is batch)
    ///
    /// Pseudo code parameters:
    ///
    ///   \code 
    ///     // Layer Data
    ///     Tensor running_mean,running_var,gamma,beta;
    ///     // Temorary Data kept between FW and BW
    ///     Tensor mean,var;
    ///     // Workspace
    ///     Tensor ws;
    ///   \endcode
    ///
    ///  Actual pseudo code calcultions
    ///   Affine, Train
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_calculate_batch_stats(x,mean,var,ws)
    ///     enqueue_update_running_stats(0.1,0.9,mean,running_mean,
    ///                                  0.1 * m/(m-1),0.9,var,running_var,ws);
    ///     enqueue_forward_affine(x,y, gamma,beta, mean, var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_affine(true,x,dy,mean,var,gamma,&dx,&dgamma,&dbeta,ws);
    ///  \endcode
    ///
    ///   Affine, Test (fixed batch)
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_forward_affine(x,y, gamma,beta, running_mean, running_var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_affine(false,x,dy,running_mean,runnig_var,gamma,&dx,&dgamma,&dbeta,ws);
    ///  \endcode
    ///
    ///   Without affine, Train
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_calculate_batch_stats(x,mean,var,ws)
    ///     enqueue_update_running_stats(0.1,0.9,mean,running_mean,
    ///                                  0.1 * m/(m-1),0.9,var,running_var,ws);
    ///     enqueue_forward_direct(x,y, mean, var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_direct(true,x,dy,mean,var,dx,ws);
    ///  \endcode
    ///
    ///   without affine, Test (fixed batch)
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_forward_direct(x,y, running_mean, running_var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_direct(false,x,dy,running_mean,runnig_var,dx,ws);
    ///  \endcode
    ////


    class BatchNormFwdBwd {
    public:
        virtual ~BatchNormFwdBwd() {}
        
        ///
        /// Workspace size needed for intermediate results of computations
        ///
        virtual size_t workspace() = 0;

        
        ///
        /// Compute batch mean and variance for input x
        ///
        /// Note \a mean and \a var shoudl have Shape(features) where features is x.shape()[1]
        ///
        virtual void enqueue_calculate_batch_stats(Tensor &x,Tensor &mean,Tensor &var,Tensor &ws,ExecutionContext const &e) = 0;
        
        ///
        /// Update running sums as 
        /// \code
        ///   running_mean = running_mean_factor * running_mean + batch_mean_factor * batch_mean;
        ///   running_var = running_var_factor * running_var + batch_var_factor * batch_var;
        /// \endcode
        ///
        virtual void enqueue_update_running_stats(float batch_mean_factor,float running_mean_factor,
                                                  Tensor &batch_mean,Tensor &running_mean,
                                                  float batch_var_factor,float running_var_factor,
                                                  Tensor &batch_var,Tensor &running_var,
                                                  Tensor &ws,ExecutionContext const &e) = 0;

        ///
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps)
        ///
        /// Note mean/var can be taken from batch or from global running stats as per user request
        ///
        virtual void enqueue_forward_direct(Tensor &x,Tensor &y,
                                            Tensor &mean,Tensor &var,float eps,
                                            Tensor &ws,ExecutionContext const &e) = 0;
        ///
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps), save 1/sqrt(var + eps) as rstd
        ///
        /// Useful for pytorch that uses rstd for LayerNorm output
        ///
        /// Note mean/var can be taken from batch or from global running stats as per user request
        ///
        virtual void enqueue_forward_get_rstd(  Tensor &x,Tensor &y,
                                                Tensor &mean,Tensor &var,float eps,
                                                Tensor &rstd,Tensor &ws,
                                                ExecutionContext const &e) = 0;
        ///
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps) * gamma + beta 
        ///
        /// Notes:
        /// - mean/var can be taken from batch or from global running stats as per user request
        /// - mean/var and gamma/beta are converted to single y=ax+b and than computation is done in a single step
        ///
        virtual void enqueue_forward_affine(Tensor &x,Tensor &y,
                                            Tensor &gamma,Tensor &beta,
                                            Tensor &mean,Tensor &var,
                                            float eps,
                                            Tensor &ws,ExecutionContext const &e) = 0;

        ///
        /// Perform backpropogation calculations
        ///
        /// training_mode - assumes that mean/var were calculated on batches of X - they need to be kept from forward stage
        ///   otherwise mean/var considered constant values
        ///
        /// gamma/beta affine transofrmation after BN
        ///
        /// dy - top gradient for backpropogation
        /// dx - calculate backpropogation on X
        /// dgamma - calculate backpropogation gradient for gamma scale
        /// dbeta - calculate backpropogation gradient for beta scale
        /// ws - worksspace 
        ///
        virtual void enqueue_backward_affine(bool training_mode,
                                             Tensor &x,Tensor &dy,
                                             Tensor &mean,Tensor &var,
                                             Tensor &gamma,
                                             Tensor *dx,float fx_factor,
                                             Tensor *dgamma,float dgamma_factor,
                                             Tensor *dbeta,float dbeta_factor,
                                             float eps,
                                             Tensor &ws,ExecutionContext const &e) = 0;

        ///
        /// Perform backpropogation calculations for BN without affine addtition Gamma/Beta
        ///
        /// training_mode - assumes that mean/var were calculated on batches of X - they need to be kept from forward stage
        ///   otherwise mean/var considered constant values
        ///
        /// dy - top gradient for backpropogation
        /// dx - calculate backpropogation on X 
        /// ws - worksspace 
        ///
        virtual void enqueue_backward_direct(bool training_mode,
                                             Tensor &x,Tensor &dy,
                                             Tensor &mean,Tensor &var,
                                             Tensor &dx,float dx_factor,
                                             float eps,
                                             Tensor &ws,ExecutionContext const &e) = 0;

        ///
        /// Perform backpropogation calculations for BN without affine addtition Gamma/Beta and using rstd instread of var
        ///
        /// assumes that mean/std are always used - no testing mode
        ///
        /// dy - top gradient for backpropogation
        /// dx - calculate backpropogation on X 
        /// ws - worksspace 
        ///
        virtual void enqueue_backward_rstd(  Tensor &x,Tensor &dy,
                                             Tensor &mean,Tensor &rstd,
                                             Tensor &dx,float dx_factor,
                                             Tensor &ws,ExecutionContext const &e) = 0;

        static std::unique_ptr<BatchNormFwdBwd> create(Context &ctx,Shape const &s,DataType dt=float_data);
        
    };

} // core
} // dlprim
