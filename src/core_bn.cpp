#include <dlprim/core_ops.hpp>
namespace dlprim {
namespace core {
    class BatchNorm2DImpl : public BatchNorm2DFwdBwd {
    public:
        virtual ~BatchNorm2DImpl() {}
        BatchNorm2DImpl(Context &ctx,Shape s,DataType dtype);
        ///
        /// Workspace size needed for intermediate results of computations
        ///
        virtual size_t workspace()
        {
            return ws_;
        }

        
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
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps) * gamma + beta 
        ///
        /// Notes:
        /// - mean/var can be taken from batch or from global running stats as per user request
        /// - mean/var and gamma/beta are converted to single y=ax+b and than computation is done in a single step
        ///
        virtual void enqueue_forward_affine(Tensor &x,Tensor &y,
                                            Tensor &gamma,Tensor &beta,
                                            Tensor &mean,Tensor &var,
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
                                             Tensor &gamma,Tensor &beta,
                                             Tensor *dx,float fx_factor,
                                             Tensor *dgamma,float dgamma_factor,
                                             Tensor *dbeta,float dbeta_factor,
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
                                             Tensor &ws,ExecutionContext const &e) = 0;

        static std::unique_ptr<BatchNorm2DFwdBwd> create(Context &ctx,Shape const &s,DataType dt=float_data);
    };

