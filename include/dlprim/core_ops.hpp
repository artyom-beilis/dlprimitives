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

    class Scale {
    public:
        Scale(Context &ctx,DataType dtype=float_data);
        void enqueue(float s,Tensor &t,ExecutionContext const &ec);
    private:
        cl::Kernel k_;
    };

    ///
    /// Performs batch normalization computations
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
    ///     enqueue_backward_affine(true,x,dy,mean,var,gamma,beta,&dx,&dgamma,&dbeta,ws);
    ///  \endcode
    ///
    ///   Affine, Test (fixed batch)
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_forward_affine(x,y, gamma,beta, running_mean, running_var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_affine(false,x,dy,running_mean,runnig_var,gamma,beta,&dx,&dgamma,&dbeta,ws);
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


    class BatchNorm2DFwdBwd {
    public:
        virtual ~BatchNorm2DFwdBwd() {}
        
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
                                             Tensor &gamma,Tensor &beta,
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

        static std::unique_ptr<BatchNorm2DFwdBwd> create(Context &ctx,Shape const &s,DataType dt=float_data);
        
    };

    ///
    /// Set to zero tensor - OpenCL only
    ///
    void fill_tensor(Context &ctx,ExecutionContext const &e,Tensor &t,double value);


    ///
    /// Type of random distribution
    ///
    enum RandomDistribution {
        rnd_uniform = 0,
        rnd_normal  = 1
    };
    ///
    /// Fill tensor with random numbers using provided distribution
    ///
    /// \param t tesnor to fill
    /// \param philox_seed - 64 bit seed for philox-2x4-10 algorithm
    /// \param philox_seq  - counter for RNG to start. Note each philox counter item
    ///     generated 4 random numbers. So if you have tensor of size 100, that 25 items
    ///     will be used [philox_seq, philox_seq + 25)
    /// \param distribution type
    /// \param p1 - min value for uniform and mu for normal
    /// \param p2 - max value for uniform and sigma for normal
    ///
    void fill_random(Context &ctx,ExecutionContext const &e,Tensor &t,cl_ulong philox_seed,cl_ulong philox_seq,RandomDistribution dist,float p1,float p2);

    ///
    /// Add bias to t over dimentsion 1: t[:,i,:,:] = b[i]
    ///
    void add_bias(Context &ctx,ExecutionContext const &e,Tensor &t,Tensor &b);
   
    
    ///
    /// Class for copying a slice of an tensor
    ///
    class SliceCopy {
    public:
        SliceCopy(Context &ctx,DataType dtype=float_data);
        ~SliceCopy();
        
        ///
        /// Copy one part of tensor to another over single dimentsion dim, lets say if target and source are 4d tensors
        /// and dim == 1 then it is equivalent of following in numpy:
        ///
        /// \code
        /// target[:,taget_offset:target_offset + slice,:,:] *=target_scale
        /// target[:,taget_offset:target_offset + slice,:,:] += source[:,source_offset:source_offset+slice,:,:]
        /// \endcode
        ///
        void tensor_slice_copy(int dim,size_t slice,
                               Tensor &target,size_t target_offset,
                               Tensor &source,size_t source_offset,
                               float target_scale,ExecutionContext const &e);
    private:
        cl::Kernel kernel_;
        DataType dtype_;
    };

    
    

} // core
} // dprim
