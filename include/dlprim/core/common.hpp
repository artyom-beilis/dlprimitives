#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    ///
    /// Scale tensor by factor inplace, if s==0 fills with zero
    /// so nan is not propagated of s==0
    ///
    class Scale {
    public:
        Scale(Context &ctx,DataType dtype=float_data);
        void enqueue(float s,Tensor &t,ExecutionContext const &ec);
    private:
        cl::Kernel k_;
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
} // dlprim
