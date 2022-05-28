///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once 
#include <dlprim/context.hpp>
#include <dlprim/shape.hpp>
#include <memory>
namespace dlprim {

    class Tensor;    
    ///
    /// Definition of Tensor without actual memory/object
    ///
    class TensorSpecs {
    public:
        ///
        /// Specifications defined by shape, data type,
        ///
        /// flag trainable marks that specific tensor must not participate in gradient decent calculations
        /// it is non-trainable parameter - for example Batch Normalization's running_var/running_mean
        /// 
        ///
        TensorSpecs(Shape const &s=Shape(),DataType d=float_data,bool trainable = true) :
            shape_(s),
            dtype_(d)
        {
            is_trainable_ = trainable && is_floating_point_data_type(d);
        }
        
        bool operator==(TensorSpecs const &other) const
        {
            return shape_ == other.shape_ && dtype_ == other.dtype_ && is_trainable_ == other.is_trainable_;
        }
        
        bool operator!=(TensorSpecs const &other) const
        {
            return !(*this == other);
        }

        /// get tensor shape
        Shape const &shape() const
        {
            return shape_;
        }

        void shape(Shape const &s)
        {
            shape_=s;
        }

        ///
        /// return if tensor need to participate in gradient decent
        ///
        bool is_trainable() const
        {
            return is_trainable_;
        }

        ///
        /// Mark tensor as one that does not participate in gradients calculations
        ///
        void freeze()
        {
            is_trainable_ = false;
        }

        ///
        /// set - non-trainable property
        ///
        void is_trainable(bool v)
        {
            is_trainable_ = v;
        }

        ///
        /// Get reuired memory size for the tensor
        ///
        size_t memory_size() const
        {
            return shape_.total_size() * size_of_data_type(dtype_);
        }


        DataType dtype() const
        {
            return dtype_;
        }
    private:
        friend class Tensor;
        Shape shape_;
        DataType dtype_;
        bool is_trainable_;
    };


    ///
    /// Central Data Contrainer - Tensor
    ///
    /// Note all this object data is reference counted - copying is cheap but be aware that modifications
    /// of one tensor affect other
    ///
    class Tensor {
    public:
       
        /// 
        /// Create a tensor for specific context and allocate the device memory for it.
        /// 
        Tensor(Context &ctx,Shape const &s,DataType d=float_data,bool is_trainable=true);
        
        ///
        /// Create a tensor from external buffer
        ///
        Tensor(cl::Buffer const &buffer,cl_ulong offset,Shape const &s,DataType d=float_data,bool is_trainable=true);

        ///
        /// Create null tensor, binding such a tensor to kernel will pass NULL pointer
        ///
        Tensor();
        ///
        /// Copy construtor - uses reference counting points to same memory
        ///
        Tensor(Tensor const &) = default;
        ///
        /// Assignment - uses reference counting points to same memory
        ///
        Tensor &operator=(Tensor const &) = default;
        Tensor(Tensor &&) = default;
        Tensor &operator=(Tensor &&) = default;
        ~Tensor() {}
        
        TensorSpecs const &specs() const
        {
            return *specs_;
        }

        /// get tensor shape
        Shape const &shape() const
        {
            return specs_->shape();
        }

        ///
        /// return if tensor need to participate in gradient decent
        ///
        bool is_trainable() const
        {
            return specs_->is_trainable();
        }

        ///
        /// Get reuired memory size for the tensor
        ///
        size_t memory_size() const
        {
            return shape().total_size() * size_of_data_type(dtype());
        }


        DataType dtype() const
        {
            return specs_->dtype();
        }
        

        ///
        /// Reshape the tensor, the only requirement that ns.total_size() <= shape().total_size()
        ///
        void reshape(Shape const &ns);
        
        ///
        /// Get cl::Buffer for the tensor
        ///
        cl::Buffer &device_buffer() 
        { 
            return buffer_;
        }
        ///
        /// Get offset - you should always bind both buffer and offset since there is no pointer arithmetics 
        /// at host in OpenCL and same memory may be used for several tensors
        ///
        /// Always uses 64 bit ulong even of the device 32 bit. 
        ///
        cl_ulong device_offset() 
        {
            return offset_; 
        }
        ///
        /// Get a pointer to CPU memory - uses lazy allocation on demand
        ///
        void *host_data();


        ///
        /// Create tensor over all avalible size for data type d
        ///
        Tensor workspace_as_type(DataType d=float_data) const
        {
            size_t size = memory_size() / size_of_data_type(d);
            return sub_tensor(0,Shape(size),d);
        }
        
        ///
        /// Create tensor on the memory of existing tensor
        ///
        /// \param offset - memory offset in \a d units, i.e. if new tensor has float_data and offset=2 than address offset is 8 bytes
        /// \param s - shape of new tensor
        /// \prarm d - new tensor type
        /// \param trainable - mark as trainable tensor
        ///
        Tensor sub_tensor_target_offset(size_t offset,Shape const &s,DataType d=float_data,bool trainable = true) const
        {
            size_t bytes = offset * size_of_data_type(d);
            int this_sizeof = size_of_data_type(dtype());
            DLPRIM_CHECK(bytes % this_sizeof == 0);
            return sub_tensor(bytes / this_sizeof,s,d,trainable);
        }
        ///
        /// Create tensor on the memory of existing tensor
        ///
        /// \param offset - memory offset in the units of the data type of this tensor, if this tensor has type uint16_data and offset is 16 that the offset is 32 bytes
        /// \param s - shape of new tensor
        /// \prarm d - new tensor type
        /// \param trainable - mark as trainable tensor
        ///
        Tensor sub_tensor(size_t offset,Shape const &s,DataType d=float_data,bool trainable = true) const;

        ///
        /// Create a tensor with same memory but shape isn't connected to original - it is alias to 
        /// same data but with ability to modify shape
        ///
        Tensor alias() const
        {
            return sub_tensor(0,shape(),dtype(),is_trainable());
        }
        ///
        /// same as t=alias(); t.reshape(s); return t;
        ///
        Tensor alias(Shape const &new_shape) const
        {
            Tensor t = alias();
            t.reshape(new_shape);
            return t;
        }

        ///
        /// get pointer to the host pointer and cast to relevant type
        ///
        template<typename T>
        T *data()
        {
            DLPRIM_CHECK(TypeTraits<T>::data_type == dtype());
            return static_cast<T*>(host_data());
        }

        ///
        /// Copy external host memory to device, sync - for synchronoys copy
        ///
        void to_device(ExecutionContext const &c,void *host_memory,bool sync=true);
        ///
        /// Copy device memory to external host memory, sync - for synchronoys copy
        ///
        void to_host(ExecutionContext const &c,void *host_memory,bool sync=true);
        ///
        /// Copy host memory to device, sync - for synchronoys copy
        ///
        void to_device(ExecutionContext const &c,bool sync=true);
        ///
        /// Copy device memory to host, sync - for synchronoys copy
        ///
        void to_host(ExecutionContext const &c,bool sync=true);

        ///
        /// Assign buffer and offset as kernel argumnets, at position pos and pos+1, pos incrementeded twice
        ///
        void set_arg(cl::Kernel &k,int &pos)
        {
            k.setArg(pos++,device_buffer());
            k.setArg(pos++,device_offset());
        }

    private:
		struct HostMem;
        std::shared_ptr<TensorSpecs> specs_;
        std::shared_ptr<HostMem> host_;
        bool cpu_tensor_;
        int offset_;
        cl::Buffer buffer_;
        size_t capacity_;
        size_t full_capacity_;
    };

    ///
    /// Pair of tensor and its gradient for backpropogation
    ///
    struct TensorAndGradient {
        bool requires_gradient=true; ///<- set to false to prevent computations of the gradient
        float accumulate_gradient=0.0; ///<- accumulate factor. 0 - gradient is overrwritten, 1.0 is fully accumulated
        Tensor data; /// value
        Tensor diff; /// its gradient
    };

    inline std::ostream &operator<<(std::ostream &out,TensorSpecs const &ts)
    {
        out << '[' << ts.shape() << ",dtype=" << data_type_to_string(ts.dtype()) << ']';
        return out;
    }

    inline std::ostream &operator<<(std::ostream &out,Tensor const &ts)
    {
        out << ts.specs();
        return out;
    }

}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

