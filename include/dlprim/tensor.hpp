#pragma once 
#include <dlprim/context.hpp>
#include <dlprim/shape.hpp>
#include <memory>
namespace dlprim {
    
    ///
    /// Definition of Tensor without actual memory/object
    ///
    class TensorSpecs {
    public:
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

        Shape const &shape() const
        {
            return shape_;
        }

        bool is_trainable() const
        {
            return is_trainable_;
        }

        size_t memory_size() const
        {
            return shape_.total_size() * size_of_data_type(dtype_);
        }


        DataType dtype() const
        {
            return dtype_;
        }
    protected:
        Shape shape_;
        DataType dtype_;
        bool is_trainable_;
    };

    inline std::ostream &operator<<(std::ostream &out,TensorSpecs const &ts)
    {
        out << '[' << ts.shape() << ",dtype=" << data_type_to_string(ts.dtype()) << ']';
        return out;
    }

    class Tensor : public TensorSpecs {
    public:
        
        Tensor(Context &ctx,Shape const &s,DataType d=float_data);

        Tensor();
        Tensor(Tensor const &) = default;
        Tensor &operator=(Tensor const &) = default;
        Tensor(Tensor &&) = default;
        Tensor &operator=(Tensor &&) = default;
        ~Tensor() {}

        void reshape(Shape const &ns);
        
        cl::Buffer &device_buffer() 
        { 
            return buffer_;
        }
        size_t device_offset() 
        {
            return offset_; 
        }
        void *host_data();
        
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

        template<typename T>
        T *data()
        {
            DLPRIM_CHECK(TypeTraits<T>::data_type == dtype_);
            return static_cast<T*>(host_data());
        }

        
        void to_device(ExecutionContext const &c,bool sync=true);
        void to_host(ExecutionContext const &c,bool sync=true);

        void set_arg(cl::Kernel &k,int &pos)
        {
            k.setArg(pos++,device_buffer());
            k.setArg(pos++,int(device_offset()));
        }

    private:
		struct HostMem;
        std::shared_ptr<HostMem> host_;
        bool cpu_tensor_;
        int offset_;
        cl::Buffer buffer_;
        size_t capacity_;
        size_t full_capacity_;
    };

    struct TensorAndGradient {
        bool requires_gradient=true;
        float accumulate_gradient=0.0;
        Tensor data;
        Tensor diff;
    };


}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

