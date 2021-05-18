#pragma once 
#include <dlprim/context.hpp>
#include <dlprim/shape.hpp>
#include <memory>
namespace dlprim {
    
    class TensorSpecs {
    public:
        TensorSpecs(Shape const &s=Shape(),DataType d=float_data) :
            shape_(s),
            dtype_(d)
        {
        }
        
        bool operator==(TensorSpecs const &other) const
        {
            return shape_ == other.shape_ && dtype_ == other.dtype_;
        }
        
        bool operator!=(TensorSpecs const &other) const
        {
            return !(*this == other);
        }

        Shape const &shape() const
        {
            return shape_;
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
        
        cl::Buffer &device_buffer() 
        { 
            DLPRIM_CHECK(!cpu_tensor_);
            return buffer_;
        }
        size_t device_offset() 
        {
            DLPRIM_CHECK(!cpu_tensor_);
            return offset_; 
        }
        void *host_data();

        template<typename T>
        T *data()
        {
            DLPRIM_CHECK(TypeTraits<T>::data_type == dtype_);
            return static_cast<T*>(host_data());
        }

        
        void to_device(ExecutionContext const &c,bool sync=true);
        void to_host(ExecutionContext const &c,bool sync=true);

    private:
        std::shared_ptr<void> host_;
        bool cpu_tensor_;
        int offset_;
        cl::Buffer buffer_;
    };

}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

