#pragma once
#include <array>
#include <dlprim/definitions.hpp>
#include <ostream>

namespace dlprim {

    ///
    /// Tensor shape
    ///
    class Shape {
    public:
        Shape() : shape_{},size_(0) {}
        Shape(size_t b): shape_({b}),size_(1) {}
        Shape(size_t b,size_t c): shape_({b,c}),size_(2) {}
        Shape(size_t b,size_t c,size_t h): shape_({b,c,h}),size_(3) {}
        Shape(size_t b,size_t c,size_t h,size_t w): shape_({b,c,h,w}),size_(4) {}
       
        ///
        /// Initialize from pair of iterators
        ///
        template<typename It>
        static Shape from_range(It begin, It end)
        {
            Shape s;
            while(begin!=end) {
                if(s.size_ >= max_tensor_dim)
                    throw ValidationError("Unsupported tensor size");
                s.shape_[s.size_++] = *begin++;
            }
            return s;
        }
        
        bool operator==(Shape const &other) const
        {
            if(size_ != other.size_)
                return false;
            for(int i=0;i<size_;i++)
                if(shape_[i] != other.shape_[i])
                    return false;
            return true;
        }
        bool operator!=(Shape const &other) const
        {
            return !(*this == other);
        }

        ///
        /// Total number of elements in shape without the first one - batch
        ///
        size_t size_no_batch() const
        {
            if(size_ <= 0)
                return 0;
            size_t r=1;
            for(int i=1;i<size_;i++) {
                r*=shape_[i];
            }
            return r;
        }
        ///
        /// Total number of elements - product of all items
        ///
        size_t total_size() const
        {
            if(size_ == 0)
                return 0;
            size_t r=1;
            for(int i=0;i<size_;i++) {
                r*=size_t(shape_[i]);
            }
            return r;
        }
        ///
        /// dimetions count of the shape
        ///
        int size() const
        {
            return size_;
        }
        ///
        /// specific dimension
        ///
        size_t operator[](int i) const
        {
            return shape_[i];
        }
    private:
        std::array<size_t,max_tensor_dim> shape_;
        int size_;
    };

    inline std::ostream &operator<<(std::ostream &o,Shape const &s)
    {
        o << '(';
        for(int i=0;i<s.size();i++) {
            if(i > 0)
                o<<',';
            o << s[i];
        }
        o << ')';
        return o;
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
