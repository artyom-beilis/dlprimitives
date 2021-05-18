#pragma once
#include <array>
#include <dlprim/definitions.hpp>
#include <ostream>

namespace dlprim {

    class Shape {
    public:
        Shape() : shape_{},size_(0) {}
        Shape(int b): shape_({b}),size_(1) {}
        Shape(int b,int c): shape_({b,c}),size_(2) {}
        Shape(int b,int c,int h): shape_({b,c,h}),size_(3) {}
        Shape(int b,int c,int h,int w): shape_({b,c,h,w}),size_(4) {}
        template<typename It>
        Shape(It begin, It end) : size_(0)
        {
            while(begin!=end) {
                if(size_ >= max_tensor_dim)
                    throw ValidatioError("Unsupported tensor size");
                shape_[size_++] = *begin++;
            }
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

        size_t size_no_batch() const
        {
            if(size_ <= 1)
                return 0;
            size_t r=1;
            for(int i=1;i<size_;i++) {
                r*=shape_[i];
            }
            return r;
        }
        size_t total_size() const
        {
            if(size_ == 0)
                return 0;
            size_t r=1;
            for(int i=0;i<size_;i++) {
                r*=shape_[i];
            }
            return r;
        }
        int size() const
        {
            return size_;
        }
        int operator[](int i) const
        {
            return shape_[i];
        }
    private:
        std::array<int,max_tensor_dim> shape_;
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
