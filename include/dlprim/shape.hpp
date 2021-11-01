#pragma once
#include <array>
#include <vector>
#include <dlprim/definitions.hpp>
#include <ostream>
#include <sstream>

namespace dlprim {

    class Shape;
    
    std::ostream &operator<<(std::ostream &o,Shape const &s);


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
        Shape(size_t b,size_t c,size_t d,size_t h,size_t w): shape_({b,c,d,h,w}),size_(5) {}
       
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
        size_t &operator[](int i)
        {
            return shape_[i];
        }
        ///
        /// specific dimension
        ///
        size_t operator[](int i) const
        {
            return shape_[i];
        }
        ///
        /// Split the shape accordint to axis - before axis and after
        /// for example:
        /// - [2,3,4,5] split axis==2 -> [6,4,5]
        /// - [2,3,4,5] split axis == 0 -> [1,2,60]
        /// - [2,3] split axis == 2 -> [6,1,1]
        Shape split_and_merge_over_axis(int axis) const
        {
            size_t d0 = 1,d1 = 1,d2=1;
            for(int i=0;i<size_;i++) {
                if(i < axis)
                    d0*=shape_[i];
                else if(i == axis)
                    d1*=shape_[i];
                else
                    d2*=shape_[i];
            }
            return Shape(d0,d1,d2);
        }

        ///
        /// Add dimention=1 at axis location, for example for Shape(2,3).unsqueeze(0) == Shape(1,2,3)
        ///
        Shape unsqueeze(int axis) const;

        ///
        /// Compute strides needed for broadcasting this shape to target shape
        ///
        Shape broadcast_strides(Shape const &target) const;

        size_t const *begin() const
        {
            return &shape_[0];
        }
        size_t const *end() const
        {
            return begin() + size_;
        }

    private:
        std::array<size_t,max_tensor_dim> shape_;
        int size_;
    };

    /// calculate numpy style broadcast shape
    Shape broadcast(Shape const &ain,Shape const &bin);
    
    ///
    /// Broadcast shapes numpy style and remove planes that can be merged.
    ///
    /// For example:
    ///   shrink([2,3,4],[2,1,1]) -> [2,12],[2,1]
    ///   shrink([2,3,4],[2,3,4]) -> [24],[24]
    ///   shrink([2,3,4],[1]) -> [24],[1]
    ///   shrink([2,3,4],[3,1]) -> [2,3,4],[1,3,1]
    ///   shrink([2,3,4,5],[3,1,1]) -> [2,3,20],[1,3,1]
    ///   shrink([2,3,4,5],[1,3,4,1]) -> [2,12,5],[1,12,1]
    ///
    void shrink_broadcast_ranges(std::vector<Shape> &shapes);


};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
