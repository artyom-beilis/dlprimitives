///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/shape.hpp>
#include <iostream>
#include <algorithm>

namespace dlprim {
    Shape Shape::unsqueeze(int axis) const
    {
        if(axis < 0)
            axis = axis + size_ + 1;
        DLPRIM_CHECK(0 <= axis && axis<=size_);
        DLPRIM_CHECK(size_+1 <= max_tensor_dim);
        Shape r;
        for(int i=0;i<axis;i++)
            r.shape_[i] = shape_[i];
        r.shape_[axis] = 1;
        for(int i=axis;i<size_;i++)
            r.shape_[i+1] = shape_[i];
        r.size_ = size_ + 1;
        return r;
    }

    Shape Shape::reshape(std::vector<int> const &dims) const
    {
        std::vector<int> rshape;
        int calc_axis = -1;
        for(int i=0;i<int(dims.size());i++) {
            if(dims[i] == -1) {
                if(calc_axis != -1)
                    throw ValidationError("At most one index may be -1 in reshape");
                calc_axis = i;
                rshape.push_back(1);
            }
            else if(dims[i] == 0) {
                if(i >= size())
                    throw ValidationError("0 reshape for index too large");
                rshape.push_back(shape_[i]);
            }
            else {
                rshape.push_back(dims[i]);
            }
        }
        if(calc_axis != -1) {
            size_t provided = 1;
            for(auto const &d : rshape)
                provided *= d;
            if(total_size() % provided != 0 || total_size() < provided)
                throw ValidationError("Can't computed deduced shape, original shape isn't multiple of provided");
            rshape[calc_axis] = total_size() / provided;
        }
        Shape res = from_range(rshape.begin(),rshape.end());
        if(res.total_size() != total_size()) {
            std::ostringstream ss;
            ss << "Reshape from " << *this << " to " << res << " invalid total size" << std::endl;
            throw ValidationError(ss.str());
        }
        return res;
    }
    
    dlprim::Shape Shape::squeeze() const
    {
        std::vector<int> dims;
        for(int i=0;i<size();i++)
            if(shape_[i] == 1)
                dims.push_back(i);
        return squeeze(dims);
    }

    Shape Shape::squeeze(std::vector<int> dims) const
    {
        std::vector<size_t> squeezed;

        for(auto &axis : dims) {
            if (axis < 0) {
                axis = axis + size();
            }
            DLPRIM_CHECK(axis < size());
        }
        std::sort(dims.begin(),dims.end());

        int pos = 0;
        for(int i=0;i<size();i++) {
            if(pos < int(dims.size()) && i==dims[pos]) {
                DLPRIM_CHECK(shape_[i] == 1);
                pos++;
            }
            else {
                squeezed.push_back(shape_[i]);
            }
        }
        auto squeezed_shape = dlprim::Shape::from_range(squeezed.begin(),squeezed.end());
        if(squeezed_shape.size() == 0) {
            squeezed_shape = dlprim::Shape(1);
        }
        DLPRIM_CHECK(squeezed_shape.total_size() == total_size());
        return squeezed_shape;
    }

    Shape Shape::broadcast_strides(Shape const &target) const
    {
        DLPRIM_CHECK(size() <= target.size());
        Shape strides = target;
        size_t stride = 1;
        for(int i=strides.size()-1,pos=size_ - 1;i>=0;i--,pos--) {
            if(pos >= 0) {
                if(shape_[pos] == target[i]) {
                    strides[i] = stride;
                    stride *= target[i];
                }
                else if(shape_[pos] == 1) {
                    strides[i] = 0;
                }
                else {
                    std::ostringstream ss;
                    ss << "Can't broadcast " << *this << " to " << target;
                    throw ValidationError(ss.str());
                }
            }
            else {
                strides[i] = 0;
            }
        }
        return strides;
    }

    std::ostream &operator<<(std::ostream &o,Shape const &s)
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

    /// calculate numpy style broadcast shape
    Shape broadcast(Shape const &ain,Shape const &bin)
    {
        Shape a=ain,b=bin;
        while(a.size() < b.size())
            a=a.unsqueeze(0);
        while(b.size() < a.size())
            b=b.unsqueeze(0);

        Shape r=a;
        for(int i=0;i<a.size();i++) {
            if(a[i] == b[i])
                r[i]=a[i];
            else if(a[i] == 1)
                r[i]=b[i];
            else if(b[i] == 1)
                r[i]=a[i];
            else {
                std::ostringstream ss;
                ss << "Non broadcatable shapes" << ain << " and " << bin;
                throw ValidationError(ss.str());
            }
        }
        return r;
    }

    void shrink_broadcast_ranges(std::vector<Shape> &shapes)
    {
        // Calculate broadcasted shape and strides
        Shape broadcasted = shapes[0];
        for(size_t i=1;i<shapes.size();i++) {
            broadcasted = broadcast(broadcasted,shapes[i]);
        }
        std::vector<Shape> strides(shapes.size());
        for(size_t i=0;i<shapes.size();i++) {
            strides[i] = shapes[i].broadcast_strides(broadcasted);
        }

        /// find dimentions that can be converted to a single
        std::vector<bool> squeezable(broadcasted.size(),true);
        int squeezed=0;
        for(int i=0;i<broadcasted.size()-1;i++) {
            if(!squeezable[i])
                continue;
            for(size_t j=0;j<shapes.size();j++) {
                if(strides[j][i+1]*broadcasted[i+1] != strides[j][i])
                {
                    squeezable[i] = false;
                    break;
                }
            }
            if(squeezable[i])
                squeezed++;
        }
        int final_dim = broadcasted.size() - squeezed;

        for(size_t i=0;i<shapes.size();i++) {
            Shape input=shapes[i];
            while(input.size() < broadcasted.size()) {
                input = input.unsqueeze(0);
            }
            std::array<size_t,max_tensor_dim> values;
            for(int i=0,pos=0;i<final_dim;i++) {
                values[i] = input[pos];
                while(pos + 1 < broadcasted.size() && squeezable[pos]) {
                    values[i]*=input[pos+1];
                    pos++;
                }
                pos++;
            }
            Shape result = Shape::from_range(&values[0],&values[0] + final_dim);
            shapes[i] = result;
        }
    }

} // dlprim
