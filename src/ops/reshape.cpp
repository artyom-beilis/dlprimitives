#include <dlprim/ops/reshape.hpp>

namespace dlprim {
    ReshapeBase::ReshapeBase(Context const &c) : Operator(c) 
    {
    }
    ReshapeBase::~ReshapeBase()
    {
    }
    void ReshapeBase::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &parameters,size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        parameters.clear();
        ws = 0;
        Shape s = new_shape(in[0].shape());
        DLPRIM_CHECK(in[0].shape().total_size() == s.total_size());
        out.assign({TensorSpecs(s,in[0].dtype(),in[0].is_trainable())});
    }
    void ReshapeBase::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        ws = 0;
        out.assign({new_shape(in[0])});
        DLPRIM_CHECK(out[0].total_size() == in[0].total_size());
    }
};
