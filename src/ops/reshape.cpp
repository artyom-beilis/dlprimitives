///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/reshape.hpp>
#include <dlprim/json.hpp>

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
    SqueezeConfig SqueezeConfig::from_json(json::value const &v)
    {
        SqueezeConfig cfg;
        cfg.dims = v.get("dims",cfg.dims);
        if(!cfg.dims.empty())
            cfg.all = false;
        cfg.all = v.get("all",cfg.all);
        if(cfg.all && !cfg.dims.empty())
            throw ValidationError("Can't use both squeeze all and squeeze by dims");
        return cfg;
    }
    ReshapeConfig ReshapeConfig::from_json(json::value const &v)
    {
        ReshapeConfig cfg;
        cfg.dims = v.get("dims",cfg.dims);
        if(cfg.dims.empty())
            throw ValidationError("Empty reshape dims");
        return cfg;
    }
};
