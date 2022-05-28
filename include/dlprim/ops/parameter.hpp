///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {
    struct ParameterConfig {
        TensorSpecs spec;
        static ParameterConfig from_json(json::value const &v);
    };

    class Parameter : public Operator {
    public:
        Parameter(Context &ctx,ParameterConfig const &cfg) : Operator(ctx), cfg_(cfg)
        {
        }
        virtual ~Parameter()
        {
        }
        virtual char const *operator_type() const
        {
            return "Parameter";
        }
        virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace)
        {
            workspace = 0;
            DLPRIM_CHECK(in.empty());
            parameters.assign({cfg_.spec});
            out.assign({cfg_.spec});
        }

        virtual void reshape(std::vector<Shape> const &,
                             std::vector<Shape> &out,
                             size_t &ws)
        {
            ws = 0;
            out.assign({cfg_.spec.shape()});
        }

		virtual void forward(std::vector<Tensor> &,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &,
                             ExecutionContext const &q)
        {
            copy_and_scale(output.at(0),parameters.at(0),0.0,q);
        }

        virtual void backward(std::vector<TensorAndGradient> &,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &,
                              ExecutionContext const &q)
        {
            if(parameters.at(0).requires_gradient) {
                copy_and_scale(parameters[0].diff,output.at(0).diff,parameters[0].accumulate_gradient,q);
            }
        }
    private:
        void copy_and_scale(Tensor &tgt,Tensor &src,float accum,ExecutionContext const &q);
        ParameterConfig cfg_;

    };
}
