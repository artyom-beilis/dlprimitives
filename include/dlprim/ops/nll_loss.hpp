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
    namespace json { class value; }

    struct NLLLossConfig {
        enum Reduction {
            reduce_none,
            reduce_sum,
            reduce_mean
        };
        Reduction reduce = reduce_mean;
        static NLLLossConfig from_json(json::value const &v);
    };

    class NLLLoss : public Operator {
    public:
        NLLLoss(Context &ctx,NLLLossConfig const &cfg=NLLLossConfig());
        virtual ~NLLLoss();
        virtual char const *operator_type() const
        {
            return "NLLLoss";
        }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);

        virtual void backward(  std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &,
                                Tensor &,
                                ExecutionContext const &e);

    private:
        template<typename Index>
        void forwad_cpu(Tensor &x,Tensor &lbl,Tensor &y);
        template<typename Index>
        void backward_cpu(Tensor &dx,Tensor &lbl,Tensor &dy,float accum);

        NLLLossConfig cfg_;
    };

}// dlprim

