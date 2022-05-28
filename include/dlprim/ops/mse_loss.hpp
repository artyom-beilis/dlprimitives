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
    namespace core { class PointwiseOperationBroadcastReduce; }

    struct MSELossConfig {
        enum Reduction {
            reduce_none,
            reduce_sum,
            reduce_mean
        };
        Reduction reduce = reduce_mean;
        static MSELossConfig from_json(json::value const &v);
    };

    class MSELoss : public Operator {
    public:
        MSELoss(Context &ctx,MSELossConfig const &cfg=MSELossConfig());
        virtual ~MSELoss();
        virtual char const *operator_type() const
        {
            return "MSELoss";
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
        void setup_gpu(std::vector<TensorSpecs> in,std::vector<TensorSpecs> out,size_t &workspace);
        void forward_cpu(Tensor &a,Tensor &b,Tensor &y);
        void backward_cpu(Tensor &dy,Tensor &a,Tensor &b,Tensor &dx,float scale,float accum);

        MSELossConfig cfg_;
        DataType dtype_;
        std::unique_ptr<core::PointwiseOperationBroadcastReduce> fwd_;
    };

}// dlprim

