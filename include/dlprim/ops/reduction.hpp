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
    namespace core { class PointwiseOperationBroadcastReduce; }
    namespace json { class value; }

    struct ReductionConfig {
        enum Method {
            sum,
            sumsq,
            abssum,
            mean
        };
        Method method = sum;
        float output_scale = 1.0;
        bool keep_dim = true;
        int start_axis = 0;
        std::vector<int> dims;
        static ReductionConfig from_json(json::value const &v);
    };

    
    class Reduction : public Operator {
    public:
        Reduction(Context &ctx,ReductionConfig const &cfg = ReductionConfig());
        virtual ~Reduction();
        
        virtual char const *operator_type() const
        {
            return "Reduction";
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
        struct FWDData;
        struct BWDData;
        struct FWDBase;
        struct BWDBase;
        struct SumFwd;
        struct SumBwd;
        struct Sum2Fwd;
        struct Sum2Bwd;
        struct L1Fwd;
        struct L1Bwd;
        template<typename Op>
        void iterate(Op &op);
        void forward_cpu(Tensor &input,Tensor &output);
        void backward_cpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float accum);
        
        void forward_gpu(Tensor &input,Tensor &output,Tensor &ws,ExecutionContext const &q);
        void backward_gpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float accum,ExecutionContext const &q);
        void config_broadcast(TensorSpecs x,TensorSpecs y);
        void calc_shapes();
        float get_coeff();

        ReductionConfig cfg_;
        std::unique_ptr<core::PointwiseOperationBroadcastReduce> broadcast_;
        std::vector<int> dims_to_reduce_;
        Shape y_strides_,full_y_,squeezed_y_,x_shape_;
        DataType dtype_;
    };

}
