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
    class Scal;

    struct SoftmaxConfig {
        bool log=false;
        static SoftmaxConfig from_json(json::value const &v);
    };

    class SoftmaxBase {
    public:
        
    protected:
        bool setup_kernel_params(int sm_range);

        int wg_size_ = 0;
        int items_per_wi_ = 0;
        int sm_range_ = -1;
        int nd_range_;
    };
    
    class Softmax : public Operator, public SoftmaxBase {
    public:
        Softmax(Context &ctx,SoftmaxConfig const &cfg=SoftmaxConfig());
        virtual ~Softmax();
        
        virtual char const *operator_type() const
        {
            return "Softmax";
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
        void forward_cpu(Tensor &input,Tensor &output);
        void backward_cpu(Tensor &dx,Tensor &y,Tensor &dy,float accum);
        SoftmaxConfig cfg_;
        DataType dtype_;
    };

    class SoftmaxWithLoss : public Operator, public SoftmaxBase {
    public:
        SoftmaxWithLoss(Context &ctx,SoftmaxConfig const &cfg=SoftmaxConfig());
        virtual ~SoftmaxWithLoss();
        
        virtual char const *operator_type() const
        {
            return "SoftmaxWithLoss";
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
        template<typename IndexType>
        void forward_cpu_loss(Tensor &input,Tensor &label,Tensor &loss);
        template<typename IndexType>
        void backward_cpu_loss(Tensor &x,Tensor &dx,Tensor &label,Tensor &loss,float factor);
        void backward_gpu_loss(Tensor &input,Tensor &diff, Tensor &label,Tensor &output,float factor, ExecutionContext const &ctx);
        void forward_gpu_loss(Tensor &input,Tensor &label,Tensor &output,ExecutionContext const &ctx);
        void setup_kernel(int sm_range);
        DataType dtype_;
        std::string itype_;
        cl::Kernel kernel_,kernel_bwd_;
        std::unique_ptr<Scal> scal_;
    };
}
