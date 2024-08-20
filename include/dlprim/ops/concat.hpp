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
    namespace core { class SliceCopy; class Scale; }

    struct ConcatConfig {
        int dim = 1;
        static ConcatConfig from_json(json::value const &v);
    };


    struct SliceConfig {
        int dim = 1;
        int begin = 0;
        int end = -1;
        static SliceConfig from_json(json::value const &v);
    };

    class ConcatSliceBase  {
    protected:
        template<typename Cp>
        static void copy_cpu(size_t &offset,int dim,Tensor &in,Tensor &out,Cp cp);

    };

    class Concat : public Operator, public ConcatSliceBase {
    public:
        Concat(Context &ctx,ConcatConfig const &config);
        virtual ~Concat();
        virtual char const *operator_type() const
        {
            return "Concat";
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

        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);
    private:

        ConcatConfig cfg_;
        DataType dtype_;
        std::unique_ptr<core::SliceCopy> copy_;
    };

    class Slice : public Operator, public ConcatSliceBase {
    public:
        Slice(Context &ctx,SliceConfig const &config);
        virtual ~Slice();
        virtual char const *operator_type() const
        {
            return "Slice";
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

        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);
    private:

        SliceConfig cfg_;
        DataType dtype_;
        std::unique_ptr<core::SliceCopy> copy_;
        std::unique_ptr<core::Scale> scale_;
    };
}
