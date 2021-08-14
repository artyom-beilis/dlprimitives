#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    namespace core { class SliceCopy; }

    struct ConcatConfig {
        int dim = 1;
        static ConcatConfig from_json(json::value const &v);
    };
    class Concat : public Operator {
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
        template<typename Cp>
        void copy_cpu(size_t &offset,Tensor &in,Tensor &out,Cp cp);

        ConcatConfig cfg_;
        DataType dtype_;
        std::unique_ptr<core::SliceCopy> copy_;
    };
}
