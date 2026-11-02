#pragma once

#include<dlprim/context.hpp>
#include<dlprim/tensor.hpp>

namespace dlprim {
namespace core {
class Select {
public:
    enum Op {
        ASSIGN = 0,
        ADD = 1
    };
    enum Target {
        X = 0,
        Y = 1
    };
    virtual ~Select() = default;
    virtual void enqueue(Tensor&x, Tensor&y, Tensor&idx, int64_t axis, Tensor *flag, double alpha, ExecutionContext e, bool negIdx=true) = 0;
    static std::unique_ptr<Select> create(Context &ctx, int dim, DataType dtype, DataType itype, Op op, Target target, bool scal);
};
} // core
} // dlprim