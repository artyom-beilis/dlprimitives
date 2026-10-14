#pragma once

#include<dlprim/core/common.hpp>
#include<dlprim/gpu/program_cache.hpp>

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
    virtual void enqueue(Tensor&x, Tensor&y, Tensor&idx, Shape xstrds, Shape ystrds, Shape istrds, uint64_t axis, Tensor *flag, double alpha, ExecutionContext e) = 0;
    static std::unique_ptr<Select> create(Context &ctx, int dim, DataType dtype, DataType itype, Op op, Target target, bool scal);
};
} // core
} // dlprim