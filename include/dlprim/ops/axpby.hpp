#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {
    class AXPBY {
    public:
        AXPBY(Context &ctx,DataType dt=float_data);
        ~AXPBY();
        void apply(float a,Tensor &x,float b,Tensor &y,Tensor &z,ExecutionContext const &e);
    private:
        Context ctx_;
        cl::Kernel kernel_;
    };
} // namespace
