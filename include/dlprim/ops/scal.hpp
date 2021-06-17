#pragma once
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/tensor.hpp>

namespace dlprim {
    class Scal {
    public:
        Scal(Context &ctx,DataType dt);
        ~Scal();
        void scale(float s,Tensor &t,ExecutionContext const &ec);
    private:
        Context ctx_;
        cl::Kernel k_;
    };
}
