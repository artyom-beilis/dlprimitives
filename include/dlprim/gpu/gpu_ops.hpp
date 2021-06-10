#pragma once
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/tensor.hpp>

namespace dlprim {
    namespace gpu {
        class Scal {
        public:
            Scal(Context &ctx,DataType dt) {
                DLPRIM_CHECK(dt==float_data);
                cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"scal");
                k_ = cl::Kernel(prog,"sscal");
            }
            void scale(float s,Tensor &t,ExecutionContext const &ec)
            {
                int p = 0;
                int size = t.shape().total_size();
                int wg;
                if(size >= 1024)
                    wg = 256;
                else
                    wg = 64;
                k_.setArg(p++,int(size));
                k_.setArg(p++,s);
                k_.setArg(p++,t.device_buffer());
                k_.setArg(p++,int(t.device_offset()));
                cl::NDRange l(wg);
                cl::NDRange g=gpu::round_range(size,l);
                ec.queue().enqueueNDRangeKernel(k_,cl::NullRange,g,l,ec.events(),ec.event("sscal"));
            }
        private:
            cl::Kernel k_;
        };
    }
}
