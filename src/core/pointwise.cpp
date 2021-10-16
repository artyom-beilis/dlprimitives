#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {
    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<float>  ws,
                             std::string const &code,
                             ExecutionContext const &e)
    {
        Context ctx(e);
        Shape ref;
        DLPRIM_CHECK(xs.size() + ys.size() > 0);
        if(xs.empty())
            ref = ys[0].shape();
        else
            ref = xs[0].shape();

        for(size_t i=0;i<xs.size();i++) {
            DLPRIM_CHECK(ref == xs[i].shape());
        }
        for(size_t i=0;i<ys.size();i++) {
            DLPRIM_CHECK(ref == ys[i].shape());
        }
        std::ostringstream params,loads,saves;
        for(size_t i=0;i<xs.size();i++) {
            params<<", __global dtype const *px" << i<< ", ulong px"<<i<<"_offset ";
            loads<<"dtype x"<<i<<"=px"<<i<<"[index]; ";
        }
        for(size_t i=0;i<ys.size();i++) {
            params<<", __global dtype *py" << i<< ", ulong py"<<i<<"_offset ";
            loads<<"dtype y"<<i<<";";
            saves<<"py"<<i<<"[index]=y"<<i<<"; ";
        }
        for(size_t i=0;i<ws.size();i++) {
            params<<", dtype w" <<i;
        }

        std::ostringstream code_fixed;
        for(size_t i=0;i<code.size();i++)
            if(code[i]=='\n')
                code_fixed << "\\\n";
            else
                code_fixed << code[i];
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"pointwise",
                                                                           "#PARAMS",params.str(),
                                                                           "#LOADS",loads.str(),
                                                                           "#SAVES",saves.str(),
                                                                           "#CALC",code_fixed.str());
        cl::Kernel k(prog,"exec");
        cl_ulong total = ref.total_size();
        int p=0;
        k.setArg(p++,total);
        for(Tensor &x:xs)
            x.set_arg(k,p);
        for(Tensor &y:ys)
            y.set_arg(k,p);
        for(float w:ws)
            k.setArg(p++,w);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(total),cl::NullRange,e.events(),e.event("pointwise_exec"));
    }
} // core
} // dlprim

