#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {
    void bind_as_dtype(cl::Kernel &k,int &p,double value,DataType dt)
    {
        switch(dt) {
        case  double_data:  k.setArg(p++,double(value)); break;
        case  float_data:   k.setArg(p++,float(value)); break;
        case  int64_data:   k.setArg(p++,cl_long(value)); break;
        case  int32_data:   k.setArg(p++,cl_int(value)); break;
        case  int16_data:   k.setArg(p++,cl_short(value)); break;
        case  int8_data:    k.setArg(p++,cl_char(value)); break;
        case  uint64_data:  k.setArg(p++,cl_long(value)); break;
        case  uint32_data:  k.setArg(p++,cl_int(value)); break;
        case  uint8_data:   k.setArg(p++,cl_uchar(value)); break;
        default:
            throw  NotImplementedError("Unsupported bind as type:" + data_type_to_opencl_type(dt));
        }
    }
    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<double>  ws,
                             std::string const &code,
                             ExecutionContext const &e)
    {
        Context ctx(e);
        Shape ref;
        DataType ref_type = float_data;
        DLPRIM_CHECK(xs.size() + ys.size() > 0);
        if(xs.empty()) {
            ref = ys[0].shape();
            ref_type = ys[0].dtype();
        }
        else {
            ref = xs[0].shape();
            ref_type = xs[0].dtype();
        }

        for(size_t i=0;i<xs.size();i++) {
            DLPRIM_CHECK(ref == xs[i].shape());
            DLPRIM_CHECK(ref_type == xs[i].dtype());
        }
        for(size_t i=0;i<ys.size();i++) {
            DLPRIM_CHECK(ref == ys[i].shape());
            DLPRIM_CHECK(ref_type == ys[i].dtype());
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
                                                                           "dtype",data_type_to_opencl_type(ref_type),
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
        for(double w:ws)
            bind_as_dtype(k,p,w,ref_type);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(total),cl::NullRange,e.events(),e.event("pointwise_exec"));
    }
} // core
} // dlprim

