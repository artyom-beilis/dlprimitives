///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/parameter.hpp>
#include <dlprim/json.hpp>
#include <my_cblas.hpp>
#include <dlprim/core/pointwise.hpp>
namespace dlprim {
    ParameterConfig ParameterConfig::from_json(json::value const &v)
    {
        std::vector<int> dims = v.get<std::vector<int> >("shape");
        DataType dtype = string_to_data_type(v.get("dtype","float"));
        bool is_trainable = v.get("is_trainable",true);
        ParameterConfig cfg;
        cfg.spec = TensorSpecs(Shape::from_range(dims.begin(),dims.end()),dtype,is_trainable);
        return cfg;
    }

    void Parameter::copy_and_scale(Tensor &tgt,Tensor &src,float accum,ExecutionContext const &q)
    {
        DLPRIM_CHECK(tgt.specs() == src.specs());
        if(ctx_.is_cpu_context()) {
            if(accum == 0) {
                memcpy(tgt.host_data(),src.host_data(),tgt.memory_size());
            }
            else {
                size_t N = tgt.shape().total_size();
                if(accum != 1.0)
                    cblas_sscal(N,accum, tgt.data<float>(),1);

                cblas_saxpy(N,1.0f,src.data<float>(),1,tgt.data<float>(),1);
            }
        }
        else {
            if(accum == 0)
                core::pointwise_operation({src},{tgt},{},"y0=x0;",q);
            else
                core::pointwise_operation({src,tgt},{tgt},{accum},"y0=x0+w0*x1;",q);
        }
    }
}
