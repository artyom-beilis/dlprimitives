#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/bias.hpp>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
    BWBias::~BWBias() {}

    BWBias::BWBias(Context &ctx,Shape const &sp,DataType dt) 
    {
        if(ctx.is_cpu_context())
            return;

        impl_ = std::move(core::BiasBackwardFilter::create(ctx,sp,dt));
    }

    size_t BWBias::workspace() const
    {
        if(impl_.get())
            return impl_->workspace();
        return 0;
    }
    void BWBias::backward(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e)
    {
        DLPRIM_CHECK(dy.shape().size() >= 2);
        DLPRIM_CHECK(dw.shape().size() == 1);
        DLPRIM_CHECK(dw.shape().total_size() == dy.shape()[1]);
        if(impl_.get())
            impl_->enqueue(dy,dw,ws,beta,e);
        else
            backward_cpu(dy,dw,beta);
    }
    void BWBias::backward_cpu(Tensor &dy,Tensor &dw,float beta)
    {
        int batches = dy.shape()[0];
        int features = dw.shape()[0];
        int rows_columns = dy.shape().size_no_batch() / dy.shape()[1];
        float *dyp = dy.data<float>();
        float *dwp = dw.data<float>();
        if(beta == 0)
            memset(dwp,0,features*sizeof(float));
        else
            cblas_sscal(features,beta,dwp,1);
        if(rows_columns == 1) {
            for(int b=0;b<batches;b++) {
                cblas_saxpy(features,1.0,dyp,1,dwp,1);
                dyp+=features;
            }
        }
        else {
            for(int b=0;b<batches;b++) {
                for(int f=0;f<features;f++) {
                    for(int pix=0;pix<rows_columns;pix++) {
                        dwp[f] += *dyp++;
                    }
                }
            }
        }
    }

}
