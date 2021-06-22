#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
    BWBias::~BWBias() {}

    BWBias::BWBias(Context &ctx,int batch,int rows_columns,DataType dt) :
        ctx_(ctx),
        batch_(batch),
        rows_columns_(rows_columns)
    {
        DLPRIM_CHECK(dt == float_data);
        int total_size = batch_ * rows_columns_;
        two_stage_reduction_ = false;
        if(ctx_.is_cpu_context())
            return;
        
        if(total_size > 256 * 16) {
            two_stage_reduction_ = true;
            wg_ = 256;
            items_per_wi_ = 16;
            int reduce_1st = wg_ * items_per_wi_;
            size2_ = (total_size + reduce_1st - 1) / reduce_1st;
            if(size2_ >= 256)
                wg2_ = 256;
            else if(size2_ >= 128)
                wg2_ = 128;
            else
                wg2_ = 64;
            items_per_wi2_ = (size2_ + wg2_ - 1) / wg2_;
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"bwd_bias",
                    "WG_SIZE",wg2_,
                    "ITEMS_PER_WI",items_per_wi2_,
                    "SIZE_2D",size2_);
            kernel2_ = cl::Kernel(prog,"bwd_bias");
        }
        else {
            two_stage_reduction_ = false;
            size2_ = 1;
            if(total_size <= 64)
                wg_ = 64;
            else if(total_size <= 128)
                wg_ = 128;
            else
                wg_ = 256;
            items_per_wi_ = (total_size + wg_ - 1) / wg_;

        }
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"bwd_bias",
                "WG_SIZE",wg_,
                "ITEMS_PER_WI",items_per_wi_,
                "SIZE_2D",rows_columns_
                );
        kernel_ = cl::Kernel(prog,"bwd_bias");
    }

    size_t BWBias::workspace(int features) const
    {
        if(two_stage_reduction_)
            return features * size2_ * size_of_data_type(float_data);
        return 0;
    }
    void BWBias::backward_gpu(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e)
    {
        int total_size = dy.shape()[0] * rows_columns_;
        int features = dw.shape()[0];
        if(two_stage_reduction_) {
            cl::NDRange l(wg_,1);
            cl::NDRange g=gpu::round_range(wg_ * size2_,features,l);
            int p=0;
            kernel_.setArg(p++,features);
            kernel_.setArg(p++,total_size);
            kernel_.setArg(p++,dy.device_buffer());
            kernel_.setArg(p++,int(dy.device_offset()));
            kernel_.setArg(p++,ws.device_buffer());
            kernel_.setArg(p++,int(ws.device_offset()));
            kernel_.setArg(p++,size2_);
            kernel_.setArg(p++,0.0f);
            auto ec1 = e.generate_series_context(0,2);
            auto ec2 = e.generate_series_context(0,2);
            e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,ec1.events(),ec1.event("bwd_bias_a"));
            p=0;
            kernel2_.setArg(p++,features);
            kernel2_.setArg(p++,size2_);
            kernel2_.setArg(p++,ws.device_buffer());
            kernel2_.setArg(p++,int(ws.device_offset()));
            kernel2_.setArg(p++,dw.device_buffer());
            kernel2_.setArg(p++,int(dw.device_offset()));
            kernel2_.setArg(p++,1);
            kernel2_.setArg(p++,beta);
            e.queue().enqueueNDRangeKernel(kernel2_,cl::NullRange,cl::NDRange(wg2_,features),cl::NDRange(wg2_,1),ec2.events(),ec2.event("bwd_bias_b"));
        }
        else {
            int norm_size = (total_size + items_per_wi_ - 1) / items_per_wi_;
            cl::NDRange l(wg_,1);
            cl::NDRange g=gpu::round_range(norm_size,features,l);
           
            int p=0;
            kernel_.setArg(p++,features);
            kernel_.setArg(p++,total_size);
            kernel_.setArg(p++,dy.device_buffer());
            kernel_.setArg(p++,int(dy.device_offset()));
            kernel_.setArg(p++,dw.device_buffer());
            kernel_.setArg(p++,int(dw.device_offset()));
            kernel_.setArg(p++,1);
            kernel_.setArg(p++,beta);
            e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,e.events(),e.event("bwd_bias"));
        }
    }
    void BWBias::backward_cpu(Tensor &dy,Tensor &dw,float beta)
    {
        int batches = dy.shape()[0];
        int features = dw.shape()[0];
        float *dyp = dy.data<float>();
        float *dwp = dw.data<float>();
        if(beta == 0)
            memset(dwp,0,features*sizeof(float));
        else
            cblas_sscal(features,beta,dwp,1);
        if(rows_columns_ == 1) {
            for(int b=0;b<batches;b++) {
                cblas_saxpy(features,1.0,dyp,1,dwp,1);
                dyp+=features;
            }
        }
        else {
            for(int b=0;b<batches;b++) {
                for(int f=0;f<features;f++) {
                    for(int pix=0;pix<rows_columns_;pix++) {
                        dwp[f] += *dyp++;
                    }
                }
            }
        }
    }

}
