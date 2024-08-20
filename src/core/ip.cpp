///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    class IPForwardImpl : public IPForward {
    public:
        IPForwardImpl(Context &ctx,IPSettings const &cfg,bool bias,StandardActivations activation)
        {
            int batch = cfg.optimal_batch_size;
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                ctx,cfg.dtype,false,true,
                batch,cfg.outputs,cfg.inputs,
                (bias ? gpu::GEMM::bias_N : gpu::GEMM::no_bias),
                activation            
            ));
        }
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y,ExecutionContext const &e)
        {
            int batch = x.shape()[0];
            int inps  = x.shape().size_no_batch();
            int outs  = y.shape()[1];
            int bias_offset = bias ? bias->device_offset() : 0;
            cl::Buffer *bias_buffer = bias ? &bias->device_buffer() :  nullptr;
            gemm_->gemm(batch,outs,inps,
                    x.device_buffer(),x.device_offset(),inps,
                    w.device_buffer(),w.device_offset(),inps,
                    y.device_buffer(),y.device_offset(),outs,
                    bias_buffer,bias_offset,0.0f,
                    y.shape().total_size(),
                    e);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };

    std::unique_ptr<IPForward> IPForward::create(Context &ctx,IPSettings const &config,bool bias,StandardActivations activation)
    {
        std::unique_ptr<IPForward> r(new IPForwardImpl(ctx,config,bias,activation));
        return r;
    }




    class IPBackwardDataImpl : public IPBackwardData {
    public:
        IPBackwardDataImpl(Context &ctx,IPSettings const &cfg)
        {
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                        ctx,cfg.dtype,false,false,
                        cfg.optimal_batch_size,cfg.inputs,cfg.outputs,
                        gpu::GEMM::no_bias,
                        StandardActivations::identity            
                        ));
        }
        virtual void enqueue(Tensor &dx,Tensor &M,Tensor &dy,float factor,ExecutionContext const &e) 
        {
            int outputs = dy.shape()[1];
            int inputs  = dx.shape().size_no_batch();
            gemm_->gemm(dy.shape()[0],inputs,outputs,
                        dy.device_buffer(),
                        dy.device_offset(),
                        outputs,
                        M.device_buffer(),
                        M.device_offset(),
                        M.shape()[1],
                        dx.device_buffer(),
                        dx.device_offset(),
                        inputs,
                        nullptr,0,
                        factor,
                        dx.shape().total_size(),
                        e);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };

    std::unique_ptr<IPBackwardData> IPBackwardData::create(Context &ctx,IPSettings const &config)
    {
        std::unique_ptr<IPBackwardData> r(new IPBackwardDataImpl(ctx,config));
        return r;
    }

    class IPBackwardFilterImpl : public IPBackwardFilter {
    public:
        IPBackwardFilterImpl(Context &ctx,IPSettings const &config)
        {
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                        ctx,config.dtype,true,false,
                        config.outputs,config.inputs,config.optimal_batch_size,
                        gpu::GEMM::no_bias,
                        StandardActivations::identity            
            ));
        }
        virtual void enqueue(Tensor &x,Tensor &dM,Tensor &dy,float factor,ExecutionContext const &e)
        {
            int outputs = dy.shape()[1];
            int inputs = x.shape().size_no_batch();
            gemm_->gemm(outputs,inputs,dy.shape()[0],
                                dy.device_buffer(),
                                dy.device_offset(),
                                outputs,
                                x.device_buffer(),
                                x.device_offset(),
                                inputs,
                                dM.device_buffer(),
                                dM.device_offset(),
                                dM.shape()[1],
                                nullptr,0,
                                factor,
                                dM.shape().total_size(),
                                e);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };
    
    std::unique_ptr<IPBackwardFilter> IPBackwardFilter::create(Context &ctx,IPSettings const &config)
    {
        std::unique_ptr<IPBackwardFilter> r(new IPBackwardFilterImpl(ctx,config));
        return r;
    }


    ///
    /// Calculate filter
    ///
    class BiasBackwardFilterImpl : public BiasBackwardFilter {
    public:
        BiasBackwardFilterImpl(Context &ctx,Shape const &shape,DataType dt=float_data) :
            batch_(shape[0]),
            features_(shape[1]),
            rows_columns_(shape.size_no_batch() / shape[1]),
            two_stage_reduction_(false),
            dt_(dt)
        {
            DLPRIM_CHECK(dt == float_data);
            int total_size = batch_ * rows_columns_;
            two_stage_reduction_ = false;

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
                cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"bwd_bias",
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
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"bwd_bias",
                    "WG_SIZE",wg_,
                    "ITEMS_PER_WI",items_per_wi_,
                    "SIZE_2D",rows_columns_
                    );
            kernel_ = cl::Kernel(prog,"bwd_bias");

        }
        virtual size_t workspace()
        {
            if(two_stage_reduction_)
                return features_ * size2_ * size_of_data_type(float_data);
            return 0;
        }
        virtual void enqueue(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e)
        {
            DLPRIM_CHECK(features_ == int(dw.shape()[0]));
            int total_size = dy.shape()[0] * rows_columns_;
            if(two_stage_reduction_) {
                Tensor float_ws = ws.workspace_as_type(dt_);
                cl::NDRange l(wg_,1);
                cl::NDRange g=gpu::round_range(wg_ * size2_,features_,l);
                int p=0;
                kernel_.setArg(p++,features_);
                kernel_.setArg(p++,total_size);
                dy.set_arg(kernel_,p);
                float_ws.set_arg(kernel_,p);
                kernel_.setArg(p++,size2_);
                kernel_.setArg(p++,0.0f);
                auto ec1 = e.generate_series_context(0,2);
                auto ec2 = e.generate_series_context(1,2);
                e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,ec1.events(),ec1.event("bwd_bias_a"));
                p=0;
                kernel2_.setArg(p++,features_);
                kernel2_.setArg(p++,size2_);
                float_ws.set_arg(kernel2_,p);
                dw.set_arg(kernel2_,p); 
                kernel2_.setArg(p++,1);
                kernel2_.setArg(p++,beta);
                e.queue().enqueueNDRangeKernel(kernel2_,cl::NullRange,cl::NDRange(wg2_,features_),cl::NDRange(wg2_,1),ec2.events(),ec2.event("bwd_bias_b"));
            }
            else {
                int norm_size = (total_size + items_per_wi_ - 1) / items_per_wi_;
                cl::NDRange l(wg_,1);
                cl::NDRange g=gpu::round_range(norm_size,features_,l);
               
                int p=0;
                kernel_.setArg(p++,features_);
                kernel_.setArg(p++,total_size);
                dy.set_arg(kernel_,p);
                dw.set_arg(kernel_,p);
                kernel_.setArg(p++,1);
                kernel_.setArg(p++,beta);
                e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,g,l,e.events(),e.event("bwd_bias"));
            }
        }
    private:
        int batch_;
        int features_;
        int rows_columns_;
        int wg_;
        int items_per_wi_;

        int wg2_;
        int items_per_wi2_;
        int size2_;

        bool two_stage_reduction_;

        cl::Kernel kernel_;
        cl::Kernel kernel2_;
        DataType dt_;
    };
    std::unique_ptr<BiasBackwardFilter> BiasBackwardFilter::create(Context &ctx,Shape const &sp,DataType dt)
    {
        std::unique_ptr<BiasBackwardFilter> r(new BiasBackwardFilterImpl(ctx,sp,dt));
        return r;
    }

    void add_bias(Tensor &t,Tensor &bias,ExecutionContext const &e)
    {
        Context ctx(e);

        DLPRIM_CHECK(t.dtype() == float_data);
        DLPRIM_CHECK(t.shape().size() >= 2);
        DLPRIM_CHECK(t.shape()[1] == bias.shape().total_size());

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"fwd_bias");
        cl::Kernel k(prog,"fwd_bias");

        Shape const &s = t.shape();
        int B = s[0];
        int F = s[1];
        int RC = 1;
        if(s.size() >= 3)
            RC *= s[2];
        if(s.size() >= 4)
            RC *= s[3];

        int p = 0;
        k.setArg(p++,B);
        k.setArg(p++,F);
        k.setArg(p++,RC);
        t.set_arg(k,p);
        bias.set_arg(k,p);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(RC,F,B),cl::NullRange,e.events(),e.event("fwd_bias"));
    }


} // core
} // dlprim

