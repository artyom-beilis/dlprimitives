///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/pool.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>
namespace dlprim {
namespace core {

    class Pooling2DFWBDImpl {
    public:
        size_t workspace() { return 0; }
        Pooling2DFWBDImpl(Context &ctx,bool avg,int k[2],int p[2],int s[2],bool inc_pad,DataType dt) :
            scal_(ctx,dt),
            avg_(avg)
        {
            DLPRIM_CHECK(dt == float_data);
            wg_size_ = 8;
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"pooling",
                                        "WG_SIZE",wg_size_,
                                        "POOL_H",k[0],
                                        "POOL_W",k[1],
                                        "STRIDE_H",s[0],
                                        "STRIDE_W",s[1],
                                        "PAD_H",p[0],
                                        "PAD_W",p[1],
                                        "POOL_MODE",int(avg),
                                        "COUNT_INCLUDE_PAD",int(inc_pad));
            kernel_ = cl::Kernel(prog,"pooling");
            bwd_kernel_ = cl::Kernel(prog,"pooling_bw");
        }

        void forward(Tensor &in,Tensor &out,ExecutionContext const &ctx)
        {
            int bc = in.shape()[0]*in.shape()[1];

            int in_h = in.shape()[2];
            int in_w = in.shape()[3];

            int out_h = out.shape()[2];
            int out_w = out.shape()[3];

            int p=0;
            kernel_.setArg(p++,bc);
            kernel_.setArg(p++,in_h);
            kernel_.setArg(p++,in_w);
            kernel_.setArg(p++,out_h);
            kernel_.setArg(p++,out_w);
            in.set_arg(kernel_,p);
            out.set_arg(kernel_,p);

            cl::NDRange wg(wg_size_,wg_size_,1);
            cl::NDRange gr = gpu::round_range(out_h,out_w,bc,wg);

            ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("pooling"));
        }

        void backward(Tensor *x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ex)
        {
            int bc = dx.shape()[0]*dx.shape()[1];

            int in_h = dx.shape()[2];
            int in_w = dx.shape()[3];

            int out_h = dy.shape()[2];
            int out_w = dy.shape()[3];

            int p=0;

            auto ec1 = ex.generate_series_context(0,2);
            auto ec2 = ex.generate_series_context(1,2);

            scal_.enqueue(factor,dx,ec1);

            bwd_kernel_.setArg(p++,bc);
            bwd_kernel_.setArg(p++,in_h);
            bwd_kernel_.setArg(p++,in_w);
            bwd_kernel_.setArg(p++,out_h);
            bwd_kernel_.setArg(p++,out_w);
            if(!avg_) {
                DLPRIM_CHECK(x!=nullptr);
                x->set_arg(bwd_kernel_,p);
            }
            dy.set_arg(bwd_kernel_,p);
            dx.set_arg(bwd_kernel_,p);

            cl::NDRange wg(wg_size_,wg_size_,1);
            cl::NDRange gr = gpu::round_range(out_h,out_w,bc,wg);

            ex.queue().enqueueNDRangeKernel(bwd_kernel_,cl::NullRange,gr,wg,ec2.events(),ec2.event("pooling_bw"));
        }
    private:
        Scale scal_;
        bool avg_;
        int wg_size_;
        cl::Kernel kernel_;
        cl::Kernel bwd_kernel_;
    };

    class GlobalPoolingFWBWImpl  {
    public:
        GlobalPoolingFWBWImpl(Context &ctx,bool avg,Shape const &sh,DataType dt=float_data) 
        {
            avg_ = avg;
            DLPRIM_CHECK(dt == float_data);
            DLPRIM_CHECK(sh.size() == 4);
            int sm_range = sh[2]*sh[3];
            if(sm_range <= 64)
                wg_size_ = 64;
            else if(sm_range <= 128)
                wg_size_ = 128;
            else 
                wg_size_ = 256;
            items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;

            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"global_pooling",
                    "POOL_MODE",int(avg),
                    "WG_SIZE",wg_size_,
                    "ENABLE_BWD",1,
                    "ITEMS_PER_WI",items_per_wi_
                    );
            kernel_ = cl::Kernel(prog,"global_pooling");
            kernel_bwd_ = cl::Kernel(prog,"global_pooling_bwd");
            sm_range_ = sm_range;
            int mpl = wg_size_ * items_per_wi_;
            nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
        }

        void forward(Tensor &input,Tensor &output,ExecutionContext const &ctx)
        {
            Shape in_shape = input.shape();
            int over = in_shape[2] * in_shape[3];
            DLPRIM_CHECK(over == sm_range_);
            int p=0;
            kernel_.setArg(p++,int(in_shape[0]*in_shape[1]));
            kernel_.setArg(p++,sm_range_);
            kernel_.setArg(p++,float(1.0f / (in_shape[2]*in_shape[3])));
            input.set_arg(kernel_,p);
            output.set_arg(kernel_,p);

            cl::NDRange gr(in_shape[0]*in_shape[1],nd_range_);
            cl::NDRange wg(1,wg_size_);
            ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("global_pooling"));
        }
        void backward(Tensor *x,Tensor &dx,Tensor &dy, float factor,ExecutionContext const &ctx)
        {
            Shape in_shape = dx.shape();
            int over = in_shape[2] * in_shape[3];
            DLPRIM_CHECK(over == sm_range_);
            int p=0;
            kernel_bwd_.setArg(p++,int(in_shape[0]*in_shape[1]));
            kernel_bwd_.setArg(p++,sm_range_);
            kernel_bwd_.setArg(p++,float(1.0f / (in_shape[2]*in_shape[3])));
            if(!avg_) {
                DLPRIM_CHECK(x!=nullptr);
                x->set_arg(kernel_bwd_,p);
            }
            dx.set_arg(kernel_bwd_,p);
            dy.set_arg(kernel_bwd_,p);
            kernel_bwd_.setArg(p++,factor);

            cl::NDRange gr(in_shape[0]*in_shape[1],nd_range_);
            cl::NDRange wg(1,wg_size_);
            ctx.queue().enqueueNDRangeKernel(kernel_bwd_,cl::NullRange,gr,wg,ctx.events(),ctx.event("global_pooling_bwd"));
        }
    private:
        cl::Kernel kernel_;
        cl::Kernel kernel_bwd_;
        int wg_size_;
        int items_per_wi_;
        int sm_range_;
        int nd_range_;
        bool avg_;
    };

    template<typename Impl>
    class ForwardImpl : public Pooling2DForward, public Impl {
    public:
        using Impl::Impl;
        size_t workspace() { return 0; }
        virtual void enqueue(Tensor &X,Tensor &Y,ExecutionContext const &e)
        {
            this->forward(X,Y,e);
        }
    };

    template<typename Impl>
    class BackwardMax : public MaxPooling2DBackward, public Impl {
    public:
        using Impl::Impl;
        size_t workspace() { return 0; }
        virtual void enqueue(Tensor &X,Tensor &dX,Tensor &dY,float factor,ExecutionContext const &e)
        {
            this->backward(&X,dX,dY,factor,e);
        }
    };

    template<typename Impl>
    class BackwardAvg : public AvgPooling2DBackward, public Impl {
    public:
        using Impl::Impl;
        size_t workspace() { return 0; }
        virtual void enqueue(Tensor &dX,Tensor &dY,float factor,ExecutionContext const &e)
        {
            this->backward(nullptr,dX,dY,factor,e);
        }
    };

    std::unique_ptr<Pooling2DForward> Pooling2DForward::create_max_pooling(Context &ctx,int k[2],int p[2],int s[2],DataType dt)
    {
        std::unique_ptr<Pooling2DForward> r(new ForwardImpl<Pooling2DFWBDImpl>(ctx,false,k,p,s,false,dt));
        return r;
    }
    std::unique_ptr<Pooling2DForward> Pooling2DForward::create_avg_pooling(Context &ctx,int k[2],int p[2],int s[2],bool cip,DataType dt)
    {
        std::unique_ptr<Pooling2DForward> r(new ForwardImpl<Pooling2DFWBDImpl>(ctx,true,k,p,s,cip,dt));
        return r;
    }
   
    std::unique_ptr<Pooling2DForward> Pooling2DForward::create_global_max_pooling(Context &ctx,Shape const &in_shape,DataType dt)
    {
        std::unique_ptr<Pooling2DForward> r(new ForwardImpl<GlobalPoolingFWBWImpl>(ctx,false,in_shape,dt));
        return r;
    }
    std::unique_ptr<Pooling2DForward> Pooling2DForward::create_global_avg_pooling(Context &ctx,Shape const &in_shape,DataType dt)
    {
        std::unique_ptr<Pooling2DForward> r(new ForwardImpl<GlobalPoolingFWBWImpl>(ctx,true,in_shape,dt));
        return r;
    }

    std::unique_ptr<MaxPooling2DBackward>  MaxPooling2DBackward::create(Context &ctx,int k[2],int p[2],int s[2],DataType dt)
    {
        std::unique_ptr<MaxPooling2DBackward> r(new BackwardMax<Pooling2DFWBDImpl>(ctx,false,k,p,s,false,dt));
        return r;
    }

    std::unique_ptr<AvgPooling2DBackward>  AvgPooling2DBackward::create(Context &ctx,int k[2],int p[2],int s[2],bool cip,DataType dt)
    {
        std::unique_ptr<AvgPooling2DBackward> r(new BackwardAvg<Pooling2DFWBDImpl>(ctx,true,k,p,s,cip,dt));
        return r;
    }

    std::unique_ptr<MaxPooling2DBackward>  MaxPooling2DBackward::create_global(Context &ctx,Shape const &s,DataType dt)
    {
        std::unique_ptr<MaxPooling2DBackward> r(new BackwardMax<GlobalPoolingFWBWImpl>(ctx,false,s,dt));
        return r;
    }

    std::unique_ptr<AvgPooling2DBackward>  AvgPooling2DBackward::create_global(Context &ctx,Shape const &s,DataType dt)
    {
        std::unique_ptr<AvgPooling2DBackward> r(new BackwardAvg<GlobalPoolingFWBWImpl>(ctx,true,s,dt));
        return r;
    }


} // core
} // dlprim

