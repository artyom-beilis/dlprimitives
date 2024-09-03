///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/conv.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    Shape Conv2DBase::get_output_shape(Convolution2DConfigBase const &config,Shape const &in)
    {
        DLPRIM_CHECK(in.size() == 4);
        int batch = in[0];
        DLPRIM_CHECK(int(in[1]) == config.channels_in);
        int ihw[2] = { int(in[2]), int(in[3]) };
        int ohw[2];
        for(int i=0;i<2;i++)        
            ohw[i] = (ihw[i] + 2 * config.pad[i] - config.dilate[i] * (config.kernel[i] - 1) - 1) /  config.stride[i] + 1;
        DLPRIM_CHECK(ohw[0] > 0);
        DLPRIM_CHECK(ohw[1] > 0);
        return Shape(batch,config.channels_out,ohw[0],ohw[1]);
    }

    Shape Conv2DBase::get_output_shape_transposed(Convolution2DConfigBase const &config,Shape const &in,int output_pad[2])
    {
        DLPRIM_CHECK(in.size() == 4);
        int batch = in[0];
        DLPRIM_CHECK(int(in[1]) == config.channels_in);
        int ihw[2] = { int(in[2]), int(in[3]) };
        int ohw[2];
        for(int i=0;i<2;i++)        
            ohw[i] = (ihw[i] - 1) * config.stride[i] - 2 * config.pad[i] + config.dilate[i] * (config.kernel[i] - 1) + output_pad[i] + 1;
        DLPRIM_CHECK(ohw[0] > 0);
        DLPRIM_CHECK(ohw[1] > 0);
        return Shape(batch,config.channels_out,ohw[0],ohw[1]);
    }

    class Conv2DForwardGEMM : public Conv2DForward {
    public:
        virtual char const *algo() const
        {
            return "gemm";
        }
        virtual void enqueue(Tensor &x,Tensor &W,Tensor *bias,Tensor &y, Tensor &,float factor,ExecutionContext const &e) 
        {
            cl::Buffer *bias_buffer = nullptr;
            int bias_offset = 0;
            if(bias) {
                bias_buffer = &bias->device_buffer();
                bias_offset = bias->device_offset();
            }
            int batch = x.shape()[0];

            int M = config_.channels_out / config_.groups;
            int N = y.shape()[2]*y.shape()[3];
            int K = im2col_width_;
            
           
            gemm_->gemm(M,N*batch,K,
                W.device_buffer(),
                W.device_offset(),
                K,
                x.device_buffer(),
                x.device_offset(),
                K,
                y.device_buffer(),
                y.device_offset(),
                N,
                bias_buffer,
                bias_offset,
                factor,
                y.shape().total_size(),
                e);
        }

        Conv2DForwardGEMM(Context &ctx,Conv2DSettings const &config,bool bias,StandardActivations activation = StandardActivations::identity) :
            config_(config)
        {
            im2col_width_ = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1]; 
            Shape const &in = config_.shape;
            Shape out = get_output_shape(config_,in);
            int out_h = out[2];
            int out_w = out[3];
            int M = config_.channels_out / config_.groups;
            int N = out_h*out_w * out[0];
            int K = im2col_width_;

            auto gemm = gpu::GEMM::get_optimal_conv_gemm(
                ctx,config_.dtype,GemmOpMode::forward,
                false,true,
                M,N,K,
                config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                (bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
                activation,
                out_h * out_w
            );
            gemm_ = std::move(gemm);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
        Conv2DSettings config_;
        int im2col_width_;
    };
    
    class Conv2DBackwardDataGEMM : public Conv2DBackwardData {
    public:
        Conv2DBackwardDataGEMM(Context &ctx,Conv2DSettings const &config) :
            config_(config)
        {
            int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
            Shape const &in = config_.shape;
            Shape out = get_output_shape(config_,config_.shape);
            int im2col_rows = out[2]*out[3]*out[0];

            auto bwd = gpu::GEMM::get_optimal_conv_gemm(
                    ctx,config_.dtype,
                    GemmOpMode::backward_data,
                    true,false,
                    im2col_rows,kernel_cols,config_.channels_out / config_.groups,
                    config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                    config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                    gpu::GEMM::no_bias,
                    StandardActivations::identity,
                    out[2] * out[3]
                );


            gemm_ = std::move(bwd);
        }
        virtual char const *algo() const { return "gemm"; }
        virtual void enqueue(Tensor &dx,Tensor &K,Tensor &dy,Tensor &,float factor,ExecutionContext const &e)
        {
            int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
            int im2col_rows = dy.shape()[2]*dy.shape()[3]*dy.shape()[0];
            gemm_->gemm(
                im2col_rows,
                kernel_cols,
                config_.channels_out / config_.groups,
                dy.device_buffer(),
                dy.device_offset(),
                im2col_rows,
                K.device_buffer(),
                K.device_offset(),
                kernel_cols,
                dx.device_buffer(),
                dx.device_offset(),
                kernel_cols,
                nullptr,  // no bias for BW
                0,
                factor,
                dx.shape().total_size(),
                e);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
        Conv2DSettings config_;
    };

    class Conv2DBackwardFilterGEMM: public Conv2DBackwardFilter  {
    public:
        Conv2DBackwardFilterGEMM(Context &ctx,Conv2DSettings const &config) : config_(config) 
        {
            Shape const &in = config_.shape;
            Shape out = get_output_shape(config_,config_.shape);

            int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
            int im2col_rows = out[2]*out[3]*out[0];

            auto bwd = gpu::GEMM::get_optimal_conv_gemm(
                    ctx,config_.dtype,
                    GemmOpMode::backward_filter,
                    false,false,
                    config_.channels_out / config_.groups,kernel_cols,im2col_rows,
                    config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                    config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                    gpu::GEMM::no_bias,
                    StandardActivations::identity,
                    out[2]* out[3]
                );

            gemm_ = std::move(bwd);
        }

        virtual char const *algo() const { return "gemm"; }
        virtual void enqueue(Tensor &x,Tensor &dK,Tensor &dy,Tensor &,float factor,ExecutionContext const &e) 
        {
            int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
            int im2col_rows = dy.shape()[2]*dy.shape()[3]*dy.shape()[0];
            gemm_->gemm(
                config_.channels_out / config_.groups,  // M
                kernel_cols,                            // N
                im2col_rows,                            // K
                dy.device_buffer(),
                dy.device_offset(),
                im2col_rows,
                x.device_buffer(),
                x.device_offset(),
                kernel_cols,
                dK.device_buffer(),
                dK.device_offset(),
                kernel_cols,
                nullptr,  // no bias for BW
                0,
                factor,
                dK.shape().total_size(),
                e);
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
        Conv2DSettings config_;
        
    };


    class Conv2DForwardWinograd : public Conv2DForward {
    public:
        virtual char const *algo() const
        {
            return "winograd";
        }
        virtual size_t workspace()
        {
            return sizeof(float)*16 * config_.channels_in * config_.channels_out;
        }
        virtual void enqueue(Tensor &in,Tensor &W,Tensor *bias,Tensor &out, Tensor &ws,float factor,ExecutionContext const &ec)
        {
            int B = in.shape()[0];
            int N = config_.channels_out;
            int C = in.shape()[1];
            int h = in.shape()[2];
            int w = in.shape()[3];

            int p=0;

            Tensor float_ws = ws.workspace_as_type(float_data);
            conv_kernel_.setArg(p++,config_.channels_out);
            conv_kernel_.setArg(p++,config_.channels_in);
            W.set_arg(conv_kernel_,p);
            float_ws.set_arg(conv_kernel_,p);

            p=0;
            conv_.setArg(p++,B);
            conv_.setArg(p++,N);
            conv_.setArg(p++,C);
            conv_.setArg(p++,h);
            conv_.setArg(p++,w);

            in.set_arg(conv_,p);
            float_ws.set_arg(conv_,p);
            if(bias) {
                bias->set_arg(conv_,p);
            }
            out.set_arg(conv_,p);
            conv_.setArg(p++,factor);

            auto ec1 = ec.generate_series_context(0,2);
            auto ec2 = ec.generate_series_context(1,2);

            cl::NDRange l1(8,8);
            cl::NDRange g1 = gpu::round_range(config_.channels_out,config_.channels_in,l1);
            ec.queue().enqueueNDRangeKernel(conv_kernel_,cl::NullRange,g1,l1,ec.events(),ec1.event("winograd_3to4_kernel"));
            cl::NDRange l2(256,1);
            int tiles = ((w + 1) / 2 * (h + 1) / 2 * B + 31)/32;
            cl::NDRange g2(tiles * 256,(N + 31) / 32);
            ec.queue().enqueueNDRangeKernel(conv_,cl::NullRange,g2,l2,ec.events(),ec2.event("winograd_3x3_main"));
        }

        Conv2DForwardWinograd(Context &ctx,Conv2DSettings const &config,bool bias,StandardActivations activation = StandardActivations::identity) :
            config_(config)
        {
            int off = ctx.is_amd() ? 0 : 1;
            int toff = 1;
            int local_mem_size = ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
            if(local_mem_size < 40960) {
                off = 0;
                toff = 0;
            }
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"winograd_fwd",
                                            "ACTIVATION",int(activation),
                                            "STRIDE_OFFSET",off,
                                            "TR_STRIDE_OFFSET",toff,
                                            "BIAS",int(bias));
            conv_kernel_ = cl::Kernel(prog,"winconv_calc_gkgt_3x3");
            conv_ = cl::Kernel(prog,"winconv_3x3");
        }
    private:
        cl::Kernel conv_,conv_kernel_;
        Conv2DSettings config_;
    };

    class Conv2DBackwardDataWinograd : public Conv2DBackwardData {
    public:
        virtual char const *algo() const
        {
            return "winograd";
        }
        virtual size_t workspace()
        {
            return sizeof(float)*16 * config_.channels_in * config_.channels_out;
        }
        Conv2DBackwardDataWinograd(Context &ctx,Conv2DSettings const &config) :
            config_(config),
            s_(ctx,config.dtype)
        {
            int off = ctx.is_amd() ? 0 : 1;
            int toff = 1;
            int local_mem_size = ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
            if(local_mem_size < 40960) {
                off = 0;
                toff = 0;
            }
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"winograd_bwd_data",
                        "STRIDE_OFFSET",off,"TR_STRIDE_OFFSET",toff);
            conv_kernel_bwd_ = cl::Kernel(prog,"winconv_calc_gkgt_3x3");
            bw_conv_data_ = cl::Kernel(prog,"winconv_3x3_bwd_data");
        }
        virtual void enqueue(Tensor &dx,Tensor &K,Tensor &dy,Tensor &ws_bytes,float factor,ExecutionContext const &ec)
        {
            auto ec1 = ec.generate_series_context(0,3);
            auto ec2 = ec.generate_series_context(1,3);
            auto ec3 = ec.generate_series_context(2,3);

            int B = dx.shape()[0];
            int N = config_.channels_out;
            int C = dx.shape()[1];
            int h = dx.shape()[2];
            int w = dx.shape()[3];

            int p=0;
            Tensor ws = ws_bytes.workspace_as_type(float_data);

            conv_kernel_bwd_.setArg(p++,config_.channels_out);
            conv_kernel_bwd_.setArg(p++,config_.channels_in);
            K.set_arg(conv_kernel_bwd_,p);
            ws.set_arg(conv_kernel_bwd_,p);
            
            p=0;
            bw_conv_data_.setArg(p++,B);
            bw_conv_data_.setArg(p++,N);
            bw_conv_data_.setArg(p++,C);
            bw_conv_data_.setArg(p++,h);
            bw_conv_data_.setArg(p++,w);

            dx.set_arg(bw_conv_data_,p);
            ws.set_arg(bw_conv_data_,p);
            dy.set_arg(bw_conv_data_,p);
            
            s_.enqueue(factor,dx,ec1);

            cl::NDRange l1(8,8);
            cl::NDRange g1 = gpu::round_range(config_.channels_out,config_.channels_in,l1);
            ec.queue().enqueueNDRangeKernel(conv_kernel_bwd_,cl::NullRange,g1,l1,ec2.events(),ec2.event("winograd_3to4_kernel"));

            cl::NDRange l2(256,1);
            int tiles = ((w + 1) / 2 * (h + 1) / 2 * B + 31)/32;
            cl::NDRange g2(tiles * 256,(C + 31) / 32);
            ec.queue().enqueueNDRangeKernel(bw_conv_data_,cl::NullRange,g2,l2,ec.events(),ec2.event("winograd_3x3_main_bwd"));
        }
    private:
        Conv2DSettings config_;
        Scale s_;
        cl::Kernel conv_kernel_bwd_, bw_conv_data_;
    };

    class Conv2DBackwardFilterWinograd: public Conv2DBackwardFilter  {
    public:
        virtual char const *algo() const
        {
            return "winograd";
        }

        Conv2DBackwardFilterWinograd(Context &ctx,Conv2DSettings const &config) : 
            config_(config),
            s_(ctx,config.dtype)
        {
            int h = config.shape[2];
            int w = config.shape[3];
            int winograd_work_items = (config_.channels_in / 32) * (config_.channels_out / 32) * 256;
            reduce_k_ = winograd_work_items < ctx.estimated_core_count();
            int off = ctx.is_amd() ? 0 : 1;
            int toff = 1;
            int local_mem_size = ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
            if(local_mem_size < 40960) {
                off = 0;
                toff = 0;
            }
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"winograd_bwd_filter",
                                                                            "IMG_H",h,"IMG_W",w,
                                                                            "STRIDE_OFFSET",off,
                                                                            "TR_STRIDE_OFFSET",toff);
            bw_conv_filter_ = cl::Kernel(prog,"winconv_3x3_bwd_filter");
        }
        virtual void enqueue(Tensor &x,Tensor &dK,Tensor &dy,Tensor &,float factor,ExecutionContext const &ec) 
        {
            int B = x.shape()[0];
            int N = config_.channels_out;
            int C = config_.channels_in;
            int p=0;
            bw_conv_filter_.setArg(p++,B);
            bw_conv_filter_.setArg(p++,N);
            bw_conv_filter_.setArg(p++,C);
            x.set_arg(bw_conv_filter_,p);
            dK.set_arg(bw_conv_filter_,p);
            dy.set_arg(bw_conv_filter_,p);
            bw_conv_filter_.setArg(p++,factor);
            
            cl::NDRange wg(256,1,1);
            int g1 = 256 * ((C+31)/32);
            int g2 = (N+31)/32;
            cl::NDRange gr(g1,g2,reduce_k_ ? 8 : 1);
            if(reduce_k_) {
                ExecutionContext ec1 = ec.generate_series_context(0,2);
                ExecutionContext ec2 = ec.generate_series_context(1,2);
                s_.enqueue(factor,dK,ec1);
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec2.events(),ec2.event("winograd_bwd_filter"));
            }
            else {
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec.events(),ec.event("winograd_bwd_filter"));
            }
        }
    private:
        Conv2DSettings config_;
        Scale s_;
        cl::Kernel bw_conv_filter_;
        bool reduce_k_;
    };

    struct DepthwiseSeparableBase {
        static int get_opt_val(int x)
        {
            if(x <= 2)
                return 1;
            if(x <= 4)
                return 2;
            if(x <= 8)
                return 4;
            if(x <= 16)
                return 8;
            return 16;
        }
        constexpr static int ds_patch_rows = 2;
        constexpr static int ds_patch_cols = 2;
    };

    constexpr int DepthwiseSeparableBase::ds_patch_rows;
    constexpr int DepthwiseSeparableBase::ds_patch_cols;


    class Conv2DForwardDepthwiseSeparable : public Conv2DForward, public DepthwiseSeparableBase {
    public:
        virtual char const *algo() const
        {
            return "depthwise_separable";
        }
        virtual void enqueue(Tensor &in,Tensor &W,Tensor *bias,Tensor &out, Tensor &,float factor,ExecutionContext const &ec)
        {
            int batch = in.shape()[0];
            int height = in.shape()[2];
            int width = in.shape()[3];
            int p=0;
            conv_.setArg(p++,batch);
            conv_.setArg(p++,height);
            conv_.setArg(p++,width);
            in.set_arg(conv_,p);
            W.set_arg(conv_,p);
            if(bias) {
                bias->set_arg(conv_,p);
            }
            out.set_arg(conv_,p);
            conv_.setArg(p++,factor);
            
            int gW = (width+1)/2;
            int gH = (height+1)/2;

            int lW = get_opt_val(gW);
            int lH = get_opt_val(gH);
            int lD = 1;
            if(lW * lH < 64)
                lD = 64 / (lW * lH);
            
            cl::NDRange wg(lH,lW,lD);
            cl::NDRange gr=gpu::round_range(gH,gW,batch*config_.channels_in,wg);
                        
            ec.queue().enqueueNDRangeKernel(conv_,cl::NullRange,gr,wg,ec.events(),ec.event("sep_conv"));
        }

        Conv2DForwardDepthwiseSeparable(Context &ctx,Conv2DSettings const &config,bool bias,StandardActivations activation = StandardActivations::identity) :
            config_(config)
        {
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"depthwise_separable_conv",
                                    "ACTIVATION",int(activation),
                                    "BIAS",int(bias),
                                    "PATCH_ROWS",ds_patch_rows,
                                    "PATCH_COLS",ds_patch_cols,
                                    "KERN",config_.kernel[0],
                                    "CHANNELS",config_.channels_in);
            conv_ = cl::Kernel(prog,"conv");
        }
    private:
        Conv2DSettings config_;
        cl::Kernel conv_;
    };
    
    class Conv2DBackwardDataDepthwiseSeparable  : public Conv2DBackwardData, public DepthwiseSeparableBase {
    public:
        virtual char const *algo() const
        {
            return "depthwise_separable";
        }
        virtual void enqueue(Tensor &dx,Tensor &K,Tensor &dy,Tensor &,float factor,ExecutionContext const &ec)
        {
            int total = 1;
            if(factor != 1.0f ) {
                total++;
                ExecutionContext ec1 = ec.generate_series_context(0,2);
                s_.enqueue(factor,dx,ec1);
            }
            ExecutionContext ec2 = ec.generate_series_context(1,total);
            int batch = dx.shape()[0];
            int height = dx.shape()[2];
            int width = dx.shape()[3];
            int p=0;
            bw_conv_data_.setArg(p++,batch);
            bw_conv_data_.setArg(p++,height);
            bw_conv_data_.setArg(p++,width);
            dx.set_arg(bw_conv_data_,p);
            K.set_arg(bw_conv_data_,p);
            dy.set_arg(bw_conv_data_,p);
            
            int gW = (width+1)/2;
            int gH = (height+1)/2;

            int lW = get_opt_val(gW);
            int lH = get_opt_val(gH);
            int lD = 1;
            if(lW * lH < 64)
                lD = 64 / (lW * lH);
            
            cl::NDRange wg(lH,lW,lD);
            cl::NDRange gr=gpu::round_range(gH,gW,batch*config_.channels_in,wg);
            
            ec.queue().enqueueNDRangeKernel(bw_conv_data_,cl::NullRange,gr,wg,ec2.events(),ec2.event("sep_conv_bw_data"));
        }

        Conv2DBackwardDataDepthwiseSeparable(Context &ctx,Conv2DSettings const &config) :
            config_(config),
            s_(ctx,config.dtype)
        {
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"depthwise_separable_conv",
                                    "PATCH_ROWS",ds_patch_rows,
                                    "PATCH_COLS",ds_patch_cols,
                                    "KERN",config_.kernel[0],
                                    "CHANNELS",config_.channels_in);
            bw_conv_data_ = cl::Kernel(prog,"backward_data_conv");
        }
    private:
        Conv2DSettings config_;
        Scale s_;
        cl::Kernel bw_conv_data_;
    };

    class Conv2DBackwardFilterDepthwiseSeparable: public Conv2DBackwardFilter  {
    public:
        virtual char const *algo() const
        {
            return "depthwise_separable";
        }

        virtual size_t workspace()
        {
            return config_.kernel[0] * config_.kernel[1] * config_.channels_in * second_reduce_ * sizeof(float);
        }

        Conv2DBackwardFilterDepthwiseSeparable(Context &ctx,Conv2DSettings const &config) : 
            config_(config)
        {
            int total = config.shape[0] * config.shape[2] * config.shape[3];
            int second_size = (total + 255) / 256;
            if(second_size < 64) {
                if(total >= 256)
                    dwsc_bw_filter_wg_ = 256;
                else if(total >= 128)
                    dwsc_bw_filter_wg_ = 128;
                else
                    dwsc_bw_filter_wg_ = 64;
                second_reduce_ = 1;
            }
            else {
                dwsc_bw_filter_wg_ = 256;
                if(second_size >= 256)
                    second_reduce_ = 256;
                else if(second_size >= 128)
                    second_reduce_ = 128;
                else
                    second_reduce_ = 64;
                
            }
            
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,
                        "depthwise_separable_bw_filter",
                        "KERN",config_.kernel[0],
                        "WG_SIZE",dwsc_bw_filter_wg_,
                        "SECOND_REDUCE_SIZE",second_reduce_,
                        "CHANNELS",config_.channels_in);

            bw_conv_filter_ = cl::Kernel(prog,"conv_bw_filter");
            if(second_reduce_ > 1)
                reduce_ = cl::Kernel(prog,"reduce");
        }
        virtual void enqueue(Tensor &x,Tensor &dK,Tensor &dy,Tensor &ws_bytes,float factor,ExecutionContext const &ec) 
        {
            int kitems = dK.shape().total_size();
            int batch = x.shape()[0];
            int height = x.shape()[2];
            int width = x.shape()[3];
            int p=0;
            Tensor ws = ws_bytes.workspace_as_type(float_data);
            bw_conv_filter_.setArg(p++,batch);
            bw_conv_filter_.setArg(p++,height);
            bw_conv_filter_.setArg(p++,width);
            x.set_arg(bw_conv_filter_,p);
            if(second_reduce_ == 1) {
                dK.set_arg(bw_conv_filter_,p);
            }
            else {
                ws.set_arg(bw_conv_filter_,p);
            }
            dy.set_arg(bw_conv_filter_,p);
            if(second_reduce_ == 1)
               bw_conv_filter_.setArg(p++,factor);
            
            cl::NDRange wg(dwsc_bw_filter_wg_,1);
            cl::NDRange gr(dwsc_bw_filter_wg_ * second_reduce_,kitems);

            if(second_reduce_ == 1) {
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec.events(),ec.event("sep_conv_bw_filter"));
            }
            else  {
                auto ec1 = ec.generate_series_context(0,2);
                auto ec2 = ec.generate_series_context(1,2);
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec1.events(),ec1.event("sep_conv_bw_filter"));
                p=0;
                int reduce_items = dK.shape().total_size();
                ws.set_arg(reduce_,p);
                dK.set_arg(reduce_,p);
                reduce_.setArg(p++,factor);
                ec.queue().enqueueNDRangeKernel(reduce_,
                            cl::NullRange,
                            cl::NDRange(second_reduce_,reduce_items),
                            cl::NDRange(second_reduce_,1),
                            ec2.events(),ec2.event("sep_conv_bw_filter_reduce"));

            }
        }
    private:
        Conv2DSettings config_;
        cl::Kernel bw_conv_filter_;
        cl::Kernel reduce_;
        int dwsc_bw_filter_wg_;
        int second_reduce_;
    };


    static bool is_depthwise_separable_compatible(Conv2DSettings const &config)
    {
        return 
            config.kernel[0] == config.kernel[1] 
            && config.pad[0] == config.pad[1] 
            && config.dilate[0] == config.dilate[1]
            && config.stride[0] == config.stride[1]
            && config.groups == config.channels_in
            && config.channels_in > 1
            && config.dilate[0] == 1
            && config.stride[0] == 1
            && config.kernel[0] % 2 == 1
            && config.kernel[0] / 2 == config.pad[0];
    }
    static bool is_winograd_compatible(Context &ctx,Conv2DSettings const &config)
    {
        if(!(ctx.is_amd() || ctx.is_nvidia() || ctx.is_intel()))
            return false;
        if(ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() < 32768)
            return false;
        return 
            config.kernel[0] == config.kernel[1] 
            && config.pad[0] == config.pad[1] 
            && config.dilate[0] == config.dilate[1]
            && config.stride[0] == config.stride[1]
            && config.groups == 1
            && config.dilate[0] == 1
            && config.stride[0] == 1
            && config.kernel[0] == 3
            && config.pad[0] == 1;
    }

    std::unique_ptr<Conv2DForward> Conv2DForward::create(Context &ctx,
                                                         Conv2DSettings const &config,
                                                         bool bias,
                                                         StandardActivations activation,
                                                         std::string const &algo)
    {
        bool win = is_winograd_compatible(ctx,config);
        bool dsc = is_depthwise_separable_compatible(config);

        bool win_recommended = win && config.channels_in >= 8 && config.channels_out >= 8;
        bool dsc_recommended = dsc;

        bool auto_algo = algo == "auto" || algo == "";
        
        std::unique_ptr<Conv2DForward> r;
        if((algo == "winograd" && win) || (auto_algo && win_recommended)) {
            r.reset(new Conv2DForwardWinograd(ctx,config,bias,activation));
        }
        else if((algo == "depthwise_separable" && dsc) || (auto_algo && dsc_recommended)) {
            r.reset(new Conv2DForwardDepthwiseSeparable(ctx,config,bias,activation));
        }
        else { // substitude and default
            r.reset(new Conv2DForwardGEMM(ctx,config,bias,activation));
        }
        return r;
    }
    std::unique_ptr<Conv2DBackwardData> Conv2DBackwardData::create(Context &ctx,Conv2DSettings const &config,std::string const &algo)
    {
        bool win = is_winograd_compatible(ctx,config);
        bool dsc = is_depthwise_separable_compatible(config);

        bool win_recommended = win && config.channels_in >= 8 && config.channels_out >= 8;
        bool dsc_recommended = dsc;

        bool auto_algo = algo == "auto" || algo == "";
        
        std::unique_ptr<Conv2DBackwardData> r;
        if((algo == "winograd" && win) || (auto_algo && win_recommended)) {
            r.reset(new Conv2DBackwardDataWinograd(ctx,config));
        }
        else if((algo == "depthwise_separable" && dsc) || (auto_algo && dsc_recommended)) {
            r.reset(new Conv2DBackwardDataDepthwiseSeparable(ctx,config));
        }
        else { // substitude and default
            r.reset(new Conv2DBackwardDataGEMM(ctx,config));
        }
        return r;
    }
    std::unique_ptr<Conv2DBackwardFilter> Conv2DBackwardFilter::create(Context &ctx,Conv2DSettings const &config,std::string const &algo)
    {
        bool win = is_winograd_compatible(ctx,config); // no need to test min channels in/out since it is more optimal in any case
        bool dsc = is_depthwise_separable_compatible(config);

        bool win_recommended = win;
        bool dsc_recommended = dsc;

        bool auto_algo = algo == "auto" || algo == "";
        
        std::unique_ptr<Conv2DBackwardFilter> r;
        if((algo == "winograd" && win) || (auto_algo && win_recommended)) {
            r.reset(new Conv2DBackwardFilterWinograd(ctx,config));
        }
        else if((algo == "depthwise_separable" && dsc) || (auto_algo && dsc_recommended)) {
            r.reset(new Conv2DBackwardFilterDepthwiseSeparable(ctx,config));
        }
        else { // substitude and default
            r.reset(new Conv2DBackwardFilterGEMM(ctx,config));
        }
        return r;
    }



} // core
} // dlprim

