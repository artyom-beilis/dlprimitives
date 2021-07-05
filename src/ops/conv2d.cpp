#include <dlprim/ops/conv2d.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/ops/activation.hpp>
#include <my_cblas.hpp>

namespace dlprim {
   

    Convolution2DConfig Convolution2DConfig::from_json(json::value const &v)
    {
        Convolution2DConfig cfg;
        cfg.channels_in = v.get("channels_in",cfg.channels_in);
        cfg.channels_out = v.get<int>("channels_out");
        utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
        utils::get_1dNd_from_json(v,"stride",cfg.stride);
        utils::get_1dNd_from_json(v,"dilate",cfg.dilate);
        utils::get_1dNd_from_json(v,"pad",cfg.pad);
        cfg.groups = v.get("groups",cfg.groups);
        cfg.bias = v.get("bias",cfg.bias);
        cfg.activation = utils::activation_from_json(v); 
        return cfg;
    }
    

    Shape Convolution2D::get_output_shape(Shape const &in)
    {
        DLPRIM_CHECK(in.size() == 4);
        int batch = in[0];
        DLPRIM_CHECK(int(in[1]) == config_.channels_in);
        int ihw[2] = { int(in[2]), int(in[3]) };
        int ohw[2];
        for(int i=0;i<2;i++)        
            ohw[i] = (ihw[i] + 2 * config_.pad[i] - config_.dilate[i] * (config_.kernel[i] - 1) - 1) /  config_.stride[i] + 1;
        DLPRIM_CHECK(ohw[0] > 0);
        DLPRIM_CHECK(ohw[1] > 0);
        return Shape(batch,config_.channels_out,ohw[0],ohw[1]);
    }

    int Convolution2D::get_im2col_width()
    {
        return config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
    }


    Convolution2D::Convolution2D(Context &ctx,Convolution2DConfig const &cfg) :
        Operator(ctx),
        config_(cfg),
        dtype_(float_data)
    {
        DLPRIM_CHECK(config_.channels_out > 0);
        DLPRIM_CHECK(dtype_==float_data);
        DLPRIM_CHECK(config_.channels_in  % config_.groups == 0);
        DLPRIM_CHECK(config_.channels_out % config_.groups == 0);
        out_h_ = out_w_ = -1;
        in_h_ = in_w_ = -1;
    }
    
    Convolution2D::~Convolution2D()
    {
    }

    bool Convolution2D::is_depthwise_separable_conv()
    {
        return 
            config_.kernel[0] == config_.kernel[1] 
            && config_.pad[0] == config_.pad[1] 
            && config_.dilate[0] == config_.dilate[1]
            && config_.stride[0] == config_.stride[1]
            && config_.groups == config_.channels_in
            && config_.channels_in > 1
            && config_.dilate[0] == 1
            && config_.stride[0] == 1
            && config_.kernel[0] % 2 == 1
            && config_.kernel[0] / 2 == config_.pad[0];
    }

    size_t Convolution2D::setup_winograd_conv(int h,int w)
    {
        if(ctx_.is_cpu_context())
            return 0;
        {
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"winograd_fwd",
                                            "ACTIVATION",int(config_.activation),
                                            "BIAS",int(config_.bias));
            conv_kernel_ = cl::Kernel(prog,"winconv_calc_gkgt_3x3");
            conv_ = cl::Kernel(prog,"winconv_3x3");
        }
        {
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"winograd_bwd_data");
            conv_kernel_bwd_ = cl::Kernel(prog,"winconv_calc_gkgt_3x3");
            bw_conv_data_ = cl::Kernel(prog,"winconv_3x3_bwd_data");
        }
        {
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"winograd_bwd_filter",
                                                                            "IMG_H",h,"IMG_W",w);
            bw_conv_filter_ = cl::Kernel(prog,"winconv_3x3_bwd_filter");
        }
        size_t res = sizeof(float)*16 * config_.channels_in * config_.channels_out;
        return res;
    }

    bool Convolution2D::is_winograd_candidate()
    {
        if(!ctx_.is_amd() && !ctx_.is_nvidia())
            return false;
        return 
            config_.channels_in >= 8
            && config_.channels_out >= 8
            && config_.kernel[0] == config_.kernel[1] 
            && config_.pad[0] == config_.pad[1] 
            && config_.dilate[0] == config_.dilate[1]
            && config_.stride[0] == config_.stride[1]
            && config_.groups == 1
            && config_.dilate[0] == 1
            && config_.stride[0] == 1
            && config_.kernel[0] == 3
            && config_.pad[0] == 1;
    }

    void Convolution2D::setup_depthwise_separable_conv(Shape const &s)
    {
        if(ctx_.is_cpu_context())
            return;
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"depthwise_separable_conv",
                                    "ACTIVATION",int(config_.activation),
                                    "BIAS",int(config_.bias),
                                    "PATCH_ROWS",ds_patch_rows,
                                    "PATCH_COLS",ds_patch_cols,
                                    "KERN",config_.kernel[0],
                                    "CHANNELS",config_.channels_in);
        conv_ = cl::Kernel(prog,"conv");
        bw_conv_data_ = cl::Kernel(prog,"backward_data_conv");

        
        int total = s[0] * s[2] * s[3];
        if(total >= 256)
            dwsc_bw_filter_wg_ = 256;
        else if(total >= 128)
            dwsc_bw_filter_wg_ = 128;
        else
            dwsc_bw_filter_wg_ = 64;
        cl::Program const &prog2 = gpu::Cache::instance().get_program(ctx_,
                    "depthwise_separable_bw_filter",
                    "KERN",config_.kernel[0],
                    "WG_SIZE",dwsc_bw_filter_wg_,
                    "CHANNELS",config_.channels_in);

        bw_conv_filter_ = cl::Kernel(prog2,"conv_bw_filter");

        use_ds_conv_=true;
    }
    
    void Convolution2D::setup(std::vector<TensorSpecs> const &in,
                              std::vector<TensorSpecs> &out,
                              std::vector<TensorSpecs> &params,
                              size_t &workspace)
    {
        DLPRIM_CHECK(in.size() == 1);
        Shape in_shape = in[0].shape();
        DLPRIM_CHECK(in_shape.size() == 4);
        int chn   = in_shape[1];
        if(config_.channels_in == -1) {
            config_.channels_in = chn;
        }
        workspace = 0;

        DLPRIM_CHECK(config_.channels_in  % config_.groups == 0);
        DLPRIM_CHECK(config_.channels_out % config_.groups == 0);

        Shape output_shape = get_output_shape(in_shape);
        out.assign({TensorSpecs(output_shape,dtype_)});

        use_ds_conv_=false;
        use_winograd_=false;

        Shape params_shape(config_.channels_out,
                           config_.channels_in / config_.groups,
                           config_.kernel[0],
                           config_.kernel[1]);
        
        params.push_back(TensorSpecs(params_shape,dtype_));
        if(config_.bias) 
            params.push_back(TensorSpecs(Shape(config_.channels_out),dtype_));

        if(mode_ == CalculationsMode::train) {
            if(config_.bias) {
                bwd_bias_.reset(new BWBias(ctx_,in[0].shape()[0],output_shape[2]*output_shape[3],dtype_));
            }
            if(config_.activation != StandardActivations::identity)
                activation_ = std::move(Activation::get_bwd_op(ctx_,config_.activation,in[0]));
        }

        if(ctx_.is_cpu_context()) {
            ws_size_ = workspace =  output_shape[2] * output_shape[3] * size_of_data_type(dtype_) * get_im2col_width();
            return;
        }
        else {
             size_t ws = 0;
             if(bwd_bias_.get()) {
                 ws = bwd_bias_->workspace(config_.channels_out);
             }
             ws_size_ = workspace = std::max(ws,workspace);
        }

        if(mode_ != CalculationsMode::predict)
                scal_.reset(new Scal(ctx_,dtype_));

        use_ds_conv_ = is_depthwise_separable_conv();
        use_winograd_ = is_winograd_candidate();
        if(use_winograd_) {
            size_t ws = setup_winograd_conv(in[0].shape()[2],in[0].shape()[3]);
            ws_size_ = workspace = std::max(ws,workspace);
        }

        get_gemm(in[0].shape(),out[0].shape());
    }
    
    void Convolution2D::get_gemm(Shape const &in,Shape const &out)
    {
        if(out_h_ == out[2] && out_w_ == out[3] && in_h_ == in[2] && out_w_ == in[3])
            return;
        out_h_ = out[2];
        out_w_ = out[3];
        in_h_ = in[2];
        in_w_ = in[3];
       
        
        if(ctx_.is_cpu_context())
            return;
        
        if(!use_ds_conv_){
            if(!use_winograd_) {
                int M = config_.channels_out / config_.groups;
                int N = out_h_*out_w_ * out[0];
                int K = get_im2col_width(); 

                auto gemm = gpu::GEMM::get_optimal_conv_gemm(
                    ctx_,dtype_,GemmOpMode::forward,
                    false,true,
                    M,N,K,
                    config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                    config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                    (config_.bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
                    config_.activation,
                    out_h_ * out_w_
                );

                gemm_ = std::move(gemm);
            }

            if(mode_ != CalculationsMode::predict) {
                int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
                int im2col_rows = out[2]*out[3]*out[0];
                auto bw = gpu::GEMM::get_optimal_conv_gemm(
                        ctx_,dtype_,
                        GemmOpMode::backward_filter,
                        false,false,
                        config_.channels_out / config_.groups,kernel_cols,im2col_rows,
                        config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                        config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                        gpu::GEMM::no_bias,
                        StandardActivations::identity,
                        out_h_ * out_w_
                    );
                bwd_weights_gemm_ = std::move(bw);

                auto bwd = gpu::GEMM::get_optimal_conv_gemm(
                        ctx_,dtype_,
                        GemmOpMode::backward_data,
                        true,false,
                        im2col_rows,kernel_cols,config_.channels_out / config_.groups,
                        config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
                        config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                        gpu::GEMM::no_bias,
                        StandardActivations::identity,
                        out_h_ * out_w_
                    );
                bwd_data_gemm_ = std::move(bwd);
            }
        }
        else {
            setup_depthwise_separable_conv(in);
        }
    }

    void Convolution2D::reshape(std::vector<Shape> const &in,
                               std::vector<Shape> &out)
    {
        DLPRIM_CHECK(in.size() == 1);
        out.assign({get_output_shape(in[0])});
        if(activation_) {
            std::vector<Shape> tmp;
            activation_->reshape(out,tmp);
        }
        if(bwd_bias_) {
            int rc = out[0][2]*out[0][3];
            int b = out[0][0];
            if(bwd_bias_->batch() < b || bwd_bias_->rows_columns() != rc) {
                bwd_bias_.reset(new BWBias(ctx_,b,rc,dtype_));
            }
        }
        if(use_winograd_) {
            setup_winograd_conv(in[0][2],in[0][3]);
        }
        get_gemm(in[0],out[0]);
    }
    void Convolution2D::forward(std::vector<Tensor> &in,std::vector<Tensor> &out,std::vector<Tensor> &parameters,Tensor &ws,
            ExecutionContext const &ectx)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(out.size() == 1);
        Shape in_shape = in[0].shape();
        Shape out_shape = out[0].shape();
        DLPRIM_CHECK(out_shape == get_output_shape(in_shape));
        DLPRIM_CHECK(parameters.size()==(1u+unsigned(config_.bias)));
        DLPRIM_CHECK(parameters[0].shape() == Shape(config_.channels_out,config_.channels_in / config_.groups,config_.kernel[0],config_.kernel[1]));
        Tensor &W = parameters[0];
        Tensor *bias = nullptr;
        if(config_.bias) {
            DLPRIM_CHECK(parameters[1].shape() == Shape(config_.channels_out)); 
            bias = &parameters[1];
        }

        if(ctx_.is_cpu_context()) {
            DLPRIM_CHECK(ws.shape().total_size() > 0);
            forward_cpu(in[0],out[0],W,bias,ws.host_data());
        }
        else {
            if(use_winograd_)
                forward_winograd_gpu(in[0],out[0],W,ws,bias,ectx);
            else
                forward_gpu(in[0],out[0],W,bias,ectx);
        }
    }

    int Convolution2D::get_opt_val(int x)
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


    
    void Convolution2D::forward_winograd_gpu(Tensor &in,Tensor &out,Tensor &W,Tensor &ws,Tensor *bias,ExecutionContext const &ec)
    {
        int B = in.shape()[0];
        int N = config_.channels_out;
        int C = in.shape()[1];
        int h = in.shape()[2];
        int w = in.shape()[3];

        int p=0;
        conv_kernel_.setArg(p++,config_.channels_out);
        conv_kernel_.setArg(p++,config_.channels_in);
        conv_kernel_.setArg(p++,W.device_buffer());
        conv_kernel_.setArg(p++,int(W.device_offset()));
        conv_kernel_.setArg(p++,ws.device_buffer());
        conv_kernel_.setArg(p++,int(ws.device_offset()));

        p=0;
        conv_.setArg(p++,B);
        conv_.setArg(p++,N);
        conv_.setArg(p++,C);
        conv_.setArg(p++,h);
        conv_.setArg(p++,w);

        conv_.setArg(p++,in.device_buffer());
        conv_.setArg(p++,int(in.device_offset()));
        conv_.setArg(p++,ws.device_buffer());
        conv_.setArg(p++,int(ws.device_offset()));
        if(bias) {
            conv_.setArg(p++,bias->device_buffer());
            conv_.setArg(p++,int(bias->device_offset()));
        }
        conv_.setArg(p++,out.device_buffer());
        conv_.setArg(p++,int(out.device_offset()));
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

    void Convolution2D::forward_gpu(Tensor &in,Tensor &out,Tensor &W,Tensor *bias,ExecutionContext const &ec)
    {
        cl::Buffer *bias_buffer = nullptr;
        int bias_offset = 0;
         if(config_.bias) {
            bias_buffer = &bias->device_buffer();
            bias_offset = bias->device_offset();
        }
        int batch = in.shape()[0];
        int height = in.shape()[2];
        int width = in.shape()[3];
        
        if(use_ds_conv_) {
            int p=0;
            conv_.setArg(p++,batch);
            conv_.setArg(p++,height);
            conv_.setArg(p++,width);
            conv_.setArg(p++,in.device_buffer());
            conv_.setArg(p++,int(in.device_offset()));
            conv_.setArg(p++,W.device_buffer());
            conv_.setArg(p++,int(W.device_offset()));
            if(config_.bias) {
                conv_.setArg(p++,bias->device_buffer());
                conv_.setArg(p++,bias_offset);
            }
            conv_.setArg(p++,out.device_buffer());
            conv_.setArg(p++,int(out.device_offset()));
            
            int gW = (width+1)/2;
            int gH = (height+1)/2;

            int lW = get_opt_val(gW);
            int lH = get_opt_val(gH);
            int lD = 1;
            if(lW * lH < 64)
                lD = 64 / (lW * lH);
            
            cl::NDRange wg(lD,lH,lW);
            cl::NDRange gr=gpu::round_range(batch*config_.channels_in,gH,gW,wg);
            ec.queue().enqueueNDRangeKernel(conv_,cl::NullRange,gr,wg,ec.events(),ec.event("sep_conv"));
        }
        else {
            int M = config_.channels_out / config_.groups;
            int N = out.shape()[2]*out.shape()[3];
            int K = get_im2col_width(); 
            
           
            gemm_->gemm(M,N*batch,K,
                W.device_buffer(),
                W.device_offset(),
                K,
                in.device_buffer(),
                in.device_offset(),
                K,
                out.device_buffer(),
                out.device_offset(),
                N,
                bias_buffer,
                bias_offset,
                0.0f,
                out.shape().total_size(),
                ec);
        }
    }

    namespace details {
        
        struct Im2ColOp {
            template<typename DType>
            static void copy(DType &img,DType &im2col)
            {
                im2col = img;
            }
            template<typename DType>
            static void copy_if(DType &img,DType &im2col,bool cond)
            {
                if(cond) {
                    im2col = img;
                }
                else {
                    im2col = DType();
                }
            }
            template<typename DType>
            static void pad_zero(DType &im2col)
            {
                im2col = DType();
            }
        };
        struct Col2ImOp {
            template<typename DType>
            static void copy(DType &img,DType &im2col)
            {
                img += im2col;
            }
            template<typename DType>
            static void copy_if(DType &img,DType &im2col,bool cond)
            {
                if(cond) {
                    img += im2col;
                }
            }
            template<typename DType>
            static void pad_zero(DType &)
            {
            }
        };

        template<int K,int S,int P,typename Op,typename Float>
        void im2col_fast(Shape const &in,Shape const &outs,Float *img_in,Float *mat_in)
        {
            int rows = outs[2];
            int cols = outs[3];
            int src_rows = in[2];
            int src_cols = in[3];
            int channels_in = in[1];
            for(int chan = 0;chan < channels_in;chan ++) {
                for(int r=P;r<rows-P;r++) {
                    for(int c=P;c<cols-P;c++) {
                        int mat_row = r * cols + c;
                        int mat_col = chan * (K*K);
                        Float *mat = mat_in + mat_row * channels_in * (K*K) + mat_col;
                        int y_pos = -P + r * S; 
                        int x_pos = -P + c * S;
                        Float *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;


                        for(int dy = 0;dy < K ;dy++, img += src_cols) {
                            for(int dx=0;dx < K ;dx++) {
                                Op::copy(img[dx],*mat);
                                mat++;
                            }
                        }
                    }
                }
                if(P>0) {
                    for(int r=0;r<rows;r++) {
                        for(int c=0;c<cols;c++) {
                            if(c==P && r>=P && r<rows-P) 
                                c=cols-P;

                            int mat_row = r * cols + c;
                            int mat_col = chan * (K*K);
                            Float *mat = mat_in + mat_row * channels_in * (K*K) + mat_col;
                            int y_pos = -P + r * S;
                            int x_pos = -P + c * S;
                            Float *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;


                            for(int dy = 0;dy < K ;dy++, img += src_cols) {
                                int y = y_pos + dy;
                                if(y >= 0 && y < src_rows) {
                                    for(int dx=0;dx < K ;dx++) {
                                        int x = x_pos + dx;
                                        Op::copy_if(img[dx],*mat,(x >= 0 && x < src_cols));
                                        mat++;
                                    }
                                }
                                else {
                                    for(int dx=0;dx < K ;dx++) {
                                        Op::pad_zero(*mat);
                                        mat++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // details


    template<typename Op,typename DType>
    void Convolution2D::im2col(Shape const &in,Shape const &outs,DType *img_in,DType *mat_in)
    {
        int kern_h   = config_.kernel[0];
        int kern_w   = config_.kernel[1];
        int pad_h    = config_.pad[0];
        int pad_w    = config_.pad[1];
        int dilate_h = config_.dilate[0];
        int dilate_w = config_.dilate[1];
        int stride_h = config_.stride[0];
        int stride_w = config_.stride[1];

        if(dilate_h == 1 && dilate_h == 1 && kern_h == kern_w && pad_h == pad_w && stride_h == stride_w) {
            int k=kern_w, p = pad_w, s = stride_w;
            if(s < 10 && p < 10) {
                int combine = k*100 + s * 10 + p;
                switch(combine) {
                case 1142: details::im2col_fast<11,4,2,Op>(in,outs,img_in,mat_in); return;
                case  311: details::im2col_fast< 3,1,1,Op>(in,outs,img_in,mat_in); return;
                case  512: details::im2col_fast< 5,1,2,Op>(in,outs,img_in,mat_in); return;
                case  321: details::im2col_fast< 3,2,1,Op>(in,outs,img_in,mat_in); return;
                case  723: details::im2col_fast< 7,2,3,Op>(in,outs,img_in,mat_in); return;
                case  110: details::im2col_fast< 1,1,0,Op>(in,outs,img_in,mat_in); return;
                case  120: details::im2col_fast< 1,2,0,Op>(in,outs,img_in,mat_in); return;
                }
            }
        }

        int rows = outs[2];
        int cols = outs[3];
        int src_rows = in[2];
        int src_cols = in[3];
        int channels_in = in[1];
        for(int chan = 0;chan < channels_in;chan ++) {
            for(int r=0;r<rows;r++) {
                for(int c=0;c<cols;c++) {
                    int mat_row = r * cols + c;
                    int mat_col = chan * (kern_h * kern_w);
                    DType *mat = mat_in + mat_row * channels_in * (kern_h * kern_w) + mat_col;
                    int y_pos = -pad_h + r * stride_h;
                    int x_pos = -pad_w + c * stride_w;
                    DType *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;

                    for(int dy = 0;dy < kern_h * dilate_h ;dy+= dilate_h, img += src_cols * dilate_h) {
                        int y = y_pos + dy;
                        if(y >= 0 && y < src_rows) {
                            for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
                                int x = x_pos + dx;
                                Op::copy_if(img[dx],*mat,(x >= 0 && x < src_cols));
                                mat++;
                            }
                        }
                        else {
                            for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
                                Op::pad_zero(*mat);
                                mat++;
                            }
                        }
                    }
                }
            }
        }
    }
   
    void Convolution2D::scale_cpu(Tensor &t,float v)
    {
        size_t items = t.shape().total_size();
        float  *ptr = t.data<float>();
        if(v == 0)
            memset(ptr,0,items * sizeof(float));
        else
            cblas_sscal(items,v,ptr,1);
    }
    
    void Convolution2D::forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,void *ws)
    {
        fwd_bwd_cpu(GemmOpMode::forward,in,out,M,bias,ws);
        cpu::apply_activation(out.data<float>(),out.shape().total_size(),config_.activation);
    }
    void Convolution2D::backward_data_cpu(Tensor &dy,Tensor &K,Tensor &dx,Tensor &ws,float factor)
    {
        scale_cpu(dx,factor);
        fwd_bwd_cpu(GemmOpMode::backward_data,dx,dy,K,nullptr,ws.host_data());
    }
    void Convolution2D::backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dK,Tensor &ws,float factor)
    {
        scale_cpu(dK,factor);
        fwd_bwd_cpu(GemmOpMode::backward_filter,x,dy,dK,nullptr,ws.host_data());
    }
    void Convolution2D::backward_data_gpu(Tensor &dy,Tensor &K,Tensor &dx,Tensor &ws,float factor,ExecutionContext const &ec)
    {
        if(use_ds_conv_ || use_winograd_) {
            ExecutionContext ec1 = ec.generate_series_context(0,2 + use_winograd_);
            scal_->scale(factor,dx,ec1);
            if(use_winograd_) {
                int B = dx.shape()[0];
                int N = config_.channels_out;
                int C = dx.shape()[1];
                int h = dx.shape()[2];
                int w = dx.shape()[3];

                int p=0;
                conv_kernel_bwd_.setArg(p++,config_.channels_out);
                conv_kernel_bwd_.setArg(p++,config_.channels_in);
                conv_kernel_bwd_.setArg(p++,K.device_buffer());
                conv_kernel_bwd_.setArg(p++,int(K.device_offset()));
                conv_kernel_bwd_.setArg(p++,ws.device_buffer());
                conv_kernel_bwd_.setArg(p++,int(ws.device_offset()));

                p=0;
                bw_conv_data_.setArg(p++,B);
                bw_conv_data_.setArg(p++,N);
                bw_conv_data_.setArg(p++,C);
                bw_conv_data_.setArg(p++,h);
                bw_conv_data_.setArg(p++,w);

                bw_conv_data_.setArg(p++,dx.device_buffer());
                bw_conv_data_.setArg(p++,int(dx.device_offset()));
                bw_conv_data_.setArg(p++,ws.device_buffer());
                bw_conv_data_.setArg(p++,int(ws.device_offset()));
                bw_conv_data_.setArg(p++,dy.device_buffer());
                bw_conv_data_.setArg(p++,int(dy.device_offset()));
                auto ec2 = ec.generate_series_context(1,3);
                auto ec3 = ec.generate_series_context(2,3);

                cl::NDRange l1(8,8);
                cl::NDRange g1 = gpu::round_range(config_.channels_out,config_.channels_in,l1);
                ec.queue().enqueueNDRangeKernel(conv_kernel_bwd_,cl::NullRange,g1,l1,ec2.events(),ec2.event("winograd_3to4_kernel"));
                cl::NDRange l2(256,1);
                int tiles = ((w + 1) / 2 * (h + 1) / 2 * B + 31)/32;
                cl::NDRange g2(tiles * 256,(C + 31) / 32);
                ec.queue().enqueueNDRangeKernel(bw_conv_data_,cl::NullRange,g2,l2,ec.events(),ec2.event("winograd_3x3_main_bwd"));
            }
            else { // use_ds_conv_
                ExecutionContext ec2 = ec.generate_series_context(1,2);
                int batch = dx.shape()[0];
                int height = dx.shape()[2];
                int width = dx.shape()[3];
                int p=0;
                bw_conv_data_.setArg(p++,batch);
                bw_conv_data_.setArg(p++,height);
                bw_conv_data_.setArg(p++,width);
                bw_conv_data_.setArg(p++,dx.device_buffer());
                bw_conv_data_.setArg(p++,int(dx.device_offset()));
                bw_conv_data_.setArg(p++,K.device_buffer());
                bw_conv_data_.setArg(p++,int(K.device_offset()));
                bw_conv_data_.setArg(p++,dy.device_buffer());
                bw_conv_data_.setArg(p++,int(dy.device_offset()));
                
                int gW = (width+1)/2;
                int gH = (height+1)/2;

                int lW = get_opt_val(gW);
                int lH = get_opt_val(gH);
                int lD = 1;
                if(lW * lH < 64)
                    lD = 64 / (lW * lH);
                
                cl::NDRange wg(lD,lH,lW);
                cl::NDRange gr=gpu::round_range(batch*config_.channels_in,gH,gW,wg);
                ec.queue().enqueueNDRangeKernel(bw_conv_data_,cl::NullRange,gr,wg,ec2.events(),ec2.event("sep_conv_bw_data"));
            }
            return;
        }

        int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
        int im2col_rows = dy.shape()[2]*dy.shape()[3]*dy.shape()[0];
        bwd_data_gemm_->gemm(
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
            ec);
    }

    void Convolution2D::backward_filter_gpu(Tensor &dy,Tensor &x,Tensor &dK,float factor,ExecutionContext const &ec)
    {
        int winograd_work_items = (config_.channels_in / 32) * (config_.channels_out / 32) * 256;
        if(use_winograd_) {
            bool reduce_k = winograd_work_items < ctx_.estimated_core_count();
            int B = x.shape()[0];
            int N = config_.channels_out;
            int C = config_.channels_in;
            int p=0;
            bw_conv_filter_.setArg(p++,B);
            bw_conv_filter_.setArg(p++,N);
            bw_conv_filter_.setArg(p++,C);
            bw_conv_filter_.setArg(p++,x.device_buffer());
            bw_conv_filter_.setArg(p++,int(x.device_offset()));
            bw_conv_filter_.setArg(p++,dK.device_buffer());
            bw_conv_filter_.setArg(p++,int(dK.device_offset()));
            bw_conv_filter_.setArg(p++,dy.device_buffer());
            bw_conv_filter_.setArg(p++,int(dy.device_offset()));
            bw_conv_filter_.setArg(p++,factor);
            
            cl::NDRange wg(256,1);
            int g1 = 256 * ((C+31)/32);
            int g2 = (N+31)/32;
            cl::NDRange gr(g1,g2,reduce_k ? 8 : 1);
            if(reduce_k) {
                ExecutionContext ec1 = ec.generate_series_context(0,2);
                ExecutionContext ec2 = ec.generate_series_context(1,2);
                scal_->scale(factor,dK,ec1);
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec2.events(),ec2.event("winograd_bwd_filter"));
            }
            else {
                ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec.events(),ec.event("winograd_bwd_filter"));
            }
            return;
        }
        else if(use_ds_conv_) {
            int kitems = dK.shape().total_size();
            int batch = x.shape()[0];
            int height = x.shape()[2];
            int width = x.shape()[3];
            int p=0;
            bw_conv_filter_.setArg(p++,batch);
            bw_conv_filter_.setArg(p++,height);
            bw_conv_filter_.setArg(p++,width);
            bw_conv_filter_.setArg(p++,x.device_buffer());
            bw_conv_filter_.setArg(p++,int(x.device_offset()));
            bw_conv_filter_.setArg(p++,dK.device_buffer());
            bw_conv_filter_.setArg(p++,int(dK.device_offset()));
            bw_conv_filter_.setArg(p++,dy.device_buffer());
            bw_conv_filter_.setArg(p++,int(dy.device_offset()));
            bw_conv_filter_.setArg(p++,factor);
            
            cl::NDRange wg(dwsc_bw_filter_wg_,1);
            cl::NDRange gr(dwsc_bw_filter_wg_,kitems);
            ec.queue().enqueueNDRangeKernel(bw_conv_filter_,cl::NullRange,gr,wg,ec.events(),ec.event("sep_conv_bw_filter"));
            return;
        }
        int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
        int im2col_rows = dy.shape()[2]*dy.shape()[3]*dy.shape()[0];
        bwd_weights_gemm_->gemm(
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
            ec);
    }

    void Convolution2D::fwd_bwd_cpu(GemmOpMode mode,Tensor &in,Tensor &out,Tensor &W,Tensor *bias_tensor,void *ws)
    {
        int batch = in.shape()[0];
        float *imcols = static_cast<float *>(ws);
        float *kernel = W.data<float>();
        int im2col_rows = out.shape()[2]*out.shape()[3];
        int kernel_cols = config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
        int in_size_no_batch = in.shape().size_no_batch();
        int out_size_no_batch = out.shape().size_no_batch();
        int step_groups_out = config_.channels_out / config_.groups;
        int step_groups_in  = config_.channels_in  / config_.groups;
        int step_kernel = step_groups_out * step_groups_in * config_.kernel[0] * config_.kernel[1];
        Shape  in_shape(in.shape()[0],in.shape()[1]/config_.groups,in.shape()[2],in.shape()[3]);
        Shape out_shape(out.shape()[0],out.shape()[1]/config_.groups,out.shape()[2],out.shape()[3]);
        for(int b=0;b<batch;b++) {
            for(int g=0;g<config_.groups;g++) {
                float *img = in.data<float>()  + in_size_no_batch *b + g * step_groups_in * in.shape()[2] * in.shape()[3];
                float *omg = out.data<float>() + out_size_no_batch*b + g * step_groups_out * out.shape()[2] * out.shape()[3];
                switch(mode) {
                case GemmOpMode::forward: {
                        im2col<details::Im2ColOp>(in_shape,out_shape,img,imcols);
                        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasTrans,
                                config_.channels_out / config_.groups,im2col_rows,kernel_cols,
                                1.0f,
                                kernel + step_kernel * g,kernel_cols,
                                imcols,kernel_cols,
                                0.0f,
                                omg,
                                im2col_rows);
                        if(config_.bias) {
                            float *bias = bias_tensor->data<float>() + g * step_groups_out;
                            int plane_size = out.shape()[2]*out.shape()[3];
                            for(int i=0;i<step_groups_out;i++) {
                                cblas_saxpy(plane_size,1.0f,bias + i,0,omg + plane_size*i,1);
                            }
                        }
                    }
                    break;
                case GemmOpMode::backward_filter: {
                        im2col<details::Im2ColOp>(in_shape,out_shape,img,imcols);
                        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,
                                config_.channels_out / config_.groups,kernel_cols,im2col_rows,
                                1.0f,
                                omg,im2col_rows,
                                imcols,kernel_cols,
                                1.0f,
                                kernel + step_kernel * g,kernel_cols
                                );
                    }
                    break;
                case GemmOpMode::backward_data: {
                        cblas_sgemm(CblasRowMajor,CblasTrans, CblasNoTrans,
                                im2col_rows, kernel_cols, config_.channels_out / config_.groups,
                                1.0f,
                                omg,im2col_rows,
                                kernel + step_kernel * g,kernel_cols,
                                0.0f,
                                imcols,kernel_cols
                                );
                        im2col<details::Col2ImOp>(in_shape,out_shape,img,imcols);
                    }
                    break;
                } // switch
            }
        }
    }

    void Convolution2D::backward(std::vector<TensorAndGradient> &input,
                                 std::vector<TensorAndGradient> &output,
                                 std::vector<TensorAndGradient> &parameters,
                                 Tensor &workspace,
                                 ExecutionContext const &e)
    {
        int steps =     2*int(input[0].requires_gradient) 
                        + 2*int(parameters[0].requires_gradient)
                        + int(config_.bias && parameters[1].requires_gradient)
                        + int(config_.activation != StandardActivations::identity);
        int step = 0;
        if(config_.activation != StandardActivations::identity) {
            std::vector<TensorAndGradient> tmp({output[0]}),empty;
            tmp[0].requires_gradient = true;
            tmp[0].accumulate_gradient = 0.0;
            activation_->backward(tmp,tmp,empty,workspace,e.generate_series_context(step++,steps));
        }
        if(config_.bias && parameters[1].requires_gradient) {
            bwd_bias_->backward(output[0].diff,
                                parameters[1].diff,
                                workspace,
                                parameters[1].accumulate_gradient,
                                e.generate_series_context(step++,steps));
        }
        if(parameters[0].requires_gradient) {
            auto ec1 = e.generate_series_context(step++,steps);
            auto ec2 = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                backward_filter_gpu(output[0].diff,input[0].data,parameters[0].diff,
                                    parameters[0].accumulate_gradient,ec2);
            }
            else {
                backward_filter_cpu(output[0].diff,input[0].data,parameters[0].diff,workspace,
                                    parameters[0].accumulate_gradient);
            }
        }
        if(input[0].requires_gradient) {
            auto ec1 = e.generate_series_context(step++,steps);
            auto ec2 = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                backward_data_gpu(output[0].diff,parameters[0].data,input[0].diff,workspace,
                                    input[0].accumulate_gradient,ec2);
            }
            else {
                backward_data_cpu(output[0].diff,parameters[0].data,input[0].diff,workspace,
                                    input[0].accumulate_gradient);
            }
        }
    }


} // dlprim
