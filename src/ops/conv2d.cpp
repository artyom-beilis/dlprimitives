#include <dlprim/ops/conv2d.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/ops/activation.hpp>
#include <cblas.h>
#include <boost/compute/event.hpp>

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

        DLPRIM_CHECK(config_.channels_in  % config_.groups == 0);
        DLPRIM_CHECK(config_.channels_out % config_.groups == 0);

        Shape output_shape = get_output_shape(in_shape);
        out.assign({TensorSpecs(output_shape,dtype_)});

        Shape params_shape(config_.channels_out,
                           config_.channels_in / config_.groups,
                           config_.kernel[0],
                           config_.kernel[1]);
        
        params.push_back(TensorSpecs(params_shape,dtype_));
        if(config_.bias) 
            params.push_back(TensorSpecs(Shape(config_.channels_out),dtype_));

        if(mode_ == CalculationsMode::train) {
            if(config_.bias)
                bwd_bias_.reset(new BWBias(ctx_,in[0].shape()[0],out_h_ * out_w_,dtype_));
            if(config_.activation != StandardActivations::identity)
                activation_ = std::move(Activation::get_bwd_op(ctx_,config_.activation,in[0]));
        }

        if(ctx_.is_cpu_context()) {
            ws_size_ = workspace =  output_shape[2] * output_shape[3] * size_of_data_type(dtype_) * get_im2col_width();
        }
        else {
             ws_size_ = workspace = 0;
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
        
        if(config_.groups > 1) {
            int M = config_.channels_out / config_.groups;
            int N = out_h_*out_w_;
            int K = get_im2col_width(); 

            auto gemm = gpu::GEMM::get_optimal_conv_gemm(
                ctx_,dtype_,false,true,
                M,N,K,
                config_.kernel,config_.dilate,config_.pad,config_.stride,
                config_.channels_in / config_.groups,in[2],in[3],out[2],out[3],
                (config_.bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
                config_.activation,
                out_h_ * out_w_
            );

            gemm_ = std::move(gemm);
        }
        else {
            int M = config_.channels_out;
            int N = out_h_*out_w_ * out[0];
            int K = get_im2col_width(); 

            auto gemm = gpu::GEMM::get_optimal_conv_gemm(
                ctx_,dtype_,false,true,
                M,N,K,
                config_.kernel,config_.dilate,config_.pad,config_.stride,
                config_.channels_in,in[2],in[3],out[2],out[3],
                (config_.bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
                config_.activation,
                out_h_ * out_w_
            );

            gemm_ = std::move(gemm);
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
            if(config_.groups > 1)
                forward_gpu_grouped(in[0],out[0],W,bias,ectx);
            else
                forward_gpu(in[0],out[0],W,bias,ectx);
        }
    }
    


    void Convolution2D::forward_gpu_grouped(Tensor &in,Tensor &out,Tensor &W,Tensor *bias,ExecutionContext const &ec)
    {
        int M = config_.channels_out / config_.groups;
        int N = out.shape()[2]*out.shape()[3];
        int K = get_im2col_width(); 
        int batch = in.shape()[0];
        int groups = config_.groups;
        
        cl::Buffer *bias_buffer = nullptr;
        int bias_offset = 0;
        
        if(config_.bias) {
            bias_buffer = &bias->device_buffer();
            bias_offset = bias->device_offset();
        }
        int chstep_in  = config_.channels_in  / config_.groups;
        int chstep_out = config_.channels_out / config_.groups;
        int step_input  = chstep_in  *  in.shape()[2]  * in.shape()[3];
        int step_output = chstep_out * out.shape()[2] * out.shape()[3];
        int step_kernel = chstep_out * chstep_in * config_.kernel[0] * config_.kernel[1];

        int index = 0;
        for(int b=0;b<batch;b++) {
            for(int g=0;g<groups;g++,index++) {
                ExecutionContext ectmp = ec.generate_series_context(index,batch*groups);
                gemm_->gemm(M,N,K,
                    W.device_buffer(),
                    W.device_offset() + g * step_kernel,
                    K,
                    in.device_buffer(),
                    in.device_offset() + index * step_input,
                    K,
                    out.device_buffer(),
                    out.device_offset() + index * step_output,
                    N,
                    bias_buffer,
                    bias_offset + g* chstep_out,
                    0.0f,
                    ectmp.queue(),ectmp.events(),ectmp.event("conv_gemm"));
            }
        }
    }

    void Convolution2D::forward_gpu(Tensor &in,Tensor &out,Tensor &W,Tensor *bias,ExecutionContext const &ec)
    {
        int batch = in.shape()[0];
        int M = config_.channels_out;
        int N = out.shape()[2]*out.shape()[3];
        int K = get_im2col_width(); 
        
        cl::Buffer *bias_buffer = nullptr;
        int bias_offset = 0;
        
        if(config_.bias) {
            bias_buffer = &bias->device_buffer();
            bias_offset = bias->device_offset();
        }
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
            ec.queue(),ec.events(),ec.event("conv_gemm"));
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
        fwd_bwd_cpu(OpMode::forward,in,out,M,bias,ws);
        cpu::apply_activation(out.data<float>(),out.shape().total_size(),config_.activation);
    }
    void Convolution2D::backward_data_cpu(Tensor &dy,Tensor &K,Tensor &dx,Tensor &ws,float factor)
    {
        scale_cpu(dx,factor);
        fwd_bwd_cpu(OpMode::backward_data,dx,dy,K,nullptr,ws.host_data());
    }
    void Convolution2D::backward_filter_cpu(Tensor &dy,Tensor &x,Tensor &dK,Tensor &ws,float factor)
    {
        scale_cpu(dK,factor);
        fwd_bwd_cpu(OpMode::backward_filter,x,dy,dK,nullptr,ws.host_data());
    }

    void Convolution2D::fwd_bwd_cpu(OpMode mode,Tensor &in,Tensor &out,Tensor &W,Tensor *bias_tensor,void *ws)
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
                case OpMode::forward: {
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
                case OpMode::backward_filter: {
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
                case OpMode::backward_data: {
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
        int steps = int(input[0].requires_gradient) 
                        + int(parameters[0].requires_gradient)
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
                                parameters[1].accumulate_gradient,
                                e.generate_series_context(step++,steps));
        }
        if(parameters[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                //backward_filter_gpu(output[0].diff,input[0].data,parameters[0].diff,
                //                    parameters[0].accumulate_gradient,ec);
            }
            else {
                backward_filter_cpu(output[0].diff,input[0].data,parameters[0].diff,workspace,
                                    parameters[0].accumulate_gradient);
            }
        }
        if(input[0].requires_gradient) {
            auto ec = e.generate_series_context(step++,steps);
            if(!ctx_.is_cpu_context()) {
                //backward_data_gpu(output[0].diff,parameters[0].data,input[0].diff,
                //                    input[0].accumulate_gradient,ec);
            }
            else {
                backward_data_cpu(output[0].diff,parameters[0].data,input[0].diff,workspace,
                                    input[0].accumulate_gradient);
            }
        }
    }


} // dlprim
