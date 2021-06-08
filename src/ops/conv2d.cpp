#include <dlprim/ops/conv2d.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
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

        if(ctx_.is_cpu_context()) {
            ws_size_ = workspace =  output_shape[2] * output_shape[3] * size_of_data_type(dtype_) * get_im2col_width();
            return;
        }
        
        get_gemm(in[0].shape(),out[0].shape());

        ws_size_ = workspace =  0;

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
            int N = out_h_*out_w_ * out[0];
            int K = get_im2col_width(); 

            auto gemm = gpu::GEMM::get_optimal_conv_gemm(
                ctx_,dtype_,false,true,
                M,N,K,
                config_.kernel,config_.dilate,config_.pad,config_.stride,config_.groups,
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
                config_.kernel,config_.dilate,config_.pad,config_.stride,1,
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
            forward_gpu(in[0],out[0],W,bias,ectx);
        }
    }
    

    void Convolution2D::forward_gpu(Tensor &in,Tensor &out,Tensor &W,Tensor *bias,ExecutionContext const &ec)
    {
        int batch = in.shape()[0];
        int M = config_.channels_out / config_.groups;
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
        template<int K,int S,int P,typename Float>
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
                                *mat++ = img[dx];
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
                                        *mat++ = (x >= 0 && x < src_cols) ? img[dx] : 0;
                                    }
                                }
                                else {
                                    for(int dx=0;dx < K ;dx++) {
                                        *mat++ = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // details


    void Convolution2D::im2col(Shape const &in,Shape const &outs,float *img_in,float *mat_in)
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
                case 1142: details::im2col_fast<11,4,2>(in,outs,img_in,mat_in); return;
                case  311: details::im2col_fast< 3,1,1>(in,outs,img_in,mat_in); return;
                case  512: details::im2col_fast< 5,1,2>(in,outs,img_in,mat_in); return;
                case  321: details::im2col_fast< 3,2,1>(in,outs,img_in,mat_in); return;
                case  723: details::im2col_fast< 7,2,3>(in,outs,img_in,mat_in); return;
                case  110: details::im2col_fast< 1,1,0>(in,outs,img_in,mat_in); return;
                case  120: details::im2col_fast< 1,2,0>(in,outs,img_in,mat_in); return;
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
                    float *mat = mat_in + mat_row * channels_in * (kern_h * kern_w) + mat_col;
                    int y_pos = -pad_h + r * stride_h;
                    int x_pos = -pad_w + c * stride_w;
                    float *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;

                    for(int dy = 0;dy < kern_h * dilate_h ;dy+= dilate_h, img += src_cols * dilate_h) {
                        int y = y_pos + dy;
                        if(y >= 0 && y < src_rows) {
                            for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
                                int x = x_pos + dx;
                                *mat++ = (x >= 0 && x < src_cols) ? img[dx] : 0;
                            }
                        }
                        else {
                            for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
                                *mat++ = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    void Convolution2D::forward_cpu(Tensor &in,Tensor &out,Tensor &W,Tensor *bias_tensor,void *ws)
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
                im2col(in_shape,out_shape,img,imcols);
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
        }
        cpu::apply_activation(out.data<float>(),out.shape().total_size(),config_.activation);
    }
} // dlprim
