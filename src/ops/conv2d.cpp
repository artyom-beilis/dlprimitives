#include <dlprim/ops/conv2d.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
#include <cblas.h>
#include <boost/compute/event.hpp>

#define MERGED_GEMM

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
        DLPRIM_CHECK(config_.groups == 1);
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

#ifndef MERGED_GEMM
        
        ws_size_ = workspace = in_shape[0] * get_im2col_width() * output_shape[2] * output_shape[3] * size_of_data_type(dtype_);

        cl::Program const &prog2 = gpu::Cache::instance().get_program(ctx_,"im2col",
                                        "CHANNELS",config_.channels_in / config_.groups,
                                        "KERN_H",config_.kernel[0],
                                        "KERN_W",config_.kernel[1],
                                        "PAD_H",config_.pad[0],
                                        "PAD_W",config_.pad[1],
                                        "STRIDE_H",config_.stride[0],
                                        "STRIDE_W",config_.stride[1],
                                        "DILATE_H",config_.dilate[0],
                                        "DILATE_W",config_.dilate[1]);

        im2col_kernel_ = cl::Kernel(prog2,"im2col");
#else
        ws_size_ = workspace =  0;
#endif        

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
        
        int M = config_.channels_out;
        int N = out_h_*out_w_ * out[0];
        int K = get_im2col_width(); 

#ifndef MERGED_GEMM
        auto gemm = gpu::GEMM::get_optimal_gemm(
            ctx_,dtype_,false,true,
            M,N*out[0],K,
            (config_.bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
            config_.activation,
            out_h_ * out_w_
        );
#else
        auto gemm = gpu::GEMM::get_optimal_conv_gemm(
            ctx_,dtype_,false,true,
            M,N*out[0],K,
            config_.kernel,config_.dilate,config_.pad,config_.stride,
            config_.channels_in,in[2],in[3],out[2],out[3],
            (config_.bias ? gpu::GEMM::bias_M : gpu::GEMM::no_bias),
            config_.activation,
            out_h_ * out_w_
        );
#endif        
        gemm_ = std::move(gemm);
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
        else
            forward_gpu(in[0],out[0],W,bias,ectx);
    }

    #ifdef MERGED_GEMM
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
            ec.queue(),ec.events(),ec.event("conv_gemm"));
    }

    #else

    void Convolution2D::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ec)
    {
        int batch = in.shape()[0];

        int ind = 0;
        im2col_kernel_.setArg(ind++,batch);
        im2col_kernel_.setArg(ind++,int(in.shape()[2]));
        im2col_kernel_.setArg(ind++,int(in.shape()[3]));

        im2col_kernel_.setArg(ind++,int(out.shape()[2]));
        im2col_kernel_.setArg(ind++,int(out.shape()[3]));
        im2col_kernel_.setArg(ind++,in.device_buffer());
        im2col_kernel_.setArg(ind++,int(in.device_offset()));
        im2col_kernel_.setArg(ind++,workspace_.device_buffer());
        im2col_kernel_.setArg(ind++,int(workspace_.device_offset()));
        
        ExecutionContext ec1 = ec.generate_series_context(0,2);

        int bc = config_.channels_in * batch;
        int rows = out.shape()[2];
        int cols = out.shape()[3];
        cl::NDRange lr(1,8,8);
        cl::NDRange gr=gpu::round_range(bc,rows,cols,lr);

        std::cerr << bc << " " << rows << " " << cols << " " << ws_size_ << std::endl;

        ec.queue().enqueueNDRangeKernel(im2col_kernel_,cl::NullRange,gr,lr,
                                                       ec1.events(),ec1.event("im2col"));
        
        ind = 0;
        int M = config_.channels_out;
        int N = out.shape()[2]*out.shape()[3];
        int K = get_im2col_width(); 
        
        cl::Buffer *bias_buffer = nullptr;
        int bias_offset = 0;
        
        if(config_.bias) {
            Tensor &bias = parameters()[1];
            bias_buffer = &bias.device_buffer();
            bias_offset = bias.device_offset();
        }


        ExecutionContext ec2 = ec.generate_series_context(1,2);

        gemm_->gemm(M,N*batch,K,
            parameters()[0].device_buffer(),
            parameters()[0].device_offset(),
            K,
            workspace_.device_buffer(),
            workspace_.device_offset(),
            K,
            out.device_buffer(),
            out.device_offset(),
            N,
            bias_buffer,
            bias_offset,
            ec.queue(),ec2.events(),ec2.event("gemm"));
    }

    #endif


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

        int rows = outs[2];
        int cols = outs[3];
        int src_rows = in[2];
        int src_cols = in[3];
        for(int chan = 0;chan < config_.channels_in;chan ++) {
            for(int r=0;r<rows;r++) {
                for(int c=0;c<cols;c++) {
                    int mat_row = r * cols + c;
                    int mat_col = chan * (kern_h * kern_w);
                    float *mat = mat_in + mat_row * config_.channels_in * (kern_h * kern_w) + mat_col;
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
        int kernel_cols = config_.channels_in * config_.kernel[0] * config_.kernel[1];
        int in_size_no_batch = in.shape().size_no_batch();
        int out_size_no_batch = out.shape().size_no_batch();
        for(int b=0;b<batch;b++) {
            float *img = in.data<float>() + in_size_no_batch*b;
            float *omg = out.data<float>() + out_size_no_batch*b;
            im2col(in.shape(),out.shape(),img,imcols);
			cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasTrans,
					config_.channels_out,im2col_rows,kernel_cols,
					1.0f,
					kernel,kernel_cols,
					imcols,kernel_cols,
					0.0f,
					omg,
					im2col_rows);
            if(config_.bias) {
                float *bias = bias_tensor->data<float>();
                int plane_size = out.shape()[2]*out.shape()[3];
                for(int i=0;i<config_.channels_out;i++) {
                    cblas_saxpy(plane_size,1.0f,bias + i,0,omg + plane_size*i,1);
                }
            }
        }
        cpu::apply_activation(out.data<float>(),out.shape().total_size(),config_.activation);
    }
} // dlprim
