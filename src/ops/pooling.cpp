///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/pooling.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/core/pool.hpp>
#include <my_cblas.hpp>

namespace dlprim {
PoolingBase PoolingBase::from_json(json::value const &v)
{
    PoolingBase cfg;
    char const *names[] = { "max", "avg" };
    cfg.mode = utils::parse_enum(v,"mode",names,cfg.mode);
    return cfg;
}

Pooling2DConfig Pooling2DConfig::from_json(json::value const &v)
{
    Pooling2DConfig cfg;
    static_cast<PoolingBase &>(cfg) = PoolingBase::from_json(v);
    utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
    utils::get_1dNd_from_json(v,"stride",cfg.stride);
    utils::get_1dNd_from_json(v,"pad",cfg.pad);
    cfg.ceil_mode = v.get("ceil_mode",cfg.ceil_mode);
    cfg.count_include_pad = v.get("count_include_pad",cfg.count_include_pad);
    return cfg;
}


Pooling2D::Pooling2D(Context &ctx,Pooling2DConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
    DLPRIM_CHECK(config_.kernel[0] > 0 && config_.kernel[1] > 0);
    DLPRIM_CHECK(config_.stride[0] > 0 && config_.stride[1] > 0);
    DLPRIM_CHECK(config_.pad[0] >= 0 && config_.pad[1] >= 0);
    DLPRIM_CHECK(config_.max <= config_.mode  && config_.mode <= config_.avg);
}

Pooling2D::~Pooling2D()
{
}

void Pooling2D::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    Shape ins = in[0].shape();
    Shape outs = calc_shape(ins);
    out.assign({TensorSpecs(outs,dtype_)});
    p.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    if(config_.mode == PoolingBase::max) {
        fwd_ = std::move(core::Pooling2DForward::create_max_pooling(
                ctx_,
                config_.kernel,config_.pad,config_.stride,
                dtype_));
        bwd_ = std::move(core::MaxPooling2DBackward::create(
                ctx_,
                config_.kernel,config_.pad,config_.stride,
                dtype_));
    }
    else {
        fwd_ = std::move(core::Pooling2DForward::create_avg_pooling(
                ctx_,
                config_.kernel,config_.pad,config_.stride,config_.count_include_pad,
                dtype_));
        bwd_ = std::move(core::AvgPooling2DBackward::create(
                ctx_,
                config_.kernel,config_.pad,config_.stride,config_.count_include_pad,
                dtype_));
    }
    ws =std::max(fwd_->workspace(),bwd_->workspace());
}

int Pooling2D::calc_output_size(int in_size,int dim)
{
    return core::calc_pooling_output_size(in_size,  config_.kernel[dim],
                                                    config_.pad[dim],
                                                    config_.stride[dim],
                                                    config_.ceil_mode);
}

Shape Pooling2D::calc_shape(Shape ins)
{
    DLPRIM_CHECK(ins.size()==4);
    int oh = calc_output_size(ins[2],0);
    int ow = calc_output_size(ins[3],1);
    return Shape(ins[0],ins[1],oh,ow);
}

void Pooling2D::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape ins = in[0];
    out.assign({calc_shape(ins)});
    if(ctx_.is_cpu_context())
        ws = 0;
    else
        ws = std::max(fwd_->workspace(),bwd_->workspace());
}

template<typename Dtype>
struct Pooling2D::MaxRedcue {
    static constexpr Dtype init_val = -std::numeric_limits<Dtype>::max();
    static Dtype apply(Dtype a,Dtype b) { return std::max(a,b); };
    static Dtype norm_valid(Dtype a,int ,int,int,int ) { return a; }
    static Dtype norm_full(Dtype a) { return a; }
};

template<typename Dtype>
struct Pooling2D::AveReduceValid
{
    AveReduceValid(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    static Dtype norm_valid(Dtype a,int  dr,int dc,int,int) { return a * (Dtype(1)/(dr*dc)); }
    Dtype norm_full(Dtype a) { return a * factor; }
};

template<typename Dtype>
struct Pooling2D::AveReduceFull
{
    AveReduceFull(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    Dtype norm_valid(Dtype a,int,int,int dr_p,int dc_p) { return a / (dr_p*dc_p); }
    Dtype norm_full(Dtype a) { return a * factor; }
};


void Pooling2D::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(output[0].shape() == calc_shape(input[0].shape()));
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        switch(config_.mode) {
        case Pooling2DConfig::max:
            forward_cpu<float>(input[0],output[0],MaxRedcue<float>());
            break;
        case Pooling2DConfig::avg:
            {
                float factor = 1.0f / (config_.kernel[0]*config_.kernel[1]);
                if(config_.count_include_pad)
                    forward_cpu<float>(input[0],output[0],AveReduceFull<float>(factor));
                else
                    forward_cpu<float>(input[0],output[0],AveReduceValid<float>(factor));
            }
            break;
        }
    }
    else {
        forward_gpu(input[0],output[0],e);
    }
}

void Pooling2D::backward_cpu_max(Tensor &xt,Tensor &dxt,Tensor &dyt,float factor)
{
    float *dx = dxt.data<float>();
    float *dy = dyt.data<float>();
    float *x  = xt.data<float>();
    
    size_t N = xt.shape().total_size();
    if(factor == 0)
        memset(dx,0,sizeof(float)*N);
    else 
        cblas_sscal(N,factor,dx,1);

    int bc = xt.shape()[0]*xt.shape()[1];
    
    int in_h = xt.shape()[2];
    int in_w = xt.shape()[3];

    int out_h = dyt.shape()[2];
    int out_w = dyt.shape()[3];
    
    for(int bc_count = 0;bc_count < bc;bc_count ++, x+= in_h*in_w,dx+= in_h*in_w,dy+= out_h*out_w) {
        for(int out_r=0;out_r<out_h;out_r++) {
            for(int out_c = 0;out_c <out_w;out_c++) {
                int row0 = out_r * config_.stride[0] - config_.pad[0];
                int col0 = out_c * config_.stride[1] - config_.pad[1];
                int row1 = row0 + config_.kernel[0];
                int col1 = col0 + config_.kernel[1];
                
                float val = -std::numeric_limits<float>::max();
                int pos=0;

                row0 = std::max(0,row0);
                col0 = std::max(0,col0);
                row1 = std::min(row1,in_h);
                col1 = std::min(col1,in_w);
                
                for(int r=row0;r<row1;r++) {
                    for(int c=col0;c<col1;c++) {
                        float tmp = x[r*in_w + c];
                        if(tmp > val) {
                            pos = r*in_w + c;
                            val = tmp;
                        }
                    }
                }

                dx[pos] += dy[out_r*out_w + out_c];
            }
        }
    }
}

template<typename Reduce>
void Pooling2D::backward_cpu_ave(Tensor &dxt,Tensor &dyt,float factor,Reduce rop)
{
    float *dx = dxt.data<float>();
    float *dy = dyt.data<float>();
    
    size_t N = dxt.shape().total_size();
    if(factor == 0)
        memset(dx,0,sizeof(float)*N);
    else 
        cblas_sscal(N,factor,dx,1);

    int bc = dxt.shape()[0]*dxt.shape()[1];
    
    int in_h = dxt.shape()[2];
    int in_w = dxt.shape()[3];

    int out_h = dyt.shape()[2];
    int out_w = dyt.shape()[3];
    
    for(int bc_count = 0;bc_count < bc;bc_count ++,dx+= in_h*in_w,dy+= out_h*out_w) {
        for(int out_r=0;out_r<out_h;out_r++) {
            for(int out_c = 0;out_c <out_w;out_c++) {
                int row0 = out_r * config_.stride[0] - config_.pad[0];
                int col0 = out_c * config_.stride[1] - config_.pad[1];
                int row1 = row0 + config_.kernel[0];
                int col1 = col0 + config_.kernel[1];

                int dr_with_pad = std::min(row1,in_h + config_.pad[0]) - std::max(-config_.pad[0],row0);
                int dc_with_pad = std::min(col1,in_w + config_.pad[1]) - std::max(-config_.pad[1],col0);

                row0 = std::max(0,row0);
                col0 = std::max(0,col0);
                row1 = std::min(row1,in_h);
                col1 = std::min(col1,in_w);
                
                float dy_norm = rop.norm_valid(dy[out_r*out_w + out_c],row1-row0,col1-col0,dr_with_pad,dc_with_pad);
                for(int r=row0;r<row1;r++) {
                    for(int c=col0;c<col1;c++) {
                        dx[r*in_w + c] += dy_norm;
                    }
                }
            }
        }
    }
}

void Pooling2D::backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ex)
{
    bwd_->enqueue(x,dx,dy,factor,ex);
}


template<typename Dtype,typename Reduce>
void Pooling2D::forward_cpu(Tensor &in,Tensor &out,Reduce rop)
{
    Dtype *src = in.data<Dtype>();
    Dtype *tgt = out.data<Dtype>();
    
    int bc = in.shape()[0]*in.shape()[1];
    
    int in_h = in.shape()[2];
    int in_w = in.shape()[3];

    int out_h = out.shape()[2];
    int out_w = out.shape()[3];
    
    for(int bc_count = 0;bc_count < bc;bc_count ++, src+= in_h*in_w,tgt+= out_h*out_w) {
        for(int out_r=0;out_r<out_h;out_r++) {
            for(int out_c = 0;out_c <out_w;out_c++) {
                int row0 = out_r * config_.stride[0] - config_.pad[0];
                int col0 = out_c * config_.stride[1] - config_.pad[1];
                int row1 = row0 + config_.kernel[0];
                int col1 = col0 + config_.kernel[1];
                
                Dtype val = rop.init_val;

                int dr_with_pad = std::min(row1,in_h + config_.pad[0]) - std::max(-config_.pad[0],row0);
                int dc_with_pad = std::min(col1,in_w + config_.pad[1]) - std::max(-config_.pad[1],col0);
                
                row0 = std::max(0,row0);
                col0 = std::max(0,col0);
                row1 = std::min(row1,in_h);
                col1 = std::min(col1,in_w);
                
                for(int r=row0;r<row1;r++) {
                    for(int c=col0;c<col1;c++) {
                        val = rop.apply(val,src[r*in_w + c]);
                    }
                }
                int dr = row1 - row0;
                int dc = col1 - col0;
                if(dr == config_.kernel[0] && dc == config_.kernel[1])
                    val = rop.norm_full(val);
                else
                    val = rop.norm_valid(val,dr,dc,dr_with_pad,dc_with_pad);
                tgt[out_r*out_w + out_c] = val;
            }
        }
    }
}

void Pooling2D::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx)
{
    fwd_->enqueue(in,out,ctx);
}

void Pooling2D::backward(std::vector<TensorAndGradient> &input,
                         std::vector<TensorAndGradient> &output,
                         std::vector<TensorAndGradient> &/*parameters*/,
                         Tensor &/*workspace*/,
                         ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!input[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[0].data.shape().size()==4);
    DLPRIM_CHECK(input[0].diff.shape().size()==4);
    DLPRIM_CHECK(output[0].data.shape().size()==4);

    DLPRIM_CHECK(input[0].data.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].data.shape()[1] == output[0].diff.shape()[1]);
    
    DLPRIM_CHECK(input[0].diff.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].diff.shape()[1] == output[0].diff.shape()[1]);

    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].data.dtype() == dtype_);
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        if(config_.mode == Pooling2DConfig::max) {
            backward_cpu_max(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient);
        }
        else {
            float factor = 1.0f / (config_.kernel[0]*config_.kernel[1]);
            if(config_.count_include_pad)
                backward_cpu_ave(input[0].diff,output[0].diff,input[0].accumulate_gradient,AveReduceFull<float>(factor));
            else
                backward_cpu_ave(input[0].diff,output[0].diff,input[0].accumulate_gradient,AveReduceValid<float>(factor));
        }
    }
    else {
        backward_gpu(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient,ctx);
    }

}





GlobalPooling::GlobalPooling(Context &ctx,GlobalPoolingConfig const &cfg) :
    Operator(ctx),
    cfg_(cfg),
    dtype_(float_data)
{
}
GlobalPooling::~GlobalPooling() {}

void GlobalPooling::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0].shape(); 
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in_shape.size() == 4);
    out.assign({TensorSpecs(Shape(in_shape[0],in_shape[1],1,1),in[0].dtype())});
    par.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    ws = setup_kernel(in_shape);
}

size_t GlobalPooling::setup_kernel(Shape const &in_shape)
{
    if(cfg_.mode == PoolingBase::max) {
        fwd_ = std::move(core::Pooling2DForward::create_global_max_pooling(ctx_,in_shape,dtype_));
        bwd_ = std::move(core::MaxPooling2DBackward::create_global(ctx_,in_shape,dtype_));
    }
    else {
        fwd_ = std::move(core::Pooling2DForward::create_global_avg_pooling(ctx_,in_shape,dtype_));
        bwd_ = std::move(core::AvgPooling2DBackward::create_global(ctx_,in_shape,dtype_));
    }
    return std::max(fwd_->workspace(),bwd_->workspace());
}


void GlobalPooling::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 4);
    ws = 0;
    out.assign({Shape(in[0][0],in[0][1],1,1)});
    if(ctx_.is_cpu_context())
        return;
    ws=setup_kernel(in[0]);
}

void GlobalPooling::forward_cpu(Tensor &input,Tensor &output)
{
    Shape in_shape = input.shape();
    float *in  = input.data<float>();
    float *out = output.data<float>();
    size_t total = in_shape[0]*in_shape[1];
    size_t over = in_shape[2]*in_shape[3];
    if(cfg_.mode == PoolingBase::max) {
        for(size_t i=0;i<total;i++) {
            float start = *in++;
            for(size_t i=1;i<over;i++)
                start = std::max(start,*in++);
            *out++= start;
        }
    }
    else {
        float factor = 1.0f / over;
        for(size_t i=0;i<total;i++) {
            float sum = 0;
            for(size_t i=0;i<over;i++)
                sum += *in++;
            *out++= sum * factor;
        }
    }
}


void GlobalPooling::backward_cpu(Tensor &xt,Tensor &dxt,Tensor &dyt,float scale)
{
    Shape in_shape = xt.shape();
    size_t total_dx = in_shape.total_size();
    float *x  = xt.data<float>();
    float *dx = dxt.data<float>();
    float *dy = dyt.data<float>();
    size_t total = in_shape[0]*in_shape[1];
    size_t over = in_shape[2]*in_shape[3];

    if(scale == 0)
        memset(dx,0,total_dx*sizeof(float));
    else
        cblas_sscal(total_dx,scale,dx,1);

    if(cfg_.mode == PoolingBase::max) {
        for(size_t i=0;i<total;i++) {
            size_t index = 0;
            for(size_t j=1;j<over;j++) {
                if(x[j] > x[index])
                    index = j;
            }
            float store = *dy++;
            for(size_t j=0;j<over;j++)
                *dx++ += store * (j == index);
            x+=over;
        }
    }
    else {
        for(size_t i=0;i<total;i++) {
            float store = dy[i] / over;
            for(size_t j=0;j<over;j++) {
                *dx++ += store;
            }
        }
    }

}
void GlobalPooling::backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ctx)
{
    bwd_->enqueue(x,dx,dy,factor,ctx);
}


void GlobalPooling::forward_gpu(Tensor &input, Tensor &output, ExecutionContext const &ctx)
{
    fwd_->enqueue(input,output,ctx);
}


void GlobalPooling::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==4);
    DLPRIM_CHECK(output[0].shape().size()==4);
    DLPRIM_CHECK(input[0].shape()[0] == output[0].shape()[0]);
    DLPRIM_CHECK(input[0].shape()[1] == output[0].shape()[1]);
    DLPRIM_CHECK(1 == output[0].shape()[2]);
    DLPRIM_CHECK(1 == output[0].shape()[3]);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],output[0]);
    }
    else {
        forward_gpu(input[0],output[0],ctx);
    }
}

void  GlobalPooling::backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &,
                              Tensor &,
                              ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!output[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[0].data.shape().size()==4);
    DLPRIM_CHECK(output[0].diff.shape().size()==4);
    DLPRIM_CHECK(input[0].diff.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].diff.shape()[1] == output[0].diff.shape()[1]);
    DLPRIM_CHECK(input[0].data.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].data.shape()[1] == output[0].diff.shape()[1]);
    DLPRIM_CHECK(1 == output[0].diff.shape()[2]);
    DLPRIM_CHECK(1 == output[0].diff.shape()[3]);
    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].data.dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        backward_cpu(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient);
    }
    else {
        backward_gpu(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient,ctx);
    }
}



} // dlprim


