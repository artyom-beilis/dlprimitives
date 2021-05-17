#include <dlprim/functions.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <cblas.h>

namespace dlprim {
SoftMax::~SoftMax() {}

SoftMax::SoftMax(Context &ctx,SoftMaxConfig const &) : 
    Operator(ctx),
    dtype_(float_data),wg_size_(0),items_per_wi_(0),sm_range_(-1)
{
}

void SoftMax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    out = in;
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0].shape()[1]);
}

void SoftMax::setup_kernel(int sm_range)
{
    if(sm_range_ == sm_range)
        return;
    if(sm_range <= 64)
        wg_size_ = 64;
    else if(sm_range <= 128)
        wg_size_ = 128;
    else 
        wg_size_ = 256;
    items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;

    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"softmax","WG_SIZE",wg_size_,"ITEMS_PER_WI",items_per_wi_);
    kernel_ = cl::Kernel(prog,"softmax");
    sm_range_ = sm_range;
    int mpl = wg_size_ * items_per_wi_;
    nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
}


void SoftMax::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 2);
    out = in;
    if(ctx_.is_cpu_context())
        return;
    setup_kernel(in[0][1]);
}

void SoftMax::forward_cpu(Tensor &input,Tensor &output)
{
    Shape in_shape = input.shape();
    float *in  = input.data<float>();
    float *out = output.data<float>();
    for(int i=0;i<in_shape[0];i++) {
        float maxv = in[0];
        for(int j=1;j<in_shape[1];j++)
            maxv = std::max(in[j],maxv);
        float sum = 0.0f;
        for(int j=0;j<in_shape[1];j++) 
            sum += out[j] = expf(in[j] - maxv);
        float factor = 1.0f/sum;
        for(int j=0;j<in_shape[1];j++) 
            out[j] *= factor;
        in += in_shape[1];
        out+= in_shape[1];
    }

}

void SoftMax::forward_gpu(Tensor &input, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(in_shape[1] == sm_range_);
    kernel_.setArg(0,in_shape[0]);
    kernel_.setArg(1,sm_range_);
    kernel_.setArg(2,input.device_buffer());
    kernel_.setArg(3,int(input.device_offset()));
    kernel_.setArg(4,output.device_buffer());
    kernel_.setArg(5,int(output.device_offset()));
    
    cl::NDRange gr(in_shape[0],nd_range_);
    cl::NDRange wg(1,wg_size_);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event());
}

void SoftMax::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2);
    DLPRIM_CHECK(input[0].shape() == output[0].shape());
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],output[0]);
    }
    else {
        forward_gpu(input[0],output[0],ctx);
    }
}

void SoftMax::backward_data(std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            std::vector<Tensor> &,
                            ExecutionContext const &)
{
    throw NotImplementedError("softmax::backward_data not implemented");
}


ElementwiseConfig ElementwiseConfig::from_json(json::value const &v)
{
    ElementwiseConfig cfg;
    std::string op = v.get<std::string>("operation","sum");
    if(op == "sum")
        cfg.op = elementwise_sum;
    else if(op == "prod")
        cfg.op = elementwise_prod;
    else if(op == "max")
        cfg.op = elementwise_max;
    else
        throw ValidatioError("Unsupported Elementwise operation " + op);
    cfg.coeff[0] = v.get("coef1",1.0f);
    cfg.coeff[1] = v.get("coef2",1.0f);
    cfg.activation = utils::activation_from_json(v);
    return cfg;
}

Elementwise::Elementwise(Context &ctx,ElementwiseConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
}

Elementwise::~Elementwise()
{
}

void Elementwise::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    DLPRIM_CHECK(in[1].dtype() == dtype_);
    DLPRIM_CHECK(in[0].shape() == in[1].shape());
    out.assign({in[0]});
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"eltwise",
                                        "ACTIVATION",int(config_.activation),
                                        "ELTOP",int(config_.op));
    kernel_ = cl::Kernel(prog,"eltwise");
}

void Elementwise::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0] == in[1]);
    out.assign({in[0]});
}

void Elementwise::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(input[0].shape() == input[1].shape());
    DLPRIM_CHECK(output[0].shape() == input[0].shape());
    
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(input[1].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    if(ctx_.is_cpu_context()) {
        forward_cpu(input[0],input[1],output[0]);
    }
    else {
        forward_gpu(input[0],input[1],output[0],e);
    }
}

void Elementwise::forward_cpu(Tensor &a,Tensor &b,Tensor &c)
{
    size_t size = a.shape().total_size();
    float *ap=a.data<float>();
    float *bp=b.data<float>();
    float *cp=c.data<float>();
    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        memcpy(cp,bp,sizeof(float)*size);
        cblas_saxpby(size,config_.coeff[0],ap,1,config_.coeff[1],cp,1); 
        break;
    case ElementwiseConfig::elementwise_prod:
        {
            float c = config_.coeff[0] * config_.coeff[1];
            for(size_t i=0;i<size;i++) 
                cp[i] = c * ap[i] * bp[i];
        }
        break;
    case ElementwiseConfig::elementwise_max:
        {
            float c1 = config_.coeff[0];
            float c2 = config_.coeff[1];
            for(size_t i=0;i<size;i++) 
                cp[i] = std::max(c1 * ap[i], c2 * bp[i]);
        }
        break;
    }
    cpu::apply_activation(cp,size,config_.activation);
}

void Elementwise::forward_gpu(Tensor &a,Tensor &b,Tensor &c,ExecutionContext const &ctx)
{
    int p=0;
    int size = a.shape().total_size();
    kernel_.setArg(p++,size);
    kernel_.setArg(p++,a.device_buffer());
    kernel_.setArg(p++,int(a.device_offset()));
    kernel_.setArg(p++,b.device_buffer());
    kernel_.setArg(p++,int(b.device_offset()));
    kernel_.setArg(p++,c.device_buffer());
    kernel_.setArg(p++,int(c.device_offset()));
    kernel_.setArg(p++,config_.coeff[0]);
    kernel_.setArg(p++,config_.coeff[1]);
    
    cl::NDRange gr((size + 255) / 256 * 256);
    cl::NDRange wg(256);
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event());
    
}

void Elementwise::backward_data(std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                ExecutionContext const &)
{
    throw NotImplementedError("Elementwise::backward_data not implemented");
}


Pooling2DConfig Pooling2DConfig::from_json(json::value const &v)
{
    Pooling2DConfig cfg;
    cfg.activation = utils::activation_from_json(v);
    utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
    utils::get_1dNd_from_json(v,"stride",cfg.stride);
    utils::get_1dNd_from_json(v,"pad",cfg.pad);
    cfg.count_include_pad = v.get("count_include_pad",cfg.count_include_pad);
    char const *names[] = { "max", "avg" };
    cfg.mode = utils::parse_enum(v,"mode",names,cfg.mode);
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

void Pooling2D::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    Shape ins = in[0].shape();
    Shape outs = calc_shape(ins);
    out.assign({TensorSpecs(outs,dtype_)});
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    wg_size_ = 256;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"pooling",
                                        "WG_SIZE",wg_size_,
                                        "POOL_H",config_.kernel[0],
                                        "POOL_W",config_.kernel[1],
                                        "STRIDE_H",config_.stride[0],
                                        "STRIDE_W",config_.stride[1],
                                        "PAD_H",config_.pad[0],
                                        "PAD_W",config_.pad[1],
                                        "POOL_MODE",int(config_.mode),
                                        "COUNT_INCLUDE_PAD",int(config_.count_include_pad),
                                        "ACTIVATION",int(config_.activation));
    kernel_ = cl::Kernel(prog,"pooling");
}

int Pooling2D::calc_output_size(int in_size,int dim)
{
    int padded_size = in_size + config_.pad[dim]*2;
    DLPRIM_CHECK(padded_size >= config_.kernel[dim]);
    return (padded_size - config_.kernel[dim]) / config_.stride[dim] + 1;
}

Shape Pooling2D::calc_shape(Shape ins)
{
    DLPRIM_CHECK(ins.size()==4);
    int oh = calc_output_size(ins[2],0);
    int ow = calc_output_size(ins[3],1);
    return Shape(ins[0],ins[1],oh,ow);
}

void Pooling2D::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==1);
    Shape ins = in[0];
    out.assign({calc_shape(ins)});
}

template<typename Dtype>
struct Pooling2D::MaxRedcue {
    static constexpr Dtype init_val = -std::numeric_limits<Dtype>::max();
    static Dtype apply(Dtype a,Dtype b) { return std::max(a,b); };
    static Dtype norm_valid(Dtype a,int ,int ) { return a; }
    static Dtype norm_full(Dtype a) { return a; }
};

template<typename Dtype>
struct Pooling2D::AveReduceValid
{
    AveReduceValid(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    static Dtype norm_valid(Dtype a,int  dr,int dc) { return a * (Dtype(1)/(dr*dc)); }
    Dtype norm_full(Dtype a) { return a * factor; }
};

template<typename Dtype>
struct Pooling2D::AveReduceFull
{
    AveReduceFull(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    Dtype norm_valid(Dtype a,int,int) { return a * factor; }
    Dtype norm_full(Dtype a) { return a * factor; }
};


void Pooling2D::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, ExecutionContext const &e)
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
                    val = rop.norm_valid(val,dr,dc);
                tgt[out_r*out_w + out_c] = val;
            }
        }
    }
    
    cpu::apply_activation(out.data<Dtype>(),out.shape().total_size(),config_.activation);
}

void Pooling2D::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx)
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
    kernel_.setArg(p++,in.device_buffer());
    kernel_.setArg(p++,int(in.device_offset()));
    kernel_.setArg(p++,out.device_buffer());
    kernel_.setArg(p++,int(out.device_offset()));

    
    cl::NDRange gr((bc + wg_size_ - 1) / wg_size_ * wg_size_,out_h,out_w);
    cl::NDRange wg(wg_size_,1,1);
    
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event());
    
}

void Pooling2D::backward_data(  std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                std::vector<Tensor> &,
                                ExecutionContext const &)
{
    throw NotImplementedError("Elementwise::backward_data not implemented");
}




} // dlprim


