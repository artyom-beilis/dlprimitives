#include <dlprim/ops/elementwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>

#include <my_cblas.hpp>

namespace dlprim {
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
        throw ValidationError("Unsupported Elementwise operation " + op);
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

void Elementwise::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    DLPRIM_CHECK(in[1].dtype() == dtype_);
    DLPRIM_CHECK(in[0].shape() == in[1].shape());
    out.assign({in[0]});
    p.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"eltwise",
                                        "ACTIVATION",int(config_.activation),
                                        "ELTOP",int(config_.op));
    kernel_ = cl::Kernel(prog,"eltwise");
    kernel_bwd_ = cl::Kernel(prog,"eltwise_bwd");
}

void Elementwise::reshape(std::vector<Shape> const &in,std::vector<Shape> &out)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0] == in[1]);
    out.assign({in[0]});
}

void Elementwise::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
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

struct Elementwise::SumTraits 
{
    static void left(float &,float &da,float &,float &,float &dc,float c1,float)
    {
        da += c1*dc;
    }
    static void right(float &,float &,float &,float &db,float &dc,float ,float c2)
    {
        db += c2*dc;
    }
};

struct Elementwise::ProdTraits 
{
    static void left(float &,float &da,float &b,float &,float &dc,float c1,float c2)
    {
        da += c1*c2*b*dc;
    }
    static void right(float &a,float &,float &,float &db,float &dc,float c1,float c2)
    {
        db += c1*c2*a*dc;
    }
};

struct Elementwise::MaxTraits 
{
    static void left(float &a,float &da,float &b,float &,float &dc,float c1,float c2)
    {
        if(a*c1 >= b*c2) {
            da += c1*dc;
        }
    }
    static void right(float &a,float &,float &b,float &db,float &dc,float c1,float c2)
    {
        if(a*c1 < b*c2) {
            db += c2*dc;
        }
    }
};

template<typename Traits>
void Elementwise::backward_cpu( Tensor &at,Tensor &dat,
                                Tensor &bt,Tensor &dbt,
                                Tensor &ct,Tensor &dct,
                                bool left,bool right,
                                float beta_a,float beta_b)
{
    size_t size = at.shape().total_size();
    
    float *a = at.data<float>();
    float *b = bt.data<float>();
    float *c = ct.data<float>();
    float *da = dat.data<float>();
    float *db = dbt.data<float>();
    float *dc = dct.data<float>();

    cpu::apply_activation_diff(size,c,dc,dc,config_.activation);

    if(beta_a == 0)
        memset(da,0,sizeof(float)*size);
    else
        cblas_sscal(size,beta_a,da,1);

    if(beta_b == 0)
        memset(db,0,sizeof(float)*size);
    else
        cblas_sscal(size,beta_b,db,1);

    float c1 = config_.coeff[0];
    float c2 = config_.coeff[1];

    if(left && right) {
        for(size_t i=0;i<size;i++) {
            Traits::left(a[i],da[i],b[i],db[i],dc[i],c1,c2);
            Traits::right(a[i],da[i],b[i],db[i],dc[i],c1,c2);
        }
    }
    else if(left) {
        for(size_t i=0;i<size;i++) {
            Traits::left(a[i],da[i],b[i],db[i],dc[i],c1,c2);
        }
    }
    else if(right) {
        for(size_t i=0;i<size;i++) {
            Traits::right(a[i],da[i],b[i],db[i],dc[i],c1,c2);
        }
    }
}

void Elementwise::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(input[0].diff.shape() == input[1].diff.shape());
    DLPRIM_CHECK(output[0].diff.shape() == input[0].diff.shape());
    
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[1].diff.dtype() == dtype_);
    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);

    DLPRIM_CHECK(input[0].diff.shape() == input[1].diff.shape());
    DLPRIM_CHECK(output[0].diff.shape() == input[0].diff.shape());
    
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[1].diff.dtype() == dtype_);
    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);

    if(!input[0].requires_gradient && !input[1].requires_gradient)
        return;
    if(ctx_.is_cpu_context()) {
        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
            backward_cpu<SumTraits>(input[0].data,input[0].diff,
                                    input[1].data,input[1].diff,
                                    output[0].data,output[0].diff,
                                    input[0].requires_gradient,input[1].requires_gradient,
                                    input[0].accumulate_gradient,input[1].accumulate_gradient);
            break;
        case ElementwiseConfig::elementwise_prod:
            backward_cpu<ProdTraits>(input[0].data,input[0].diff,
                                    input[1].data,input[1].diff,
                                    output[0].data,output[0].diff,
                                    input[0].requires_gradient,input[1].requires_gradient,
                                    input[0].accumulate_gradient,input[1].accumulate_gradient);
            break;
        case ElementwiseConfig::elementwise_max:
            backward_cpu<MaxTraits>(input[0].data,input[0].diff,
                                    input[1].data,input[1].diff,
                                    output[0].data,output[0].diff,
                                    input[0].requires_gradient,input[1].requires_gradient,
                                    input[0].accumulate_gradient,input[1].accumulate_gradient);
        }
    }
    else {
        backward_gpu(input[0].data,input[0].diff,
                     input[1].data,input[1].diff,
                     output[0].data,output[0].diff,
                     input[0].requires_gradient,input[1].requires_gradient,
                     input[0].accumulate_gradient,input[1].accumulate_gradient,
                     e);
    }

}
void Elementwise::backward_gpu( Tensor &a,Tensor &da,
                                Tensor &b,Tensor &db,
                                Tensor &c,Tensor &dc,
                                bool left,bool right,
                                float beta_a,float beta_b,
                                ExecutionContext const &e)

{
    int p=0;
    int sel = int(left) + (int(right) << 1);
    int size = a.shape().total_size();
    kernel_bwd_.setArg(p++,size);
    kernel_bwd_.setArg(p++,sel);

    kernel_bwd_.setArg(p++,a.device_buffer());
    kernel_bwd_.setArg(p++,int(a.device_offset()));
    kernel_bwd_.setArg(p++,da.device_buffer());
    kernel_bwd_.setArg(p++,int(da.device_offset()));

    kernel_bwd_.setArg(p++,b.device_buffer());
    kernel_bwd_.setArg(p++,int(b.device_offset()));
    kernel_bwd_.setArg(p++,db.device_buffer());
    kernel_bwd_.setArg(p++,int(db.device_offset()));

    kernel_bwd_.setArg(p++,c.device_buffer());
    kernel_bwd_.setArg(p++,int(c.device_offset()));
    kernel_bwd_.setArg(p++,dc.device_buffer());
    kernel_bwd_.setArg(p++,int(dc.device_offset()));

    kernel_bwd_.setArg(p++,config_.coeff[0]);
    kernel_bwd_.setArg(p++,config_.coeff[1]);

    kernel_bwd_.setArg(p++,beta_a);
    kernel_bwd_.setArg(p++,beta_b);
    
    cl::NDRange gr((size + 255) / 256 * 256);
    cl::NDRange wg(256);
    e.queue().enqueueNDRangeKernel(kernel_bwd_,cl::NullRange,gr,wg,e.events(),e.event("eltwise_bwd"));
    
}






void Elementwise::forward_cpu(Tensor &a,Tensor &b,Tensor &c)
{
    size_t size = a.shape().total_size();
    float *ap=a.data<float>();
    float *bp=b.data<float>();
    float *cp=c.data<float>();
    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        {
            float c0 = config_.coeff[0];
            float c1 = config_.coeff[1];
            for(size_t i=0;i<size;i++) 
                cp[i] = c0 * ap[i] + c1 * bp[i];
        }
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
    ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("eltwise"));
    
}

}
