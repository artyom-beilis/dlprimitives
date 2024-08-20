///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/elementwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/core/pointwise.hpp>
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
    config_(config)
{
}

Elementwise::~Elementwise()
{
}

void Elementwise::setup_bwd_gpu(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,size_t &ws)
{
    bool same_bwd = in[0] == in[1];
    TensorSpecs y_spec = out[0];
    TensorSpecs l_spec = in[0];
    TensorSpecs r_spec = in[1];
    std::string reduce_init = "reduce_y0 = 0;";
    std::string reduce_both_init = reduce_init + " reduce_y1 = 0;";
    std::string reduce = "reduce_y0 += y0;";
    std::string reduce_both = reduce + " reduce_y1 += y1;";
    std::string activation = "x1=" + activation_backward_equation(config_.activation,"x1","x0") + ";\n";
    bwd_both_.reset();

    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        bwd_l_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec},{l_spec},1,y_spec.dtype(),
                        activation + 
                        "y0 = x1 * w0;",reduce_init,reduce));

        bwd_r_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec},{r_spec},1,y_spec.dtype(),
                        activation + 
                        "y0 = x1 * w0;",reduce_init,reduce));

        if(same_bwd) {
            bwd_both_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                            ctx_,
                            {y_spec,y_spec},{l_spec,r_spec},2,y_spec.dtype(),
                            activation + 
                            "y0 = x1 * w0; y1=x1 * w1;",reduce_both_init,reduce_both));
        }

        break;
    case ElementwiseConfig::elementwise_prod:
        bwd_l_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec,r_spec},{l_spec},1,y_spec.dtype(),
                        activation + 
                        "y0 = x1 * w0 * x2;",reduce_init,reduce));

        bwd_r_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec,l_spec},{r_spec},1,y_spec.dtype(),
                        activation + 
                        "y0 = x1 * w0 * x2;",reduce_init,reduce));
        if(same_bwd) {
            bwd_both_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                            ctx_,
                            {y_spec,y_spec,l_spec,r_spec},{l_spec,r_spec},1,y_spec.dtype(),
                            activation + 
                            "y0 = x1 * x3 * w0; y1=x1 * x2 * w0;",reduce_both_init,reduce_both));
        }
        break;
    case ElementwiseConfig::elementwise_max:
        bwd_l_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec,l_spec,r_spec},{l_spec},2,y_spec.dtype(),
                        activation + 
                        "y0 = x2 * w0 >= x3 * w1 ? w0 * x1 : 0 ;",reduce_init,reduce));

        bwd_r_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                        ctx_,
                        {y_spec,y_spec,l_spec,r_spec},{r_spec},2,y_spec.dtype(),
                        activation + 
                        "y0 = x2 * w0 <  x3 * w1 ? w1 * x1 : 0 ;",reduce_init,reduce));

        if(same_bwd) {
            bwd_both_ = std::move(core::PointwiseOperationBroadcastReduce::create(
                            ctx_,
                            {y_spec,y_spec,l_spec,r_spec},{l_spec,r_spec},2,y_spec.dtype(),
                            activation + 
                            R"xx(
                                y0 = y1 = 0;  
                                if(x2 * w0 >=  x3 * w1) {
                                    y0 = w0 * x1; 
                                }
                                else  {
                                    y1 = w1 * x1;
                                }
                            )xx",
                            reduce_both_init,reduce_both));
        }

    }
    ws = std::max(bwd_l_->workspace(),bwd_r_->workspace());
    if(same_bwd) {
        ws = std::max(ws,bwd_both_->workspace());
    }
}

void Elementwise::backward_gpu( Tensor &a,Tensor &da,
                                Tensor &b,Tensor &db,
                                Tensor &c,Tensor &dc,
                                Tensor &ws,
                                bool left,bool right,
                                float beta_a,float beta_b,
                                ExecutionContext const &e)

{
    ExecutionContext ec_l,ec_r;
    if(left && right) {
        ec_l = e.generate_series_context(0,2);
        ec_r = e.generate_series_context(1,2);
    }
    else {
        ec_l = ec_r = e;
    }
    if(bwd_both_ && left && right) {
        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
                bwd_both_->enqueue({c,dc},{da,db},ws,
                                   {config_.coeff[0],config_.coeff[1]},
                                   {1,1},{beta_a,beta_b},e);
            break;
        case ElementwiseConfig::elementwise_prod:
                bwd_both_->enqueue({c,dc,a,b},{da,db},ws,
                                   {config_.coeff[0]*config_.coeff[1]},
                                   {1,1},{beta_a,beta_b},e);
            break;
        case ElementwiseConfig::elementwise_max:
                bwd_both_->enqueue({c,dc,a,b},{da,db},ws,
                                   {config_.coeff[0],config_.coeff[1]},
                                   {1,1},{beta_a,beta_b},e);
        }
        return;
    }
    
    if(left){
        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
                bwd_l_->enqueue({c,dc},{da},ws,
                                   {config_.coeff[0]},
                                   {1},{beta_a},ec_l);
            break;
        case ElementwiseConfig::elementwise_prod:
                bwd_l_->enqueue({c,dc,b},{da},ws,
                                   {config_.coeff[0]*config_.coeff[1]},
                                   {1},{beta_a},ec_l);
            break;
        case ElementwiseConfig::elementwise_max:
                bwd_l_->enqueue({c,dc,a,b},{da},ws,
                                   {config_.coeff[0],config_.coeff[1]},
                                   {1},{beta_a},ec_l);
        }
    }
    
    if(right) {
        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
                bwd_r_->enqueue({c,dc},{db},ws,
                                   {config_.coeff[1]},
                                   {1},{beta_b},ec_r);
            break;
        case ElementwiseConfig::elementwise_prod:
                bwd_r_->enqueue({c,dc,a},{db},ws,
                                   {config_.coeff[0]*config_.coeff[1]},
                                   {1},{beta_b},ec_r);
            break;
        case ElementwiseConfig::elementwise_max:
                bwd_r_->enqueue({c,dc,a,b},{db},ws,
                                   {config_.coeff[0],config_.coeff[1]},
                                   {1},{beta_b},ec_r);
        }
    }
}


void Elementwise::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    dtype_ = in[0].dtype();
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    DLPRIM_CHECK(in[1].dtype() == dtype_);
    out.assign({broadcast(in[0].shape(),in[1].shape())});
    p.clear();
    ws = 0;
    if(ctx_.is_cpu_context())
        return;
    
    setup_bwd_gpu(in,out,ws);
}

void Elementwise::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    out.assign({broadcast(in[0],in[1])});
    ws = 0;
    if(ctx_.is_cpu_context())
        return;

    std::vector<TensorSpecs> ins={TensorSpecs(in[0],dtype_),TensorSpecs(in[1],dtype_)};
    std::vector<TensorSpecs> outs={TensorSpecs(out[0],dtype_)};
    setup_bwd_gpu(ins,outs,ws);
}

void Elementwise::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(output[0].shape() == broadcast(input[0].shape(),input[1].shape()));
    
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

void Elementwise::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &ws,
                          ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(broadcast(input[0].data.shape(),input[1].data.shape())==output[0].data.shape());
    
    DLPRIM_CHECK(input[0].data.dtype() == dtype_);
    DLPRIM_CHECK(input[1].data.dtype() == dtype_);
    DLPRIM_CHECK(output[0].data.dtype() == dtype_);

    DLPRIM_CHECK(broadcast(input[0].diff.shape(),input[1].diff.shape())==output[0].diff.shape());
    
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[1].diff.dtype() == dtype_);
    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);

    if(!input[0].requires_gradient && !input[1].requires_gradient)
        return;
    if(ctx_.is_cpu_context()) {
        backward_cpu(input[0].data,input[0].diff,
                     input[1].data,input[1].diff,
                     output[0].data,output[0].diff,
                     input[0].requires_gradient,input[1].requires_gradient,
                     input[0].accumulate_gradient,input[1].accumulate_gradient);
    }
    else {
        backward_gpu(input[0].data,input[0].diff,
                     input[1].data,input[1].diff,
                     output[0].data,output[0].diff,
                     ws,
                     input[0].requires_gradient,input[1].requires_gradient,
                     input[0].accumulate_gradient,input[1].accumulate_gradient,
                     e);
    }

}

template<>
struct Elementwise::StridePos<1> {
    static size_t calc(Shape const &index,Shape const &stride)
    {
        return index[0]*stride[0];
    }
    template<typename F>
    static void loop(Shape s,F &f)
    {
        Shape index(0);
        for(index[0]=0;index[0]<s[0];index[0]++) {
            f(index);
        }
    }
};

template<>
struct Elementwise::StridePos<2> {
    static size_t calc(Shape const &index,Shape const &stride)
    {
        return index[0]*stride[0]+index[1]*stride[1];
    }
    template<typename F>
    static void loop(Shape s,F &f)
    {
        Shape index(0,0);
        for(index[0]=0;index[0]<s[0];index[0]++) {
            for(index[1]=0;index[1]<s[1];index[1]++) {
                f(index);
            }
        }
    }
};

template<>
struct Elementwise::StridePos<3> {
    static size_t calc(Shape const &index,Shape const &stride)
    {
        return index[0]*stride[0]+index[1]*stride[1]+index[2]*stride[2];
    }
    template<typename F>
    static void loop(Shape s,F &f)
    {
        Shape index(0,0,0);
        for(index[0]=0;index[0]<s[0];index[0]++) {
            for(index[1]=0;index[1]<s[1];index[1]++) {
                for(index[2]=0;index[2]<s[2];index[2]++) {
                    f(index);
                }
            }
        }
    }
};

template<>
struct Elementwise::StridePos<4> {
    static size_t calc(Shape const &index,Shape const &stride)
    {
        return index[0]*stride[0]+index[1]*stride[1]+index[2]*stride[2]+index[3]*stride[3];
    }
    template<typename F>
    static void loop(Shape s,F &f)
    {
        Shape index(0,0,0);
        for(index[0]=0;index[0]<s[0];index[0]++) {
            for(index[1]=0;index[1]<s[1];index[1]++) {
                for(index[2]=0;index[2]<s[2];index[2]++) {
                    for(index[3]=0;index[3]<s[3];index[3]++) {
                        f(index);
                    }
                }
            }
        }
    }
};

template<>
struct Elementwise::StridePos<5> {
    static size_t calc(Shape const &index,Shape const &stride)
    {
        return index[0]*stride[0]+index[1]*stride[1]+index[2]*stride[2]+index[3]*stride[3]+index[4]*stride[4];
    }
    template<typename F>
    static void loop(Shape s,F &f)
    {
        Shape index(0,0,0);
        for(index[0]=0;index[0]<s[0];index[0]++) {
            for(index[1]=0;index[1]<s[1];index[1]++) {
                for(index[2]=0;index[2]<s[2];index[2]++) {
                    for(index[3]=0;index[3]<s[3];index[3]++) {
                        for(index[4]=0;index[4]<s[4];index[4]++) {
                            f(index);
                        }
                    }
                }
            }
        }
    }
};

template<int dim,typename F>
void Elementwise::loop_strides_dim(Shape s,float *a,Shape a_strides,float *b,Shape b_strides,float *c,F const &func)
{
    auto f = [&](Shape const &index) {
        float x0 = a[StridePos<dim>::calc(index,a_strides)];
        float x1 = b[StridePos<dim>::calc(index,b_strides)];
        float y0 = func(x0,x1);
        *c++ = y0;
    };
    StridePos<dim>::loop(s,f);
}

template<typename F>
void Elementwise::loop_strides(Shape s,float *a,Shape a_strides,float *b,Shape b_strides,float *c,F const &func)
{
    switch(s.size()) {
    case 1: loop_strides_dim<1>(s,a,a_strides,b,b_strides,c,func); return;
    case 2: loop_strides_dim<2>(s,a,a_strides,b,b_strides,c,func); return;
    case 3: loop_strides_dim<3>(s,a,a_strides,b,b_strides,c,func); return;
    case 4: loop_strides_dim<4>(s,a,a_strides,b,b_strides,c,func); return;
    case 5: loop_strides_dim<5>(s,a,a_strides,b,b_strides,c,func); return;
    }
    throw ValidationError("Invalid shape size from broadcasting");
}


template<int dim,typename F,typename R>
void Elementwise::loops_reduce_dim(Shape s,float *a,Shape as,float *r, Shape rs,F const &func,R const &reduce)
{
    auto f = [&](Shape const &index) {
        float x0 = a[StridePos<dim>::calc(index,as)];
        float y0 = func(x0);
        size_t pos = StridePos<dim>::calc(index,rs);
        r[pos] = reduce(r[pos],y0);
    };
    StridePos<dim>::loop(s,f);
}

template<typename F,typename R>
void Elementwise::loops_reduce(Shape s,float *a,Shape as,float *r,Shape rs,F const &func,R const &reduce)
{
    switch(s.size()) {
    case 1: loops_reduce_dim<1>(s,a,as,r,rs,func,reduce); return;
    case 2: loops_reduce_dim<2>(s,a,as,r,rs,func,reduce); return;
    case 3: loops_reduce_dim<3>(s,a,as,r,rs,func,reduce); return;
    case 4: loops_reduce_dim<4>(s,a,as,r,rs,func,reduce); return;
    case 5: loops_reduce_dim<5>(s,a,as,r,rs,func,reduce); return;
    }
    throw ValidationError("Invalid shape size from broadcasting");
}





template<int dim,typename F,typename R>
void Elementwise::loops_reduce_dim(Shape s,float *a,Shape as,float *b,Shape bs,float *r, Shape rs,F const &func,R const &reduce)
{
    auto f = [&](Shape const &index) {
        float x0 = a[StridePos<dim>::calc(index,as)];
        float x1 = b[StridePos<dim>::calc(index,bs)];
        float y0 = func(x0,x1);
        size_t pos = StridePos<dim>::calc(index,rs);
        r[pos] = reduce(r[pos],y0);
    };
    StridePos<dim>::loop(s,f);
}

template<typename F,typename R>
void Elementwise::loops_reduce(Shape s,float *a,Shape as,float *b,Shape bs,float *r,Shape rs,F const &func,R const &reduce)
{
    switch(s.size()) {
    case 1: loops_reduce_dim<1>(s,a,as,b,bs,r,rs,func,reduce); return;
    case 2: loops_reduce_dim<2>(s,a,as,b,bs,r,rs,func,reduce); return;
    case 3: loops_reduce_dim<3>(s,a,as,b,bs,r,rs,func,reduce); return;
    case 4: loops_reduce_dim<4>(s,a,as,b,bs,r,rs,func,reduce); return;
    case 5: loops_reduce_dim<5>(s,a,as,b,bs,r,rs,func,reduce); return;
    }
    throw ValidationError("Invalid shape size from broadcasting");
}

template<int dim,typename F,typename R>
void Elementwise::loops_reduce_dim(Shape s,float *a,Shape as,float *b,Shape bs,float *c,Shape cs,float *r, Shape rs,F const &func,R const &reduce)
{
    auto f = [&](Shape const &index) {
        float x0 = a[StridePos<dim>::calc(index,as)];
        float x1 = b[StridePos<dim>::calc(index,bs)];
        float x2 = c[StridePos<dim>::calc(index,cs)];
        float y0 = func(x0,x1,x2);
        size_t pos = StridePos<dim>::calc(index,rs);
        r[pos] = reduce(r[pos],y0);
    };
    StridePos<dim>::loop(s,f);
}

template<typename F,typename R>
void Elementwise::loops_reduce(Shape s,float *a,Shape as,float *b,Shape bs,float *c,Shape cs,float *r,Shape rs,F const &func,R const &reduce)
{
    switch(s.size()) {
    case 1: loops_reduce_dim<1>(s,a,as,b,bs,c,cs,r,rs,func,reduce); return;
    case 2: loops_reduce_dim<2>(s,a,as,b,bs,c,cs,r,rs,func,reduce); return;
    case 3: loops_reduce_dim<3>(s,a,as,b,bs,c,cs,r,rs,func,reduce); return;
    case 4: loops_reduce_dim<4>(s,a,as,b,bs,c,cs,r,rs,func,reduce); return;
    case 5: loops_reduce_dim<5>(s,a,as,b,bs,c,cs,r,rs,func,reduce); return;
    }
    throw ValidationError("Invalid shape size from broadcasting");
}





void Elementwise::forward_cpu(Tensor &a,Tensor &b,Tensor &c)
{
    size_t size = c.shape().total_size();
    float *ap=a.data<float>();
    float *bp=b.data<float>();
    float *cp=c.data<float>();

    std::vector<Shape> shrank = {a.shape(),b.shape()};
    shrink_broadcast_ranges(shrank);
    Shape shrank_c = broadcast(shrank[0],shrank[1]);

    Shape as = shrank[0].broadcast_strides(shrank_c);
    Shape bs = shrank[1].broadcast_strides(shrank_c);
    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        {
            float c0 = config_.coeff[0];
            float c1 = config_.coeff[1];
            loop_strides(shrank_c,ap,as,bp,bs,cp,[=](float x0,float x1) {
                return x0*c0 + x1*c1;
            });
        }
        break;
    case ElementwiseConfig::elementwise_prod:
        {
            float w = config_.coeff[0] * config_.coeff[1];
            loop_strides(shrank_c,ap,as,bp,bs,cp,[=](float x0,float x1) {
                return x0*x1*w;
            });
        }
        break;
    case ElementwiseConfig::elementwise_max:
        {
            float c0 = config_.coeff[0];
            float c1 = config_.coeff[1];
            loop_strides(shrank_c,ap,as,bp,bs,cp,[=](float x0,float x1) {
                return std::max(x0*c0,x1*c1);
            });
        }
        break;
    }
    cpu::apply_activation(c.data<float>(),size,config_.activation);
}

void Elementwise::forward_gpu(Tensor &a,Tensor &b,Tensor &c,ExecutionContext const &ctx)
{
    std::string op;
    switch(config_.op) {
    case ElementwiseConfig::elementwise_sum:
        op="y0=x0*w0+x1*w1;\n";
        break;
    case ElementwiseConfig::elementwise_prod:
        op="y0=x0*x1*w0*w1;\n";
        break;
    case ElementwiseConfig::elementwise_max:
        op="y0=max(x0*w0,x1*w1);\n";
        break;
    default:
        throw ValidationError("Invalid activation values");
    }

    std::ostringstream code;
    code << op;
    if(config_.activation != StandardActivations::identity) {
        code << "y0 = " << activation_equation(config_.activation,"y0") << ";\n";
    }
    core::pointwise_operation_broadcast({a,b},{c},{config_.coeff[0],config_.coeff[1]},code.str(),ctx);
    
}


void Elementwise::backward_cpu( Tensor &at,Tensor &dat,
                                Tensor &bt,Tensor &dbt,
                                Tensor &ct,Tensor &dct,
                                bool left,bool right,
                                float beta_a,float beta_b)
{
    float *a = at.data<float>();
    float *b = bt.data<float>();
    float *c = ct.data<float>();
    float *da = dat.data<float>();
    float *db = dbt.data<float>();
    float *dc = dct.data<float>();

    cpu::apply_activation_diff(ct.shape().total_size(),c,dc,dc,config_.activation);

    std::vector<Shape> shrank = {at.shape(),bt.shape(),ct.shape()};
    shrink_broadcast_ranges(shrank);
    Shape res_shape = shrank[2];

    Shape as = shrank[0].broadcast_strides(res_shape);
    Shape bs = shrank[1].broadcast_strides(res_shape);
    Shape cs = shrank[2].broadcast_strides(res_shape);


    float c1 = config_.coeff[0];
    float c2 = config_.coeff[1];
    float c12 = c1*c2;

    if(left) {
        if(beta_a == 0)
            memset(da,0,sizeof(float)*at.shape().total_size());
        else
            cblas_sscal(at.shape().total_size(),beta_a,da,1);

        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
            loops_reduce(res_shape,dc,cs,da,as,
                        [&](float dy) { return dy * c1; },
                        [&](float a, float b) { return a+b; });
            break;
        case ElementwiseConfig::elementwise_prod:
            loops_reduce(res_shape,dc,cs,b,bs,da,as,
                        [&](float dC,float  B) { return c12*B*dC; },
                        [&](float a, float b) { return a+b; });
            break;
        case ElementwiseConfig::elementwise_max:
            loops_reduce(res_shape,a,as,b,bs,dc,cs,da,as,
                        [&](float a,float b,float dc) { return a*c1 >= b*c2 ? c1 * dc : 0; },
                        [&](float a, float b) { return a+b; });
            break;
        }
    }

    if(right) {
        if(beta_b == 0)
            memset(db,0,sizeof(float)*bt.shape().total_size());
        else
            cblas_sscal(bt.shape().total_size(),beta_b,db,1);
        switch(config_.op) {
        case ElementwiseConfig::elementwise_sum:
            loops_reduce(res_shape,dc,cs,db,bs,
                        [&](float dy) { return dy * c2; },
                        [&](float a, float b) { return a+b; });
            break;
        case ElementwiseConfig::elementwise_prod:
            loops_reduce(res_shape,dc,cs,a,as,db,bs,
                        [&](float dC,float  A) { return c12*A*dC; },
                        [&](float a, float b) { return a+b; });
            break;
        case ElementwiseConfig::elementwise_max:
            loops_reduce(res_shape,a,as,b,bs,dc,cs,db,bs,
                        [&](float a,float b,float dc) { return a*c1 < b*c2 ? c2 * dc : 0; },
                        [&](float a, float b) { return a+b; });
            break;
        }
    }
}




}
