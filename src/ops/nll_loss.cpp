///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/nll_loss.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>

namespace dlprim {

NLLLossConfig NLLLossConfig::from_json(json::value const &v)
{
    NLLLossConfig cfg;
    char const *names[] = { "none", "sum", "mean" };
    cfg.reduce = utils::parse_enum(v,"reduce",names,cfg.reduce);
    return cfg;
}

NLLLoss::NLLLoss(Context &ctx,NLLLossConfig const &cfg) : 
    Operator(ctx),
    cfg_(cfg)
{
}
NLLLoss::~NLLLoss(){}
void NLLLoss::setup(std::vector<TensorSpecs> const &in,
                   std::vector<TensorSpecs> &out,
                   std::vector<TensorSpecs> &parameters,
                   size_t &workspace)
{
    DLPRIM_CHECK(in.size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in[0].shape().size()==2);
    DLPRIM_CHECK(in[1].shape().size()==1);
    DLPRIM_CHECK(in[1].shape()[0] == in[0].shape()[0]);

    size_t dim = cfg_.reduce == NLLLossConfig::reduce_none ? in[0].shape()[0] : 1;
    out = { TensorSpecs(Shape( dim ),float_data) };
    parameters.clear();
    workspace = 0;
}

void NLLLoss::reshape(std::vector<Shape> const &in,
                     std::vector<Shape> &out,
                     size_t &ws)
{
    size_t dim = cfg_.reduce == NLLLossConfig::reduce_none ? in[0][0] : 1;
    out = { Shape(dim) };
    ws = 0;
}

template<typename Index>
void NLLLoss::forwad_cpu(Tensor &tx,Tensor &tlbl,Tensor &ty)
{
    float *x = tx.data<float>();
    Index const *lbl = tlbl.data<Index>();
    float *y = ty.data<float>();
    int batch = tx.shape()[0];
    int chan = tx.shape()[1];
    if(cfg_.reduce == NLLLossConfig::reduce_none) {
        for(int b=0;b<batch;b++) {
            int index = static_cast<int>(lbl[b]);
            float yval = 0;
            if(0<=index && index<chan)
                yval = -x[b*chan+index];
            *y++=yval;
        }
    }
    else {
        float sum = 0.0f;
        for(int b=0;b<batch;b++) {
            int index = static_cast<int>(lbl[b]);
            if(0<= index && index < chan)
                sum += -x[b*chan+index];
        }
        if(cfg_.reduce == NLLLossConfig::reduce_mean)
            sum *= 1.0f / batch;
        y[0]=sum;
    }
}

void NLLLoss::forward(std::vector<Tensor> &input,
                     std::vector<Tensor> &output,
                     std::vector<Tensor> &parameters,
                     Tensor &workspace,
                     ExecutionContext const &q)
{
    Tensor x=input.at(0);
    Tensor lbl=input.at(1);
    Tensor y=output.at(0);
    if(ctx_.is_opencl_context()) {
        float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/x.shape()[0] : 1.0f;
        core::nll_loss_forward(x,lbl,y,
                                cfg_.reduce != NLLLossConfig::reduce_none,
                                scale,
                                q);
    }
    else {
        switch(lbl.dtype()) {
        case float_data: forwad_cpu<float>(x,lbl,y); break;
        case int32_data: forwad_cpu<int>(x,lbl,y); break;
        default:
            throw NotImplementedError("NLLLoss label must be either int or float");
        }
    }
}

template<typename Index>
void NLLLoss::backward_cpu(Tensor &tdx,Tensor &tlbl,Tensor &tdy,float accum)
{
    float *dx = tdx.data<float>();
    Index const *lbl = tlbl.data<Index>();
    float *dy = tdy.data<float>();
    int batch = tdx.shape()[0];
    int chan = tdx.shape()[1];
    if(accum == 0)
        memset(dx,0,sizeof(float)*batch*chan);
    if(cfg_.reduce == NLLLossConfig::reduce_none) {
        for(int b=0;b<batch;b++) {
            int index = static_cast<int>(lbl[b]);
            float dyval = *dy++;
            for(int c=0;c<chan;c++) {
                float dxval = (c==index) ? -dyval : 0.0f;
                *dx = accum * *dx + dxval;
                dx++;
            }
        }
    }
    else {
        float dyval = *dy;
        if(cfg_.reduce == NLLLossConfig::reduce_mean)
            dyval /= batch;
        for(int b=0;b<batch;b++) {
            int index = static_cast<int>(lbl[b]);
            for(int c=0;c<chan;c++) {
                float dxval = (c==index) ? -dyval : 0.0f;
                *dx = accum * *dx + dxval;
                dx++;
            }
        }
    }
}

void NLLLoss::backward(  std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &e)
{
    if(!input.at(0).requires_gradient)
        return;
    Tensor dx = input[0].diff;
    Tensor lbl = input.at(1).data;
    Tensor dy = output.at(0).diff;
    float accum = input[0].accumulate_gradient;
    if(ctx_.is_opencl_context()) {
        float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/dx.shape()[0] : 1.0f;
        core::nll_loss_backward(dx,lbl,dy,
                            cfg_.reduce != NLLLossConfig::reduce_none,
                            scale,
                            accum,
                            e);
    }
    else {
        switch(lbl.dtype()) {
        case float_data: backward_cpu<float>(dx,lbl,dy,accum); break;
        case int32_data: backward_cpu<int>(dx,lbl,dy,accum); break;
        default:
            throw NotImplementedError("NLLLoss label must be either int or float");
        }
    }
}

} // dlprim
