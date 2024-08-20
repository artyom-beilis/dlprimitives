///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/pointwise.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/json.hpp>
#include <cmath>
#include <my_cblas.hpp>
namespace dlprim {
    PointwiseBase::PointwiseBase(Context &ctx) : Operator(ctx) {}
    PointwiseBase::~PointwiseBase(){}
    void PointwiseBase::setup(std::vector<TensorSpecs> const &in,
                       std::vector<TensorSpecs> &out,
                       std::vector<TensorSpecs> &parameters,
                       size_t &workspace)
    {
        DLPRIM_CHECK(in.size() == 1);
        out.assign({in[0]});
        parameters.clear();
        workspace = 0;
    }
    void PointwiseBase::reshape(std::vector<Shape> const &in,
                         std::vector<Shape> &out,
                         size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        out.assign({in[0]});
        ws = 0;
    }
	void PointwiseBase::forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &,
                             ExecutionContext const &q)
    {
        DLPRIM_CHECK(input.size() == output.size());
        DLPRIM_CHECK(input[0].shape() == output[0].shape());
        DLPRIM_CHECK(input[0].dtype() == output[0].dtype());
        if(ctx_.is_cpu_context()) {
            forward_cpu(input[0],output[0]);
        }
        else {
            forward_gpu(input[0],output[0],q);
        }
    }

    void PointwiseBase::backward(std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &,
                                Tensor &,
                                ExecutionContext const &q)
    {
        DLPRIM_CHECK(input.size()==1);
        DLPRIM_CHECK(output.size()==1); 
        if(!input[0].requires_gradient)
            return;
        DLPRIM_CHECK(input[0].diff.shape() == output[0].diff.shape());
        DLPRIM_CHECK(input[0].diff.shape() == output[0].data.shape());
        float accum = input[0].accumulate_gradient;
        if(ctx_.is_cpu_context()) {
            backward_cpu(input[0].data,input[0].diff,output[0].data,output[0].diff,accum);
        }
        else {
            backward_gpu(input[0].data,input[0].diff,output[0].data,output[0].diff,accum,q);
        }
    }
    void PointwiseBase::forward_cpu(Tensor &x,Tensor &y) 
    {
        size_t n = x.shape().total_size();
        float *px = x.data<float>();
        float *py = y.data<float>();
        forward_cpu_float(n,px,py);
    }
    void PointwiseBase::backward_cpu(Tensor &x,Tensor &dx,Tensor &y,Tensor &dy,float beta)
    {
        size_t n = x.shape().total_size();
        backward_cpu_float(n,x.data<float>(),dx.data<float>(),y.data<float>(),dy.data<float>(),beta);
    }

    ThresholdConfig ThresholdConfig::from_json(json::value const &v)
    {
        ThresholdConfig r;
        r.threshold = v.get("threshold",r.threshold);
        return r;
    }
    void Threshold::forward_cpu_float(size_t n,float const *x,float *y)
    {
        float th = cfg_.threshold;
        for(size_t i=0;i<n;i++)
            y[i] = x[i] > th;
    }
    void Threshold::backward_cpu_float(size_t n,float const *,float *dx,float const *,float const *,float beta)
    {
        if(beta == 0)
            memset(dx,0,n*sizeof(float));
        else
            cblas_sscal(n,beta,dx,1);
    }
    void Threshold::forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q)
    {
        core::pointwise_operation({x},{y},{cfg_.threshold},"y0 = x0 > w0 ? 1 : 0;",q);
    }
    void Threshold::backward_gpu(Tensor &,Tensor &dx,Tensor &,Tensor &,float beta,ExecutionContext const &q)
    {
        core::pointwise_operation({dx},{dx},{beta},"y0 = w0 > 0 ? x0 * w0 : 0;",q);
    }


    HardtanhConfig HardtanhConfig::from_json(json::value const &v)
    {
        HardtanhConfig r;
        r.min_val = v.get("min_val",r.min_val);
        r.max_val = v.get("max_val",r.max_val);
        return r;
    }
    void Hardtanh::forward_cpu_float(size_t n,float const *x,float *y)
    {
        float min_val = cfg_.min_val;
        float max_val = cfg_.max_val;
        for(size_t i=0;i<n;i++)
            y[i] = std::max(min_val,std::min(max_val,x[i]));
    }
    void Hardtanh::backward_cpu_float(size_t n,float const *x,float *dx,float const *y,float const *dy,float beta)
    {
        float min_val = cfg_.min_val;
        float max_val = cfg_.max_val;
        if(beta == 0) {
            for(size_t i=0;i<n;i++)
                dx[i] = ((min_val <= x[i] && x[i] <= max_val) ? dy[i] : 0);
        }
        else {
            for(size_t i=0;i<n;i++) {
                dx[i] = dx[i] * beta + ((min_val <= x[i] && x[i] <= max_val) ? dy[i] : 0);
            }
        }
    }
    void Hardtanh::forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q)
    {
        core::pointwise_operation({x},{y},{cfg_.min_val,cfg_.max_val},
                                    "y0=max(w0,min(w1,x0));",q);
    }
    void Hardtanh::backward_gpu(Tensor &x,Tensor &dx,Tensor &,Tensor &dy,float beta,ExecutionContext const &q)
    {
        core::pointwise_operation({x,dy,dx},{dx},{cfg_.min_val,cfg_.max_val,beta},
                                    "y0 = (w2 != 0 ? (x2 * w2) : 0) +  ((w0 <= x0 && x0 <= w1) ? x1 : 0);",
                                    q);
    }


    void Abs::forward_cpu_float(size_t n,float const *x,float *y)
    {
        for(size_t i=0;i<n;i++)
            y[i] = fabs(x[i]);
    }
    void Abs::backward_cpu_float(size_t n,float const *x,float *dx,float const *y,float const *dy,float beta)
    {
        if(beta == 0) {
            for(size_t i=0;i<n;i++)
                dx[i] =  x[i] >= 0 ? dy[i] : -dy[i];
        }
        else {
            for(size_t i=0;i<n;i++) {
                dx[i] = dx[i] * beta + (x[i] >= 0 ? dy[i] : -dy[i]);
            }
        }
    }
    void Abs::forward_gpu(Tensor &x,Tensor &y,ExecutionContext const &q)
    {
        core::pointwise_operation({x},{y},{},
                                    "y0=x0 >= 0 ? x0 : -x0;",q);
    }
    void Abs::backward_gpu(Tensor &x,Tensor &dx,Tensor &,Tensor &dy,float beta,ExecutionContext const &q)
    {
        core::pointwise_operation({x,dy,dx},{dx},{beta},
                                    "y0 = (w0 != 0 ? (x2 * w0) : 0) +  ((x0 >= 0) ? x1 : -x1);",
                                    q);
    }


}
