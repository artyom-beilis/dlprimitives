///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/interpolation.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {
InterpolationConfig InterpolationConfig::from_json(json::value const &v)
{
    InterpolationConfig cfg;
    cfg.out_h = v.get<int>("out_h",-1);
    cfg.out_w = v.get<int>("out_w",-1);
    cfg.scale_y = v.get<double>("scale_y",-1.0);
    cfg.scale_x = v.get<double>("scale_x",-1.0);
    cfg.align_corners = v.get<bool>("align_corners",false);
    std::string method = v.get<std::string>("method");
    if(method == "nearest")
        cfg.method = InterpolateType::nearest;
    else if(method == "nearest-exact")
        cfg.method = InterpolateType::nearest_exact;
    else if(method == "bilinear")
        cfg.method = InterpolateType::bilinear;
    else
       throw ValidationError("Unsupported interpolation method " + method); 
    return cfg;
}

Interpolation::Interpolation(Context &ctx,InterpolationConfig config) :
    Operator(ctx),
    config_(config)
{
    DLPRIM_CHECK((config_.scale_y <= 0) == (config_.scale_x <= 0));
    DLPRIM_CHECK((config_.out_h <= 0) == (config_.out_w <= 0));
    DLPRIM_CHECK((config_.scale_x <= 0) != (config_.out_w <= 0));
    DLPRIM_CHECK(config_.method == InterpolateType::bilinear || config_.align_corners == false);
}

Interpolation::~Interpolation()
{
}

int Interpolation::calc_size(int in,double scale)
{
    return int(in * scale);
}

Shape Interpolation::calc_size(Shape in_shape)
{
    Shape out_shape = in_shape;
    if(config_.out_h > 0) {
        out_shape[2] = config_.out_h;
        out_shape[3] = config_.out_w;
    }
    else {
        out_shape[2] = calc_size(in_shape[2],config_.scale_y);
        out_shape[3] = calc_size(in_shape[3],config_.scale_x);
    }
    return out_shape;
}

void Interpolation::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0].shape();
    DLPRIM_CHECK(in_shape.size() == 4);
    Shape out_shape=calc_size(in_shape);
    TensorSpecs outs(out_shape,in[0].dtype());
    out.assign({outs});
    p.clear();
    ws = 0;
}

void Interpolation::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0];
    DLPRIM_CHECK(in_shape.size() == 4);
    Shape out_shape = calc_size(in_shape);
    out.assign({out_shape});
    ws = 0;
}

namespace {
    struct HandleFWD {
        static void handle(
                    float &y,
                    float &x00,float &x01,
                    float &x10,float &x11,
                    float w_r0, float w_r1,float w_c0,float w_c1)
        {
            y = w_r0 * (w_c0 * x00 + w_c1 *x01)
              + w_r1 * (w_c0 * x10 + w_c1 *x11);
        }
    };
    struct HandleBWD {
        static void handle(
                    float &y,
                    float &x00,float &x01,
                    float &x10,float &x11,
                    float w_r0, float w_r1,float w_c0,float w_c1)
        {
            float val = y;
            x00 += val * w_r0 * w_c0;
            x01 += val * w_r0 * w_c1;
            x10 += val * w_r1 * w_c0;
            x11 += val * w_r1 * w_c1;
        }
    };
}


float Interpolation::calc_bin_scale(float scale,int x_size,int y_size)
{
    if(config_.align_corners) {
        if(y_size <= 1)
            return 0;
        return double(x_size-1)/(y_size-1);
    }
    if(scale > 0)
        return 1.0/scale;
    return double(x_size)/y_size;
}

float Interpolation::calc_bin_src(int dst_index,float scale)
{
    if(config_.align_corners)
        return scale * dst_index;
    else
        return std::max(scale * (dst_index + 0.5f) - 0.5f,0.0f);
}

std::tuple<int,int,float,float> Interpolation::calc_bin_src_weight(int dst_intex,float scale,int size)
{
    float p0f = calc_bin_src(dst_intex,scale);
    int p0 = p0f;
    int dp = (p0 < size-1) ? 1 : 0;
    int p1 = p0 + dp;
    float w1 = p0f - p0;
    float w0 = 1 - w1;
    return std::make_tuple(p0,p1,w0,w1);
}

template<typename OpHandle>
void Interpolation::bilinear_fwd_bwd_cpu(Tensor &in,Tensor &out)
{
    Shape x_shape = in.shape();
    Shape y_shape = out.shape();
    float *x=in.data<float>();
    float *y=out.data<float>();
    int bc = x_shape[0] * x_shape[1];
    int srcH = x_shape[2];
    int srcW = x_shape[3];
    int tgtH = y_shape[2];
    int tgtW = y_shape[3];
    {
        float vs = calc_bin_scale(config_.scale_y,x_shape[2],y_shape[2]);
        float hs = calc_bin_scale(config_.scale_x,x_shape[3],y_shape[3]);
        std::vector<std::tuple<int,int,float,float>> src_r(tgtH), src_c(tgtW);
        for(int r=0;r<tgtH;r++)
            src_r[r] = calc_bin_src_weight(r,vs,srcH);
        for(int c=0;c<tgtW;c++)
            src_c[c] = calc_bin_src_weight(c,hs,srcW);
        for(int i=0;i<bc;i++) {
            for(int r=0;r<tgtH;r++) {
                auto rdata = src_r[r];
                int src_r0 = std::get<0>(rdata);
                int src_r1 = std::get<1>(rdata);
                float w_r0 = std::get<2>(rdata);
                float w_r1 = std::get<3>(rdata);
                for(int c=0;c<tgtW;c++) {
                    auto cdata = src_c[c];
                    int src_c0 = std::get<0>(cdata);
                    int src_c1 = std::get<1>(cdata);
                    float w_c0 = std::get<2>(cdata);
                    float w_c1 = std::get<3>(cdata);
                    OpHandle::handle(
                                y[r * tgtW + c],
                                x[src_r0 * srcW + src_c0],
                                x[src_r0 * srcW + src_c1],
                                x[src_r1 * srcW + src_c0],
                                x[src_r1 * srcW + src_c1],
                                w_r0,w_r1,w_c0,w_c1);
                }
            }
            x+=srcW*srcH;
            y+=tgtW*tgtH;
        }
    }
}



void Interpolation::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(input[0].shape().size()==4);
    DLPRIM_CHECK(output[0].shape() == calc_size(input[0].shape()));
    DLPRIM_CHECK(input[0].dtype() == output[0].dtype());
    Context ctx(e);
    if(e.is_cpu_context()) {
        if(config_.method == InterpolateType::bilinear)
            bilinear_fwd_bwd_cpu<HandleFWD>(input[0],output[0]);
        else
            forward_cpu(input[0],output[0]);
    }
    else {
        core::interpolate2d(input[0],output[0],config_.scale_y,config_.scale_x,config_.method,config_.align_corners,e);
    }
}

void Interpolation::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!input[0].requires_gradient)
        return;
 
    DLPRIM_CHECK(input[0].diff.shape().size()==4);
    DLPRIM_CHECK(output[0].diff.shape() == calc_size(input[0].diff.shape()));
    DLPRIM_CHECK(input[0].diff.dtype() == output[0].diff.dtype());
    float accum = input[0].accumulate_gradient;
    if(e.is_cpu_context()) {
        float *dx = input[0].diff.data<float>();
        size_t len = input[0].diff.shape().total_size();
        if(accum == 0) {
            memset(dx,0,sizeof(float)*len);
        }
        else {
            cblas_sscal(len,accum,dx,1);
        }
        if(config_.method == InterpolateType::bilinear) {
            bilinear_fwd_bwd_cpu<HandleBWD>(input[0].diff,output[0].diff);
        }
        else {
            backward_cpu(input[0].diff,output[0].diff);
        }
    }
    else {
        core::interpolate2d_backward(input[0].diff,output[0].diff,config_.scale_y,config_.scale_x,config_.method,config_.align_corners,accum,e);
    }
}


void Interpolation::forward_cpu(Tensor &in,Tensor &out)
{
    Shape x_shape = in.shape();
    Shape y_shape = out.shape();
    float *x=in.data<float>();
    float *y=out.data<float>();
    int bc = x_shape[0] * x_shape[1];
    int srcH = x_shape[2];
    int srcW = x_shape[3];
    int tgtH = y_shape[2];
    int tgtW = y_shape[3];
    switch(config_.method) {
    case InterpolateType::nearest:
    case InterpolateType::nearest_exact:
        {
            float vs = config_.scale_y > 0 ? 1/config_.scale_y : float(x_shape[2])/y_shape[2];
            float hs = config_.scale_x > 0 ? 1/config_.scale_x : float(x_shape[3])/y_shape[3];
            float offset = config_.method == InterpolateType::nearest ? 0.0f : 0.5f; 
            std::vector<int> src_r(tgtH,0);
            std::vector<int> src_c(tgtW,0);
            for(int r=0;r<tgtH;r++)
                src_r[r] = std::min(int((r + offset) * vs),srcH-1);
            for(int c=0;c<tgtW;c++)
                src_c[c] = std::min(int((c + offset) * hs),srcW-1);
            for(int i=0;i<bc;i++) {
                for(int r=0;r<tgtH;r++) {
                    for(int c=0;c<tgtW;c++) {
                        y[r * tgtW + c] = x[src_r[r] * srcW + src_c[c]];
                    }
                }
                x+=srcW*srcH;
                y+=tgtW*tgtH;
            }
        }
        break;
    default:
        throw ValidationError("Unimplemented method");
    }
}

void Interpolation::backward_cpu(Tensor &in,Tensor &out)
{
    Shape dx_shape = in.shape();
    Shape dy_shape = out.shape();
    float *dx=in.data<float>();
    float *dy=out.data<float>();
    int bc = dx_shape[0] * dx_shape[1];
    int srcH = dx_shape[2];
    int srcW = dx_shape[3];
    int tgtH = dy_shape[2];
    int tgtW = dy_shape[3];
    switch(config_.method) {
    case InterpolateType::nearest:
    case InterpolateType::nearest_exact:
        {
            float vs = config_.scale_y > 0 ? config_.scale_y : float(dy_shape[2])/dx_shape[2];
            float hs = config_.scale_x > 0 ? config_.scale_x : float(dy_shape[3])/dx_shape[3];
            float offset = config_.method == InterpolateType::nearest ? 0.0f : 0.5f; 
            std::vector<std::pair<int,int>> tgt_r(srcH,std::pair<int,int>(0,0));
            std::vector<std::pair<int,int>> tgt_c(srcW,std::pair<int,int>(0,0));
            for(int r=0;r<srcH;r++) {
                int r0 = std::min(int(std::ceil(float(r  )*vs-offset)),tgtH);
                int r1 = std::min(int(std::ceil(float(r+1)*vs-offset)),tgtH);
                tgt_r[r]=std::make_pair(r0,r1);
            }
            for(int c=0;c<srcW;c++) {
                int c0 = std::min(int(std::ceil(float(c  )*hs-offset)),tgtW);
                int c1 = std::min(int(std::ceil(float(c+1)*hs-offset)),tgtW);
                tgt_c[c]=std::make_pair(c0,c1);
            }
            for(int i=0;i<bc;i++) {
                for(int r=0;r<srcH;r++) {
                    for(int c=0;c<srcW;c++) {
                        float grad = 0;
                        for(int tr=tgt_r[r].first;tr<tgt_r[r].second;tr++) {
                            for(int tc=tgt_c[c].first;tc<tgt_c[c].second;tc++) {
                                grad += dy[tr*tgtW+tc];
                            }
                        }
                        dx[r*srcW+c] += grad;
                    }
                }
                dx+=srcW*srcH;
                dy+=tgtW*tgtH;
            }
        }
        break;
    default:
        throw ValidationError("Unimplemented method");
    }
}


} // dlprim
