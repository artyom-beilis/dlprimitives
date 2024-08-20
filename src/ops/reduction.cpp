///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/reduction.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/core/pointwise.hpp>
#include <my_cblas.hpp>
#include <cmath>

namespace dlprim {
    ReductionConfig ReductionConfig::from_json(json::value const &v)
    {
        ReductionConfig cfg;
        char const *names[] = { "sum", "sumsq", "abssum","mean"};
        cfg.method = utils::parse_enum(v,"method",names,cfg.method);
        cfg.output_scale = v.get("output_scale",cfg.output_scale);
        cfg.keep_dim = v.get("keep_dim",cfg.keep_dim);
        if(v.find("dims").is_undefined()) {
            cfg.start_axis = v.get("start_axis",cfg.start_axis);
        }
        else if (v.find("start_axis").is_undefined()){
            cfg.dims = v.get("dims",cfg.dims);
            if(cfg.dims.empty()) {
                throw ValidationError("Using dims in Reduction requires providing at least 1 dimension");
            }
        }
        else {
            throw ValidationError("Can't use both dims and start_axis in Reduction");
        }
        return cfg;
    }

    
    Reduction::Reduction(Context &ctx,ReductionConfig const &cfg) : Operator(ctx), cfg_(cfg)
    {
    }
    Reduction::~Reduction() {}

    void Reduction::setup(  std::vector<TensorSpecs> const &in,
                            std::vector<TensorSpecs> &out,
                            std::vector<TensorSpecs> &params,
                            size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        x_shape_ = in[0].shape();
        dtype_ = in[0].dtype();
        dims_to_reduce_.assign(x_shape_.size(),0);
        if(cfg_.dims.empty()) {
            int start_axis = cfg_.start_axis;
            if(start_axis < 0)
                start_axis += x_shape_.size();
            DLPRIM_CHECK(0<= start_axis && start_axis < x_shape_.size());
            for(int axis = start_axis;axis < x_shape_.size();axis++) {
                dims_to_reduce_[axis] = 1;
            }
        }
        else {
            for(int axis : cfg_.dims) { 
                if(axis < 0)
                    axis += x_shape_.size();
                DLPRIM_CHECK(0<= axis && axis < x_shape_.size());
                dims_to_reduce_[axis] = 1;
            }
        }

        calc_shapes();
        out.assign({TensorSpecs(squeezed_y_,dtype_)});
        params.clear();
        ws = 0;
        if(!ctx_.is_cpu_context()) {
            config_broadcast(in[0],TensorSpecs(full_y_,dtype_));
            ws = broadcast_->workspace();
        }
    }
    
    void Reduction::reshape(std::vector<Shape> const &in,
                            std::vector<Shape> &out,
                            size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1 && in[0].size() == x_shape_.size());
        x_shape_ = in[0];
        calc_shapes();
        out.assign({squeezed_y_});
        ws = 0;
        if(!ctx_.is_cpu_context()) {
            config_broadcast(TensorSpecs(in[0],dtype_),TensorSpecs(full_y_,dtype_));
            ws = broadcast_->workspace();
        }
    }
	void Reduction::forward( std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &q)
    {
        DLPRIM_CHECK(input.size() == 1 && output.size() == 1);
        DLPRIM_CHECK(input[0].shape() == x_shape_ && output[0].shape() == squeezed_y_);

        if(ctx_.is_cpu_context()) {
            forward_cpu(input[0],output[0]);
        }
        else {
            forward_gpu(input[0],output[0],workspace,q);
        }
    }
    

    void Reduction::backward(   std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &,
                                Tensor &,
                                ExecutionContext const &q)
    {
        if(!input.at(0).requires_gradient)
            return;
        float accum = input[0].accumulate_gradient;
        Tensor &x = input[0].data;
        Tensor &dx = input[0].diff;
        Tensor &y = output[0].data;
        Tensor &dy = output[0].diff;
        if(ctx_.is_cpu_context()) {
            backward_cpu(x,dx,y,dy,accum);
        }
        else {
            backward_gpu(x,dx,y,dy,accum,q);
        }
    }

    void Reduction::calc_shapes()
    {
        std::vector<uint64_t> full,squeezed;
        for(int i=0;i<x_shape_.size();i++) {
            if(dims_to_reduce_[i]) {
                full.push_back(1);
                if(cfg_.keep_dim)
                    squeezed.push_back(1);
            }
            else {
                full.push_back(x_shape_[i]);
                squeezed.push_back(x_shape_[i]);
            }
        }
        if(squeezed.empty())
            squeezed.push_back(1);

        full_y_ = Shape::from_range(full.begin(),full.end());
        squeezed_y_ = Shape::from_range(squeezed.begin(),squeezed.end());
        std::vector<uint64_t> strides(full_y_.size());
        size_t step = 1;
        for(int i=full_y_.size() - 1;i>=0;i--) {
            if(dims_to_reduce_[i])
                strides[i] = 0;
            else
                strides[i] = step;
            step *= full_y_[i];
        }
        y_strides_ = Shape::from_range(strides.begin(),strides.end());
    }

    struct Reduction::FWDData { float *x,*y; };
    struct Reduction::BWDData { float *x,*dx,*y,*dy; float scale; };
    struct Reduction::FWDBase {
        float *x,*y;
        FWDBase(FWDData const &d) : x(d.x), y(d.y) {}
    };
    struct Reduction::BWDBase {
        float *x,*dx,*y,*dy;
        float scale;
        BWDBase(BWDData const &d) : x(d.x), dx(d.dx), y(d.y), dy(d.dy),scale(d.scale) {}
    };

    struct Reduction::SumFwd : public FWDBase {
        using FWDBase::FWDBase;
        void operator()(size_t x_offset,size_t y_offset)
        {
            y[y_offset]+=x[x_offset];
        }
    };
    struct Reduction::SumBwd : public BWDBase {
        using BWDBase::BWDBase;
        void operator()(size_t x_offset,size_t y_offset)
        {
            dx[x_offset] += dy[y_offset] * scale;
        }
    };

    struct Reduction::Sum2Fwd : public Reduction::FWDBase {
        using FWDBase::FWDBase;
        void operator()(size_t dx,size_t dy)
        {
            y[dy]+=x[dx]*x[dx];
        }
    };
    struct Reduction::Sum2Bwd : public Reduction::BWDBase {
        using BWDBase::BWDBase;
        void operator()(size_t x_offset,size_t y_offset)
        {
            dx[x_offset] += 2*x[x_offset] * dy[y_offset] * scale;
        }
    };
    struct Reduction::L1Fwd : public Reduction::FWDBase{
        using FWDBase::FWDBase;
        void operator()(size_t x_offset,size_t y_offset)
        {
            y[y_offset]+=fabs(x[x_offset]);
        }
    };
    struct Reduction::L1Bwd : public Reduction::BWDBase {
        using BWDBase::BWDBase;
        void operator()(size_t x_offset,size_t y_offset)
        {
            float xv = x[x_offset];
            dx[x_offset] += scale * (xv > 0 ? 1: (xv < 0 ? -1 : 0)) *dy[y_offset];
        }
    };

    template<typename Op>
    void Reduction::iterate(Op &op)
    {
        Shape a = x_shape_;
        Shape y_strides = y_strides_;
        size_t x=0;
        switch(a.size()) {
        case 1:
            for(size_t i0=0;i0<a[0];i0++) {
                size_t y0=y_strides[0]*i0;
                op(x++,y0);
            }
            break;
        case 2:
            for(size_t i0=0;i0<a[0];i0++) {
                size_t y0 = y_strides[0]*i0;
                for(size_t i1=0;i1<a[1];i1++) {
                    size_t y1 = y_strides[1]*i1;
                    op(x++,y0+y1);
                }
            }
            break;
        case 3:
            for(size_t i0=0;i0<a[0];i0++) {
                size_t y0 = y_strides[0]*i0;
                for(size_t i1=0;i1<a[1];i1++) {
                    size_t y1 = y_strides[1]*i1;
                    for(size_t i2=0;i2<a[2];i2++) {
                        size_t y2 = y_strides[2]*i2;
                        op(x++,y0+y1+y2);
                    }
                }
            }
            break;
        case 4:
            for(size_t i0=0;i0<a[0];i0++) {
                size_t y0 = y_strides[0]*i0;
                for(size_t i1=0;i1<a[1];i1++) {
                    size_t y1 = y_strides[1]*i1;
                    for(size_t i2=0;i2<a[2];i2++) {
                        size_t y2 = y_strides[2]*i2;
                        for(size_t i3=0;i3<a[3];i3++) {
                            size_t y3 = y_strides[3]*i3;
                            op(x++,y0+y1+y2+y3);
                        }
                    }
                }
            }
            break;
        case 5:
            for(size_t i0=0;i0<a[0];i0++) {
                size_t y0 = y_strides[0]*i0;
                for(size_t i1=0;i1<a[1];i1++) {
                    size_t y1 = y_strides[1]*i1;
                    for(size_t i2=0;i2<a[2];i2++) {
                        size_t y2 = y_strides[2]*i2;
                        for(size_t i3=0;i3<a[3];i3++) {
                            size_t y3 = y_strides[3]*i3;
                            for(size_t i4=0;i4<a[4];i4++) {
                                size_t y4 = y_strides[4]*i4;
                                op(x++,y0+y1+y2+y3+y4);
                            }
                        }
                    }
                }
            }
            break;
        default:
            throw ValidationError("Invalid dim");
        }
    }

    
    void Reduction::forward_cpu(Tensor &tx,Tensor &ty)
    {
        float coeff = get_coeff();
        float *x=tx.data<float>();
        float *y=ty.data<float>();
        memset(y,0,sizeof(float)*full_y_.total_size());
        FWDData data{x,y};
        switch(cfg_.method) {
        case ReductionConfig::sum:
        case ReductionConfig::mean:
            { 
                SumFwd op(data);
                iterate(op);
            }
            break;
        case ReductionConfig::sumsq:
            {
                Sum2Fwd op(data);
                iterate(op);
            }
            break;
        case ReductionConfig::abssum:
            {
                L1Fwd op(data);
                iterate(op);
            }
        }
        if(coeff != 1.0f)
            cblas_sscal(full_y_.total_size(),coeff,y,1);
    }

    float Reduction::get_coeff()
    {
        if(cfg_.method == ReductionConfig::mean)
            return cfg_.output_scale * squeezed_y_.total_size() / x_shape_.total_size();
        else
            return cfg_.output_scale;
    }

    void Reduction::config_broadcast(TensorSpecs x,TensorSpecs y)
    {
        char const *calc = nullptr;
        switch(cfg_.method) {
        case ReductionConfig::mean:
        case ReductionConfig::sum:
            calc = "y0 = x0;";
            break;
        case ReductionConfig::sumsq:
            calc = "y0 = x0*x0;";
            break;
        case ReductionConfig::abssum:
            calc = "y0=(x0 < 0) ? -x0 : x0;";
            break;
        default:
            throw ValidationError("Internal Error config broadcast");
        }
        broadcast_ = std::move(core::PointwiseOperationBroadcastReduce::create(ctx_,
                            {x},{y},0,float_data,
                            calc,
                            "reduce_y0 = 0;",
                            "reduce_y0 += y0;"));
    }
    void Reduction::forward_gpu(Tensor &x,Tensor &y,Tensor &ws,ExecutionContext const &q)
    {
        Tensor Y = y.alias();
        Y.reshape(full_y_);
        broadcast_->enqueue({x},{Y},ws,{},{get_coeff()},{0},q);
    }
    void Reduction::backward_gpu(Tensor &x,Tensor &dx,Tensor &inp_y,Tensor &inp_dy,float accum,ExecutionContext const &q)
    {
        Tensor y = inp_y.alias();
        y.reshape(full_y_);
        Tensor dy = inp_dy.alias();
        dy.reshape(full_y_);
        float coeff = get_coeff();
        switch(cfg_.method) {
        case ReductionConfig::sum:
        case ReductionConfig::mean:
            core::pointwise_operation_broadcast({dy,dx},{dx},{coeff,accum},
                        "y0 = (w1 == 0 ? 0 : w1*x1) + w0 * x0;",q);
            break;
        case ReductionConfig::sumsq:
            core::pointwise_operation_broadcast({dy,dx,x},{dx},{coeff,accum},
                        "y0 = (w1 == 0 ? 0 : w1*x1) + w0 * 2 * x2 * x0;",q);
            break;
        case ReductionConfig::abssum:
            core::pointwise_operation_broadcast({dy,dx,x},{dx},{coeff,accum},
                        "y0 = (w1 == 0 ? 0 : (w1*x1)) + w0 * (x2 > 0.0 ? 1.0 : (x2 < 0.0 ? -1.0: 0.0)) * x0;",q);
            break;
        };
    }

    void Reduction::backward_cpu(Tensor &tx,Tensor &tdx,Tensor &ty,Tensor &tdy,float accum)
    {
        float *x=tx.data<float>();
        float *dx=tdx.data<float>();
        float *y=ty.data<float>();
        float *dy = tdy.data<float>();
        float coeff = get_coeff();
        if(accum == 0)
            memset(dx,0,sizeof(float)*tdx.shape().total_size());
        else if(accum != 1)
            cblas_sscal(tdx.shape().total_size(),accum,dx,1);
        BWDData data{x,dx,y,dy,coeff};
        switch(cfg_.method) {
        case ReductionConfig::sum:
        case ReductionConfig::mean:
            { 
                SumBwd op(data);
                iterate(op);
            }
            break;
        case ReductionConfig::sumsq:
            {
                Sum2Bwd op(data);
                iterate(op);
            }
            break;
        case ReductionConfig::abssum:
            {
                L1Bwd op(data);
                iterate(op);
            }
        }
    }
}

