///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    void softmax_forward(Tensor &x,Tensor &y,bool log_softmax,ExecutionContext const &e)
    {
        DLPRIM_CHECK(x.shape().size() == 2 || x.shape().size() == 3);
        DLPRIM_CHECK(x.dtype() == float_data);
        DLPRIM_CHECK(y.shape()==x.shape());
        DLPRIM_CHECK(y.dtype() == x.dtype());
        int sm_range=x.shape()[1];

        int wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
        
        int items_per_wi = (sm_range + wg_size - 1) / wg_size;

        int mpl = wg_size * items_per_wi;
        int nd_range = (sm_range + mpl - 1) / mpl * wg_size;
        Context ctx(e);

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"softmax",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "LOG_SM",int(log_softmax));
        cl::Kernel kernel(prog,"softmax");
        Shape in_shape = x.shape();
        int b0 = in_shape[0];
        int b2 = in_shape.size() == 3 ? in_shape[2] : 1;
        int p = 0;
        kernel.setArg(p++,b0);
        kernel.setArg(p++,sm_range);
        kernel.setArg(p++,b2);
        x.set_arg(kernel,p);
        y.set_arg(kernel,p);

        cl::NDRange gr(b0,nd_range,b2);
        cl::NDRange wg(1,wg_size,1);
        e.queue().enqueueNDRangeKernel(kernel,cl::NullRange,gr,wg,e.events(),e.event("softmax"));
    }

    void softmax_backward(Tensor &dx,Tensor &y,Tensor &dy,bool log_softmax,float factor,ExecutionContext const &e)
    {
        DLPRIM_CHECK(dx.shape().size() == 2 || dx.shape().size() == 3);
        DLPRIM_CHECK(dx.dtype() == float_data);
        DLPRIM_CHECK(dy.shape() == dx.shape());
        DLPRIM_CHECK(dy.dtype() == dx.dtype());
        DLPRIM_CHECK(y.shape() == dx.shape());
        DLPRIM_CHECK(y.dtype() == dx.dtype());

        int sm_range=dx.shape()[1];

        int wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
        
        int items_per_wi = (sm_range + wg_size - 1) / wg_size;

        int mpl = wg_size * items_per_wi;
        int nd_range = (sm_range + mpl - 1) / mpl * wg_size;
        Context ctx(e);

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"softmax",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "LOG_SM",int(log_softmax));
        cl::Kernel kernel(prog,"softmax_backward");
        Shape in_shape = dx.shape();
        int b0 = in_shape[0];
        int b2 = in_shape.size() == 3 ? in_shape[2] : 1;
        int p = 0;
        kernel.setArg(p++,b0);
        kernel.setArg(p++,sm_range);
        kernel.setArg(p++,b2);
        dx.set_arg(kernel,p);
        y.set_arg(kernel,p);
        dy.set_arg(kernel,p);
        kernel.setArg(p++,factor);

        cl::NDRange gr(b0,nd_range,b2);
        cl::NDRange wg(1,wg_size,1);
        e.queue().enqueueNDRangeKernel(kernel,cl::NullRange,gr,wg,e.events(),e.event("softmax"));
    }

    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_forward(Tensor &x,Tensor &lbl,Tensor &y,bool reduce,float scale,ExecutionContext const &e)
    {
        DLPRIM_CHECK(x.shape().size() == 2);
        DLPRIM_CHECK(x.dtype() == float_data);
        DLPRIM_CHECK(y.shape()==(reduce ? Shape(1) : Shape(x.shape()[0])));
        DLPRIM_CHECK(y.dtype() == x.dtype());
        int sm_range=x.shape()[0];

        int wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
        
        int items_per_wi = (sm_range + wg_size - 1) / wg_size;

        std::string itype;
        switch(lbl.dtype()) {
        case int32_data: itype = "int"; break;
        case int64_data: itype = "long"; break;
        case float_data: itype = "float"; break;
        default: throw NotImplementedError("Unsupported type");
        }

        Context ctx(e);

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"nll_loss_fwd",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "REDUCE",int(reduce),
                            "itype",itype);

        cl::Kernel kernel(prog,"nll_loss_forward");
        Shape in_shape = x.shape();
        int p = 0;
        kernel.setArg(p++,int(in_shape[0]));
        kernel.setArg(p++,int(in_shape[1]));
        x.set_arg(kernel,p);
        lbl.set_arg(kernel,p);
        y.set_arg(kernel,p);
        kernel.setArg(p++,scale);
        cl::NDRange wg(wg_size,1,1);
        e.queue().enqueueNDRangeKernel(kernel,cl::NullRange,wg,wg,e.events(),e.event("nll_loss_fwd"));
    }
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_backward(Tensor &dx,Tensor &lbl,Tensor &dy,bool reduce,float scale,float factor,ExecutionContext const &e)
    {
        DLPRIM_CHECK(dx.shape().size() == 2);
        DLPRIM_CHECK(dx.dtype() == float_data);
        DLPRIM_CHECK(dy.shape()==(reduce ? Shape(1) : Shape(dx.shape()[0])));
        DLPRIM_CHECK(dy.dtype() == dx.dtype());
        std::string itype;
        switch(lbl.dtype()) {
        case int32_data: itype = "int"; break;
        case int64_data: itype = "long"; break;
        case float_data: itype = "float"; break;
        default: throw NotImplementedError("Unsupported type");
        }

        Context ctx(e);

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"nll_loss_bwd",
                            "REDUCE",int(reduce),
                            "itype",itype);
        cl::Kernel kernel(prog,"nll_loss_backward");
        Shape in_shape = dx.shape();
        int p = 0;
        kernel.setArg(p++,int(in_shape[0]));
        kernel.setArg(p++,int(in_shape[1]));
        dx.set_arg(kernel,p);
        lbl.set_arg(kernel,p);
        dy.set_arg(kernel,p);
        kernel.setArg(p++,scale);
        kernel.setArg(p++,factor);
        cl::NDRange nd(in_shape[1],in_shape[0]);
        e.queue().enqueueNDRangeKernel(kernel,cl::NullRange,nd,cl::NullRange,e.events(),e.event("nll_loss_bwd"));
    }

    
} // core
} // dlprim

