#include <dlprim/core/interpolate.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim { namespace core {
    ///
    /// Interpolate forward y size should be floor(src_size * scale), scale_x,scale_y can be -1 to calculate automatically
    ///
    void interpolate2d_intern(Tensor &x,Tensor &y,double scale_y,double scale_x,InterpolateType method,bool bool_align_corners,ExecutionContext const &e,int fwd_bilinear)
    {
        float offset = 0;
        int align_corners = bool_align_corners;
        DLPRIM_CHECK(method == InterpolateType::nearest || method == InterpolateType::nearest_exact || method == InterpolateType::bilinear);
        if(method  == InterpolateType::nearest_exact)
            offset = 0.5f;

        bool bilinear = method == InterpolateType::bilinear;

        Shape x_shape = x.shape();
        Shape y_shape = y.shape();
        DLPRIM_CHECK(x_shape.size() == 4 && y_shape.size()==4);
        DLPRIM_CHECK(x_shape[0] == y_shape[0] && x_shape[1] == y_shape[1]);
        int srcH = x_shape[2], srcW = x_shape[3];
        int tgtH = y_shape[2], tgtW = y_shape[3];
        DLPRIM_CHECK((scale_y <= 0) == (scale_x <= 0));

        if(bilinear && align_corners) {
            scale_y = tgtH >= 1 ?  double(srcH-1)/(tgtH-1) : 0;
            scale_x = tgtW >= 1 ?  double(srcW-1)/(tgtW-1) : 0;
        }
        else {
            if(scale_y <= 0) {
                scale_y = double(srcH) / tgtH;
                scale_x = double(srcW) / tgtW;
            }
            else {
                DLPRIM_CHECK(tgtH == int(srcH * scale_y));
                DLPRIM_CHECK(tgtW == int(srcW * scale_x));
                scale_x = 1.0/scale_x;
                scale_y = 1.0/scale_y;
            }
        }
        
        Context ctx(e);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"interpolate_2d");
        cl::Kernel k(prog,(bilinear ? "bilinear" : "nearest_fwd"));

        int bc = x_shape[0]*x_shape[1];
        int p=0;
        int work_per_thread = std::min(bc,64);
        if(bilinear)
            k.setArg(p++,fwd_bilinear);
        k.setArg(p++,bc);
        k.setArg(p++,work_per_thread);
        k.setArg(p++,srcH);
        k.setArg(p++,srcW);
        k.setArg(p++,tgtH);
        k.setArg(p++,tgtW);
        k.setArg(p++,float(scale_y));
        k.setArg(p++,float(scale_x));
        if(bilinear)
            k.setArg(p++,align_corners);
        else
            k.setArg(p++,offset);
        x.set_arg(k,p);
        y.set_arg(k,p);
        
        int gr_size = (bc + work_per_thread - 1) / work_per_thread;
        cl::NDRange gr(tgtH,tgtW,gr_size);
        char const *name = (bilinear ? (fwd_bilinear ? "bilinear_fwd" : "bilinear_bwd") : "nearest_fwd");
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,cl::NullRange,e.events(),e.event(name));
    }

    void interpolate2d(Tensor &x,Tensor &y,double scale_y,double scale_x,InterpolateType method,bool bool_align_corners,ExecutionContext const &e)
    {
        interpolate2d_intern(x,y,scale_y,scale_x,method,bool_align_corners,e,true);
    }

    void interpolate2d_backward(Tensor &dx,Tensor &dy,double scale_y,double scale_x,InterpolateType method,bool align_corners,float factor,ExecutionContext const &e)
    {
        if(method == InterpolateType::bilinear) {
            scale_tensor(factor,dx,e.generate_series_context(0,2));
            interpolate2d_intern(dx,dy,scale_y,scale_x,method,align_corners,e.generate_series_context(1,2),false);
            return;
        }
        float offset = 0;
        DLPRIM_CHECK(method == InterpolateType::nearest || method == InterpolateType::nearest_exact);
        if(method  == InterpolateType::nearest_exact)
            offset = 0.5f;

        Shape x_shape = dx.shape();
        Shape y_shape = dy.shape();
        DLPRIM_CHECK(x_shape.size() == 4 && y_shape.size()==4);
        DLPRIM_CHECK(x_shape[0] == y_shape[0] && x_shape[1] == y_shape[1]);
        int srcH = x_shape[2], srcW = x_shape[3];
        int tgtH = y_shape[2], tgtW = y_shape[3];
        DLPRIM_CHECK((scale_y < 0) == (scale_x < 0));
        if(scale_y < 0) {
            scale_y = double(tgtH) / srcH;
            scale_x = double(tgtW) / srcW;
        }
        else {
            DLPRIM_CHECK(tgtH == int(srcH * scale_y));
            DLPRIM_CHECK(tgtW == int(srcW * scale_x));
        }
        
        Context ctx(e);
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"interpolate_2d");
        cl::Kernel k(prog,"nearest_bwd");

        int bc = x_shape[0]*x_shape[1];
        int p=0;
        int work_per_thread = std::min(bc,64);
        k.setArg(p++,bc);
        k.setArg(p++,work_per_thread);
        k.setArg(p++,srcH);
        k.setArg(p++,srcW);
        k.setArg(p++,tgtH);
        k.setArg(p++,tgtW);
        k.setArg(p++,float(scale_y));
        k.setArg(p++,float(scale_x));
        k.setArg(p++,offset);
        dx.set_arg(k,p);
        dy.set_arg(k,p);
        k.setArg(p++,factor);
        
        int gr_size = (bc + work_per_thread - 1) / work_per_thread;
        cl::NDRange gr(srcH,srcW,gr_size);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,cl::NullRange,e.events(),e.event("nearest_bwd"));
    }


}} // namespace dlprim::core
