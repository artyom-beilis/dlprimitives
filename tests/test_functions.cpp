#include <dlprim/functions.hpp>
#include <stdlib.h>
#include <math.h>

#include "test.hpp"
#include <iostream>

namespace dp = dlprim;

void fill_random(dp::Tensor &t)
{
    float *p = t.data<float>();
    for(size_t i=0;i<t.shape().total_size();i++) {
        p[i]=random()*2.0f / RAND_MAX - 1.0f;
    }
}

void test_softmax(dp::Context &ctx)
{
    std::cout << "- Testing SoftMax" << std::endl;
    cl::CommandQueue q;
    dp::ExecutionContext e;
    if(ctx.is_gpu_context()) {
        q=cl::CommandQueue(ctx.context(),ctx.device());
        e=dp::ExecutionContext(q);
    }
    
    dp::SoftMax sm(ctx);
    std::vector<dp::TensorSpecs> in{dp::Shape(50,100)},out;
    size_t ws=10000;
    sm.setup(in,out,ws);
    TEST(out.size() == 1);
    TEST(out[0].shape() == in[0].shape());
    TESTEQ(ws,0);
    for(int features = 1; features < 1500; features = features % 2 == 0 ? features *2 - 1 : features + 1) {
        for(int bs = 1;bs < 100;bs = bs*2+1) {
            std::cout << "-- Testing for BS=" << bs << " features=" << features << std::endl;
            std::vector<dp::Shape> in_sh{dp::Shape(bs,features)},out_sh;
            sm.reshape(in_sh,out_sh);
            TEST(in_sh[0] == out_sh[0]);
            dp::Tensor a(ctx,dp::Shape(bs,features));
            dp::Tensor b(ctx,dp::Shape(bs,features));
            std::vector<dp::Tensor> va({a});
            std::vector<dp::Tensor> vb({b});
            fill_random(a);
            if(ctx.is_cpu_context()) {
                sm.forward(va,vb,e);
            }
            else {
                a.to_device(e,false);
                sm.forward(va,vb,e);
                b.to_host(e);
            }
            float *src = a.data<float>();
            float *res = b.data<float>();
            for(int i=0;i<bs;i++) {
                float s=0;
                for(int j=0;j<features;j++){
                    s+=res[i*features + j];
                }
                TEST(fabs(s - 1.0f) < 0.05f);
                int max_a_index = 0;
                int max_b_index = 0;
                for(int j=1;j<features;j++) {
                    if(src[i*features + max_a_index] < src[i*features + j])
                        max_a_index = j;
                    if(res[i*features + max_b_index] < res[i*features + j])
                        max_b_index = j;
                }
                TEST(max_a_index == max_b_index);
            }
        }
    }
}

void test_eltwise_referece()
{
    std::cout << "- Test etltwise reference " << std::endl;
    float da[]={1.2f, 1.0f,-1.0f };
    float db[]={0.5f, 0.0f,-0.5f };
    dp::Context ctx;
    dp::ElementwiseConfig cfg;
    cfg.coeff[0] = 2.0f;
    cfg.coeff[1] = -1.0f;
    
    dp::Tensor a(ctx,dp::Shape(3));
    dp::Tensor b(ctx,dp::Shape(3));
    dp::Tensor c(ctx,dp::Shape(3));
    
    memcpy(a.data<float>(),da,3*sizeof(float));
    memcpy(b.data<float>(),db,3*sizeof(float));
    std::vector<dp::TensorSpecs> tsin = {dp::TensorSpecs(dp::Shape(3)),dp::TensorSpecs(dp::Shape(3))};
    std::vector<dp::TensorSpecs> tsout;
    size_t ws;
    typedef std::vector<dp::Tensor> tv_type;
    {
        dp::Elementwise elt(ctx,cfg);
        elt.setup(tsin,tsout,ws);
        tv_type vi={a,b};
        tv_type vo={c};
        elt.forward(vi,vo,dp::ExecutionContext());
        float *r=c.data<float>();
        TEST(r[0] == 1.2f * 2.0f - 0.5f);
        TEST(r[1] == 2.0f);
        TEST(r[2] == -2.0f + 0.5f);
    }
    {
        cfg.op = dp::ElementwiseConfig::elementwise_max;
        dp::Elementwise elt(ctx,cfg);
        elt.setup(tsin,tsout,ws);
        tv_type vi={a,b};
        tv_type vo={c};
        elt.forward(vi,vo,dp::ExecutionContext());
        float *r=c.data<float>();
        TEST(r[0] == 1.2f * 2.0f);
        TEST(r[1] == 2.0f);
        TEST(r[2] == 0.5f);
    }
    {
        cfg.op = dp::ElementwiseConfig::elementwise_prod;
        dp::Elementwise elt(ctx,cfg);
        elt.setup(tsin,tsout,ws);
        tv_type vi={a,b};
        tv_type vo={c};
        elt.forward(vi,vo,dp::ExecutionContext());
        float *r=c.data<float>();
        TEST(r[0] == -1.2f);
        TEST(r[1] == 0.0f);
        TEST(r[2] == -1.0f);
    }
    {
        cfg.op = dp::ElementwiseConfig::elementwise_sum;
        cfg.activation = dp::StandardActivations::relu;
        dp::Elementwise elt(ctx,cfg);
        elt.setup(tsin,tsout,ws);
        tv_type vi={a,b};
        tv_type vo={c};
        elt.forward(vi,vo,dp::ExecutionContext());
        float *r=c.data<float>();
        TEST(r[0] == 1.2f * 2.0f - 0.5f);
        TEST(r[1] == 2.0f);
        TEST(r[2] == 0.0f);
    }
    
}

void test_eltwise(dp::Context &ctx)
{
    std::cout << "- Testing Eltwise" << std::endl;
    cl::CommandQueue q;
    dp::ExecutionContext e;
    dp::Context cpu_ctx;
    if(ctx.is_gpu_context()) {
        q=cl::CommandQueue(ctx.context(),ctx.device());
        e=dp::ExecutionContext(q);
    }
    
    dp::ElementwiseConfig::Operation ops[] = { 
        dp::ElementwiseConfig::elementwise_sum,
        dp::ElementwiseConfig::elementwise_prod,
        dp::ElementwiseConfig::elementwise_max
    };
    
    dp::StandardActivations acts[] = {
        dp::StandardActivations::identity,
        dp::StandardActivations::relu
    };
    
    for(dp::ElementwiseConfig::Operation op : ops) {
        for(dp::StandardActivations act: acts) {
            std::cout << "-- op=" << int(op) << " act=" << int(act) << std::endl;
            dp::ElementwiseConfig cfg;
            cfg.coeff[0] = float(rand()) / RAND_MAX - 0.5f;
            cfg.coeff[1] = float(rand()) / RAND_MAX - 0.5f;
            cfg.op = op;
            cfg.activation = act;
            
        
            dp::Elementwise elt(ctx,cfg);
            dp::Elementwise elt_ref(cpu_ctx,cfg);
            std::vector<dp::TensorSpecs> in{dp::Shape(50,100),dp::Shape(50,100)},out;
            size_t ws=10000;
            elt.setup(in,out,ws);
            TEST(out.size() == 1);
            TEST(out[0].shape() == in[0].shape());
            TEST(ws == 0);
            elt_ref.setup(in,out,ws);
            for(int size = 1; size < 15000; size = size % 2 == 0 ? size *2 - 1 : size + 1) {
                std::cout << "--- Testing for size =" << size << std::endl;
                std::vector<dp::Shape> in_sh{dp::Shape(size),dp::Shape(size)},out_sh;
                elt.reshape(in_sh,out_sh);
                TEST(in_sh[0] == out_sh[0]);
                elt_ref.reshape(in_sh,out_sh);
                dp::Tensor a(ctx,dp::Shape(size));
                dp::Tensor b(ctx,dp::Shape(size));
                dp::Tensor c(ctx,dp::Shape(size));
                dp::Tensor cref(cpu_ctx,dp::Shape(size));
                std::vector<dp::Tensor> vin({a,b});
                std::vector<dp::Tensor> vout({c});
                std::vector<dp::Tensor> vref({cref});
                fill_random(a);
                fill_random(b);
                elt_ref.forward(vin,vref,dp::ExecutionContext());
                if(ctx.is_cpu_context()) {
                    elt.forward(vin,vout,e);
                }
                else {
                    a.to_device(e,false);
                    b.to_device(e,false);
                    elt.forward(vin,vout,e);
                    c.to_host(e);
                }
                float *ref = cref.data<float>();
                float *res = c.data<float>();
                for(int i=0;i<size;i++) {
                    TESTEQF(res[i],ref[i],1e-5f);
                }
            }
        }
    }
}


int main(int argc,char **argv)
{
    try {
        std::string device = argc > 1 ? argv[1] : "cpu";
        dp::Context ctx(device);
        std::cout << "Using " << ctx.name() << std::endl;
        test_softmax(ctx);
        test_eltwise_referece();
        test_eltwise(ctx);
        return 0;
    }
    catch(std::exception const &e) {
        std::cerr << "Failed " << e.what() << std::endl ;
        return 1;
    }

}
