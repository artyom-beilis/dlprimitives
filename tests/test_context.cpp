#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>
#include "test.hpp"


namespace dp = dlprim;

void test_shape()
{
    std::cout << "Tesh Shape" << std::endl;
    dp::Shape s(2,3,4);
    TEST(s.unsqueeze(0) == dp::Shape(1,2,3,4));
    TEST(s.unsqueeze(3) == dp::Shape(2,3,4,1));
    TEST(s.unsqueeze(1) == dp::Shape(2,1,3,4));
    TEST(s.unsqueeze(-1) == dp::Shape(2,3,4,1));

    TEST(dp::broadcast(dp::Shape(2,3,4),dp::Shape(3,4)) == dp::Shape(2,3,4));
    TEST(dp::broadcast(dp::Shape(1,3,4),dp::Shape(5,3,1)) == dp::Shape(5,3,4));

    TEST(dp::Shape(3,4).broadcast_strides(dp::Shape(2,3,4)) == dp::Shape(0,4,1));
    TEST(dp::Shape(3,1,1).broadcast_strides(dp::Shape(8,3,32,32)) == dp::Shape(0,1,0,0));
    TEST(dp::Shape(5,1).broadcast_strides(dp::Shape(5,5)) == dp::Shape(1,0));
}

int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "Use paltform:device" << std::endl;
        return 1;
    }
    try {

        std::cout << "Basic context" << std::endl; 

        dp::Context ctx(argv[1]);
        std::cout << ctx.name() << std::endl;
        if(ctx.is_cpu_context()) {
            std::cout << "CPU - exit" << std::endl;
            return 0;
        }
        dp::Tensor a(ctx,dp::Shape(10));

        dp::ExecutionContext q = ctx.make_execution_context();
        dp::Context ctx2(q);
        dp::ExecutionContext q2 = ctx2.make_execution_context();

        if(ctx.is_opencl_context()) {
            TEST(ctx.platform()() == ctx2.platform()());
            TEST(ctx.device()() == ctx2.device()());
        }
    

        float *p = a.data<float>();

        for(unsigned i=0;i<a.shape()[0];i++)
            p[i] = -5.0 + i;
        a.to_device(q2);
        cl::Program const &prg = dp::gpu::Cache::instance().get_program(ctx,"bias","ACTIVATION",int(dp::StandardActivations::relu));
        cl::Kernel k(prg,"activation_inplace");
        int pos=0;
        k.setArg(pos++,int(a.shape().total_size()));
        a.set_arg(k,pos);
        q2.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(a.shape().total_size()),cl::NullRange,nullptr,nullptr);
        a.to_host(q2,false);
        q2.finish();
        for(unsigned i=0;i<a.shape()[0];i++) {
            TEST(p[i] == std::max(0.0,-5.0 + i));
        }
        std::cout << "Ok" << std::endl;

        test_shape();
    }
    catch(std::exception const &e) {
        std::cerr <<"Failed:"<< e.what() << std::endl;
        return 1;
    }
    return 0;

}
