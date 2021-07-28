#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dp = dlprim;
int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "Use paltform:device" << std::endl;
        return 1;
    }
    try {

        dp::Context ctx(argv[1]);
        std::cerr << ctx.name() << std::endl;
        dp::Tensor a(ctx,dp::Shape(10));

        cl::CommandQueue q(ctx.context(),ctx.device());
    

        float *p = a.data<float>();

        for(unsigned i=0;i<a.shape()[0];i++)
            p[i] = -5.0 + i;
        a.to_device(q);
        cl::Program const &prg = dp::gpu::Cache::instance().get_program(ctx,"bias","ACTIVATION",int(dp::StandardActivations::relu));
        //cl::Program const &prg = dp::gpu::Cache::build_program(ctx,"bias","ACTIVATION",int(dp::StandardActivations::relu));
        cl::Kernel k(prg,"activation_inplace");
        int pos=0;
        k.setArg(pos++,int(a.shape().total_size()));
        a.set_arg(k,pos);
        q.enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(a.shape().total_size()),cl::NullRange,nullptr,nullptr);
        a.to_host(q,false);
        q.finish();
        for(unsigned i=0;i<a.shape()[0];i++) {
            std::cerr << p[i] << std::endl;
        }
    }
    catch(std::exception const &e) {
        std::cerr <<"Failed:"<< e.what() << std::endl;
    }

    return 0;

}
