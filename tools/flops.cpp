#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dp = dlprim;

int main(int argc,char **argv)
{
    if(argc < 2) {
        std::cerr << "Usage flops PLAT:DEV" << std::endl;
        return 1;
    }
    double scale = 1;
    if(argc >= 3) {
        scale = atof(argv[2]);
    }
    int N=256*int(1024*scale);
    dp::Context ctx(argv[1]);
    auto q = ctx.make_queue();
    std::cout << "Testing on " << ctx.name() << std::endl;
    dp::Tensor t(ctx,dp::Shape(N));
    long int mem_size = int(1024*scale)*1024l*256;
    dp::Tensor halfG(ctx,dp::Shape(mem_size/4));
    int float16 = ctx.check_device_extension("cl_khr_fp16");
    std::vector<float> peaks(1+float16);
    double max_gb = 0;
    for(int half =0;half < 1 + float16;half++) {
        try {
            cl::Program const &prog = dp::gpu::Cache::instance().get_program(ctx,"benchmark","USE_HALF",half);
            cl::Kernel k1(prog,"flops_v1");
            cl::Kernel k2(prog,"flops_v2");
            cl::Kernel k4(prog,"flops_v4");
            cl::Kernel k8(prog,"flops_v8");
            cl::Kernel k16(prog,"flops_v16");
            
            if(half == 0) {
                std::cout << "Testing memory speed" << std::endl;
                for(int d=1;d<=16;d*=2) {
                    cl::Kernel ms(prog,("memspeed_v" + std::to_string(d)).c_str());
                    ms.setArg(0,halfG.device_buffer());
                    std::cout << "- Vector size " << d << std::endl;
                    std::cout << "-- Warming " << std::endl;
                    q.enqueueNDRangeKernel(ms,cl::NullRange,cl::NDRange(mem_size/4/d),cl::NullRange,nullptr,nullptr);
                    q.finish();
                    std::cout << "-- Running " << std::flush;
                    auto start = std::chrono::high_resolution_clock::now();
                    q.enqueueNDRangeKernel(ms,cl::NullRange,cl::NDRange(mem_size/4/d),cl::NullRange,nullptr,nullptr);
                    q.finish();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto secs = std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
                    double gbs = 2 * mem_size / secs * 1e-9; // read+write 
                    std::cout << "  " << gbs << " GB/s" << std::endl;
                    max_gb = std::max(max_gb,gbs);
                }
            }
 
            std::cout << "Testing flops " << (half ? "half" : "float") << std::endl;

            std::vector<cl::Kernel *> ks={&k1,&k2,&k4,&k8,&k16};
            for(unsigned i=0,vs=1;i<ks.size();i++,vs*=2)  {
                cl::Kernel &k=*ks[i];
                k.setArg(0,t.device_buffer());


                std::cout << "- Vector size " << vs << std::endl;
                std::cout << "-- Warming " << std::endl;
                q.enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(N),cl::NullRange,nullptr,nullptr);
                q.finish();
                std::cout << "-- Running " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();
                q.enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(N),cl::NullRange,nullptr,nullptr);
                q.finish();
                auto end = std::chrono::high_resolution_clock::now();
                auto secs = std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
                double flops = N * 4.0 * 10000 * vs * 2; // matrix 4 mads * float4 * 1000 * N
                float gflops = (flops / secs * 1e-9);
                std::cout << "  " << gflops << " GFlops" << std::endl;
                peaks[half]=std::max(peaks[half],gflops);
            }
        }
        catch(std::exception const &e) {
            std::cout << "Failed to run for float" << (16*(2-half)) << std::endl;
            std::cout << e.what() << std::endl;
        }
    }
    std::cout << "Summray for " << ctx.name() << std::endl;
    for(size_t half=0;half<peaks.size();half++) {
        std::cout << "Peak GFlops for " << (half ? "half" : "float") << " " << peaks[half] << std::endl;
    }
    std::cout << "Peak memory " << max_gb << " GB/s" << std::endl;
 
}

