#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/conv.hpp>
#include <iostream>
#include <sstream>
#include <cmath>

namespace dp = dlprim;

struct FlopsStats {
    double flops32;
    double flops16;
    double bps;
};

struct Metrics { double flops,bps; char const *algo="";};

class Benchmarker {
public:
    Benchmarker(std::string const &dev,FlopsStats stats,double scale,dp::DataType dt = dp::float_data) :
        ctx_(dev),
        dt_(dt),
        scale_(scale),
        seed_(0xDEADBEEF),
        seq_(0)
    {
        ref_.flops = dt == dp::float_data ? stats.flops32 : stats.flops16;
        ref_.bps = stats.bps;
        ec_ = ctx_.make_execution_context();
    }


    typedef std::chrono::high_resolution_clock clock_type;
    typedef std::chrono::time_point<clock_type> time_point_type;

    double sec_diff(time_point_type start,time_point_type end)
    {
        return std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
    }

    void run_gemm_bm()
    {
        printf("GEMM\n");
        int setups[][3] = {
            { 512,   512,  512 },
            { 1024, 1024, 1024 },
            { 1025, 1025, 1025 },
            { 2048, 2048, 2048 },
            { 2049, 2049, 2049 },
            { 64,   2048,   64 },
            { 2048,   64, 2048 },
            { 2048, 2048,   64 },
            { 2048,   64,   64 },
            { 64,   2048, 2048 },
            { 64,     64, 2048 }
        };
        for(int ta = 0; ta < 2; ta ++) {
            for(int tb = 0; tb < 2; tb ++) {
                for(unsigned setup = 0;setup < sizeof(setups)/sizeof(setups[0]);setup++) {
                    int M = setups[setup][0];
                    int N = setups[setup][1];
                    int K = setups[setup][2];
                    double approx_flop = double(M) * N * K * 2;
                    double gemm_per_sec = approx_flop / ref_.flops;
                    double target_calls = 2 / gemm_per_sec;
                    int calls = std::max(5,std::min(200,int(target_calls)));
                    int warms = std::max(1,calls / 5);

                    printf("  %c%c %4d, %4d, %4d",
                                (ta ? 'T' : 'N'), (tb ? 'T' : 'N'),
                                M,N,K);

                    fflush(stdout);
                    Metrics m = test_gemm(warms,calls,M,N,K,ta,tb);

                    double flops_per =  m.flops / ref_.flops * 100;
                    double bandw_per =  m.bps / ref_.bps * 100;
                    double max_per = std::max(flops_per,bandw_per);
                    char const *limited = (flops_per > bandw_per) ? "gflops" : "memory";
                    
                    printf("  %8.1f GFlops (%5.2f%%) %8.1f GB/s (%5.2f%%) limited by %s %5.2f%%\n",
                                m.flops * 1e-9, flops_per,
                                m.bps * 1e-9, bandw_per,
                                limited, max_per
                                );
                    fflush(stdout);
                }
            }
        }
    }



    Metrics test_gemm(int warm,int calc,int m,int n,int k,bool ta,bool tb)
    {
        std::unique_ptr<dp::gpu::GEMM> gemm(dp::gpu::GEMM::get_optimal_gemm(ctx_,dt_,ta,tb,
                                                                                 m,n,k));
        

        dp::Tensor A(ctx_,(ta?dp::Shape(k,m):dp::Shape(m,k)),dt_);
        dp::Tensor B(ctx_,(tb?dp::Shape(n,k):dp::Shape(k,n)),dt_);
        dp::Tensor C(ctx_,dp::Shape(m,n),dt_);
        
        rand(A,std::sqrt(1.0/k));
        rand(B,std::sqrt(1.0/k));
        dp::core::fill_tensor(ctx_,ec_,C,0);


        time_point_type start,end;
        for(int i=-warm;i<calc;i++) {
            if(i == 0) {
                ec_.finish();
                start = clock_type::now();
            }
            gemm->gemm(m,n,k,
                        A.device_buffer(),A.device_offset(),A.shape()[1],
                        B.device_buffer(),B.device_offset(),B.shape()[1],
                        C.device_buffer(),C.device_offset(),C.shape()[1],
                        nullptr,0,0.0f,
                        C.shape().total_size(),
                        ec_);
        }
        ec_.finish();
        end = clock_type::now();
        double seconds = sec_diff(start,end);
        double flop = double(m)*n*(k*2-1) * calc;
        double bytes = (A.memory_size() + B.memory_size() + C.memory_size()) * calc;
        Metrics met;
        met.flops = flop / seconds;
        met.bps = bytes / seconds;

        return met;
    }

    void rand(dp::Tensor &t,float sigma)
    {
        dp::core::fill_random(ctx_,ec_,t,seed_,seq_,dp::core::rnd_normal,0,sigma);
        seq_ += (t.shape().total_size() + 3)/4;
    }

    struct ConvBM {
        int kern;
        int pad;
        int stride;
        int groups;
        int c_in;
        int c_out;
        int img_size;
        char const *type;
    };


    Metrics test_conv(int warm,int calc,int op,int batch,ConvBM const &bm)
    {
        dp::Convolution2DConfigBase cfg;
        cfg.channels_in = bm.c_in;
        cfg.channels_out = bm.c_out;
        cfg.kernel[0] = cfg.kernel[1] = bm.kern;
        cfg.stride[0] = cfg.stride[1] = bm.stride;
        cfg.pad[0] = cfg.pad[1] = bm.pad;
        cfg.groups = bm.groups;
        dp::core::Conv2DSettings config(cfg,dp::Shape(batch,bm.c_in,bm.img_size,bm.img_size),dt_);
        std::unique_ptr<dp::core::Conv2DBase> conv;
        switch(op) {
        case dp::forward_data:
            conv = std::move(dp::core::Conv2DForward::create(ctx_,config,false));
            break;
        case dp::backward_data:
            conv = std::move(dp::core::Conv2DBackwardData::create(ctx_,config));
            break;
        case dp::backward_param:
            conv = std::move(dp::core::Conv2DBackwardFilter::create(ctx_,config));
            break;
        default:
            throw std::runtime_error("Invalid conv op");
        }
        dp::Shape in_shape = dp::Shape(batch,bm.c_in,bm.img_size,bm.img_size);
        dp::Shape out_shape = dp::core::Conv2DBase::get_output_shape(config,in_shape);
        std::ostringstream ss;
        dp::Tensor X(ctx_,in_shape,dt_);
        dp::Tensor Y(ctx_,out_shape,dt_);
        dp::Tensor M(ctx_,dp::Shape(bm.c_out,bm.c_in/bm.groups,bm.kern,bm.kern),dt_);
        ss << Y.shape();
        rand(X,1.0);
        rand(Y,1.0);
        rand(M,1.0/(bm.kern*bm.kern*bm.c_in/bm.groups));
        dp::Tensor ws;
        size_t ws_size = conv->workspace();
        if(ws_size > 0)
            ws = dp::Tensor(ctx_,dp::Shape(ws_size),dp::uint8_data);

        time_point_type start,end;
        for(int i=-warm;i<calc;i++) {
            if(i == 0) {
                ec_.finish();
                start = clock_type::now();
            }
            switch(op) {
                case dp::forward_data:
                    static_cast<dp::core::Conv2DForward &>(*conv).enqueue(X,M,nullptr,Y,ws,0.0,ec_);
                    break;
                case dp::backward_data:
                    static_cast<dp::core::Conv2DBackwardData &>(*conv).enqueue(X,M,Y,ws,0.0,ec_);
                    break;
                case dp::backward_param:
                    static_cast<dp::core::Conv2DBackwardFilter &>(*conv).enqueue(X,M,Y,ws,1.0,ec_);
                    break;
            }
        }
        ec_.finish();
        end = clock_type::now();
        double seconds = sec_diff(start,end);
        double flop = 2 * Y.shape().total_size() * (bm.c_in / bm.groups * bm.kern * bm.kern) * calc;
        double bytes = X.memory_size() + Y.memory_size() + M.memory_size();
        if(op == dp::backward_param)
            bytes += M.memory_size();
        bytes*=calc;
        Metrics met;
        met.flops = flop / seconds;
        met.bps = bytes / seconds;
        met.algo = conv->algo();

        return met;
    }

    void run_conv_bm()
    {
        printf("Convolution\n");
        ConvBM setups[] = {
            //  k   p   s   g  in out dim net 
            {  11,  2,  4,  1,  3, 64,224,"alexnet" },
            {   5,  2,  1,  2, 96,192, 27,"alexnet" },
            {   5,  2,  1,  1, 64,192, 27,"alexnet" },
            {   3,  1,  1,  1,384,256, 13,"alexnet" },
            
            {   7,  3,  2,  1,  3, 64,224,"resnet" },
            {   1,  0,  1,  1, 64,256, 56,"resnet" },
            {   1,  0,  1,  1, 64, 64, 56,"resnet" },
            {   3,  1,  1,  1, 64, 64, 56,"resnet" },
            {   1,  0,  2,  1,1024,2048, 14,"resnet" },
            {   1,  0,  1,  1,1024,256, 14,"resnet" },
            {   3,  1,  1,  1,256,256, 14,"resnet" },
            
            {   3,  1,  1,  1,  3, 64,224,"vgg" },
            {   3,  1,  1,  1, 64, 64,224,"vgg" },
            {   3,  1,  1,  1,512,512, 28,"vgg" },
           
            {   3,  1,  2,  1,  3, 32,224,"mobile" },
            {   3,  1,  1,144,144,144, 56,"mobile" },
            {   3,  1,  2,144,144,144, 56,"mobile" },
            {   1,  0,  1,  1,144, 24, 56,"mobile" },
            {   1,  0,  1,  1, 24,144, 56,"mobile" },
            {   1,  0,  1,  1,960,160,  7,"mobile" },
            {   1,  0,  1,  1,960,320,  7,"mobile" },
            {   3,  1,  1,960,960,960,  7,"mobile" },
#ifdef FULL_TEST 
            {   3,  1,  1,  1,192,384, 13,"alexnet" },
            {   3,  1,  1,  1,256,256, 13,"alexnet" },
            {   3,  1,  1,  2,384,256, 13,"alexnet" },
            {  11,  2,  4,  1,  3, 96,224,"alexnet" },

            {   1,  0,  1,  1,1024,512, 14,"resnet" },
            {   3,  1,  1,  1,128,128, 28,"resnet" },
            {   3,  1,  2,  1,128,128, 56,"resnet" },
            {   1,  0,  1,  1,128,512, 28,"resnet" },
            {   1,  0,  1,  1,2048,512,  7,"resnet" },
            {   1,  0,  1,  1,256,1024, 14,"resnet" },
            {   1,  0,  1,  1,256,128, 56,"resnet" },
            {   3,  1,  2,  1,256,256, 28,"resnet" },
            {   1,  0,  2,  1,256,512, 56,"resnet" },
            {   1,  0,  1,  1,256, 64, 56,"resnet" },
            {   1,  0,  2,  1,512,1024, 28,"resnet" },
            {   1,  0,  1,  1,512,128, 28,"resnet" },
            {   1,  0,  1,  1,512,2048,  7,"resnet" },
            {   1,  0,  1,  1,512,256, 28,"resnet" },
            {   3,  1,  1,  1,512,512,  7,"resnet" },
            {   3,  1,  2,  1,512,512, 14,"resnet" },
            {   3,  1,  1,  1,128,128,112,"vgg" },
            {   3,  1,  1,  1,128,256, 56,"vgg" },
            {   3,  1,  1,  1,256,256, 56,"vgg" },
            {   3,  1,  1,  1,256,512, 28,"vgg" },
            {   3,  1,  1,  1,512,512, 14,"vgg" },
            {   3,  1,  1,  1, 64,128,112,"vgg" },
            {   1,  0,  1,  1,144, 32, 28,"mobile" },
            {   1,  0,  1,  1,160,960,  7,"mobile" },
            {   1,  0,  1,  1, 16, 96,112,"mobile" },
            {   3,  1,  1,192,192,192, 28,"mobile" },
            {   3,  1,  2,192,192,192, 28,"mobile" },
            {   1,  0,  1,  1,192, 32, 28,"mobile" },
            {   1,  0,  1,  1,192, 64, 14,"mobile" },
            {   1,  0,  1,  1,320,1280,  7,"mobile" },
            {   1,  0,  1,  1, 32, 16,112,"mobile" },
            {   1,  0,  1,  1, 32,192, 28,"mobile" },
            {   3,  1,  1, 32, 32, 32,112,"mobile" },
            {   3,  1,  1,384,384,384, 14,"mobile" },
            {   1,  0,  1,  1,384, 64, 14,"mobile" },
            {   1,  0,  1,  1,384, 96, 14,"mobile" },
            {   1,  0,  1,  1,576,160,  7,"mobile" },
            {   3,  1,  1,576,576,576, 14,"mobile" },
            {   3,  1,  2,576,576,576, 14,"mobile" },
            {   1,  0,  1,  1,576, 96, 14,"mobile" },
            {   1,  0,  1,  1, 64,384, 14,"mobile" },
            {   1,  0,  1,  1, 96, 24, 56,"mobile" },
            {   1,  0,  1,  1, 96,576, 14,"mobile" },
            {   3,  1,  2, 96, 96, 96,112,"mobile" }
#endif      
        };
#ifdef FULL_TEST 
        int batches[]={16,64,128};
#else        
        int batches[]={64};
#endif        
        for(unsigned bi = 0;bi <sizeof(batches)/sizeof(batches)[0];bi++) {
            int batch = batches[bi];
            for(unsigned setup = 0;setup < sizeof(setups)/sizeof(setups[0]);setup++) {
                ConvBM bm = setups[setup];
                int out_size = (bm.img_size + 2*bm.pad - bm.kern) / bm.stride + 1;
                double approx_flop = 2 * (double(out_size)*out_size * bm.c_out*batch) * (bm.c_in / bm.groups * bm.kern * bm.kern);
                double sec_per_gemm = approx_flop / ref_.flops;
                double target_calls = 2 / sec_per_gemm;
                int calls = std::max(5,std::min(200,int(target_calls)));
                int warms = std::max(1,calls / 5);
                char const *op_name[]={"nop","forward","bwd-data","bwd-filt"};
                for(int op = dp::forward_data;op<=dp::backward_param;op++) {
                    printf("  %10s %8s b=%-2d k=%-2d p=%d s=%d in=%-4d out=%-4d g=%-3d D=%-3d",
                                bm.type,op_name[op],batch,bm.kern,bm.pad,bm.stride,bm.c_in,bm.c_out,bm.groups,bm.img_size);
                    fflush(stdout);
                    Metrics m = test_conv(warms,calls,op,batch,bm);

                    double flops_per =  m.flops / ref_.flops * 100;
                    double bandw_per =  m.bps / ref_.bps * 100;
                    double max_per = std::max(flops_per,bandw_per);
                    char const *limited = (flops_per > bandw_per) ? "gflops" : "memory";
                    
                    printf("  %8.1f GFlops (%5.2f%%) %8.1f GB/s (%5.2f%%) limited by %s %5.2f%% algo=%s\n",
                                m.flops * 1e-9, flops_per,
                                m.bps * 1e-9, bandw_per,
                                limited, max_per,
                                m.algo
                                );
                    fflush(stdout);
                }
            }
        }
    }


private:
    dp::Context ctx_;
    dp::ExecutionContext ec_;
    dp::DataType dt_;
    Metrics ref_;
    double scale_;
    cl_ulong seed_,seq_;
};


FlopsStats get_flops(std::string device, double scale)
{
    FlopsStats stats = FlopsStats();
    int N=256*int(1024*scale);
    dp::Context ctx(device);
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
                    stats.bps = 2 * mem_size / secs;
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
    stats.flops32 = peaks[0] * 1e9;
    if(peaks.size() > 1)
        stats.flops16 = peaks[1] * 1e9;
    return stats; 
}

int main(int argc,char **argv)
{
    if(argc < 2) {
        std::cerr << "Usage flops PLAT:DEV [mpl]" << std::endl;
        std::cerr << " mpl is multipier how much bigger/smaller buffers/duration to calculate\n"
                     " For example dlprim_flops 0:0 0.5\n";
        return 1;
    }
    double scale = 1;
    if(argc >= 3) {
        scale = atof(argv[2]);
    }
    FlopsStats fs = get_flops(argv[1],scale);
    Benchmarker bm(argv[1],fs,scale);
    bm.run_gemm_bm();
    bm.run_conv_bm();
}

