///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/pointwise.hpp>
#include <iostream>
#include <sstream>
#include <cmath>

#ifdef CUDA_TEST
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>
#endif

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
        report_ = fopen("report.csv","w");
		#ifndef DLPRIM_WINDOWS
        setvbuf(report_,nullptr,_IOLBF,0);
		#else
        setvbuf(report_,nullptr,_IONBF,0);
		#endif
        fprintf(report_,"Float GFlops,Half GFlops,GB/s,Device\n%1.1f,%1.1f,%1.2f,%s\n",
                stats.flops32*1e-9,stats.flops16*1e-9,stats.bps*1e-9,ctx_.name().c_str());
    }
    ~Benchmarker()
    {
        fclose(report_);
    }


    typedef std::chrono::high_resolution_clock clock_type;
    typedef std::chrono::time_point<clock_type> time_point_type;

    double sec_diff(time_point_type start,time_point_type end)
    {
        return std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
    }

    void run_gemm_bm(int index=-1)
    {
        printf("GEMM\n");
        fprintf(report_,"GEMM\n"
            "ID,Tr(A),Tr(B),M,N,K,GFlops,GFlops %%,GB/s,GB/s%%\n");
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
                    if(index != -1 && index != int(setup))
                        continue;
                    int M = setups[setup][0];
                    int N = setups[setup][1];
                    int K = setups[setup][2];
                    double approx_flop = double(M) * N * K * 2;
                    double gemm_per_sec = approx_flop / ref_.flops;
                    double target_calls = 2 / gemm_per_sec;
                    int calls = std::max(5,std::min(200,int(target_calls)));
                    int warms = std::max(1,calls / 5);

                    printf("  %c%c %2d: %4d, %4d, %4d ",
                                (ta ? 'T' : 'N'), (tb ? 'T' : 'N'),
                                setup,
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
                    fprintf(report_,"%d,%c,%c,%d,%d,%d,%1.1f,%1.2f%%,%1.2f,%1.2f\n",
                                setup,
                                (ta ? 'T' : 'N'), (tb ? 'T' : 'N'),
                                M,N,K,
                                m.flops * 1e-9, flops_per,
                                m.bps * 1e-9, bandw_per);
                }
            }
        }
    }

    std::string to_string(dp::Shape const &s)
    {
        std::ostringstream ss;
        ss<<s;
        return ss.str();
    }

    void run_br_bm(int index=-1)
    {
        printf("Broadcast/Reduce\n");
        fprintf(report_,"BC/RD\n"
            "ID,type,A,B,GFlops,GFlops %%,GB/s,GB/s%%\n");
        dp::DataType types[3]={dp::float_data,dp::int64_data,dp::int16_data};
        using dlprim::Shape;
        dp::Shape setups[][3] = {
            { Shape(64,512,24,24),  Shape(64,512,24,24),    Shape(64,512,24,24) },
            { Shape(64,512,24,24),  Shape(512,1,1),         Shape(64,512,24,24) },
            { Shape(64,512,24,24),  Shape(1,512,1,1),       Shape(1,512,1,1) },
            { Shape(64,512,24,24),  Shape(64,512,24,24),    Shape(1,512,1,1) },
            { Shape(64,512,24,24),  Shape(64,512,24,24),    Shape(64,1,1,1) },
            { Shape(256,1000),      Shape(256,1),           Shape(1) },
        };
        for(dp::DataType dtype: types) {
            for(unsigned setup = 0;setup < sizeof(setups)/sizeof(setups[0]);setup++) {
                if(index != -1 && index != int(setup))
                    continue;
                Shape A = setups[setup][0];
                Shape B = setups[setup][1];
                Shape C = setups[setup][2];
                double total = double(A.total_size()) 
                               + double(B.total_size())
                               + double(C.total_size());
                total *= dp::size_of_data_type(dtype);
                double op_per_sec = total / ref_.bps;
                double target_calls = 2 / op_per_sec;
                int calls = std::max(5,std::min(200,int(target_calls)));
                int warms = std::max(1,calls / 5);
                

                printf("  %8s %-15s %-15s %-15s ",
                            dp::data_type_to_opencl_type(dtype).c_str(),
                            to_string(A).c_str(),
                            to_string(B).c_str(),
                            to_string(C).c_str());

                fflush(stdout);
                Metrics m = test_broadcast_reduce(warms,calls,A,B,C,dtype);

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
                fprintf(report_,"%s,%s,%s,%10s,",
                            dp::data_type_to_opencl_type(dtype).c_str(),
                            to_string(A).c_str(),
                            to_string(B).c_str(),
                            to_string(C).c_str());
                fprintf(report_,"%8.1f,%8.1f\n",
                            m.flops * 1e-9,
                            m.bps * 1e-9
                            );

            }
        }
    }


    Metrics test_broadcast_reduce(int warm,int calc,dp::Shape a,dp::Shape b,dp::Shape c,dp::DataType dt)
    {
        
        dp::Tensor A(ctx_,a,dt);
        dp::Tensor B(ctx_,b,dt);
        dp::Tensor C(ctx_,c,dt);
        
        rand(A,1);
        rand(B,1);
        dp::core::fill_tensor(C,0,ec_);

        auto op = dp::core::PointwiseOperationBroadcastReduce::create(ctx_,
                    {A.specs(),B.specs()},{C.specs()},
                    1,dt,
                    "y0 = x0 + w0 * x1;",
                    "reduce_y0 = 0;",
                    "reduce_y0 += y0;");
        dp::Tensor ws;
        if(op->workspace()) {
            ws = dp::Tensor(ctx_,dp::Shape(op->workspace()),dp::uint8_data);
        }
        
        time_point_type start,end;
        for(int i=-warm;i<calc;i++) {
            if(i == 0) {
                ec_.finish();
                start = clock_type::now();
            }
            op->enqueue({A,B},{C},ws,{2},{1.0},{0.0},ec_);
        }
        ec_.finish();
        end = clock_type::now();
        double seconds = sec_diff(start,end);
        double total_shape = dp::broadcast(A.shape(),B.shape()).total_size();
        double reductions = total_shape / C.shape().total_size();
        double flop = total_shape*2 + (reductions - 1)  * C.shape().total_size();
        double bytes = A.memory_size() + B.memory_size() + C.memory_size();
        Metrics met;
        met.flops = flop  * calc / seconds;
        met.bps   = bytes * calc / seconds;

        return met;
    }
    void rand(dp::Tensor &t,float sigma)
    {
        if(t.dtype() == dp::float_data || t.dtype() == dp::half_data || t.dtype() == dp::bfloat16_data) {
            dp::core::fill_random(t,seed_,seq_,dp::core::rnd_normal,0,sigma,ec_);
            seq_ += (t.shape().total_size() + 3)/4;
        }
        else {
            size_t n=t.shape().total_size();
            if(t.dtype() == dp::int64_data) {
                int64_t *p=t.data<int64_t>();
                for(size_t i=0;i<n;i++)
                    p[i] = i%17;
            }
            else if(t.dtype() == dp::int16_data) {
                int16_t *p=t.data<int16_t>();
                for(size_t i=0;i<n;i++)
                    p[i] = i%17;
            }
            t.to_device(ec_);
        }
    }

#ifdef CUDA_TEST
    Metrics test_gemm(int warm,int calc,int m,int n,int k,bool ta,bool tb)
    {
        static bool init = false;
        if(!init) {
            cublasInit();
            init=true;
        }
        cublasHandle_t h;
        int status;
        if((status=cublasCreate(&h))!=0)
    		throw std::runtime_error(std::string("Failed to create cublas:") + std::to_string(status));

        float *A,*B,*C;
        cudaMalloc((void**)&A,m*k*sizeof(float));
        cudaMalloc((void**)&B,k*n*sizeof(float));
        cudaMalloc((void**)&C,m*n*sizeof(float));

        rand(A,m*k,std::sqrt(1.0/k));
        rand(B,n*k,std::sqrt(1.0/k));
        zero(C,m*n);        

        time_point_type start,end;
        for(int i=-warm;i<calc;i++) {
            if(i == 0) {
                cudaDeviceSynchronize();
                start = clock_type::now();
            }
            cublasStatus_t status;
            float alpha = 1.0f;
            float beta = 0.0f;
            status = cublasSgemm(h,(tb ? CUBLAS_OP_T : CUBLAS_OP_N), (ta ? CUBLAS_OP_T : CUBLAS_OP_N),
                n,m,k,
                &alpha,
                B,(!tb?n:k),
                A,(!ta?k:m),
                &beta,
                C,n);
            if(status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("sgemm");
        }
        cudaDeviceSynchronize();
        end = clock_type::now();
        double seconds = sec_diff(start,end);
        double flop = double(m)*n*(k*2-1) * calc;
        double bytes = ((size_t)m*k + (size_t)k*n + (size_t)m*n) * sizeof(float) * calc;
        Metrics met;
        met.flops = flop / seconds;
        met.bps = bytes / seconds;

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        return met;

    }
    void rand(void *p,size_t N,float sigma)
    {
        std::vector<float> v(N);
        for(size_t i=0;i<N;i++)
            v[i]=((double)random()/RAND_MAX - 0.5) * 2 * sigma;
        cudaMemcpy(p,v.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    }
    void zero(void *p,size_t N)
    {
        cudaMemset(p,0,sizeof(float)*N);
    }
#else

    Metrics test_gemm(int warm,int calc,int m,int n,int k,bool ta,bool tb)
    {
        std::unique_ptr<dp::gpu::GEMM> gemm(dp::gpu::GEMM::get_optimal_gemm(ctx_,dt_,ta,tb,
                                                                                 m,n,k));
        

        dp::Tensor A(ctx_,(ta?dp::Shape(k,m):dp::Shape(m,k)),dt_);
        dp::Tensor B(ctx_,(tb?dp::Shape(n,k):dp::Shape(k,n)),dt_);
        dp::Tensor C(ctx_,dp::Shape(m,n),dt_);
        
        rand(A,std::sqrt(1.0/k));
        rand(B,std::sqrt(1.0/k));
        dp::core::fill_tensor(C,0,ec_);


        time_point_type warmp,start,end;
        for(int i=-warm - 1;i<calc;i++) {
            if(i == -warm) {
                ec_.finish();
                warmp = clock_type::now();
            }
            if(i == 0) {
                ec_.finish();
                start = clock_type::now();
                double time_per_kern = sec_diff(warmp,start) / warm;
                calc = std::max(5,int(ceil(1/time_per_kern)));
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
#endif

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

#ifdef CUDA_TEST

#define check(expression)                               \
do{                                                          \
	cudnnStatus_t status = (expression);                     \
	if (status != CUDNN_STATUS_SUCCESS) {                    \
		std::ostringstream ss; ss << "Error on line " << __LINE__ << ": "      \
		<< cudnnGetErrorString(status) << std::endl; \
		throw std::runtime_error(ss.str()); \
	}                                                        \
}while(0)

    template<typename T>
    void get_first(T *v,int n)
    {
        static size_t constexpr mem_limit = 1024ull*1024*256;
        for(int i=0;i<n;i++) {
            if(v[i].memory < mem_limit) {
                v[0] = v[i];
                return;
            }
        }
        v[0]=v[n-1];
    }
    
    Metrics test_conv(int warm,int calc,int op,int batch,ConvBM const &bm)
    {
        cudnnHandle_t handle;
        cudnnConvolutionDescriptor_t desc;
		check(cudnnCreate(&handle));
        cudnnTensorDescriptor_t X_d,Y_d;
        cudnnFilterDescriptor_t M_d;
		check(cudnnCreateTensorDescriptor(&X_d));
		check(cudnnCreateFilterDescriptor(&M_d));
        
        check(cudnnSetTensor4dDescriptor(X_d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch,bm.c_in,bm.img_size,bm.img_size));
        check(cudnnSetFilter4dDescriptor(M_d,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,bm.c_out,bm.c_in/bm.groups,bm.kern,bm.kern));
        
        check(cudnnCreateConvolutionDescriptor(&desc));
        check(cudnnSetConvolution2dDescriptor(desc,bm.pad,bm.pad,bm.stride,bm.stride,
                    1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        check(cudnnSetConvolutionGroupCount(desc,bm.groups));

        dp::Shape in_shape(batch,bm.c_in,bm.img_size,bm.img_size);
        dp::Shape k_shape(bm.c_out,bm.c_in/bm.groups,bm.kern*bm.kern);
        int shape[4]={};
        check(cudnnGetConvolution2dForwardOutputDim(desc,X_d,M_d,&shape[0],&shape[1],&shape[2],&shape[3]));
        int o_size = shape[2];
        //int o_size = 1 + ( bm.img_size + 2*bm.pad - bm.kern )/bm.stride;
        dp::Shape out_shape(batch,bm.c_out,o_size,o_size);
		check(cudnnCreateTensorDescriptor(&Y_d));
        check(cudnnSetTensor4dDescriptor(Y_d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch,bm.c_out,o_size,o_size));


        static constexpr int max_algo = 4;

        cudnnConvolutionFwdAlgoPerf_t perf_fwd[max_algo];
        cudnnConvolutionBwdDataAlgoPerf_t perf_bwd_data[max_algo];
        cudnnConvolutionBwdFilterAlgoPerf_t perf_bwd_filter[max_algo];
        int algs = 0;
        size_t ws = 0;
        char const *aname = nullptr;
        int algo_id = 0;

        switch(op) {
        case dp::forward_data:
		    check(cudnnGetConvolutionForwardAlgorithm_v7(handle,X_d,M_d,desc,Y_d,max_algo,&algs,perf_fwd));
            get_first(perf_fwd,algs);
            ws = perf_fwd[0].memory;
            algo_id = perf_fwd[0].algo;
            {
                static char const *names[] = {
                    "IMPLICIT_GEMM",
                    "IMPLICIT_PRECOMP_GEMM",
                    "GEMM",
                    "DIRECT",
                    "FFT",
                    "FFT_TILING",
                    "WINOGRAD",
                    "WINOGRAD_NONFUSED"

                };
                if(algo_id < int(sizeof(names)/sizeof(names[0])))
                    aname = names[algo_id];
            }
            break;
        case dp::backward_data:
		    check(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,M_d,Y_d,desc,X_d,max_algo,&algs,perf_bwd_data));
            get_first(perf_bwd_data,algs);
            ws = perf_bwd_data[0].memory;
            algo_id = perf_bwd_data[0].algo;
            {
                static char const *names[] = {
                    "0" /* non-deterministic */,
                        "1",
                        "FFT",
                        "FFT_TILING",
                        "WINOGRAD",
                        "WINOGRAD_NONFUSED"
                };
                if(algo_id < int(sizeof(names)/sizeof(names[0])))
                    aname = names[algo_id];
            }
            break;
        case dp::backward_param:
		    check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,X_d,Y_d,desc,M_d,max_algo,&algs,perf_bwd_filter));
            get_first(perf_bwd_filter,algs);
            ws = perf_bwd_filter[0].memory;
            algo_id = perf_bwd_filter[0].algo;
            {
                static char const *names[] = {
                    "0" /* non-deterministic */,
                        "1",
                        "FFT",
                        "3" /* non-deterministic */,
                        "WINOGRAD" /* not implemented */,
                        "WINOGRAD_NONFUSED",
                        "FFT_TILING"
                };
                if(algo_id < int(sizeof(names)/sizeof(names[0])))
                    aname = names[algo_id];
            }
            break;
        default:
            throw std::runtime_error("Invalid conv op");
        }

        float *X=nullptr,*Y=nullptr,*M=nullptr;
        void *W=nullptr;
        cudaMalloc((void**)&X,in_shape.total_size()*sizeof(float));
        cudaMalloc((void**)&Y,out_shape.total_size()*sizeof(float));
        cudaMalloc((void**)&M,k_shape.total_size()*sizeof(float));
        if(ws > 0) {
            cudaMalloc(&W,ws);
        }

        rand(X,in_shape.total_size(),1.0);
        rand(Y,out_shape.total_size(),1.0);
        rand(M,k_shape.total_size(),1.0 / (bm.kern*bm.kern*bm.c_in/bm.groups));


        time_point_type start,end;
        for(int i=-warm;i<calc;i++) {
            if(i == 0) {
                cudaDeviceSynchronize();
                start = clock_type::now();
            }
            float alpha = 1.0;
            float beta  = 0.0;
            switch(op) {
                case dp::forward_data:
                    check(cudnnConvolutionForward(handle,&alpha,X_d,X,M_d,M,desc,perf_fwd[0].algo,W,ws,&beta,Y_d,Y));
                    break;
                case dp::backward_data:
                    check(cudnnConvolutionBackwardData(handle,&alpha,M_d,M,Y_d,Y,desc,perf_bwd_data[0].algo,W,ws,&beta,X_d,X));
                    break;
                case dp::backward_param:
                    beta = 1.0;
                    check(cudnnConvolutionBackwardFilter(handle,&alpha,X_d,X,Y_d,Y,desc,perf_bwd_filter[0].algo,W,ws,&beta,M_d,M));
                    break;
            }
        }
        cudaDeviceSynchronize();
        end = clock_type::now();
        double seconds = sec_diff(start,end);
        double flop = 2 * out_shape.total_size() * (bm.c_in / bm.groups * bm.kern * bm.kern) * calc;
        double bytes = in_shape.total_size() + out_shape.total_size() + k_shape.total_size();
        if(op == dp::backward_param)
            bytes += k_shape.total_size();
        bytes*=sizeof(float);
        bytes*=calc;
        Metrics met;
        met.flops = flop / seconds;
        met.bps = bytes / seconds;
        if(aname)
            met.algo = aname;
        else {
            static char buf[256];
            snprintf(buf,sizeof(buf),"algo_%d",algo_id);
            met.algo = buf;
        }

        if(W)
            cudaFree(W);
        cudaFree(M);
        cudaFree(X);
        cudaFree(Y);

        return met;

    }
#else
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

        time_point_type warmp,start,end;
        for(int i=-warm - 1;i<calc;i++) {
            if(i == -warm) {
                ec_.finish();
                warmp = clock_type::now();
            }
            if(i == 0) {
                ec_.finish();
                start = clock_type::now();
                double time_per_kern = sec_diff(warmp,start) / warm;
                calc = std::max(5,int(ceil(1/time_per_kern)));
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
#endif
    void run_conv_bm(int index)
    {
        printf("Convolution\n");
        fprintf(report_,"Convolution\n"
            "#,Network,Operation,Batch,Kernel,Padding,Stride,Channles In,Channles Out,Groups,Size,GFlops,GFlops%%,GB/s,GB/s%%,Algorithm\n");
        ConvBM setups[] = {
            //  k   p   s   g  in out dim net 
            {   3,  1,  1,480,480,480, 14,"effnet" },    
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
            
            {   1,  0,  1,256,256,256,  56,"scale" },
            {   1,  0,  1,1024,1024,1024,  7,"scale" },
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
                if(index != -1 && index != int(setup))
                    continue;
                ConvBM bm = setups[setup];
                int out_size = (bm.img_size + 2*bm.pad - bm.kern) / bm.stride + 1;
                double approx_flop = 2 * (double(out_size)*out_size * bm.c_out*batch) * (bm.c_in / bm.groups * bm.kern * bm.kern);
                double sec_per_gemm = approx_flop / ref_.flops;
                double target_calls = 2 / sec_per_gemm;
                int calls = std::max(5,std::min(200,int(target_calls)));
                int warms = std::max(1,calls / 5);
                char const *op_name[]={"nop","forward","bwd-data","bwd-filt"};
                for(int op = dp::forward_data;op<=dp::backward_param;op++) {
                    printf("  %2d %10s %8s b=%-2d k=%-2d p=%d s=%d in=%-4d out=%-4d g=%-3d D=%-3d",
                                setup,bm.type,op_name[op],batch,bm.kern,bm.pad,bm.stride,bm.c_in,bm.c_out,bm.groups,bm.img_size);
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
                    fprintf(report_,"%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%1.1f,%1.2f%%,%1.2f,%1.2f%%,%s\n",
                                setup,bm.type,op_name[op],batch,bm.kern,bm.pad,bm.stride,bm.c_in,bm.c_out,bm.groups,bm.img_size,
                                m.flops * 1e-9, flops_per,
                                m.bps * 1e-9, bandw_per,
                                m.algo);
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
    FILE *report_;
};


FlopsStats get_flops(std::string device, double scale)
{
    FlopsStats stats = FlopsStats();
    int N=1024*int(1024*scale);
    dp::Context ctx(device);
    auto q = ctx.make_queue();
    std::cout << "Testing on " << ctx.name() << std::endl;
    dp::Tensor t(ctx,dp::Shape(N));
    long long int mem_size = int(1024*scale)*1024ll*256;
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
                q.enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(N/vs),cl::NullRange,nullptr,nullptr);
                q.finish();
                std::cout << "-- Running " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();
                q.enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(N/vs),cl::NullRange,nullptr,nullptr);
                q.finish();
                auto end = std::chrono::high_resolution_clock::now();
                auto secs = std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
                double flops = N * 4.0 * 10000  * 2; // matrix 4 mads * float4 * 1000 * N
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
        std::cerr << "Usage flops [-gN] [-bN] [-cN] PLAT:DEV [mpl]" << std::endl;
        std::cerr << " mpl is multipier how much bigger/smaller buffers/duration to calculate\n"
                     " For example dlprim_flops 0:0 0.5\n";
        return 1;
    }
    int gemm_index = -1;
    int conv_index = -1;
    int br_index = -1;
    static constexpr int run_all  = 0;
    static constexpr int run_gemm = 1;
    static constexpr int run_conv = 2;
    static constexpr int run_broadcast = 3;
    int test = run_all;
    if(argv[1][0] == '-') {
       if(argv[1][1] == 'g') {
            gemm_index = atoi(argv[1]+2);
            test=run_gemm;
            argv++;
            argc--;
        }
        else if(argv[1][1] == 'c') {
            test=run_conv;
            conv_index = atoi(argv[1]+2);
            argv++;
            argc--;
        }
        else if(argv[1][1] == 'b') {
            test=run_broadcast;
            br_index = atoi(argv[1]+2);
            argv++;
            argc--;
        }
        else {
            std::cerr << "Expecting g/c/b after -" << std::endl;
            return 1;
        }
    }
    double scale = 1;
    if(argc >= 3) {
        scale = atof(argv[2]);
    }
    FlopsStats fs = get_flops(argv[1],scale);
#ifdef CUDA_TEST    
    {
        int cuda_dev = -1;
        std::string ocl_dev(argv[1]);
        size_t pos = ocl_dev.find(':');
        if(pos == std::string::npos) {
            throw std::runtime_error("invalid name");
        }
        cuda_dev = atoi(ocl_dev.c_str() + pos + 1);
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties (&prop,cuda_dev)!=0) {
            throw std::runtime_error("Failed to get cuda dev props for " + std::to_string(cuda_dev));
        }
        std::cout << "Cuda Device " << prop.name << std::endl;
        if(cudaSetDevice(cuda_dev)!=0)
            throw std::runtime_error("cuda set device failed");
    }
#endif
    Benchmarker bm(argv[1],fs,scale);
    if(test == run_all) {
        bm.run_gemm_bm(gemm_index);
        bm.run_conv_bm(conv_index);
        bm.run_br_bm(br_index);
    }
    else if(test == run_broadcast) {
        bm.run_br_bm(br_index);
    }
    else if(test == run_gemm) {
        bm.run_gemm_bm(gemm_index);
    }
    else if(test == run_conv) {
        bm.run_conv_bm(conv_index);
    }
}

