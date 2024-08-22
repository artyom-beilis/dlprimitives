#include <dlprim/gpu/gemm.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include "test.hpp"
#include <my_cblas.hpp>
#include <iostream>

namespace dp = dlprim;

void test_mm(int batch,int m,int n,int k,bool ta,bool tb,float beta,dp::Context &ctx,dp::ExecutionContext &q)
{
    std::cout << "- B="<<batch << " M=" << m << " N="<<n << " K=" << k << " "<< (ta?"T":"N")<<(tb?"T":"N")<< " beta="<<beta<<std::endl;
    bool use_batch = batch > 0;
    batch = std::max(1,batch);
    dp::Tensor A(ctx,(ta?dp::Shape(batch,k,m):dp::Shape(batch,m,k)),dp::float_data);
    dp::Tensor B(ctx,(tb?dp::Shape(batch,n,k):dp::Shape(batch,k,n)),dp::float_data);
    dp::Tensor C(ctx,dp::Shape(batch,m,n),dp::float_data);
    
    dp::core::fill_random(A,0,0,dp::core::rnd_normal,-5,5,q);
    dp::core::fill_random(B,0,(A.shape().total_size() + 3)/4,dp::core::rnd_normal,-5,5,q);
    dp::core::fill_tensor(C,1,q);

    // make sure we work with ints
    dp::core::pointwise_operation({A},{A},{},"y0 = round(x0);",q);
    dp::core::pointwise_operation({B},{B},{},"y0 = round(x0);",q);

    A.to_host(q);
    B.to_host(q);
    std::vector<float> C_ref(C.shape().total_size(),1.0f);
    for(int b=0;b<batch;b++)
    {
        float *Aptr = A.data<float>() + m*k*b;
        float *Bptr = B.data<float>() + n*k*b;
        float *Cptr = C_ref.data() + m*n*b;
        using namespace dlprim;
        cblas_sgemm(CblasRowMajor,(ta?CblasTrans:CblasNoTrans),(tb?CblasTrans:CblasNoTrans),
                    m,n,k,1.0f,
                    Aptr,A.shape()[2],
                    Bptr,B.shape()[2],
                    beta,
                    Cptr,C.shape()[2]);
    }
    if(use_batch) {
        dlprim::gpu::GEMM::batch_sgemm(
            dp::float_data,ta,tb,batch,
            m,n,k,
            A.device_buffer(),A.device_offset(),m*k,A.shape()[2],
            B.device_buffer(),B.device_offset(),n*k,B.shape()[2],
            C.device_buffer(),C.device_offset(),m*n,C.shape()[2],
            beta,q);
        #if 0
        auto ptr = dlprim::gpu::GEMM::get_optimal_gemm(
            ctx,dp::float_data,
            ta,tb,m,n,k);
        for(int b=0;b<batch;b++) {
            ptr->gemm(m,n,k,
                A.device_buffer(),A.device_offset() + m*k*b,A.shape()[2],
                B.device_buffer(),B.device_offset() + n*k*b,B.shape()[2],
                C.device_buffer(),C.device_offset() + m*n*b,C.shape()[2],
                nullptr,0,
                beta,m*n,
                q);
        }
        #endif

    }
    else {
        auto ptr = dlprim::gpu::GEMM::get_optimal_gemm(
            ctx,dp::float_data,
            ta,tb,m,n,k);
        ptr->gemm(m,n,k,
            A.device_buffer(),A.device_offset(),A.shape()[2],
            B.device_buffer(),B.device_offset(),B.shape()[2],
            C.device_buffer(),C.device_offset(),C.shape()[2],
            nullptr,0,
            beta,C.shape().total_size(),
            q);
    }
    C.to_host(q);
    float *Cptr = C.data<float>();
    for(size_t i=0;i<C_ref.size();i++) {
        if(Cptr[i] != C_ref[i]) {
            std::cout << "   i=" << i << " result="<< Cptr[i] << " expected="<<C_ref[i] << std::endl;
        }
        TEST(Cptr[i] == C_ref[i]);
    }
}


int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "Use paltform:device" << std::endl;
        return 1;
    }
    try {
        dp::Context ctx(argv[1]);
        if(ctx.is_cpu_context()) {
            std::cout << "CPU - exit" << std::endl;
            return 0;
        }
        dp::ExecutionContext q = ctx.make_execution_context();
        std::cout << ctx.name() << std::endl;
        int setups[][3] = {
            { 1,       1,    1 },
            { 8,       1,    8 },
            { 1,       8,    8 },
            { 8,       8,    1 },
            { 31,     16,   31 },
            { 16,     31,   31 },
            { 31,     31,   16 },
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
        for(float beta:{0.0f,2.0f}) {
            for(int b:{0,1,5}) {
                for(int ta = 0; ta < 2; ta ++) {
                    for(int tb = 0; tb < 2; tb ++) {
                        for(unsigned setup = 0;setup < sizeof(setups)/sizeof(setups[0]);setup++) {
                            int M = setups[setup][0];
                            int N = setups[setup][1];
                            int K = setups[setup][2];
                            test_mm(b,M,N,K,ta,tb,beta,ctx,q);
                        }
                    }
                }
            }
        }
    }
    catch(std::exception const &e) {
        std::cerr <<"Failed:"<< e.what() << std::endl;
        return 1;
    }
    return 0;
}
