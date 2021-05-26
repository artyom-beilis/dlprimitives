#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace gpu {
    class StandardSGEMM : public GEMM {
    public:
        StandardSGEMM(  Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int /*K*/,
                        int bias,
                        StandardActivations act)
        {
            if(M >= 256 && N >= 256) {
                tile_size_m_ = 128;
                tile_size_n_ = 128;
                block_size_m_ = 8;
                block_size_n_ = 8;
                tile_size_k_ = 16;
                off_ = 1;
            }
            else if(M >= 128 && N>= 128) {
                tile_size_m_ = 64;
                tile_size_n_ = 64;
                block_size_m_ = 8;
                block_size_n_ = 8;
                tile_size_k_ = 16;
                off_ = 1;
            }
            else {
                tile_size_m_ = 32;
                tile_size_n_ = 32;
                block_size_m_ = 4;
                block_size_n_ = 4;
                tile_size_k_ = 32;
                off_ = 0;
            }

            cl::Program const &prog = Cache::instance().get_program(ctx,"sgemm",
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",bias,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "ACTIVATION",int(act));
            kernel_ = cl::Kernel(prog,"sgemm");
            bias_ = bias;
        }
        static int round_up_div(int x,int y)
        {
            return (x + y - 1)/y;
        }
        virtual void gemm(int M,int N,int K,
                          cl::Buffer &a,
                          int offset_a,
                          int lda,
                          cl::Buffer &b,
                          int offset_b,
                          int ldb,
                          cl::Buffer &c,
                          int offset_c,
                          int ldc,
                          cl::Buffer *bias,
                          int bias_offset,
                          cl::CommandQueue &queue,
                          std::vector<cl::Event> *events,
                          cl::Event *event)
        {
            int ind=0;
            kernel_.setArg(ind++,M);
            kernel_.setArg(ind++,N);
            kernel_.setArg(ind++,K);
            kernel_.setArg(ind++,a);
            kernel_.setArg(ind++,offset_a);
            kernel_.setArg(ind++,lda);
            kernel_.setArg(ind++,b);
            kernel_.setArg(ind++,offset_b);
            kernel_.setArg(ind++,ldb);
            kernel_.setArg(ind++,c);
            kernel_.setArg(ind++,offset_c);
            kernel_.setArg(ind++,ldc);
            if(bias_) {
                DLPRIM_CHECK(bias != nullptr);
                kernel_.setArg(ind++,*bias);
                kernel_.setArg(ind++,bias_offset);
            }
            else {
                DLPRIM_CHECK(bias == nullptr);
            }
           
            int ls0 = tile_size_m_ / block_size_m_;
            int ls1 = tile_size_n_ / block_size_n_; 
            int gs0 = round_up_div(M,tile_size_m_) * tile_size_m_ / block_size_m_;
            int gs1 = round_up_div(N,tile_size_n_) * tile_size_n_ / block_size_n_;
            cl::NDRange global(gs0,gs1);
            cl::NDRange local(ls0,ls1);
            queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,events,event);
        }

    private:
        cl::Kernel kernel_;
        int tile_size_n_,tile_size_m_,tile_size_k_;
        int block_size_n_,block_size_m_;
        int off_;
        bool bias_;
    };

    std::unique_ptr<GEMM> GEMM::get_optimal_gemm(
            Context &ctx,DataType dtype,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int bias,
            StandardActivations act)
    {
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new StandardSGEMM(ctx,trans_a,trans_b,M,N,K,bias,act));
        return g;
    }

} // gpu
} // dlprim 
