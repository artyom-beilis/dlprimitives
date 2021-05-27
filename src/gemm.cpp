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
                        StandardActivations act,
                        int im2col_chan = 0)
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
                                        "IM2COL_OCHAN",im2col_chan,
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


    class ConvSGEMM : public GEMM {
    public:
        ConvSGEMM(  Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int /*K*/,
            int kernel[2],int dilate[2],int padding[2],int stride[2],
            int src_channels,int src_rows,int src_cols,
            int tgt_rows,int tgt_cols,
                        int bias,
                        StandardActivations act,
                        int im2col_chan = 0)
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
                                        "IM2COL_OCHAN",im2col_chan,
                                        "CONVGEMM",1,
                                        "KERN_H",  kernel[0], "KERN_W",kernel[1],
                                        "DILATE_H",dilate[0], "DILATE_W",dilate[1],
                                        "PAD_H",   padding[0],"PAD_W",padding[1],
                                        "STRIDE_H",stride[0], "STRIDE_W",stride[1],
                                        "CHANNELS_IN",src_channels,
                                        "SRC_COLS",src_cols,
                                        "SRC_ROWS",src_rows,
                                        "IMG_COLS",tgt_cols,
                                        "IMG_ROWS",tgt_rows,
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
            StandardActivations act,
            int im2col_chan)
    {
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new StandardSGEMM(ctx,trans_a,trans_b,M,N,K,bias,act,im2col_chan));
        return g;
    }
    std::unique_ptr<GEMM> GEMM::get_optimal_conv_gemm(
            Context &ctx,DataType dtype,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int kernel[2],int dilate[2],int padding[2],int stride[2],
            int src_channels,int src_rows,int src_cols,
            int tgt_rows,int tgt_cols,
            int bias,
            StandardActivations act,
            int im2col_chan)
    {
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new ConvSGEMM(ctx,trans_a,trans_b,M,N,K,
            kernel,dilate,padding,stride,
            src_channels,src_rows,src_cols,
            tgt_rows,tgt_cols,
            bias,act,im2col_chan));
        return g;
    }

} // gpu
} // dlprim 
