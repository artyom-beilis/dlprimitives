#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <iostream>

namespace dlprim {
namespace gpu {
    
    class StandardSGEMMBase : public GEMM {
    public:
        StandardSGEMMBase(Context &ctx,int M,int N,int K,bool actual_gemm)
        {
            sep_scale_ = false;
            reduce_k_ = 1;
            if(ctx.check_device_extension("cl_intel_subgroups")) {
                block_size_m_ = 8;
                block_size_n_ = 8;
                tile_size_k_ = 4;
                off_ = 0;
                if(M >= 128 && N >= 128) {
                    tile_size_m_ = tile_size_n_ = 128;
                }
                else {
                    tile_size_m_ = tile_size_n_ = 64;
                }
            }
            else {
                if(ctx.is_amd() && !actual_gemm) {
					if(M >= 256 && N >= 256) {
                        tile_size_m_ = 96;
                        tile_size_n_ = 96;
                        block_size_m_ = 6;
                        block_size_n_ = 6;
                        tile_size_k_ = 16;
                        off_ = 0;
                    }
                    else if(M >= 64 && N>= 64) {
                        tile_size_m_ = 64;
                        tile_size_n_ = 64;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 16;
                        off_ = 0;
                    }
                    else if(M >= 32 && N >= 32) {
                        tile_size_m_ = 32;
                        tile_size_n_ = 32;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 32;
                        off_ = 0;
                    }
                    else if(M * N <= 256) {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 1;
                        block_size_n_ = 1;
                        tile_size_k_ = 128;
                        off_ = 0;
                    }
                    else {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 2;
                        block_size_n_ = 2;
                        tile_size_k_ = 64;
                        off_ = 0;
                    }
                }
                else {
                    if(M >= 256 && N >= 256) {
                        tile_size_m_ = 128;
                        tile_size_n_ = 128;
                        block_size_m_ = 8;
                        block_size_n_ = 8;
                        tile_size_k_ = 16;
                        off_ = ctx.is_amd() ? 0 :1;
                    }
                    else if(M >= 128 && N>= 128) {
                        tile_size_m_ = 64;
                        tile_size_n_ = 64;
                        block_size_m_ = 8;
                        block_size_n_ = 8;
                        tile_size_k_ = 16;
                        off_ = ctx.is_amd() ? 0: 1;
                    }
                    else if(M >= 32 && N >= 32) {
                        tile_size_m_ = 32;
                        tile_size_n_ = 32;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 32;
                        off_ = 0;
                    }
                    else if(M * N <= 256) {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 1;
                        block_size_n_ = 1;
                        tile_size_k_ = 128;
                        off_ = 0;
                    }
                    else {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 2;
                        block_size_n_ = 2;
                        tile_size_k_ = 64;
                        off_ = 0;
                    }
                }
                int cores = ctx.estimated_core_count();
                if(M * N / (block_size_m_ * block_size_n_) < 4 * cores && K > M*16 && K > N*16) {
                    reduce_k_ = 8;
                    set_scale(ctx);
                }
            }

        }
    protected:
        void set_scale(Context &ctx)
        {
            if(sep_scale_ == false) {
                cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"scal");
                scal_ = std::move(cl::Kernel(prog,"sscal"));
                sep_scale_ = true;
            }
        }

        void scale(size_t size,float s,cl::Buffer &x,cl_ulong x_offset,ExecutionContext const &ec)
        {
            int wg = 64;
            if(size >= 1024)
                wg = 256;
            int p=0;
            scal_.setArg(p++,int(size));
            scal_.setArg(p++,s);
            scal_.setArg(p++,x);
            scal_.setArg(p++,x_offset);
            cl::NDRange l(wg);
            cl::NDRange g=gpu::round_range(size,l);
            ec.queue().enqueueNDRangeKernel(scal_,cl::NullRange,g,l,ec.events(),ec.event("gemm_beta_scale"));
        }
        int tile_size_n_,tile_size_m_,tile_size_k_;
        int block_size_n_,block_size_m_;
        int off_;
        int reduce_k_;
        bool sep_scale_;
        cl::Kernel scal_;
    };

    class StandardSGEMM : public StandardSGEMMBase {
    public:
        StandardSGEMM(  Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int K,
                        int bias,
                        StandardActivations act,
                        int im2col_chan = 0) : 
                StandardSGEMMBase(ctx,M,N,K,true)
        {
            ///
            /// on AMD lda % 1024==0 / ldb % 1024==0 wipes cache out - so we reorder 
            //  all ops in Z-order/Morton order
            ///
            zorder_ = 0;
            if(ctx.is_amd() && (M % 1024 == 0 || N % 1024 == 0)) {
                if(M >= N && N*10 >= M)
                    zorder_ = 1;
                if(N >= M && M*10 >= N)
                    zorder_ = 1;
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
                                        "REDUCE_K",reduce_k_,
                                        "ZORDER",zorder_,
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
                          cl_ulong offset_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int ldc,
                          cl::Buffer *bias,
                          cl_ulong bias_offset,
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {

            ExecutionContext e;
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,2));
                e=ein.generate_series_context(1,2);
                beta = 1.0;
            }
            else {
                e=ein;
            }
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
            kernel_.setArg(ind++,beta);
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
            int gr0 = round_up_div(M,tile_size_m_);
            int gr1 = round_up_div(N,tile_size_n_);

            if(zorder_) {
                int gr = std::max(gr0,gr1);
                int n=1;
                while(gr > n) {
                    n<<=1;
                }
                gr0 = gr1 = n;
            }
            
            int gs0 = gr0 * ls0;
            int gs1 = gr1 * ls1;
            cl::NDRange global,local;
            if(reduce_k_ > 1) {
                global = cl::NDRange(reduce_k_,gs0,gs1);
                local =  cl::NDRange(1,ls0,ls1);
            }
            else {
                global = cl::NDRange(gs0,gs1);
                local =  cl::NDRange(ls0,ls1);
            }
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event("gemm"));
        }

    private:
        cl::Kernel kernel_;
        bool bias_;
        bool zorder_;
    };


    class ConvSGEMM : public StandardSGEMMBase {
    public:
        ConvSGEMM(  Context &ctx,
                    GemmOpMode op_mode,
                    bool atrans,bool btrans,
                    int M,int N,int K,
                    int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
                    int src_channels,int src_rows,int src_cols,
                    int tgt_rows,int tgt_cols,
                    int bias,
                    StandardActivations act,
                    int im2col_chan = 0) :
                StandardSGEMMBase(ctx,M,N,K,false)
        {
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
                                        "CONVGEMM",int(op_mode),
                                        "KERN_H",  kernel[0], "KERN_W",kernel[1],
                                        "DILATE_H",dilate[0], "DILATE_W",dilate[1],
                                        "PAD_H",   padding[0],"PAD_W",padding[1],
                                        "STRIDE_H",stride[0], "STRIDE_W",stride[1],
                                        "GROUPS",groups,
                                        "CHANNELS_IN",src_channels,
                                        "SRC_COLS",src_cols,
                                        "SRC_ROWS",src_rows,
                                        "IMG_COLS",tgt_cols,
                                        "IMG_ROWS",tgt_rows,
                                        "REDUCE_K",reduce_k_,
                                        "ACTIVATION",int(act));
            if(op_mode == GemmOpMode::backward_data) {
                set_scale(ctx);
                gemm_name_="conv_gemm_bwd_data";
            }
            else if(op_mode == GemmOpMode::backward_filter)
                gemm_name_="conv_gemm_bwd_filter";
            else
                gemm_name_="conv_gemm";
            kernel_ = cl::Kernel(prog,"sgemm");
            bias_ = bias;
            groups_ = groups;
            md_ = int(op_mode);
            k_ = kernel[0];
            pad_ = padding[0];
            s_ = stride[0];
            ci_ = src_channels;
            w_ = src_cols;
        }
        static int round_up_div(int x,int y)
        {
            return (x + y - 1)/y;
        }
        virtual void gemm(int M,int N,int K,
                          cl::Buffer &a,
                          cl_ulong offset_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int ldc,
                          cl::Buffer *bias,
                          cl_ulong bias_offset,
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {
            ExecutionContext e;
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,2));
                e=ein.generate_series_context(1,2);
                beta = 1.0;
            }
            else {
                e=ein;
            }
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
            kernel_.setArg(ind++,beta);
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
            cl::NDRange global,local;
            if(groups_ > 1 || reduce_k_ > 1) {
                global = cl::NDRange(groups_ * reduce_k_,gs0,gs1);
                local =  cl::NDRange(1,ls0,ls1);
            }
            else {
                global = cl::NDRange(gs0,gs1,1);
                local =  cl::NDRange(ls0,ls1,1);
            }
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event(gemm_name_));
        }

    private:
        char const *gemm_name_;
        cl::Kernel kernel_;
        cl::Kernel scal_;
        bool bias_;
        int groups_;
        int md_;
        int w_;
        int ci_,co_,k_,pad_,s_;
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
            GemmOpMode op_mode,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
            int src_channels,int src_rows,int src_cols,
            int tgt_rows,int tgt_cols,
            int bias,
            StandardActivations act,
            int im2col_chan)
    {
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new ConvSGEMM(ctx,op_mode,
            trans_a,trans_b,M,N,K,
            kernel,dilate,padding,stride,groups,
            src_channels,src_rows,src_cols,
            tgt_rows,tgt_cols,
            bias,act,im2col_chan));
        return g;
    }

} // gpu
} // dlprim 
