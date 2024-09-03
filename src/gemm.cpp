///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <iostream>

namespace dlprim {
namespace gpu {
    
    class StandardSGEMMBase  {
    public:
        StandardSGEMMBase(Context &ctx,int M,int N,int K,bool actual_gemm,bool batch_gemm,StandardActivations &activation)
        {
            sep_scale_ = false;
            sep_act_ = false;
            batch_gemm_ = batch_gemm;
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
                if (ctx.is_apple()) {
                    tile_size_m_ = 32;
                    tile_size_n_ = 32;
                    block_size_m_ = 4;
                    block_size_n_ = 4;
                    tile_size_k_ = 16;
                    off_ = 0;
                }
                else if (ctx.is_imagination())
                {
                   tile_size_m_ = 64;
                   tile_size_n_ = 64;
                   block_size_m_ = 8;
                   block_size_n_ = 8;
                   tile_size_k_ = 16;
                   off_ = 1;
                }
                else if(ctx.is_amd() && !actual_gemm) {
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
            }
            if(!batch_gemm_) {
                int cores = ctx.estimated_core_count();
                if(cores >= 256 && M * N / (block_size_m_ * block_size_n_) < 4 * cores && K > M*16 && K > N*16) {
                    reduce_k_ = 8;
                    set_scale(ctx,activation);
                }
            }
        }
    protected:
        static int round_up_div(int x,int y)
        {
            return (x + y - 1)/y;
        }
        void check_zorder(Context &ctx,int M,int N)
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
        }
        void calc_dims(int &gs0,int &ls0,int &gs1,int &ls1,int M,int N)
        {
            ls0 = tile_size_m_ / block_size_m_;
            ls1 = tile_size_n_ / block_size_n_; 
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
            
            gs0 = gr0 * ls0;
            gs1 = gr1 * ls1;
        }
        void set_scale(Context &ctx,StandardActivations &activation)
        {
            if(sep_scale_ == false) {
                cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"scal");
                scal_ = std::move(cl::Kernel(prog,"sscal"));
                
                if(activation != StandardActivations::identity) {
                    cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                        "ACTIVATION",int(activation));
                    cl::Kernel k(prog,"activation");
                    act_ = std::move(k);
                    activation = StandardActivations::identity;
                    sep_act_ = true;
                }

                sep_scale_ = true;
            }
        }

        void activation(size_t size,cl::Buffer &x,cl_ulong x_offset,ExecutionContext const &ec)
        {
                act_.setArg(0,cl_ulong(size));
                act_.setArg(1,x);
                act_.setArg(2,x_offset);
                act_.setArg(3,x);
                act_.setArg(4,x_offset);
                ec.queue().enqueueNDRangeKernel(act_, cl::NullRange, cl::NDRange(size),cl::NullRange,ec.events(),ec.event("activation"));
        }

        void scale(size_t size,float s,cl::Buffer &x,cl_ulong x_offset,ExecutionContext const &ec)
        {
            int wg = 64;
            if(size >= 1024)
                wg = 256;
            int p=0;
            scal_.setArg(p++,cl_ulong(size));
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
        bool sep_act_;
        bool batch_gemm_;
        cl::Kernel scal_;
        cl::Kernel act_;
        bool zorder_ = false;
    };

    class StandardSGEMM : public GEMM, public StandardSGEMMBase {
    public:
        StandardSGEMM(  Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int K,
                        int bias,
                        StandardActivations act,
                        int im2col_chan = 0) : 
                StandardSGEMMBase(ctx,M,N,K,true,false,act)
        {
            check_zorder(ctx,M,N);
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
            int kernel_runs = 1 + int(sep_act_) + int(sep_scale_);
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,kernel_runs));
                e=ein.generate_series_context(1,kernel_runs);
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

            int gs0,gs1,ls0,ls1;
            calc_dims(gs0,ls0,gs1,ls1,M,N);
           
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
            
            if(sep_act_) {
                auto e2 = ein.generate_series_context(kernel_runs-1,kernel_runs);
                activation(size_of_c,c,offset_c,e2);
            }
        }

    private:
        cl::Kernel kernel_;
        bool bias_;
    };


    class BatchSGEMM : public StandardSGEMMBase {
    public:
        BatchSGEMM(     Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int K,
                        StandardActivations &act
                        ):
                StandardSGEMMBase(ctx,M,N,K,true,true,act)
        {
            DLPRIM_CHECK(act == StandardActivations::identity);
            check_zorder(ctx,M,N);
            cl::Program const &prog = Cache::instance().get_program(ctx,"sgemm",
                                        "BATCH_GEMM",1,
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",0,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "IM2COL_OCHAN",0,
                                        "REDUCE_K",0,
                                        "ZORDER",zorder_,
                                        "ACTIVATION",0);
            kernel_ = cl::Kernel(prog,"sgemm");
            bias_ = false;
        }
        virtual void gemm(int batches,int M,int N,int K,
                          cl::Buffer &a,
                          cl_ulong offset_a,
                          int batch_stride_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int batch_stride_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int batch_stride_c,
                          int ldc,
                          float beta,
                          ExecutionContext const &e)
        {

            int ind=0;
            kernel_.setArg(ind++,batches);
            kernel_.setArg(ind++,M);
            kernel_.setArg(ind++,N);
            kernel_.setArg(ind++,K);
            kernel_.setArg(ind++,a);
            kernel_.setArg(ind++,offset_a);
            kernel_.setArg(ind++,batch_stride_a);
            kernel_.setArg(ind++,lda);
            kernel_.setArg(ind++,b);
            kernel_.setArg(ind++,offset_b);
            kernel_.setArg(ind++,batch_stride_b);
            kernel_.setArg(ind++,ldb);
            kernel_.setArg(ind++,c);
            kernel_.setArg(ind++,offset_c);
            kernel_.setArg(ind++,batch_stride_c);
            kernel_.setArg(ind++,ldc);
            kernel_.setArg(ind++,beta);
           
            int gs0,gs1,ls0,ls1;
            calc_dims(gs0,ls0,gs1,ls1,M,N);
            
            cl::NDRange global = cl::NDRange(batches,gs0,gs1);
            cl::NDRange local =  cl::NDRange(1,ls0,ls1);
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event("gemm"));
            
        }

    private:
        cl::Kernel kernel_;
        bool bias_;
        bool zorder_;
    };
    

    class ConvSGEMM : public GEMM, public StandardSGEMMBase {
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
                StandardSGEMMBase(ctx,M,N,K,false,false,act)
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
                DLPRIM_CHECK(act == StandardActivations::identity);
                set_scale(ctx,act);
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
            int kernel_runs = 1 + int(sep_scale_) + int(sep_act_);
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,kernel_runs));
                e=ein.generate_series_context(1,kernel_runs);
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

            if(sep_act_) {
                auto e2 = ein.generate_series_context(kernel_runs-1,kernel_runs);
                activation(size_of_c,c,offset_c,e2);
            }
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

    void GEMM::batch_sgemm(DataType dt,
                          bool trans_a,bool trans_b,
                          int Batch, // number of matrices
                          int M,int N,int K,
                          cl::Buffer &a,
                          cl_ulong offset_a, 
                          int batch_stride_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int batch_stride_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int batch_stride_c,
                          int ldc,
                          float beta,
                          ExecutionContext const &e)
    {
        DLPRIM_CHECK(dt == float_data);
        StandardActivations act = StandardActivations::identity;
        Context ctx(e);
        BatchSGEMM gemm_opt(ctx,trans_a,trans_b,M,N,K,act);
        gemm_opt.gemm(Batch,M,N,K,
                a,offset_a,batch_stride_a,lda,
                b,offset_b,batch_stride_b,ldb,
                c,offset_c,batch_stride_c,ldc,
                beta,
                e);

    }



} // gpu
} // dlprim 
