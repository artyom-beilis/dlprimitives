///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/context.hpp>
#include <dlprim/definitions.hpp>
namespace dlprim {
/// GPU Related tools and classes
namespace gpu {

    class GEMM {
    public:

        virtual ~GEMM() 
        {
        }

        static constexpr int no_bias = 0;
        static constexpr int bias_M =  1;
        static constexpr int bias_N =  2;

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
                          ExecutionContext const &e) = 0;

        static void batch_sgemm(
                          DataType dt,
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
                          ExecutionContext const &e);

        static std::unique_ptr<GEMM> get_optimal_gemm(
            Context &ctx,DataType dtype,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int bias = 0,
            StandardActivations act = StandardActivations::identity,
            int im2col_chan = 0);

        static std::unique_ptr<GEMM> get_optimal_conv_gemm(
            Context &ctx,DataType dtype,
            GemmOpMode op_mode,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
            int src_channels,int src_rows,int src_cols,
            int tgt_rows,int tgt_cols,
            int bias = 0,
            StandardActivations act = StandardActivations::identity,
            int im2col_chan = 0);
        
    };



}
}
