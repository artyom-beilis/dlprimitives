///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#ifndef USE_INTERNAL_BLAS
#include <cblas.h>
#else

namespace dlprim {
namespace my_cblas {
static constexpr int CblasRowMajor = 0;
static constexpr bool CblasNoTrans = false;
static constexpr bool CblasTrans = true;

void cblas_saxpby(int size,float a,float *x,int lx,float b, float *y,int ly);
void cblas_saxpy(int size,float a,float *x,int lx,float *y,int ly);
void cblas_sscal(int size,float a,float *v,int step);
void cblas_sgemm(int,bool ta,bool tb,int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float beta,float *C,int ldc);

} // my_cblas

using my_cblas::cblas_saxpby;
using my_cblas::cblas_saxpy;
using my_cblas::cblas_sscal;
using my_cblas::cblas_sgemm;
using my_cblas::CblasRowMajor;
using my_cblas::CblasNoTrans;
using my_cblas::CblasTrans;

}

#endif

