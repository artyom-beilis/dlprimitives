///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <my_cblas.hpp>
#ifdef USE_INTERNAL_BLAS
#include <string.h>
namespace dlprim {
namespace my_cblas {
    void cblas_saxpby(int size,float a,float *x,int lx,float b, float *y,int ly)
    {
        for(int i=0;i<size;i++,x+=lx,y+=ly)
            *y = a* *x + b* *y;
    }
    void cblas_saxpy(int size,float a,float *x,int lx,float *y,int ly)
    {
        for(int i=0;i<size;i++,x+=lx,y+=ly)
            *y += a * *x;
    }

    void cblas_sscal(int size,float a,float *v,int step)
    {
        for(int i=0;i<size;i++,v+=step)
            *v *= a;
    }
    void cblas_sgemm_apply_beta(int M,int N,float beta,float *C,int ldc)
    {
        if(beta == 0) {
            for(int r=0;r<M;r++)
                memset(C + ldc*r,0,N*sizeof(float));
        }
        else {
            for(int r=0;r<M;r++)
                cblas_sscal(N,beta,C + ldc*r,1);
        }
    }

    void cblas_sgemm_nn(int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float *C,int ldc)
    {
        for(int r=0;r<M;r++) {
            for(int k=0;k<K;k++) {
                float av = A[r*lda+k] * alpha;
                for(int c=0;c<N;c++) {
                    C[r*ldc+c] += av*B[k*ldb+c];
                }
            }
        }
    }
    void cblas_sgemm_tn(int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float *C,int ldc)
    {
        for(int k=0;k<K;k++) {
            for(int r=0;r<M;r++) {
                float av = A[r+lda*k] * alpha;
                for(int c=0;c<N;c++) {
                    C[r*ldc+c] += av*B[k*ldb+c];
                }
            }
        }
    }
    void cblas_sgemm_nt(int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float *C,int ldc)
    {
        for(int r=0;r<M;r++) {
            for(int c=0;c<N;c++) {
                for(int k=0;k<K;k++) {
                    float bv = B[k+ldb*c];
                    float av = A[r*lda+k] * alpha;
                    C[r*ldc+c] += av*bv;
                }
            }
        }
    }
    void cblas_sgemm_tt(int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float *C,int ldc)
    {
        for(int c=0;c<N;c++) {
            for(int k=0;k<K;k++) {
                float bv = B[k+ldb*c]*alpha;
                for(int r=0;r<M;r++) {
                    float av = A[r+lda*k];
                    C[r*ldc+c] += av*bv;
                }
            }
        }
    }

    void cblas_sgemm(int,bool ta,bool tb,int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float beta,float *C,int ldc)
    {
        cblas_sgemm_apply_beta(M,N,beta,C,ldc);
        if(ta) {
            if(tb)
                cblas_sgemm_tt(M,N,K,alpha,A,lda,B,ldb,C,ldc);
            else
                cblas_sgemm_tn(M,N,K,alpha,A,lda,B,ldb,C,ldc);
        }
        else {
            if(tb)
                cblas_sgemm_nt(M,N,K,alpha,A,lda,B,ldb,C,ldc);
            else
                cblas_sgemm_nn(M,N,K,alpha,A,lda,B,ldb,C,ldc);
        }
    }

} // cblas
} // dlprim
#endif
