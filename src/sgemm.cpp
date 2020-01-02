#include "blas.h"
#include "kernel.h"
#include <iostream>

blas_status sgemm(
    blas_operation trans_a,
    blas_operation trans_b,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc)
{
    if (trans_a == blas_operation::none && trans_b == blas_operation::none)
    {
        return sgemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    else
    {
        return blas_status::not_support;
    }
}