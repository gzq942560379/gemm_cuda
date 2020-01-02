#pragma once

enum blas_operation
{
    none,
    transpose,
    conjugate_transpose,
};

enum blas_status
{
    success = 0,
    not_support = 1,
};

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
    int ldc);

blas_status sgemm_strided_batched(
    blas_operation trans_a,
    blas_operation trans_b,
    int m,
    int n,
    int k,
    const float *alpha_p,
    const float *A,
    int lda,
    int stride_a,
    const float *B,
    int ldb,
    int stride_b,
    const float *beta_p,
    float *C,
    int ldc,
    int stride_c,
    int batch_count);