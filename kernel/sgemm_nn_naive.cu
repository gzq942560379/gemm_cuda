
#include "util.h"
#include "kernel.h"

// 2048 2048 2048 356.727ms
__global__ void mat_mul_nn_kernel(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, const float alpha, const float beta)
{
	float Cvalue = 0;
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	for (int k = 0; k < K; ++k)
		Cvalue += A[index2(k, r, lda)] * B[index2(c, k, ldb)];
	C[index2(c, r, ldc)] = C[index2(c, r, ldc)] * beta + Cvalue * alpha;
}

blas_status sgemm_nn(int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	const int Mtile = 16;
	const int Ntile = 16;
	const int Ktile = 16;
	dim3 dimBlock(Mtile, Ntile);
	dim3 dimGrid(m / Mtile, n / Ntile);
    mat_mul_nn_kernel<<<dimGrid, dimBlock>>>(m, n, k, A, lda, B, ldb, C, ldc, *alpha, *beta);
}