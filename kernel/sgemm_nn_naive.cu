
#include "util.h"
#include "kernel.h"
#include "matrix.h"
#include <cuda_runtime_api.h>

__global__ void mat_mul_nn_kernel(const Matrix A, const Matrix B, Matrix C, const float alpha, const float beta)
{

	float Cvalue = 0;
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
	for (int k = 0; k < A.cols; ++k)
		Cvalue += A.getValue(r, k) * B.getValue(k, c);
	
	Cvalue = C.getValue(r, c) * beta + Cvalue * alpha;
	C.setValue(r, c, Cvalue);
}

blas_status sgemm_nn(int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	const int Mtile = 16;
	const int Ntile = 16;
	const int Ktile = 16;

	const Matrix a(m, k, lda, const_cast<float*>(A));
	const Matrix b(k, n, ldb, const_cast<float*>(B));
	Matrix c(m, n, ldc, C);

	dim3 dimBlock(Mtile, Ntile);
	dim3 dimGrid(m / Mtile, n / Ntile);
	mat_mul_nn_kernel<<<dimGrid, dimBlock>>>(a, b, c, *alpha, *beta);
}