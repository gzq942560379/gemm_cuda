
#include "util.h"
#include "kernel.h"

template <int Mtile, int Ntile, int Ktile>
__global__ void mat_mul_nn_kernel(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, const float alpha, const float beta)
{
	int blockRow = blockIdx.x;
	int blockCol = blockIdx.y;
	int blockRowOffset = blockIdx.x * Mtile;
	int blockColOffset = blockIdx.y * Ntile;
	int row = threadIdx.x;
	int col = threadIdx.y;
	float *Csub = &C[index2(blockColOffset, blockRowOffset, ldc)];

	float Cvalue = 0;
	__shared__ float As[Ktile][Mtile];
	__shared__ float Bs[Ntile][Ktile];
	for (int kb = 0; kb < K; kb += Ktile)
	{
		const float *Asub = &A[index2(kb, blockRowOffset, lda)];
		const float *Bsub = &B[index2(blockColOffset, kb, ldb)];

		for (int colA = col; colA < Ktile; colA += Ntile)
			As[colA][row] = Asub[index2(colA, row, lda)];
		for (int rowB = row; rowB < Ktile; rowB += Mtile)
			Bs[col][rowB] = Bsub[index2(col, rowB, ldb)];

		__syncthreads();

		for (int k = 0; k < Ktile; ++k)
		{
			Cvalue += As[k][row] * Bs[col][k];
		}
		__syncthreads();
	}
	Csub[index2(col, row, ldc)] = Csub[index2(col, row, ldc)] * beta + Cvalue * alpha;
}

blas_status sgemm_nn(int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	const int Mtile = 32;
	const int Ntile = 32;
	const int Ktile = 64;
	dim3 dimBlock(Mtile, Ntile);
	dim3 dimGrid(m / Mtile, n / Ntile);
	mat_mul_nn_kernel<Mtile, Ntile, Ktile><<<dimGrid, dimBlock>>>(m, n, k, A, lda, B, ldb, C, ldc, *alpha, *beta);
}