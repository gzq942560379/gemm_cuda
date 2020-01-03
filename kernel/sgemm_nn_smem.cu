
#include "util.h"
#include "kernel.h"
#include "matrix.h"
#include <cuda_runtime_api.h>

template <int Mtile, int Ntile, int Ktile>
__global__ void mat_mul_nn_kernel(const Matrix A, const Matrix B, Matrix C, const float alpha, const float beta)
{
	int blockRow = blockIdx.x;
	int blockCol = blockIdx.y;
	int row = threadIdx.x;
	int col = threadIdx.y;
	Matrix Csub = C.getSubMatrix(blockRow, blockCol, Mtile, Ntile);

	float Cvalue = 0;
	__shared__ float As[Ktile][Mtile];
	__shared__ float Bs[Ntile][Ktile];
	for (int kb = 0; kb < (A.cols / Ktile); ++kb)
	{
		const Matrix Asub = A.getSubMatrix(blockRow, kb, Mtile, Ktile);
		const Matrix Bsub = B.getSubMatrix(kb, blockCol, Ktile, Ntile);
		for (int colA = col; colA < Ktile; colA += Ntile)
			As[colA][row] = Asub.getValue(row, colA);
		for (int rowB = row; rowB < Ktile; rowB += Mtile)
			Bs[col][rowB] = Bsub.getValue(rowB, col);
		__syncthreads();
		for (int k = 0; k < Ktile; ++k)
		{
			Cvalue += As[k][row] * Bs[col][k];
		}
		__syncthreads();
	}
	Csub.setValue(row, col, Csub.getValue(row, col) * beta + Cvalue * alpha);
}

blas_status sgemm_nn(int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	const Matrix a(m, k, lda, A);
	const Matrix b(k, n, ldb, B);
	Matrix c(m, n, ldc, C);
	const int Mtile = 16;
	const int Ntile = 16;
	const int Ktile = 32;
	dim3 dimBlock(Mtile, Ntile);
	dim3 dimGrid(m / Mtile, n / Ntile);
	mat_mul_nn_kernel<Mtile, Ntile, Ktile><<<dimGrid, dimBlock>>>(a, b, c, *alpha, *beta);
}