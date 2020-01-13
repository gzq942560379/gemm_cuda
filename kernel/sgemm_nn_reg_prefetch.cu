
#include "util.h"
#include "kernel.h"
#include "matrix.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdio>

template <int Mtile, int Ntile, int Ktile>
__global__ void mat_mul_nn_kernel(const Matrix A, const Matrix B, Matrix C, const float alpha, const float beta)
{
	const int blockRow = blockIdx.x;
	const int blockCol = blockIdx.y;

	const int blockThreadNum = blockDim.x;
	const int blockThreadId = threadIdx.x;

	const int warpNum = blockThreadNum / warpSize;
	const int warpIdx = blockThreadId / warpSize;
	const int warpThreadId = blockThreadId % warpSize;

	const int warpRows = 2;
	const int warpCols = 4;
	const int warpRowId = warpIdx % warpRows;
	const int warpColId = warpIdx / warpRows;

	const int warpThreadRows = 8;
	const int warpThreadCols = 4;
	const int warpThreadRowId = warpThreadId % warpThreadRows;
	const int warpThreadColId = warpThreadId / warpThreadRows;

	const int warp_m = Mtile / warpRows;		  //  64 / 2 = 32
	const int Warp_n = Ntile / warpCols;		  //  64 / 4 = 16
	const int thread_m = warp_m / warpThreadRows; // 32 / 8 = 4
	const int thread_n = Warp_n / warpThreadCols; // 16 / 4 = 4

	Matrix Cblock = C.getSubMatrix(blockRow, blockCol, Mtile, Ntile);
	Matrix Cwarp = Cblock.getSubMatrix(warpRowId, warpColId, warp_m, Warp_n);
	Matrix Cthread = Cwarp.getSubMatrix(warpThreadRowId, warpThreadColId, thread_m, thread_n);

	const int ldc = thread_m;
	const int Csize = thread_m * thread_n;
	float Cvalue[Csize];

	memset(Cvalue, '\0', Csize * sizeof(float));

	const int lda = Mtile;
	const int ldb = Ktile;
	const int Asize = Mtile * Ktile;
	const int Bsize = Ktile * Ntile;

	__shared__ float As[2][Asize];
	__shared__ float Bs[2][Bsize];

	float Afragment[thread_m];
	float Bfragment[thread_n];

	int kb_pre = 0;
	int cur;
	int next = kb_pre & 1;
	Matrix Ablock_pre,Bblock_pre;
	if(kb_pre < (A.cols / Ktile)){
		Ablock_pre = A.getSubMatrix(blockRow, kb_pre, Mtile, Ktile);
		Bblock_pre = B.getSubMatrix(kb_pre, blockCol, Ktile, Ntile);
		for (int fid = blockThreadId; fid < Asize; fid += blockThreadNum)
			As[next][fid] = Ablock_pre.getValue(fid % lda, fid / lda);
		for (int fid = blockThreadId; fid < Bsize; fid += blockThreadNum)
			Bs[next][fid] = Bblock_pre.getValue(fid % ldb, fid / ldb);
	}

	for (int kb = 0; kb < (A.cols / Ktile); ++kb)
	{
		__syncthreads();
		const Matrix Ablock = Ablock_pre;
		const Matrix Bblock = Bblock_pre;
		kb_pre = kb + 1;
		cur = kb & 1;
		next = kb_pre & 1;
		if(kb_pre < (A.cols / Ktile)){
			Ablock_pre = A.getSubMatrix(blockRow, kb_pre, Mtile, Ktile);
			Bblock_pre = B.getSubMatrix(kb_pre, blockCol, Ktile, Ntile);
			for (int fid = blockThreadId; fid < Asize; fid += blockThreadNum)
				As[next][fid] = Ablock_pre.getValue(fid % lda, fid / lda);
			for (int fid = blockThreadId; fid < Bsize; fid += blockThreadNum)
				Bs[next][fid] = Bblock_pre.getValue(fid % ldb, fid / ldb);
		}
		const Matrix Awarp = Ablock.getSubMatrix(warpRowId, 0, warp_m, Ktile);
		const Matrix Bwarp = Bblock.getSubMatrix(0, warpColId, Ktile, Warp_n);
		const Matrix Athread = Awarp.getSubMatrix(warpThreadRowId, 0, thread_m, Ktile);
		const Matrix Bthread = Bwarp.getSubMatrix(0, warpThreadColId, Ktile, thread_n);
		for (int k = 0; k < Ktile; ++k)
		{
			for (int tr = 0; tr < thread_m; tr++)
				Afragment[tr] = As[cur][index2(k, tr, lda)];
			for (int tc = 0; tc < thread_n; tc++)
				Bfragment[tc] = Bs[cur][index2(tc, k, ldb)];
			for (int tc = 0; tc < thread_n; tc++)
				for (int tr = 0; tr < thread_m; tr++)
					Cvalue[index2(tc, tr, ldc)] += Afragment[tr] * Bfragment[tc];
		}
	}
	__syncthreads();
	for (int tc = 0; tc < thread_n; tc++)
	{
		for (int tr = 0; tr < thread_m; tr++)
		{
			Cthread.setValue(tr, tc, Cvalue[index2(tc, tr, ldc)] * alpha + Cthread.getValue(tr, tc) * beta);
		}
	}
}

blas_status sgemm_nn(int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	const Matrix a(m, k, lda, const_cast<float*>(A));
	const Matrix b(k, n, ldb, const_cast<float*>(B));
	Matrix c(m, n, ldc, C);
	const int Mtile = 64;
	const int Ntile = 64;
	const int Ktile = 8;
	int thread_num = 256;
	mat_mul_nn_kernel<Mtile, Ntile, Ktile><<<dim3(m / Mtile, n / Ntile), dim3(thread_num)>>>(a, b, c, *alpha, *beta);
}