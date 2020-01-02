#include <iostream>
#include <sstream>

#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cstdlib>
#include <cstdio>

#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include "util.h"
#include "blas.h"
#include "cuda_helper.h"

using namespace std;

extern const int MM;
extern const int NN;
extern const int KK;
extern const float ALPHA;
extern const float BETA;

int main(int argc, char *argv[])
{
	int m = MM;
	int n = NN;
	int k = KK;
	float alpha = ALPHA;
	float beta = BETA;

	blas_operation transA = blas_operation::none;
	blas_operation transB = blas_operation::none;
	int lda = m;
	int ldb = k;
	int ldc = m;

	int size_a = m * k;
	int size_b = k * n;
	int size_c = m * n;

	vector<float> ha(size_a);
	vector<float> hb(size_b);
	vector<float> hc(size_c);
	vector<float> hc_gold(size_c);

	initialize_a_b_c_(ha, size_a, hb, size_b, hc, hc_gold, size_c);

	float *da, *db, *dc, *dc_gold;

	checkCudaError(cudaMalloc(&da, size_a * sizeof(float)));
	checkCudaError(cudaMalloc(&db, size_b * sizeof(float)));
	checkCudaError(cudaMalloc(&dc, size_c * sizeof(float)));
	checkCudaError(cudaMalloc(&dc_gold, size_c * sizeof(float)));
	checkCudaError(cudaMemcpy(da, ha.data(), sizeof(float) * size_a, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(db, hb.data(), sizeof(float) * size_b, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dc, hc.data(), sizeof(float) * size_c, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dc_gold, hc_gold.data(), sizeof(float) * size_c, cudaMemcpyHostToDevice));

	// timing
	float our_elapsed, cuda_elapsed;
	cudaEvent_t start, end;
	checkCudaError(cudaEventCreate(&start));
	checkCudaError(cudaEventCreate(&end));
	cublasHandle_t handle;
	checkCublasStatus(cublasCreate(&handle));
	checkCudaError(cudaEventRecord(start, NULL));

	cudaProfilerStart();
	checkBlasStatus(sgemm(transA, transB, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));
	cudaProfilerStop();

	checkCudaError(cudaEventRecord(end, NULL));
	checkCudaError(cudaEventSynchronize(end));
	checkCudaError(cudaEventElapsedTime(&our_elapsed, start, end));
	checkCudaError(cudaEventRecord(start, NULL));

	checkCublasStatus(cublasSgemm(handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, m, n, k, &alpha, da, lda, db, ldb, &beta, dc_gold, ldc));

	checkCudaError(cudaEventRecord(end, NULL));
	checkCudaError(cudaEventSynchronize(end));
	checkCudaError(cudaEventElapsedTime(&cuda_elapsed, start, end));

	checkCudaError(cudaMemcpy(hc.data(), dc, sizeof(float) * size_c, cudaMemcpyDeviceToHost));
	checkCudaError(cudaMemcpy(hc_gold.data(), dc_gold, sizeof(float) * size_c, cudaMemcpyDeviceToHost));

#ifdef DEBUG
	printMatrix("ha", ha.data(), m, k, lda);
	printMatrix("hb", hb.data(), k, n, ldb);
	printMatrix("hc", hc.data(), m, n, ldc);
	printMatrix("hc_glod", hc_gold.data(), m, n, ldc);
#endif

	check_s(hc_gold.data(), hc.data(), size_c);

	cout << "our time : " << our_elapsed << "ms" << endl;
	cout << "cuda time : " << cuda_elapsed << "ms" << endl;

	checkCudaError(cudaFree(da));
	checkCudaError(cudaFree(db));
	checkCudaError(cudaFree(dc));
	checkCudaError(cudaFree(dc_gold));

	return EXIT_SUCCESS;
}
