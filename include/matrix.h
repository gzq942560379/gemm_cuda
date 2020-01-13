#pragma once
#include <cuda_runtime_api.h>

struct Matrix
{
	float *values;
	int rows;
	int cols;
	int ld;
	__host__ __device__ Matrix(){};
	__host__ __device__ Matrix(int rows, int cols, int ld, float *values) : rows(rows), cols(cols), ld(ld), values(values){};
	__device__ float getValue(const int r, const int c) const
	{
		return values[index2(c, r, ld)];
	}
	__device__ inline void setValue(const int r, const int c, const float v)
	{
		values[index2(c, r, ld)] = v;
	}
	__device__ inline Matrix getSubMatrix(const int row_block_offset, const int col_block_offset, const int rows_block_size, const int cols_block_size) const
	{
		return Matrix(rows_block_size, cols_block_size, ld, &values[index2(col_block_offset * cols_block_size, row_block_offset * rows_block_size, ld)]);
	}
};