#pragma once
#include <cuda_runtime_api.h>

struct Matrix
{
	float *values;
	const float *const_values;
	const int rows;
	const int cols;
	const int ld;
	__host__ __device__ Matrix(int rows, int cols, int ld, float *values) : rows(rows), cols(cols), ld(ld), values(values), const_values(nullptr){};
	__host__ __device__ Matrix(int rows, int cols, int ld, const float *const_values) : rows(rows), cols(cols), ld(ld), values(nullptr), const_values(const_values){};
	__device__ float getValue(const int r, const int c) const
	{
		return const_values[index2(c, r, ld)];
	}
	__device__ float getValue(const int r, const int c)
	{
		return values[index2(c, r, ld)];
	}
	__device__ inline void setValue(const int r, const int c, const float v)
	{
		values[index2(c, r, ld)] = v;
	}
	__device__ inline Matrix getSubMatrix(const int row_block_offset, const int col_block_offset, const int rows_block_size, const int cols_block_size) const
	{
		return Matrix(rows_block_size, cols_block_size, ld, &const_values[index2(col_block_offset * cols_block_size, row_block_offset * rows_block_size, ld)]);
	}
	__device__ inline Matrix getSubMatrix(const int row_block_offset, const int col_block_offset, const int rows_block_size, const int cols_block_size)
	{
		return Matrix(rows_block_size, cols_block_size, ld, &values[index2(col_block_offset * cols_block_size, row_block_offset * rows_block_size, ld)]);
	}
};