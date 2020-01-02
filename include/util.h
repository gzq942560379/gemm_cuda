#pragma once

#include <cstdlib>
#include <vector>
#include <cstdio>
using std::vector;

#include "blas.h"

#define index2(y, x, ldx) ((x) + (y) * (ldx))
#define index3(z, y, x, ldy, ldx) (index2(index2(z, y, ldy), x, ldx))
#define index4(w, z, y, x, ldz, ldy, ldx) (index2(index2(index2(w, z, ldz), y, ldy), x, ldx))

bool check_s(const float *answer_data, const float *result_data, size_t size);

void initialize_a_b_c(vector<float> &ha,
                      int size_a,
                      vector<float> &hb,
                      int size_b,
                      vector<float> &hc,
                      vector<float> &hc_gold,
                      int size_c);

void initialize_a_b_c_(vector<float> &ha,
                       int size_a,
                       vector<float> &hb,
                       int size_b,
                       vector<float> &hc,
                       vector<float> &hc_gold,
                       int size_c);

void printMatrix(const char *name, float *A, int m, int n, int lda);

void _checkBlasStatus(blas_status result, char const *const func, const char *const file, int const line);
const char *_blasGetErrorEnum(blas_status status);
#define checkBlasStatus(result) _checkBlasStatus((result), #result, __FILE__, __LINE__)

