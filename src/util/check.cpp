#include "util.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

bool check_s(const float* answer_data, const float* result_data, size_t size)
{
	float norm_1 = 0.0, norm_2 = 0.0, norm_inf = 0.0;
	for (size_t i = 0; i < size; i++)
	{
		float err = fabsf(answer_data[i] - result_data[i]);
		norm_1 += err;
		norm_2 += err * err;
		norm_inf = std::max(norm_inf, err);
	}
	if (norm_inf > 1e-2)
	{
		printf("\nSignificant numeric error.\n");
		printf("inf-norm = %.16f\n\n", norm_inf);
		return false;
	}
	else
	{
		printf("\ncorrect!!!\n");
		printf("inf-norm = %.16f\n\n", norm_inf);
		return true;
	}
}

const char *status_name[] = {
    "success",
    "not support",
};

const char *_blasGetErrorEnum(blas_status status)
{
    return status_name[status];
}

void _checkBlasStatus(blas_status result, char const *const func, const char *const file,
                      int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _blasGetErrorEnum(result), func);
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}