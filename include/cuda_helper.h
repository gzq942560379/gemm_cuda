#pragma once

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#define checkCudaError(val) _checkCudaError((val), #val, __FILE__, __LINE__)

#define checkCublasStatus(val) _checkCublasStatus((val), #val, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void _checkCudaError(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void _checkCublasStatus(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "cublas status at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result),func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

