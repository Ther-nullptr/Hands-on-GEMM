#pragma once

#include <cuda_fp16.h>

template<typename T, typename T4>
void gemm(int M, int N, int K, T *a, T *b, T *c, T alpha, T beta);

// extern template void gemm<float, float4>(int M, int N, int K, float *a, float *b, float *c, float alpha, float beta);
// extern template void gemm<double, double4>(int M, int N, int K, double *a, double *b, double *c, double alpha, double beta);
// extern template void gemm<__half, __half2>(int M, int N, int K, __half *a, __half *b, __half *c, __half alpha, __half beta);
