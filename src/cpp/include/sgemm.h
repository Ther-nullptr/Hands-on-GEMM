#pragma once

#include <cuda_fp16.h>

template<typename T>
void sgemm(int M, int N, int K, T *a, T *b, T *c, T alpha, T beta);

extern template void sgemm<float>(int M, int N, int K, float *a, float *b, float *c, float alpha, float beta);
extern template void sgemm<double>(int M, int N, int K, double *a, double *b, double *c, double alpha, double beta);
extern template void sgemm<__half>(int M, int N, int K, __half *a, __half *b, __half *c, __half alpha, __half beta);
