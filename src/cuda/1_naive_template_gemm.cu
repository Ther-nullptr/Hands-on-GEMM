#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
#include "sgemm.h"
#define BLOCK_SIZE 16 // we assume that every block has equal blockDim.x and blockDim.y

// C_new = alpha * A @ B + beta * C
template <typename T, typename T4>
__global__ void matrixMul(const T *A, const T *B, T *C,
                          int M, int N, int K, T alpha, T beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int baseX = blockIdx.x * blockDim.x;
    int baseY = blockIdx.y * blockDim.y;

    T c = (0.0f);

    if (tx < M && ty < N)
    {
        for (int i = 0; i < K; i++)
        {
            c += A[tx * K + i] * B[i * N + ty];
        }
        C[tx * N + ty] = beta * C[tx * N + ty] + alpha * c; // we multiply alpha here to reduce the alpha cal num.
    }
}

template <typename T, typename T4>
void gemm(int M, int N, int K, T *a, T *b, T *c, T alpha, T beta)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / (threadsPerBlock.x), (N + threadsPerBlock.y - 1) / (threadsPerBlock.y));
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<T, T4><<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}

template void gemm<float, float4>(int M, int N, int K, float *a, float *b, float *c, float alpha = 1., float beta = 0.);
template void gemm<double, double4>(int M, int N, int K, double *a, double *b, double *c, double alpha = 1., double beta = 1.);
template void gemm<half, half2>(int M, int N, int K, half *a, half *b, half *c, half alpha = __float2half(1.), half beta = __float2half(1.));
