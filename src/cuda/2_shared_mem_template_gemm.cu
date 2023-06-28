#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>

#define DEBUG
constexpr size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
constexpr size_t BLOCK_M = 128;   // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
constexpr size_t BLOCK_N = 128;
constexpr size_t BLOCK_K = 8; // don't set 64 here, it will cause bank conflict and lower occupancy.
constexpr size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
constexpr size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

constexpr int shared_memory_A = BLOCK_M * BLOCK_K;
constexpr int shared_memory_B = BLOCK_N * BLOCK_K;
constexpr int shared_memory_element = shared_memory_A + shared_memory_B;

#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

template <typename T, typename T4>
__global__ void matrixMul(const T *A, const T *B, T *C,
                          int M, int N, int K, T alpha, T beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    T c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    T resC[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    __shared__ T subA[BLOCK_M * BLOCK_K];
    __shared__ T subB[BLOCK_N * BLOCK_K];

    T4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    T4 regA[BLOCK_M_COMPUTE / 4];

    int shiftA = baseY * K;
    int shiftB = baseX;
    int shiftC = (baseY + threadIdx.x * BLOCK_M_COMPUTE) * N + baseX + threadIdx.y * BLOCK_N_COMPUTE;
    const int baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    #ifdef DEBUG
        printf("block: (%d,%d), thread: (%d,%d), shiftA: %d, shiftB: %d, shiftC: %d, baseIdx: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, shiftA, shiftB, shiftC, baseIdx);
    #endif

    const T *baseA = A + shiftA;
    const T *baseB = B + shiftB;
    T *baseC = C + shiftC;

    int rowA = baseIdx / 2, rowB = baseIdx / (BLOCK_N / 4), colA = (baseIdx & 1) * 4, colB = (baseIdx * 4) % BLOCK_N;

    for (int i = 0; i < K; i += BLOCK_K)
    {
        regB[0] = *reinterpret_cast<const T4 *>(baseB + i * N + rowB * N + colB);
        regA[0] = *reinterpret_cast<const T4 *>(baseA + i + rowA * K + colA);
        *reinterpret_cast<T4 *>(&subB[baseIdx * 4]) = regB[0];
        subA[rowA + colA * BLOCK_M] = regA[0].x;
        subA[rowA + (colA + 1) * BLOCK_M] = regA[0].y;
        subA[rowA + (colA + 2) * BLOCK_M] = regA[0].z;
        subA[rowA + (colA + 3) * BLOCK_M] = regA[0].w;

        __syncthreads();
#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regA[0] = *reinterpret_cast<T4 *>(&subA[(threadIdx.x * BLOCK_M_COMPUTE) + ii * BLOCK_M]);
            regA[1] = *reinterpret_cast<T4 *>(&subA[(threadIdx.x * BLOCK_M_COMPUTE + 4) + ii * BLOCK_M]);

            regB[0] = *reinterpret_cast<T4 *>(&subB[threadIdx.y * BLOCK_N_COMPUTE + BLOCK_N * ii]);
            regB[1] = *reinterpret_cast<T4 *>(&subB[threadIdx.y * BLOCK_N_COMPUTE + 4 + BLOCK_N * ii]);

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4)
            *reinterpret_cast<T4 *>(&resC[i * BLOCK_M_COMPUTE + j]) = *reinterpret_cast<T4 *>(&baseC[i * N + j]);

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j++)
            resC[i * BLOCK_M_COMPUTE + j] = resC[i * BLOCK_M_COMPUTE + j] * beta + alpha * c[i * BLOCK_M_COMPUTE + j];

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4)
            *reinterpret_cast<T4 *>(&baseC[i * N + j]) = *reinterpret_cast<T4 *>(&resC[i * BLOCK_M_COMPUTE + j]);
}

template <typename T, typename T4>
void gemm(int M, int N, int K, T *a, T *b, T *c, T alpha = 1, T beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<T, T4><<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}

template void gemm<float, float4>(int M, int N, int K, float *a, float *b, float *c, float alpha = 1., float beta = 0.);
template void gemm<double, double4>(int M, int N, int K, double *a, double *b, double *c, double alpha = 1., double beta = 1.);