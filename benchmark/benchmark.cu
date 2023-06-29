// Copied from https://github.com/Cjkkkk/CUDA_gemm/blob/14b517370609d322647c55fe9136b6d81c2ba9a7/benchmark/benchmark_dense.cu

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <functional>
#include <type_traits>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "cuda_help_func.hpp"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include "cublas_wrapper.hpp"
#include "gemm.h"

#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)
#define MAXSIZE(type) (sizeof(type) * nmax * nmax)

using datatype = float;
using datatype_4 = float4;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: ./main [MAX_TEST_SIZE]\n");
        exit(0);
    }
    std::vector<int> test_sizes;
    size_t nmax = atoi(argv[1]);
    printf("MAX_TEST_SIZE: %d\n", nmax);
    bool miss_align = true, ignore_error = false;
    if (argc > 2)
    {
        ignore_error = atoi(argv[2]) == 1;
    }
    for (int i = 128; i <= nmax + 127; i += 128)
    {
        test_sizes.emplace_back(i);
    }

    nmax = test_sizes[test_sizes.size() - 1]; // we assume the last element is the largest one

    datatype *h_A = new datatype[nmax * nmax];
    datatype *h_B = new datatype[nmax * nmax];
    datatype *h_C = new datatype[nmax * nmax];
    datatype *h_C1 = new datatype[nmax * nmax];

    datatype *d_A;
    datatype *d_B;
    datatype *d_C;

    checkCudaErrors(cudaMalloc(&d_A, MAXSIZE(datatype)));
    checkCudaErrors(cudaMalloc(&d_B, MAXSIZE(datatype)));
    checkCudaErrors(cudaMalloc(&d_C, MAXSIZE(datatype)));

    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));

    std::string type_name = "";
    if (std::is_same<datatype, float>::value)
    {
        type_name = "float";
    }
    else if (std::is_same<datatype, double>::value)
    {
        type_name = "double";
    }
    else if (std::is_same<datatype, half>::value)
    {
        type_name = "half";
    }
    else
    {
        printf("Unsupported type!\n");
        exit(1);
    }

    FILE *fp;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    char buffer[32];
    std::strftime(buffer, 32, "%Y-%m-%d-%H-%M-%S", &tm);
    std::string filename = std::string(buffer) + type_name + ".csv";
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, "M, N, K, my gemm, cublas\n");
    fclose(fp);

    for (const auto &i : test_sizes)
    {
        size_t M = i;
        size_t K = i;
        size_t N = i;

        printf("\nSize M: %u, N: %u, K: %u\n", M, N, K);

        double msecPerMatrixMul[2] = {0, 0};
        double gigaFlops[2] = {0, 0};
        double flopsPerMatrixMul = 2.0 * M * N * K;

        datatype alpha = (2.0);
        datatype beta = (2.0);

        // generate data
        genRandomMatrix(h_A, M, K);
        genRandomMatrix(h_B, K, N);
        genRandomMatrix(h_C, M, N);
        copyMatrix(h_C1, h_C, M, N);

        checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(datatype), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(datatype), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float msecTotal = 0;
        int nIter = 10;

        checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(datatype), cudaMemcpyHostToDevice));

        // dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        // dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        // if (N % BLOCK_SIZE_N != 0)
        //     dimGrid.x++;
        // if (M % BLOCK_SIZE_M != 0)
        //     dimGrid.y++;

        // warm up here (not sure whether we need this or not)
        gemm<datatype, datatype_4>(M, N, K, d_A, d_B, d_C, alpha, beta);

        checkCudaErrors(cudaEventRecord(start));
        // printf("Grid Dim: (%d %d) Block Dim: (%d %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
        for (int run = 0; run < nIter; run++)
        {
            gemm<datatype, datatype_4>(M, N, K, d_A, d_B, d_C, alpha, beta);
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        msecPerMatrixMul[0] = msecTotal / nIter;
        gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
        printf("My gemm Performance=%.2f GFlop/s, Time=%.3f msec, Size=%.0f Ops,\n",
               gigaFlops[0],
               msecPerMatrixMul[0],
               flopsPerMatrixMul);

        // cublas
        checkCudaErrors(cudaMemcpy(d_C, h_C1, CSIZE(datatype), cudaMemcpyHostToDevice));
        // warmup here (not sure whether we need this or not)
        checkCuBlasErrors(
            cublasGgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha,
                        d_B, N, d_A, K, &beta, d_C, N));
        checkCudaErrors(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            checkCuBlasErrors(
                cublasGgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha,
                            d_B, N, d_A, K, &beta, d_C, N));
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        msecPerMatrixMul[1] = msecTotal / nIter;
        gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
        printf("CuBlas Performance=%.2f GFlop/s, Time=%.3f msec, Size=%.0f Ops,\n",
               gigaFlops[1],
               msecPerMatrixMul[1],
               flopsPerMatrixMul);

        // record
        fp = fopen(filename.c_str(), "a");
        fprintf(fp, "%d, %d, %d, %.2f, %.2f, %.8f\n", M, N, K, gigaFlops[0], gigaFlops[1], gigaFlops[0] / gigaFlops[1]);
        fclose(fp);

        if (!ignore_error)
        {
            checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(datatype), cudaMemcpyHostToDevice));
            gemm<datatype, datatype_4>(M, N, K, d_A, d_B, d_C, alpha, beta);
            checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(datatype), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_C, h_C1, CSIZE(datatype), cudaMemcpyHostToDevice));
            checkCuBlasErrors(
                cublasGgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha,
                            d_B, N, d_A, K, &beta, d_C, N));
            checkCudaErrors(cudaMemcpy(h_C1, d_C, CSIZE(datatype), cudaMemcpyDeviceToHost));

            for (int i = 0; i < M * N; i++)
            {
                double abs_err = 0.;
                double abs_val = 0.;
                std::tie(abs_err, abs_val) = getError(h_C1[i], h_C[i]);
                
                double dot_length = M;
                double rel_err = abs_err / abs_val / dot_length;
                bool result = checkError(i, rel_err, h_C, h_C1);
                if (!result)
                {
                    exit(1);
                }
            }
            printf("No Error\n");
        }
        else
        {
            printf("Ignore the error.\n");
        }

        printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    }
    cublasDestroy(blas_handle);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C1;
}