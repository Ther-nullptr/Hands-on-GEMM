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
#include "sgemm.h"

#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)
#define MAXSIZE(type) (sizeof(type) * nmax * nmax)

using datatype = double;
using datatype_4 = double4;

int main(int argc, char **argv)
{
    std::vector<int> test_sizes;
    size_t nmax = 256;
    printf("MAX_TEST_SIZE: %d\n", nmax);
    bool miss_align = true, ignore_error = false;

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

    size_t M = nmax;
    size_t K = nmax;
    size_t N = nmax;

    printf("\nSize M: %u, N: %u, K: %u\n", M, N, K);

    datatype alpha = 2.0;
    datatype beta = 2.0;

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

    checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(datatype), cudaMemcpyHostToDevice));

    gemm<datatype, datatype_4>(M, N, K, d_A, d_B, d_C, alpha, beta);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C1;
}