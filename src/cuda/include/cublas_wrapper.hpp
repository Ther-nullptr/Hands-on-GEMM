#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cublas_api.h>

// float version
inline cublasStatus_t cublasGgemm(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const float *A, int lda,
        const float *B, int ldb,
        const float *beta,
        float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// double version
inline cublasStatus_t cublasGgemm(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const double *alpha,
        const double *A, int lda,
        const double *B, int ldb,
        const double *beta,
        double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// half version
inline cublasStatus_t cublasGgemm(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half *alpha,
        const half *A, int lda,
        const half *B, int ldb,
        const half *beta,
        half *C, int ldc)
{
    return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}