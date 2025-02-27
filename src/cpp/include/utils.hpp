// Copied from https://github.com/Cjkkkk/CUDA_gemm/blob/14b517370609d322647c55fe9136b6d81c2ba9a7/src/cpp/utils.cpp and https://github.com/Cjkkkk/CUDA_gemm/blob/14b517370609d322647c55fe9136b6d81c2ba9a7/src/cpp/include/utils.hpp

#pragma once

#include <iostream>
#include <type_traits>
#include <tuple>
#include "bcsr.hpp"
#include "csr.hpp"


void cal_block(bcsr *, float *);
void generate_bcsr(bcsr *, float *);

void cal_nnz(csr *, float *);
void generate_csr(csr *, float *);

void genRandomMatrix(float *A, int M, int N);
void FillMatrix(float *A, float num, int M, int N);
void genFixedMatrix(float *A, int M, int N);
void genSparseMatrix(float *A, int M, int N, int sparsity);
void copyMatrix(float *des, float *src, int M, int N);

#include <time.h>

void cal_block(bcsr *mat, float *data)
{
    for (int i = 0; i < mat->m_block * mat->n_block; i++)
    {
        mat->is_block_present[i] = 0;
    }
    for (int i = 0; i < mat->m * mat->n; i++)
    {
        if (data[i] != 0)
        {
            // 计算属于哪一个block
            int m_block_idx = i / mat->n / mat->m_block_sz;
            int n_block_idx = i % mat->n / mat->n_block_sz;
            if (mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] == 0)
            {
                mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] = 1;
                mat->nnz_block_num += 1;
            }
        }
    }
}

void generate_bcsr(bcsr *mat, float *data)
{
    int ptr = 0;
    int block_ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr++] = block_ptr;
    for (int i = 0; i < mat->m_block; i += 1)
    {
        for (int j = 0; j < mat->n_block; j += 1)
        {
            if (mat->is_block_present[i * mat->n_block + j] == 1)
            {
                mat->col_idx[block_ptr++] = j;
                // copy whole block into val
                for (int i_block = 0; i_block < mat->m_block_sz; i_block++)
                {
                    for (int j_block = 0; j_block < mat->n_block_sz; j_block++)
                    {
                        mat->val[ptr++] = data[(i * mat->m_block_sz + i_block) * mat->n + (j * mat->n_block_sz + j_block)];
                    }
                }
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr++] = block_ptr;
    }
}

void cal_nnz(csr *mat, float *data)
{
    for (int i = 0; i < mat->m * mat->n; i++)
    {
        if (data[i] != 0)
        {
            mat->nnz_num += 1;
        }
    }
}

void generate_csr(csr *mat, float *data)
{
    int ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr++] = ptr;
    for (int i = 0; i < mat->m; i += 1)
    {
        for (int j = 0; j < mat->n; j += 1)
        {
            if (data[i * mat->n + j] != 0)
            {
                mat->col_idx[ptr] = j;
                mat->val[ptr] = data[i * mat->n + j];
                ptr++;
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr++] = ptr;
    }
}

void genRandomMatrix(double *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    double a = 5.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (double)rand() / ((double)RAND_MAX / a);
        }
    }
}

void genRandomMatrix(float *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    float a = 5.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (float)rand() / ((float)RAND_MAX / a);
        }
    }
}

void genRandomMatrix(half *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    float a = 5.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = __float2half(rand() /(RAND_MAX / a));
        }
    }
}

void genRandomMatrix(int8_t *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = rand() % 127;
        }
    }
}

void genOneMatrix(int8_t *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i > j ? 1 : 0;
        }
    }
}

void genRandomMatrix(int32_t *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = rand() % ((long long)(1 << 31) - 1);
        }
    }
}

void FillMatrix(float *A, float num, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = num;
        }
    }
}

void genFixedMatrix(float *A, int M, int N)
{
    float val = 0;
    for (int i = 0; i < M * N; i++)
    {
        A[i] = val++;
    }
}

void genSparseMatrix(float *A, int M, int N, int sparsity)
{
    srand(time(NULL)); // Initialization, should only be called once.
    float a = 5.0;
    int nnz = M * N * (sparsity / 100.0);
    int nnz_stride = M * N / nnz;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if ((i * N + j) % nnz_stride == 0)
                A[i * N + j] = (float)rand() / ((float)RAND_MAX / a);
            else
            {
                A[i * N + j] = 0;
            }
        }
    }
}

template <typename T>
std::tuple<T, T> getError(const T a, const T b)
{
    double abs_error = fabs(a - b);
    double abs_val = fabs(b);
    return std::make_tuple(abs_error, abs_val);
}

template <>
std::tuple<half, half> getError(const half a, const half b)
{
    double abs_error = (double)fabs(__half2float(a) - __half2float(b));
    double abs_val = (double)fabs(__half2float(b));
    return std::make_tuple(abs_error, abs_val);
}

template <typename T>
bool checkError(int i, const double rel_err, const T* h_C, const T* h_C1)
{
    double eps = 1e-6;
    if (rel_err > eps)
    {
        printf("rel_err: %f, eps: %f\n", rel_err, eps);
        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], h_C1[i], eps);
        return false;
    }
    return true;
}

template <>
bool checkError(int i, const double rel_err, const half* h_C, const half* h_C1)
{
    double eps = 1e-4;
    if (rel_err > eps)
    {
        printf("rel_err: %f, eps: %f\n", rel_err, eps);
        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, __half2float(h_C[i]), __half2float(h_C1[i]), eps);
        return false;
    }
    return true;
}

template <typename T>
void copyMatrix(T *des, T *src, size_t M, size_t N)
{
    for (int i = 0; i < M * N; i++)
    {
        des[i] = src[i];
    }
}

template <typename T>
void showMatrix(T *A, int M, int N, const char *msg)
{
    T sum = 0;
    printf("===============%s===========\n", msg);
    for (int i = 0; i < M; i++)
    {
        std::cout << i << ": ";
        for (int j = 0; j < N; j++)
        {
            if (std::is_same_v<T, int8_t>)
            {
                std::cout << int(A[i * N + j]) << " ";
            }
            else
            {
                std::cout << A[i * N + j] << " ";
            }
            sum += A[i * N + j];
        }
        std::cout << std::endl;
    }
    if (std::is_same_v<T, int8_t>)
        std::cout << int(sum) << std::endl;
    else
        std::cout << sum / (M * N) << std::endl;
    printf("============================\n");
}

template <typename T>
void transposeMatrix(T *A, size_t M, size_t N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            std::swap(A[i * M + j], A[j * N + i]);
        }
    }
}