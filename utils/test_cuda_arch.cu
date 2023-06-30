#include <cstdio>
#include <cuda_fp16.h>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

__device__ void print_arch(){
  const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}

__global__ void example()
{
   print_arch();
}

int main(){

    example<<<1,1>>>();
    cudaDeviceSynchronize();
}