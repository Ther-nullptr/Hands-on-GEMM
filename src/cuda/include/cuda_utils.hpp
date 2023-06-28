#include <cuda_fp16.h>

__half fabs(__half a)
{
    float b = __half2float(a);
    return __float2half(fabs(b));
}