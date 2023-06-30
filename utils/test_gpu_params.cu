#include <iostream>
int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM number: " << devProp.multiProcessorCount << std::endl;
    std::cout << "max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "registers per sm: " << devProp.regsPerMultiprocessor << std::endl;
    std::cout << "shared memory per sm: " << devProp.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
}