## Commands

### docker

```bash
$ docker run -it --privileged=true --gpus all -v /home/yujin:/home/yujin -v /home/yujin/.vscode-server:$HOME/.vscode-server pytorch:2.0.1-cuda11.7-cudnn8-devel-ncu /bin/bash
$ docker ps # list the docker running containers
$ docker images # list the docker images
$ docker stop <container_id>
$ docker # commit 
```

### nsight compute

```bash
$ ncu --target-processes all -o profile <binary_file> # execute profiling
$ ncu --query-metrics | grep shared # find the metrics
```

## Website

### CUDA

[传统 CUDA GEMM 不完全指北](https://zhuanlan.zhihu.com/p/584236348)
[CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)
[cuda memory access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

### version settings

[CUDA/CUDNN/Pytorch](https://zhuanlan.zhihu.com/p/632478602)
[relation of -arch and -code in nvcc 1](https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute)
[relation of -arch and -code in nvcc 2](https://kaixih.github.io/nvcc-options/)
[sm_versions](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
[CUDA driver version and CUDA runtime version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#manually-installing-from-runfile)

### profiling

1. https://zhuanlan.zhihu.com/p/463144086

2. https://zhuanlan.zhihu.com/p/463435348

3. https://zhuanlan.zhihu.com/p/463844048

### other

* use template https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl
* FP16 https://ion-thruster.medium.com/an-introduction-to-writing-fp16-code-for-nvidias-gpus-da8ac000c17f
* cublas api https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
* cuda arch https://stackoverflow.com/questions/70533382/questions-about-cuda-macro-cuda-arch
* locate a package https://askubuntu.com/questions/129022/determine-destination-location-of-apt-get-install-package
* Nsight https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
* bank conflict https://zhuanlan.zhihu.com/p/538335829 https://zhuanlan.zhihu.com/p/436395393 https://zhuanlan.zhihu.com/p/603016056