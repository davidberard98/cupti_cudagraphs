set -ex

CUDA_INSTALL_PATH=/usr/local/cuda-11.4
NVCC=$CUDA_INSTALL_PATH/bin/nvcc
CUDA_INCLUDE=""
CUPTI_INCLUDE=""
ARCH="-arch=sm_80"

$CUDA_INSTALL_PATH/bin/nvcc $ARCH -std=c++17 -c $CUDA_INCLUDE $CUPTI_INCLUDE profiling.cpp
$CUDA_INSTALL_PATH/bin/nvcc $ARCH -std=c++17 -c $CUDA_INCLUDE $CUPTI_INCLUDE main.cu
$CUDA_INSTALL_PATH/bin/nvcc $ARCH -std=c++17 $CUDA_INCLUDE $CUPTI_INCLUDE -o main main.o profiling.o -lcupti -lcuda
