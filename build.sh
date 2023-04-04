set -ex

CUDA_INSTALL_PATH=/usr/local/cuda-11.4
# CPLUS_INCLUDE_PATH=$CUDA_INSTALL_PATH/extras/CUPTI/include
# LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH
NVCC=$CUDA_INSTALL_PATH/bin/nvcc
# extras/CUPTI/include
# CUDA_INCLUDE=-I"$CUDA_INSTALL_PATH/include"
# CUPTI_INCLUDE=-I"$CUDA_INSTALL_PATH/extras/CUPTI/include"
CUDA_INCLUDE=""
CUPTI_INCLUDE=""

$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 -c $CUDA_INCLUDE $CUPTI_INCLUDE cuda_graphs_trace.cpp
$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 -c $CUDA_INCLUDE $CUPTI_INCLUDE vec.cu
$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 $CUDA_INCLUDE $CUPTI_INCLUDE -o cuda_graphs_trace vec.o cuda_graphs_trace.o -lcupti -lcuda
