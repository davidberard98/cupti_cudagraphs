set -ex

CUDA_INSTALL_PATH=/usr/local/cuda-11.6
CPLUS_INCLUDE_PATH=$CUDA_INSTALL_PATH/extras/CUPTI/include
# LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH
NVCC=$CUDA_INSTALL_PATH/bin/nvcc
# extras/CUPTI/include

$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 -c -I"/usr/local/cuda-11.6/include" -I"/usr/local/cuda-11.6/extras/CUPTI/include" cuda_graphs_trace.cpp
$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 -c -I"/usr/local/cuda-11.6/include" -I"/usr/local/cuda-11.6/extras/CUPTI/include" vec.cu
$CUDA_INSTALL_PATH/bin/nvcc -arch=sm_80 -std=c++17 -I"/usr/local/cuda-11.6/include" -I"/usr/local/cuda-11.6/extras/CUPTI/include" -o cuda_graphs_trace vec.o cuda_graphs_trace.o -L/usr/local/cuda-11.6/extras/CUPTI/lib64 -lcupti -L/usr/local/cuda-11.6/lib -L/usr/local/cuda-11.6/lib64 -lcuda
