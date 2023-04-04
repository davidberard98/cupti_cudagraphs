```
$ mkdir build
$ cd build
$ CUDACXX=/usr/local/cuda-11.4/bin/nvcc CUDA_SOURCE_DIR=/usr/local/cuda-11.4 cmake ..
$ make -j32
$ ./cupti_cudagraphs
```
