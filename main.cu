// modified copy of from NVIDIA's CUPTI cudagraph example
#include <cstddef>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <cupti.h>

#include "profiling.h"

#define N 500000 // tuned such that kernel takes a few microseconds

__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) out_d[idx]=1.23*in_d[idx];
}

#define CUPTI_CALL(call)                           \
  [&]() -> CUptiResult {                           \
    CUptiResult _status_ = call;                   \
    if (_status_ != CUPTI_SUCCESS) {               \
      const char* _errstr_ = nullptr;              \
      cuptiGetResultString(_status_, &_errstr_);   \
      std::cerr <<                  \
          "function " << #call << " failed with error " << _errstr_ << " (" << (int)_status_ << ")"; \
    }                                              \
    return _status_;                               \
  }()

#define CHECK(call)                                                      \
{                                                                        \
  const cudaError_t error = call;                                        \
  if (error != cudaSuccess) {                                            \
    printf("Error: %s:%d", __FILE__, __LINE__);                          \
    printf("code:%d, reason %s\n", error, cudaGetErrorString(error));    \
    exit(1);                                                             \
  }                                                                      \
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

struct TimerGuard {
  double start_time;
  std::string label_;
  TimerGuard(std::string label = "") : label_(label) {
    start_time = cpuSecond();
  }
  ~TimerGuard() {
    double end_time = cpuSecond();
    std::cout << " time " << label_ << ": " << int((end_time - start_time)*1e6) << " us " << std::endl;
  }
};

void startProfiling() {
  // initialize static profiling state object, see profiling.cpp
  getProfilingState();
}

int main() {
#define NSTEP 1000
#define NKERNEL 20

  int threads = 512;
  int blocks = (N + threads - 1) / threads;

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  float *in_d, *out_d;
  CHECK(cudaMalloc((float**)  &in_d, N*sizeof(float)));
  CHECK(cudaMalloc((float**) &out_d, N*sizeof(float)));

  {
    TimerGuard guard("without cuda graph");
    for(int istep=0; istep<NSTEP; istep++){
      for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
        shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        cudaStreamSynchronize(stream);
      }
    }
  }

	bool graphCreated=false;
	cudaGraph_t graph;
	cudaGraphExec_t instance;

  // Initialize cudagraph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
  }
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  graphCreated=true;

  startProfiling();

  {
    TimerGuard guard("WITH cuda graph");
    for(int istep=0; istep<NSTEP; istep++){
      std::cerr << " step " << istep << std::endl;
      CHECK(cudaGraphLaunch(instance, stream));
      cudaStreamSynchronize(stream);
    }
  }

  return 0;
}
