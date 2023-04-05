#include <cuda.h>
#include <cupti.h>

struct ProfilingState {
  ProfilingState();
  CUpti_SubscriberHandle subscriber_{0};
};

ProfilingState &getProfilingState();
