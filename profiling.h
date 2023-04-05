
#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <map>

struct ProfilingState {
  ProfilingState();
  CUpti_SubscriberHandle subscriber_ {0};
};

ProfilingState& getProfilingState();
