#include "profiling.h"

#include <iostream>

#define CUPTI_CALL(call)                           \
  [&]() -> CUptiResult {                           \
    CUptiResult _status_ = call;                   \
    if (_status_ != CUPTI_SUCCESS) {               \
      const char* _errstr_ = nullptr;              \
      cuptiGetResultString(_status_, &_errstr_);   \
      std::cerr << "function" << #call             \
                << " failed with error "           \
                << _errstr_ << " ("                \
                << (int)_status_ << ")"            \
                << std::endl;                      \
    }                                              \
    return _status_;                               \
  }()

void CUPTIAPI bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  std::cout << "  | bufferRequestedTrampoline stub" << std::endl;
}
void CUPTIAPI bufferCompletedTrampoline(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  std::cout << "  | bufferCompletedTrampoline stub" << std::endl;
}
static void CUPTIAPI callback_switchboard(
   void* /* unused */,
   CUpti_CallbackDomain domain,
   CUpti_CallbackId cbid,
   const CUpti_CallbackData* cbInfo) {
  std::cout << "  | callback_switchboard stub" << std::endl;
}



ProfilingState::ProfilingState() {
std::cout << " | initializing ProfilingState" << std::endl;


CUptiResult lastCuptiStatus_ = CUPTI_CALL(
    cuptiSubscribe(&subscriber_,
    (CUpti_CallbackFunc)callback_switchboard,
    nullptr));

// CuptiActivityProfiler.cpp:645
CUPTI_CALL(
    cuptiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline));

// CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

}

ProfilingState& getProfilingState() {
  static ProfilingState ps = ProfilingState();
  return ps;
}
