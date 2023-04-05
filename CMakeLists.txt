cmake_minimum_required(VERSION 3.17)

project(cupti_cudagraphs LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)

if (NOT CUDA_SOURCE_DIR)
    set(CUDA_SOURCE_DIR "$ENV{CUDA_SOURCE_DIR}")
    message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

if (NOT CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS "${CUDA_SOURCE_DIR}/include")
endif()

set(KINETO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/kineto/libkineto/include")


if (NOT CUPTI_INCLUDE_DIR)
    set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/extras/CUPTI/include")
endif()

set(CUPTI_LIB_NAME libcupti.so)

find_library(CUPTI_LIBRARY_PATH ${CUPTI_LIB_NAME} PATHS
		${CUDA_SOURCE_DIR}
		${CUDA_SOURCE_DIR}/extras/CUPTI/lib64
		${CUDA_SOURCE_DIR}/lib
		${CUDA_SOURCE_DIR}/lib64
		NO_DEFAULT_PATH)

if(CUPTI_LIBRARY_PATH)
  set(CUDA_cupti_LIBRARY ${CUPTI_LIBRARY_PATH})
endif()

include_directories(${PROJECT_SOURCE_DIR}/kineto/libkineto)

add_subdirectory("kineto/libkineto")
target_compile_options(kineto PRIVATE "-g")

add_executable(cupti_cudagraphs vec.cu profiling.cpp)

set_target_properties( cupti_cudagraphs
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS -arch=sm_80)
target_compile_options(cupti_cudagraphs PUBLIC -lcupti -lcuda -g)
target_include_directories(cupti_cudagraphs PUBLIC
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${CUPTI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${KINETO_INCLUDE_DIR}>)
# target_link_libraries(cupti_cudagraphs CUDA::cudart)
# target_link_libraries(cupti_cudagraphs CUDA::cuda_driver)
target_link_libraries(cupti_cudagraphs
    "${CUDA_cupti_LIBRARY}"
    fmt::fmt-header-only
    kineto)
