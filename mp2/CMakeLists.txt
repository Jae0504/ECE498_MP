cmake_minimum_required(VERSION 3.18)

project(ece498_mp2_cuda_profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_COMPILER)
  set(ECE498_CUDA_BIN_HINTS /usr/local/cuda/bin /opt/cuda/bin)
  foreach(cuda_root IN ITEMS "${CUDAToolkit_ROOT}"
                            "$ENV{CUDAToolkit_ROOT}" "$ENV{CUDA_HOME}")
    if(NOT cuda_root STREQUAL "")
      list(PREPEND ECE498_CUDA_BIN_HINTS "${cuda_root}/bin")
    endif()
  endforeach()
  find_program(ECE498_NVCC
    NAMES nvcc
    HINTS ${ECE498_CUDA_BIN_HINTS})
  if(ECE498_NVCC)
    set(CMAKE_CUDA_COMPILER "${ECE498_NVCC}" CACHE FILEPATH
        "CUDA compiler" FORCE)
  endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # NVIDIA GB10 (DGX Spark) has compute capability 12.1.
  set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "CUDA architectures")
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "CUDA compiler not found. DGX Spark needs CUDA 13+. Set CMAKE_CUDA_COMPILER to its nvcc.")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(cpu_reference STATIC cpu/reference.cpp)
target_include_directories(cpu_reference PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(cpu_reference PRIVATE
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Debug>:-O0;-g3>
  $<$<CONFIG:Release>:-O3>)

add_executable(gpu_bench
  gpu/main.cu
  gpu/gemm.cu
  gpu/gemv.cu
  gpu/attention.cu)
target_include_directories(gpu_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/gpu)
target_link_libraries(gpu_bench PRIVATE cpu_reference CUDA::cudart)
set_target_properties(gpu_bench PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF)
target_compile_options(gpu_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME gpu_correctness_fast COMMAND gpu_bench --test)
  add_test(NAME gpu_correctness_required COMMAND gpu_bench --test-required)
  set_tests_properties(gpu_correctness_fast gpu_correctness_required
    PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(gpu_correctness_required PROPERTIES
    LABELS long
    TIMEOUT 900)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME script_workflow
      COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_scripts.py)
    set_tests_properties(script_workflow PROPERTIES TIMEOUT 120)
  endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
