cmake_minimum_required(VERSION 3.18)

project(ece498_mp1_cpu_profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CPU_NATIVE "Use -march=native for Release-like CPU builds" ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import matplotlib"
  RESULT_VARIABLE MATPLOTLIB_STATUS ERROR_QUIET OUTPUT_QUIET)
if(NOT MATPLOTLIB_STATUS EQUAL 0)
  message(WARNING "Automatic Roofline plots require Matplotlib. Install with: ${Python3_EXECUTABLE} -m pip install -r ${PROJECT_SOURCE_DIR}/requirements.txt")
endif()
configure_file(include/plot_config.h.in generated/plot_config.h @ONLY)

function(configure_cpu_target target)
  target_include_directories(${target} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(${target} PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-O0;-g3>
    $<$<CONFIG:Release>:-O3;-ffp-contract=fast>
    $<$<CONFIG:RelWithDebInfo>:-O3;-g;-ffp-contract=fast>)
  if(CPU_NATIVE)
    target_compile_options(${target} PRIVATE
      $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-march=native>)
  endif()
endfunction()

add_library(cpu_kernels STATIC
  cpu/gemm.cpp
  cpu/gemv.cpp
  cpu/attention.cpp
  cpu/reference.cpp)
configure_cpu_target(cpu_kernels)
target_link_libraries(cpu_kernels PUBLIC OpenMP::OpenMP_CXX)

add_library(roofline_plot STATIC cpu/plot.cpp)
configure_cpu_target(roofline_plot)
target_include_directories(roofline_plot PRIVATE ${PROJECT_BINARY_DIR}/generated)

add_executable(cpu_bench
  cpu/main.cpp
  cpu/tests.cpp)
target_link_libraries(cpu_bench PRIVATE cpu_kernels roofline_plot)
configure_cpu_target(cpu_bench)

add_executable(roofline_bench cpu/roofline.cpp)
target_link_libraries(roofline_bench PRIVATE roofline_plot OpenMP::OpenMP_CXX)
configure_cpu_target(roofline_bench)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cpu_correctness COMMAND cpu_bench --test)
  if(Python3_Interpreter_FOUND)
    add_test(NAME threaded_execution
      COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_threaded_execution.py
        $<TARGET_FILE:cpu_bench> $<TARGET_FILE:roofline_bench>)
    set_tests_properties(threaded_execution PROPERTIES TIMEOUT 120)
    add_test(NAME csv_output
      COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_csv_output.py
        $<TARGET_FILE:cpu_bench> $<TARGET_FILE:roofline_bench>)
    set_tests_properties(csv_output PROPERTIES TIMEOUT 120)
    if(MATPLOTLIB_STATUS EQUAL 0)
      add_test(NAME automatic_roofline
        COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tests/check_roofline_output.py
          $<TARGET_FILE:cpu_bench> $<TARGET_FILE:roofline_bench>
          ${PROJECT_SOURCE_DIR}/scripts/generate_roofline.py)
      set_tests_properties(automatic_roofline PROPERTIES TIMEOUT 180)
    endif()
  endif()
endif()

message(STATUS "Roofline Python: ${Python3_EXECUTABLE} (active virtual environment takes precedence at runtime)")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CPU native optimizations: ${CPU_NATIVE}")
