cmake_minimum_required(VERSION 3.20)
project(opsparse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/options.cmake)
include(cmake/findcuda.cmake)

find_package(OpenMP REQUIRED)

set(OPSPARSE_SOURCES
    src/opsparse/core_ds/csr.cu
    src/opsparse/core_algo/spgemm/meta.cu
    src/opsparse/core_algo/spgemm/spgemm_phases.cu
    src/opsparse/core_algo/spgemm/spgemm.cu
    src/opsparse/core_algo/spgemm/cusparse_spgemm.cu
    src/opsparse/numeric/compute_flop.cu
    src/opsparse/profile/spgemm_timings.cpp
    src/opsparse/profile/perf.cpp
)

add_library(opsparse STATIC ${OPSPARSE_SOURCES})
target_include_directories(opsparse
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(opsparse PUBLIC cxx_std_17)
target_link_libraries(opsparse
    PUBLIC
        CUDA::cudart
        CUDA::cusparse
        OpenMP::OpenMP_CXX
)
set_target_properties(opsparse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(opsparse PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp -O3 --expt-relaxed-constexpr>
)

if(OPSPARSE_BUILD_BENCHMARKS)
    add_executable(opsparse_profile benchmarks/opsparse_profile.cu)
    target_link_libraries(opsparse_profile PRIVATE opsparse)
    target_include_directories(opsparse_profile PRIVATE benchmarks)

    add_executable(opsparse_reg benchmarks/opsparse_reg.cu)
    target_link_libraries(opsparse_reg PRIVATE opsparse)
    target_include_directories(opsparse_reg PRIVATE benchmarks)

    add_executable(cusparse_reg benchmarks/cusparse_reg.cu)
    target_link_libraries(cusparse_reg PRIVATE opsparse)
    target_include_directories(cusparse_reg PRIVATE benchmarks)
endif()
