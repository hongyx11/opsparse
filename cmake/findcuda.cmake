# findcuda.cmake — locate CUDA, cuSPARSE, and configure the CUDA toolchain.

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED OPSPARSE_CUDA_ARCH)
    set(OPSPARSE_CUDA_ARCH "80")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${OPSPARSE_CUDA_ARCH}")

message(STATUS "opsparse: CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")
message(STATUS "opsparse: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
