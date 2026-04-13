option(OPSPARSE_BUILD_BENCHMARKS "Build SpGEMM benchmark executables" ON)
option(OPSPARSE_BUILD_TESTS "Build unit tests" OFF)
option(OPSPARSE_BUILD_EXAMPLES "Build examples" OFF)

set(OPSPARSE_CUDA_ARCH "80" CACHE STRING "Target CUDA compute capability (e.g. 70, 80, 90)")
