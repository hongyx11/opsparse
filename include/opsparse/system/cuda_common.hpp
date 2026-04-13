#ifndef OPSPARSE_SYSTEM_CUDA_COMMON_HPP_
#define OPSPARSE_SYSTEM_CUDA_COMMON_HPP_

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

namespace opsparse {

using mint = int;
using mdouble = double;

inline bool expect_true(bool x) { return __builtin_expect(x, 1); }
inline bool expect_false(bool x) { return __builtin_expect(x, 0); }

inline void check_cuda(cudaError_t err, const char *file, int line)
{
    if (expect_false(err != cudaSuccess)) {
        std::printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        throw std::exception();
    }
}

inline void check_cusparse(cusparseStatus_t status, const std::string &msg = "")
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CuSparse error: " << msg << std::endl;
        throw std::exception();
    }
}

template <typename T>
inline void d2h(T *dst, const T *src, std::size_t bytes)
{
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

template <typename T>
inline void h2d(T *dst, const T *src, std::size_t bytes)
{
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

template <typename T>
inline void d2d(T *dst, const T *src, std::size_t bytes)
{
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
}

inline void cuda_runtime_warmup()
{
    int *d = nullptr;
    check_cuda(cudaMalloc(&d, 4), __FILE__, __LINE__);
    check_cuda(cudaFree(d), __FILE__, __LINE__);
    check_cuda(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

}  // namespace opsparse

// Error-check macros kept as macros because they require __FILE__ / __LINE__.
#define OPSPARSE_CHECK_CUDA(err) (::opsparse::check_cuda((err), __FILE__, __LINE__))
#define OPSPARSE_CHECK_CUSPARSE(status, ...) (::opsparse::check_cusparse((status), ##__VA_ARGS__))

#endif  // OPSPARSE_SYSTEM_CUDA_COMMON_HPP_
