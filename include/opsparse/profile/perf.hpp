#ifndef OPSPARSE_PROFILE_PERF_HPP_
#define OPSPARSE_PROFILE_PERF_HPP_

#include <cstddef>
#include <string>

namespace opsparse {

// Outcome of a single SpgemmReg run.
//
// The original opsparse code has NO overflow protection: mint = int (32-bit),
// so C.nnz, d_total_nnz, and d_row_flop can silently wrap. cudaMalloc on a
// wrapped (negative) nnz produces an out-of-memory error. This status enum
// makes each failure mode observable so a debug sweep can tell a real bug
// apart from the expected hardware / datatype limits.
enum class SpgemmStatus {
    ok = 0,             // multiplied successfully
    nonsquare = 1,      // A.N != B.M, skipped before any GPU work
    int32_overflow = 2, // total_flop or C.nnz would exceed INT32_MAX
    oom = 3,            // cudaMalloc returned cudaErrorMemoryAllocation
    cuda_error = 4,     // any other CUDA error
    exception = 5,      // std::exception caught at the library boundary
};

const char *to_cstr(SpgemmStatus s);

// Memory and status snapshot for a single SpgemmReg invocation.
//
// free_before / free_after are queried with cudaMemGetInfo. peak_used_bytes
// is (free_before - min(free_after, free_before)) — i.e. how much was
// still in flight at the end. For a run that releases its workspace the
// peak is observed only implicitly, so the library also exposes a running
// high-water mark updated whenever a large cudaMalloc succeeds.
struct SpgemmPerf {
    SpgemmStatus status = SpgemmStatus::ok;
    std::size_t device_total_bytes = 0;
    std::size_t free_before_bytes = 0;
    std::size_t free_after_bytes = 0;
    std::size_t peak_used_bytes = 0;
    std::size_t attempted_bytes = 0;   // last allocation attempted (useful on OOM)
    long long total_flop = 0;
    long long c_nnz = 0;               // -1 if not reached
    std::string error_msg;

    void reset();
    // Populated from cudaMemGetInfo; safe to call multiple times.
    void snapshot_free(std::size_t &out_free) const;
};

// Free / total GPU memory in bytes.
void query_cuda_memory(std::size_t &free_bytes, std::size_t &total_bytes);

}  // namespace opsparse

#endif  // OPSPARSE_PROFILE_PERF_HPP_
