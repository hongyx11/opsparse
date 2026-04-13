#include "opsparse/profile/perf.hpp"

#include <cuda_runtime.h>

namespace opsparse {

const char *to_cstr(SpgemmStatus s)
{
    switch (s) {
        case SpgemmStatus::ok:
            return "ok";
        case SpgemmStatus::nonsquare:
            return "nonsquare";
        case SpgemmStatus::int32_overflow:
            return "int32_overflow";
        case SpgemmStatus::oom:
            return "oom";
        case SpgemmStatus::cuda_error:
            return "cuda_error";
        case SpgemmStatus::exception:
            return "exception";
    }
    return "unknown";
}

void SpgemmPerf::reset()
{
    status = SpgemmStatus::ok;
    device_total_bytes = 0;
    free_before_bytes = 0;
    free_after_bytes = 0;
    peak_used_bytes = 0;
    attempted_bytes = 0;
    total_flop = 0;
    c_nnz = -1;
    error_msg.clear();
}

void SpgemmPerf::snapshot_free(std::size_t &out_free) const
{
    std::size_t total = 0;
    query_cuda_memory(out_free, total);
}

void query_cuda_memory(std::size_t &free_bytes, std::size_t &total_bytes)
{
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes = 0;
        total_bytes = 0;
    }
}

}  // namespace opsparse
