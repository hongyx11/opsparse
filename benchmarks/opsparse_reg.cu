// Debug / sweep driver for opsparse SpGEMM.
//
// Usage:
//   opsparse_reg <path-to-matrix.mtx>       // square A*A
//
// Emits tagged lines so an outer driver script can grep them:
//   STATUS,<ok|nonsquare|int32_overflow|oom|cuda_error|exception>
//   PERF,<peak_used_bytes>,<free_before>,<free_after>,<total_bytes>
//   META,<M>,<N>,<A_nnz>,<C_nnz>,<total_flop>
//   TIMING_CSV,<name>,<total_ms>,<gflops>
//
// Exit code: 0 on any clean outcome (including caught OOM/overflow); nonzero
// only on hard crash.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "opsparse/core_algo/spgemm/spgemm.hpp"
#include "opsparse/core_ds/csr.hpp"
#include "opsparse/numeric/compute_flop.hpp"
#include "opsparse/profile/perf.hpp"
#include "opsparse/profile/spgemm_timings.hpp"
#include "opsparse/system/cuda_common.hpp"

namespace {

std::string basename_of(const std::string &path)
{
    auto slash = path.find_last_of('/');
    std::string f = slash == std::string::npos ? path : path.substr(slash + 1);
    auto dot = f.rfind(".mtx");
    if (dot != std::string::npos) f.erase(dot);
    return f;
}

void print_perf(const opsparse::SpgemmPerf &perf)
{
    std::printf("PERF,%zu,%zu,%zu,%zu\n", perf.peak_used_bytes, perf.free_before_bytes, perf.free_after_bytes,
                perf.device_total_bytes);
}

}  // namespace

int main(int argc, char **argv)
{
    using namespace opsparse;

    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string name = basename_of(path);

    CSR A;
    try {
        A.construct(path);
    } catch (const std::exception &e) {
        std::printf("STATUS,%s\n", "exception");
        std::printf("ERROR,load:%s\n", e.what());
        std::fflush(stdout);
        return 0;
    }

    if (A.M != A.N) {
        std::printf("STATUS,nonsquare\n");
        std::printf("META,%d,%d,%d,-1,0\n", A.M, A.N, A.nnz);
        std::fflush(stdout);
        return 0;
    }

    try {
        A.H2D();
    } catch (const std::exception &e) {
        cudaError_t err = cudaGetLastError();
        (void)cudaGetLastError();
        const char *tag = (err == cudaErrorMemoryAllocation) ? "oom" : "cuda_error";
        std::printf("STATUS,%s\n", tag);
        std::printf("ERROR,h2d:%s\n", e.what());
        std::fflush(stdout);
        return 0;
    }

    CSR C;
    cuda_runtime_warmup();
    SpgemmMeta meta;
    SpgemmTimings timing;
    SpgemmPerf perf;

    // Warm-up is the run itself; no separate warmup loop here because a
    // sweep over 2000+ matrices is already warm enough, and we want to
    // observe the very first allocation path for debugging.
    const SpgemmStatus st = opsparse_spgemm_safe(A, A, C, meta, timing, perf);

    std::printf("STATUS,%s\n", to_cstr(st));
    print_perf(perf);
    std::printf("META,%d,%d,%d,%lld,%lld\n", A.M, A.N, A.nnz, perf.c_nnz, perf.total_flop);
    if (!perf.error_msg.empty()) std::printf("ERROR,%s\n", perf.error_msg.c_str());

    if (st == SpgemmStatus::ok) {
        const double gflops = static_cast<double>(perf.total_flop) * 2.0 / 1e9 / timing.total;
        std::printf("TIMING_CSV,%s,%.3f,%.3f\n", name.c_str(), timing.total * 1000.0, gflops);
    }

    try {
        C.release();
    } catch (...) {
    }
    try {
        A.release();
    } catch (...) {
    }
    std::fflush(stdout);
    return 0;
}
