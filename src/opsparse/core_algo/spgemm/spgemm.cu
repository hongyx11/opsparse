#include "opsparse/core_algo/spgemm/spgemm.hpp"

#include <climits>
#include <cub/cub.cuh>
#include <exception>

#include "opsparse/core_algo/spgemm/kernels/binning.cuh"
#include "opsparse/core_algo/spgemm/spgemm_phases.cuh"
#include "opsparse/numeric/compute_flop.hpp"
#include "opsparse/profile/perf.hpp"
#include "opsparse/utils/config.hpp"
#include "opsparse/utils/tsc_timer.hpp"

namespace opsparse {

void opsparse_spgemm(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing)
{
    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_numeric_binning(C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.numeric_binning = fast_clock_time() - t0;

    t0 = fast_clock_time();
    C.nnz = *meta.total_nnz;
    OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    timing.allocate = fast_clock_time() - t0;

    t0 = fast_clock_time();
    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.prefix = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.numeric = fast_clock_time() - t0;

    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

void opsparse_spgemm_reg(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing)
{
    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;

    // Fused numeric_binning + allocate + prefix
    meta.memset_all(0);
    const mint BS = 1024;
    const mint GS = div_up(C.M, BS);
    k_numeric_binning<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bin_size, meta.d_total_nnz, meta.d_max_row_nnz);
    meta.d2h_all(0);
    OPSPARSE_CHECK_CUDA(cudaStreamSynchronize(meta.stream[0]));
    C.nnz = *meta.total_nnz;

    if (*meta.max_row_nnz <= 16) {
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
        OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
        meta.bin_size[0] = C.M;
        for (int i = 1; i < kNumBin; i++) meta.bin_size[i] = 0;
        meta.bin_offset[0] = 0;
        for (int i = 1; i < kNumBin; i++) meta.bin_offset[i] = C.M;
    } else {
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for (int i = 0; i < kNumBin - 1; i++) {
            meta.bin_offset[i + 1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.h2d_bin_offset(0);
        k_numeric_binning2<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
        OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    }
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());

    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
    OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());

    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
    timing.numeric = fast_clock_time() - t0;

    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

namespace {

// Pull the last-observed CUDA error, clearing it.
cudaError_t take_last_error()
{
    cudaError_t err = cudaGetLastError();
    // Drain any further sticky error as well.
    cudaGetLastError();
    return err;
}

SpgemmStatus classify_cuda_error(cudaError_t err)
{
    if (err == cudaSuccess) return SpgemmStatus::ok;
    if (err == cudaErrorMemoryAllocation) return SpgemmStatus::oom;
    return SpgemmStatus::cuda_error;
}

}  // namespace

SpgemmStatus opsparse_spgemm_safe(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing,
                                  SpgemmPerf &perf)
{
    perf.reset();

    // Preflight: non-square composition is a skip, not a failure.
    if (A.N != B.M) {
        perf.status = SpgemmStatus::nonsquare;
        perf.error_msg = "A.N != B.M";
        return perf.status;
    }

    // Host-side flop estimate up front. If it already exceeds INT_MAX we
    // cannot trust d_row_flop (int) and the int32 atomic sums inside the
    // binning kernels, so bail out early.
    {
        long long total_flop = 0;
        try {
            total_flop = static_cast<long long>(compute_flop(A, B));
        } catch (const std::exception &e) {
            perf.status = SpgemmStatus::exception;
            perf.error_msg = std::string("compute_flop: ") + e.what();
            return perf.status;
        }
        perf.total_flop = total_flop;
        if (total_flop > static_cast<long long>(INT_MAX)) {
            perf.status = SpgemmStatus::int32_overflow;
            perf.error_msg = "total_flop exceeds INT_MAX";
            return perf.status;
        }
    }

    // Memory snapshot before any device work for this run.
    query_cuda_memory(perf.free_before_bytes, perf.device_total_bytes);
    // Drain any sticky error from earlier runs.
    (void)take_last_error();

    try {
        opsparse_spgemm_reg(A, B, C, meta, timing);
    } catch (const std::exception &e) {
        // OPSPARSE_CHECK_CUDA / OPSPARSE_CHECK_CUSPARSE throw std::exception
        // after printing. cudaGetLastError tells us whether the culprit was
        // an allocation failure.
        cudaError_t err = take_last_error();
        perf.status = classify_cuda_error(err);
        if (perf.status == SpgemmStatus::ok) perf.status = SpgemmStatus::exception;
        perf.error_msg = std::string(e.what());
        if (err != cudaSuccess) perf.error_msg += std::string(" (cuda: ") + cudaGetErrorString(err) + ")";
        // Try to salvage so the next run starts from a clean meta state.
        try {
            meta.release();
        } catch (...) {
            // ignore
        }
        query_cuda_memory(perf.free_after_bytes, perf.device_total_bytes);
        return perf.status;
    }

    // Post-run bookkeeping.
    perf.c_nnz = static_cast<long long>(C.nnz);
    // If C.nnz went negative, 32-bit total_nnz wrapped. Surface it even
    // if cudaMalloc somehow succeeded with a huge size_t.
    if (C.nnz < 0) {
        perf.status = SpgemmStatus::int32_overflow;
        perf.error_msg = "C.nnz wrapped negative (total_nnz int32 overflow)";
    }
    query_cuda_memory(perf.free_after_bytes, perf.device_total_bytes);
    if (perf.free_before_bytes >= perf.free_after_bytes) {
        perf.peak_used_bytes = perf.free_before_bytes - perf.free_after_bytes;
    }
    return perf.status;
}

}  // namespace opsparse
