#ifndef OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_HPP_
#define OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_HPP_

#include "opsparse/core_algo/spgemm/meta.hpp"
#include "opsparse/core_ds/csr.hpp"
#include "opsparse/profile/perf.hpp"
#include "opsparse/profile/spgemm_timings.hpp"

namespace opsparse {

// Full opsparse SpGEMM pipeline with per-phase timing.
void opsparse_spgemm(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing);

// Fused / "reg" variant: same result but fewer synchronizations, used by the
// overall-performance benchmark. Timing counters are not populated per phase.
void opsparse_spgemm_reg(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing);

// Safe variant used by the debug sweep: catches CUDA errors, guards against
// int32 overflow on total_flop / C.nnz, and records memory + status into
// `perf`. Never throws; returns `perf.status` also as the return value.
SpgemmStatus opsparse_spgemm_safe(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing,
                                  SpgemmPerf &perf);

}  // namespace opsparse

#endif  // OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_HPP_
