#ifndef OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_PHASES_CUH_
#define OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_PHASES_CUH_

#include "opsparse/core_algo/spgemm/meta.hpp"
#include "opsparse/core_ds/csr.hpp"
#include "opsparse/profile/spgemm_timings.hpp"

namespace opsparse {

// Host-side phase drivers. Kernels themselves live in kernels/*.cuh.
void h_compute_flop(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta);
void h_setup(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings &timing);
void h_symbolic_binning(CSR &C, SpgemmMeta &meta);
void h_symbolic(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta);
void h_numeric_binning(CSR &C, SpgemmMeta &meta);
void h_numeric_full_occu(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta);

}  // namespace opsparse

#endif  // OPSPARSE_CORE_ALGO_SPGEMM_SPGEMM_PHASES_CUH_
