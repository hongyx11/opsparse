#ifndef OPSPARSE_NUMERIC_COMPUTE_FLOP_HPP_
#define OPSPARSE_NUMERIC_COMPUTE_FLOP_HPP_

#include "opsparse/system/cuda_common.hpp"

namespace opsparse {

class CSR;

// Host (OpenMP) flop counters
long compute_flop(const mint *row_ptr_a, const mint *col_idx_a, const mint *row_ptr_b, mint M, mint *row_flop_out);
long compute_flop(const CSR &A, const CSR &B, mint *row_flop);
long compute_flop(const CSR &A, const CSR &B);

// Device flop kernel: fills d_row_flop and atomically accumulates the max.
__global__ void __launch_bounds__(1024, 2) k_compute_flop(const mint *__restrict__ d_arpt, const mint *__restrict__ d_acol,
                                                           const mint *__restrict__ d_brpt, mint M, mint *d_row_flop,
                                                           mint *d_max_row_flop);

}  // namespace opsparse

#endif  // OPSPARSE_NUMERIC_COMPUTE_FLOP_HPP_
