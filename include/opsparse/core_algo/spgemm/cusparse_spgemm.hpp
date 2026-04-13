#ifndef OPSPARSE_CORE_ALGO_SPGEMM_CUSPARSE_SPGEMM_HPP_
#define OPSPARSE_CORE_ALGO_SPGEMM_CUSPARSE_SPGEMM_HPP_

#include "opsparse/core_ds/csr.hpp"

namespace opsparse {

void cusparse_spgemm(const CSR &A, const CSR &B, CSR &C);

}  // namespace opsparse

#endif  // OPSPARSE_CORE_ALGO_SPGEMM_CUSPARSE_SPGEMM_HPP_
