#ifndef OPSPARSE_CORE_DS_CSR_HPP_
#define OPSPARSE_CORE_DS_CSR_HPP_

#include <string>

#include "opsparse/system/cuda_common.hpp"

namespace opsparse {

class CSR {
   public:
    mint M = 0;
    mint N = 0;
    mint nnz = 0;
    mint *rpt = nullptr;
    mint *col = nullptr;
    mdouble *val = nullptr;

    mint *d_rpt = nullptr;
    mint *d_col = nullptr;
    mdouble *d_val = nullptr;

    CSR() = default;
    explicit CSR(const std::string &mtx_file);
    CSR(const CSR &A);
    CSR(const CSR &A, mint M_, mint N_, mint M_start, mint N_start);
    ~CSR();

    void hrelease();
    void drelease();
    void release();
    void D2H();
    void H2D();
    bool operator==(const CSR &A);
    CSR &operator=(const CSR &A);
    void construct(const std::string &mtx_file);
};

}  // namespace opsparse

#endif  // OPSPARSE_CORE_DS_CSR_HPP_
