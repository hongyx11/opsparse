#ifndef OPSPARSE_BENCHMARKS_BENCH_COMMON_HPP_
#define OPSPARSE_BENCHMARKS_BENCH_COMMON_HPP_

#include <string>

#include "opsparse/core_ds/csr.hpp"

namespace opsparse::bench {

inline std::string resolve_matrix_path(const std::string &name)
{
    if (name.find("ER") != std::string::npos) return "../matrix/ER/" + name + ".mtx";
    if (name.find("G500") != std::string::npos) return "../matrix/G500/" + name + ".mtx";
    return "../matrix/suite_sparse/" + name + "/" + name + ".mtx";
}

inline void load_pair(const std::string &mat1, const std::string &mat2, CSR &A, CSR &B)
{
    A.construct(resolve_matrix_path(mat1));
    if (mat1 == mat2) {
        B = A;
    } else {
        B.construct(resolve_matrix_path(mat2));
        if (A.N == B.M) {
            // do nothing
        } else if (A.N < B.M) {
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        } else {
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }
}

inline void parse_args(int argc, char **argv, std::string &mat1, std::string &mat2)
{
    mat1 = "can_24";
    mat2 = "can_24";
    if (argc == 2) {
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if (argc >= 3) {
        mat1 = argv[1];
        mat2 = argv[2];
    }
}

}  // namespace opsparse::bench

#endif  // OPSPARSE_BENCHMARKS_BENCH_COMMON_HPP_
