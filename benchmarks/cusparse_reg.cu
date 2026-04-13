#include <cstdio>
#include <string>

#include "bench_common.hpp"
#include "opsparse/core_algo/spgemm/cusparse_spgemm.hpp"
#include "opsparse/numeric/compute_flop.hpp"
#include "opsparse/system/cuda_common.hpp"
#include "opsparse/utils/tsc_timer.hpp"

int main(int argc, char **argv)
{
    using namespace opsparse;

    std::string mat1, mat2;
    bench::parse_args(argc, argv, mat1, mat2);

    CSR A, B;
    bench::load_pair(mat1, mat2, A, B);
    A.H2D();
    B.H2D();

    const long total_flop = compute_flop(A, B);
    const double total_flop_G = static_cast<double>(total_flop) * 2 / 1e9;

    CSR C;
    cusparse_spgemm(A, B, C);
    C.release();

    constexpr int iter = 10;
    double t_sum = 0;
    for (int i = 0; i < iter; i++) {
        const double t0 = fast_clock_time();
        cusparse_spgemm(A, B, C);
        t_sum += fast_clock_time() - t0;
        if (i < iter - 1) C.release();
    }
    t_sum /= iter;
    std::printf("%s %lf\n", mat1.c_str(), total_flop_G / t_sum);

    A.release();
    B.release();
    C.release();
    return 0;
}
