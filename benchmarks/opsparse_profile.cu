#include <cstdio>
#include <string>

#include "bench_common.hpp"
#include "opsparse/core_algo/spgemm/spgemm.hpp"
#include "opsparse/numeric/compute_flop.hpp"
#include "opsparse/profile/spgemm_timings.hpp"
#include "opsparse/system/cuda_common.hpp"

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
    CSR C;
    cuda_runtime_warmup();
    SpgemmMeta meta;
    {
        SpgemmTimings timing;
        opsparse_spgemm(A, B, C, meta, timing);
        C.release();
    }

    constexpr int iter = 10;
    SpgemmTimings timing, bench_timing;
    for (int i = 0; i < iter; i++) {
        opsparse_spgemm(A, B, C, meta, timing);
        bench_timing += timing;
        if (i < iter - 1) C.release();
    }
    bench_timing /= iter;

    std::printf("%s ", mat1.c_str());
    bench_timing.print(total_flop * 2);

    A.release();
    B.release();
    C.release();
    return 0;
}
