#include "opsparse/profile/spgemm_timings.hpp"

#include <cstdio>

namespace opsparse {

void SpgemmTimings::operator+=(const SpgemmTimings &b)
{
    setup += b.setup;
    symbolic_binning += b.symbolic_binning;
    symbolic += b.symbolic;
    reduce += b.reduce;
    numeric_binning += b.numeric_binning;
    prefix += b.prefix;
    allocate += b.allocate;
    numeric += b.numeric;
    cleanup += b.cleanup;
    total += b.total;
}

void SpgemmTimings::operator/=(double x)
{
    setup /= x;
    symbolic_binning /= x;
    symbolic /= x;
    reduce /= x;
    numeric_binning /= x;
    prefix /= x;
    allocate /= x;
    numeric /= x;
    cleanup /= x;
    total /= x;
}

void SpgemmTimings::print(double total_flop) const
{
    const double total_flop_G = total_flop / 1e9;
    std::printf("total flop %lf\n", total_flop);
    const double sum_total =
        setup + symbolic_binning + symbolic + numeric_binning + reduce + prefix + allocate + numeric + cleanup;
    if (measure_separate) {
        std::printf("time(ms):\n");
        std::printf("    setup            %8.3lfms %6.2lf%%\n", 1000 * setup, setup / total * 100);
        std::printf("\e[1;31m    symbolic_binning %8.3lfms %6.2lf%%\n\e[0m", 1000 * symbolic_binning,
                    symbolic_binning / total * 100);
        std::printf("\e[1;31m    symbolic         %8.3lfms %6.2lf%%\n\e[0m", 1000 * symbolic, symbolic / total * 100);
        std::printf("    reduce            %8.3lfms %6.2lf%%\n", 1000 * reduce, reduce / total * 100);
        std::printf("\e[1;31m    numeric_binning  %8.3lfms %6.2lf%%\n\e[0m", 1000 * numeric_binning,
                    numeric_binning / total * 100);
        std::printf("    prefix           %8.3lfms %6.2lf%%\n", 1000 * prefix, prefix / total * 100);
        std::printf("    allocate         %8.3lfms %6.2lf%%\n", 1000 * allocate, allocate / total * 100);
        std::printf("\e[1;31m    numeric          %8.3lfms %6.2lf%%\n\e[0m", 1000 * numeric, numeric / total * 100);
        std::printf("    cleanup          %8.3lfms %6.2lf%%\n", 1000 * cleanup, cleanup / total * 100);
        std::printf("    sum_total        %8.3lfms %6.2lf%%\n", 1000 * sum_total, sum_total / total * 100);
        std::printf("    total            %8.3lfms %6.2lf%%\n", 1000 * total, total / total * 100);
        std::printf("perf(Gflops):\n");
        std::printf("    setup            %6.2lf\n", total_flop_G / setup);
        std::printf("    symbolic_binning %6.2lf\n", total_flop_G / symbolic_binning);
        std::printf("    symbolic         %6.2lf\n", total_flop_G / symbolic);
        std::printf("    reduce           %6.2lf\n", total_flop_G / reduce);
        std::printf("    numeric_binning  %6.2lf\n", total_flop_G / numeric_binning);
        std::printf("    prefix           %6.2lf\n", total_flop_G / prefix);
        std::printf("    allocate         %6.2lf\n", total_flop_G / allocate);
        std::printf("    numeric          %6.2lf\n", total_flop_G / numeric);
        std::printf("    cleanup          %6.2lf\n", total_flop_G / cleanup);
        std::printf("    total            %6.2lf\n", total_flop_G / total);
    }
}

void SpgemmTimings::reg_print(double total_flop) const
{
    const double total_flop_G = total_flop / 1e9;
    std::printf("%6.2lf\n", total_flop_G / total);
}

void SpgemmTimings::perf_print(double total_flop) const
{
    const double total_flop_G = total_flop / 1e9;
    std::printf("%6.2lf %6.2lf\n", total_flop_G / symbolic, total_flop_G / numeric);
}

void SpgemmTimings::binning_print(double total_flop) const
{
    const double total_binning_time = symbolic_binning + numeric_binning;
    std::printf("%.4le %.4lf\n", total_binning_time, 100 * total_binning_time / total);
    (void)total_flop;
}

}  // namespace opsparse
