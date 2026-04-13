#ifndef OPSPARSE_PROFILE_SPGEMM_TIMINGS_HPP_
#define OPSPARSE_PROFILE_SPGEMM_TIMINGS_HPP_

namespace opsparse {

class SpgemmTimings {
   public:
    bool measure_separate = true;
    bool measure_total = true;
    double setup = 0;
    double symbolic_binning = 0;
    double symbolic = 0;
    double reduce = 0;
    double numeric_binning = 0;
    double prefix = 0;
    double allocate = 0;
    double numeric = 0;
    double cleanup = 0;
    double total = 0;

    SpgemmTimings() = default;

    void operator+=(const SpgemmTimings &b);
    void operator/=(double x);

    void print(double total_flop) const;
    void reg_print(double total_flop) const;
    void perf_print(double total_flop) const;
    void binning_print(double total_flop) const;
};

}  // namespace opsparse

#endif  // OPSPARSE_PROFILE_SPGEMM_TIMINGS_HPP_
