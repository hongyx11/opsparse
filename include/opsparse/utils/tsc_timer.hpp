#ifndef OPSPARSE_UTILS_TSC_TIMER_HPP_
#define OPSPARSE_UTILS_TSC_TIMER_HPP_

#include <time.h>

namespace opsparse {

// Portable wall-clock timer via clock_gettime (POSIX).
// Falls back from CLOCK_MONOTONIC_RAW → CLOCK_MONOTONIC.
inline double fast_clock_time()
{
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

}  // namespace opsparse

#endif  // OPSPARSE_UTILS_TSC_TIMER_HPP_
