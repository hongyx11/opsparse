#ifndef OPSPARSE_UTILS_CONFIG_HPP_
#define OPSPARSE_UTILS_CONFIG_HPP_

namespace opsparse {

inline constexpr int kNumBin = 8;
inline constexpr int kWarpSize = 32;

inline constexpr int kPwarp = 4;
inline constexpr int kPwarpRows = 256;
inline constexpr int kPwarpTsize = 32;
inline constexpr int kPwarpBlockSize = kPwarp * kPwarpRows;

inline constexpr int kNumericPwarp = 8;
inline constexpr int kNumericPwarpRows = 128;
inline constexpr int kNumericPwarpTsize = 32;
inline constexpr int kNumericPwarpBlockSize = kNumericPwarp * kNumericPwarpRows;

// Hash strategy: single-pass CAS (true) vs. double-check (false).
inline constexpr bool kHashSingle = true;

inline constexpr int kSymbolicScaleSmall = 1;
inline constexpr int kSymbolicScaleLarge = 1;
inline constexpr int kNumericScaleLarge = 2;
inline constexpr double kNumericScale = 1.5;
inline constexpr double kThreshScale = 0.8;
inline constexpr int kHashScale = 107;

template <typename Int>
__host__ __device__ inline constexpr Int div_up(Int a, Int b)
{
    return (a + b - 1) / b;
}

}  // namespace opsparse

#endif  // OPSPARSE_UTILS_CONFIG_HPP_
