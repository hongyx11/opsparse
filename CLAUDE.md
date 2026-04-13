# CLAUDE.md — Guide for Claude Code

This repository is **opsparse**, a CUDA SpGEMM kernel library (originally the
research code behind *OpSparse: A Highly Optimized Framework for Sparse General
Matrix Multiplication on GPUs*, IEEE Access 2022).

## Build

```
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=<path-to-nvcc> \
      -DOPSPARSE_CUDA_ARCH=80
ninja -C build
```

Outputs:
- `build/libopsparse.a`   — the library
- `build/opsparse_profile` — per-phase profiling benchmark
- `build/opsparse_reg`    — fused-phase overall-performance benchmark
- `build/cusparse_reg`    — cuSPARSE SpGEMM baseline

CMake options live in `cmake/options.cmake`. CUDA + cuSPARSE discovery lives
in `cmake/findcuda.cmake`.

## Layered architecture

```
core_algo/spgemm  ──────────────── high-level SpGEMM pipeline
                    │ uses
core_ds/csr                       CSR matrix container
                    │ uses
system / utils / profile / io / numeric   foundation layers
```

- **algorithm** may include **data-structure** headers
- **data-structure** may include **foundation** headers
- foundation layers never include upward

## Directory map

```
include/opsparse/
├── system/cuda_common.hpp          CHECK macros, d2h/h2d/d2d, cuda_runtime_warmup
├── utils/config.hpp                constexpr kNumBin, kPwarp*, kHashScale, div_up
├── utils/tsc_timer.hpp             fast_clock_time (rdtsc)
├── profile/spgemm_timings.hpp      SpgemmTimings
├── io/matrix_market.hpp            banner + tokenizer (header-only)
├── numeric/compute_flop.hpp        OpenMP + CUDA flop counters
├── core_ds/csr.hpp                 CSR matrix class
└── core_algo/spgemm/
    ├── meta.hpp                    SpgemmMeta workspace
    ├── spgemm.hpp                  opsparse_spgemm / opsparse_spgemm_reg entry
    ├── cusparse_spgemm.hpp         cuSPARSE reference wrapper
    ├── spgemm_phases.cuh           host-side phase drivers (h_setup, h_symbolic, …)
    └── kernels/
        ├── setup.cuh
        ├── binning.cuh             static kernels — included from two TUs
        ├── symbolic.cuh            templated symbolic hash kernels
        └── numeric.cuh             templated numeric hash kernels

src/opsparse/   mirrors include/opsparse/ with *.cu / *.cpp implementations.

benchmarks/
├── bench_common.hpp                matrix path resolver & CLI parser
├── opsparse_profile.cu             profiling harness for opsparse_spgemm
├── opsparse_reg.cu                 throughput harness for opsparse_spgemm_reg
└── cusparse_reg.cu                 cuSPARSE baseline harness
```

## Conventions

- **C++17**, CUDA separable compilation, static library target.
- Everything lives in `namespace opsparse`; IO helpers in `opsparse::io`;
  benchmark helpers in `opsparse::bench`.
- **No C-style `#define` constants.** Tunables are `inline constexpr` in
  `utils/config.hpp`. Only `__FILE__`/`__LINE__`-capturing checks remain as
  macros (`OPSPARSE_CHECK_CUDA`, `OPSPARSE_CHECK_CUSPARSE`) in
  `system/cuda_common.hpp`.
- Kernels that are defined in `.cuh` headers and included from multiple
  translation units must be `static __global__` (see `kernels/binning.cuh`)
  so each TU has its own copy under separable compilation.
- The `kHashSingle` constexpr chooses between single-pass and double-check
  hash strategies via `if constexpr`, replacing the original
  `#ifdef HASH_SINGLE`/`HASH_MULTI` macros.

## Running benchmarks

The benchmark mains expect a `../matrix/suite_sparse/<name>/<name>.mtx` layout
relative to the binary's working directory. Use `scripts/download_matrix.sh`
to fetch `webbase-1M`:

```
cd scripts && bash download_matrix.sh && cd ..
cd build && ./opsparse_profile webbase-1M
```

## Third-party benchmarks

`third-party/benchmarks/nsparse` and `third-party/benchmarks/spECK` contain the
original external SpGEMM implementations bundled for comparison. They are
**not built by the top-level CMake** and are not refactored — use their own
Makefiles if you need to rerun them.
