The Source Code of OpSparse
========

OpSparse is a highly optimized CUDA SpGEMM (sparse × sparse matrix multiply)
framework, originally published as:

> Du, Guan, Guan, Niu, Huang, Zheng, Xie. *OpSparse: A Highly Optimized
> Framework for Sparse General Matrix Multiplication on GPUs.* IEEE Access
> vol. 10 (2022), 85960–85974. doi:10.1109/ACCESS.2022.3196940

This repository has been restructured as a reusable library (`libopsparse.a`)
plus benchmark executables. The external `nsparse` and `spECK` reference
implementations bundled for comparison now live under
`third-party/benchmarks/` and are unchanged.

## Layout

```
include/opsparse/       public headers (system, utils, profile, io, numeric,
                        core_ds, core_algo/spgemm)
src/opsparse/           library sources mirroring the include layout
benchmarks/             SpGEMM benchmark drivers (opsparse / cuSPARSE)
scripts/                download_matrix.sh
third-party/benchmarks/ unmodified nsparse and spECK
cmake/                  options.cmake, findcuda.cmake
```

See `CLAUDE.md` for the full architecture notes.

## Build

Requires CUDA ≥ 11.2, CMake ≥ 3.20, Ninja, and OpenMP.

```
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DOPSPARSE_CUDA_ARCH=80
ninja -C build
```

The default `OPSPARSE_CUDA_ARCH` is `80` (A100); set it to `70` for V100, `90`
for H100, etc. If `nvcc` is not on your `PATH`, pass
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.

Artifacts:

- `build/libopsparse.a`    — the static library
- `build/opsparse_profile` — per-phase timing benchmark
- `build/opsparse_reg`     — fused overall-throughput benchmark
- `build/cusparse_reg`     — cuSPARSE SpGEMM baseline

## Run

```
cd scripts && bash download_matrix.sh && cd ..
cd build
./opsparse_profile webbase-1M
./opsparse_reg     webbase-1M
./cusparse_reg     webbase-1M
```

The benchmark mains resolve matrix paths relative to the binary's working
directory (`../matrix/suite_sparse/<name>/<name>.mtx`), so run them from
`build/`.

## Bibtex

```
@ARTICLE{9851653,
  author={Du, Zhaoyang and Guan, Yijin and Guan, Tianchan and Niu, Dimin and Huang, Linyong and Zheng, Hongzhong and Xie, Yuan},
  journal={IEEE Access},
  title={OpSparse: A Highly Optimized Framework for Sparse General Matrix Multiplication on GPUs},
  year={2022},
  volume={10},
  pages={85960-85974},
  doi={10.1109/ACCESS.2022.3196940}}
```
