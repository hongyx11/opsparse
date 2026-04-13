# experiment_results

Bundle of opsparse SpGEMM benchmark / debug runs.

## What lives here

- `matrix_lists/` — input slices (tracked in git)
  - `matrices_lt500mb.txt` — 2808 SuiteSparse matrices whose `.mtx` is
    under 500 MB. Sourced from the sibling `mhspgemm` repo to keep
    matrix slicing consistent across SpGEMM experiments on this cluster.
  - `16matrix.txt` — 15-entry smoke-test slice used to sanity-check the
    driver pipeline before launching the full run.
- `<experiment_name>/` — per-run artifact dir (gitignored). Each contains
  its own `README.md`, `summary.csv`, raw SLURM stdout, and
  `logs/<matrix>.log` per-matrix traces.

## Dataset root

On this cluster:

```
/data/project/yuxilab/yuxihong/workspace/datasets/suitsparse/<name>/<name>.mtx
```

## Naming convention for run dirs

`<purpose>_<slice>_<hardware>_<YYYY-MM-DD>` — see the run-experiments skill
guide. Current runs:

- `debug_nnz_overflow_lt500mb_h100_2026-04-12` — first debug sweep; adds
  int32 overflow guards and an OOM-aware `SpgemmPerf` status and runs
  every matrix in `matrices_lt500mb.txt` against opsparse SpGEMM.
