#!/bin/bash
#SBATCH --account=yuxilab
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=opsparse-debug
# NOTE: --output is set by the wrapper phase so the log lands inside
# experiment_results/$EXPERIMENT/. Invoke with `bash`, not `sbatch`.
#
# opsparse SpGEMM debug sweep over SuiteSparse. Runs opsparse_reg on
# every matrix in $LIST (default: matrices_lt500mb.txt) and categorizes
# the outcome from the tagged STATUS/PERF/TIMING_CSV lines emitted by
# the benchmark.
#
# Outputs under experiment_results/$EXPERIMENT/:
#   summary.csv      index,matrix,status,exit_code,wall_s,peak_mb,c_nnz
#   timings.csv      matrix,total_ms,gflops      (ok runs only)
#   logs/<name>.log  line-buffered per-matrix stdout/stderr
#   slurm-<jobid>.out
#
# Invocation:
#   bash scripts/run_suitesparse_debug.sh
#   EXPERIMENT=debug_x bash scripts/run_suitesparse_debug.sh
#   LIST=experiment_results/matrix_lists/16matrix.txt EXPERIMENT=smoke bash scripts/run_suitesparse_debug.sh
#   GPU_TYPE=L40S EXPERIMENT=..._l40s_... bash scripts/run_suitesparse_debug.sh

set -u

REPO=/data/project/yuxilab/yuxihong/workspace/code/mcl/sc26mcl/opsparse
ROOT=/data/project/yuxilab/yuxihong/workspace/datasets/suitsparse
BIN="$REPO/build/opsparse_reg"
LIST="${LIST:-$REPO/experiment_results/matrix_lists/matrices_lt500mb.txt}"
EXPERIMENT="${EXPERIMENT:-debug_nnz_overflow_lt500mb_h100_$(date +%Y-%m-%d)}"
GPU_TYPE="${GPU_TYPE:-H100}"
LOG_DIR="$REPO/experiment_results/$EXPERIMENT"
PER_MATRIX_LOGS="$LOG_DIR/logs"
SUMMARY="$LOG_DIR/summary.csv"
TIMINGS="$LOG_DIR/timings.csv"
PER_MATRIX_TIMEOUT=120s

# -- Wrapper phase -----------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$PER_MATRIX_LOGS"
    exec sbatch \
        --gres=gpu:${GPU_TYPE}:1 \
        --output="$LOG_DIR/slurm-%j.out" \
        --export=ALL,EXPERIMENT="$EXPERIMENT",LIST="$LIST",GPU_TYPE="$GPU_TYPE" \
        "$0" "$@"
fi

# -- Execution phase (inside SLURM) ------------------------------------------

mkdir -p "$PER_MATRIX_LOGS"

: > "$SUMMARY"
echo "index,matrix,status,exit_code,wall_s,peak_mb,c_nnz" >> "$SUMMARY"
: > "$TIMINGS"
echo "matrix,total_ms,gflops" >> "$TIMINGS"

total=$(wc -l < "$LIST")
echo "Total matrices: $total"
echo "Binary:         $BIN"
echo "List:           $LIST"
echo "Experiment:     $EXPERIMENT"
echo "Node / GPU:"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
echo "Commit:         $(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo ""

i=0
ok=0
nonsquare=0
int32=0
oom=0
cuda_err=0
excpt=0
timeout_cnt=0
error_cnt=0
missing=0

while IFS= read -r name; do
    [ -z "$name" ] && continue
    i=$((i + 1))
    f="$ROOT/$name/$name.mtx"
    if [ ! -f "$f" ]; then
        echo "[$i/$total] MISSING $name"
        echo "$i,$name,missing,-,0,0,-1" >> "$SUMMARY"
        missing=$((missing + 1))
        continue
    fi

    log="$PER_MATRIX_LOGS/$name.log"

    start=$(date +%s.%N)
    timeout "$PER_MATRIX_TIMEOUT" stdbuf -oL -eL "$BIN" "$f" > "$log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')

    # Derive fields from the tagged lines in the log.
    status_line=$(grep -m1 '^STATUS,' "$log" 2>/dev/null || true)
    perf_line=$(grep -m1 '^PERF,' "$log" 2>/dev/null || true)
    meta_line=$(grep -m1 '^META,' "$log" 2>/dev/null || true)
    timing_line=$(grep -m1 '^TIMING_CSV,' "$log" 2>/dev/null || true)

    if [ $rc -eq 124 ]; then
        status=timeout
        timeout_cnt=$((timeout_cnt + 1))
    elif [ $rc -ne 0 ]; then
        status=error
        error_cnt=$((error_cnt + 1))
    elif [ -n "$status_line" ]; then
        status=$(echo "$status_line" | awk -F, '{print $2}')
        case "$status" in
            ok)             ok=$((ok + 1)) ;;
            nonsquare)      nonsquare=$((nonsquare + 1)) ;;
            int32_overflow) int32=$((int32 + 1)) ;;
            oom)            oom=$((oom + 1)) ;;
            cuda_error)     cuda_err=$((cuda_err + 1)) ;;
            exception)      excpt=$((excpt + 1)) ;;
            *)              status=unknown; error_cnt=$((error_cnt + 1)) ;;
        esac
    else
        # Ran to completion (rc=0) but no STATUS line — treat as error
        status=no_status
        error_cnt=$((error_cnt + 1))
    fi

    peak_bytes=0
    c_nnz=-1
    if [ -n "$perf_line" ]; then
        peak_bytes=$(echo "$perf_line" | awk -F, '{print $2}')
    fi
    if [ -n "$meta_line" ]; then
        c_nnz=$(echo "$meta_line" | awk -F, '{print $5}')
    fi
    peak_mb=$(awk -v b="$peak_bytes" 'BEGIN{printf "%.1f", b/(1024*1024)}')

    if [ "$status" = "ok" ] && [ -n "$timing_line" ]; then
        echo "$timing_line" | sed 's/^TIMING_CSV,//' >> "$TIMINGS"
    fi

    echo "[$i/$total] $status rc=$rc ${wall}s peak=${peak_mb}MB $name"
    echo "$i,$name,$status,$rc,$wall,$peak_mb,$c_nnz" >> "$SUMMARY"
done < "$LIST"

echo ""
echo "Done: total=$total"
echo "  ok=$ok"
echo "  nonsquare=$nonsquare"
echo "  int32_overflow=$int32"
echo "  oom=$oom"
echo "  cuda_error=$cuda_err"
echo "  exception=$excpt"
echo "  timeout=$timeout_cnt"
echo "  error=$error_cnt"
echo "  missing=$missing"
