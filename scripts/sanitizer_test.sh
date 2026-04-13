#!/bin/bash
#SBATCH --account=yuxilab
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=opsparse-sanitizer
#SBATCH --output=experiment_results/sanitizer_%j.out

set -u
REPO=/data/project/yuxilab/yuxihong/workspace/code/mcl/sc26mcl/opsparse
ROOT=/data/project/yuxilab/yuxihong/workspace/datasets/suitsparse
BIN="$REPO/build/opsparse_reg"
MATRIX=${1:-cage12}

echo "Node: $(hostname)"
nvidia-smi -L
echo "Matrix: $MATRIX"
echo "Binary: $BIN"
echo ""

echo "=== Plain run ==="
"$BIN" "$ROOT/$MATRIX/$MATRIX.mtx"
echo "rc=$?"
echo ""

echo "=== compute-sanitizer memcheck ==="
compute-sanitizer --tool memcheck "$BIN" "$ROOT/$MATRIX/$MATRIX.mtx" 2>&1
echo "rc=$?"
