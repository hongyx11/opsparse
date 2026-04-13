#!/bin/bash
#SBATCH --account=yuxilab
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=opsparse-quick

set -u
REPO=/data/project/yuxilab/yuxihong/workspace/code/mcl/sc26mcl/opsparse
ROOT=/data/project/yuxilab/yuxihong/workspace/datasets/suitsparse
BIN="$REPO/build/opsparse_reg"

echo "Node: $(hostname)"
nvidia-smi -L
echo ""

for m in cage12 pdb1HYS scircuit cant; do
    echo "=== $m ==="
    "$BIN" "$ROOT/$m/$m.mtx"
    echo "rc=$?"
    echo ""
done
