#!/bin/bash -l
#SBATCH --job-name=ma_baseline_rich
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# Baseline-ratio MapAnything scale for RICH. Env: SPLIT=train|test
set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SPLIT="${SPLIT:?set SPLIT=train or test}"
SQSH="/capstor/scratch/cscs/tnanni/datasets/rich/centered_${SPLIT}.sqsh"
MNT="/tmp/richc_${SLURM_JOB_ID}"; mkdir -p "$MNT"
cleanup() { fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT
squashfuse "$SQSH" "$MNT"

pixi run python scripts/run_ma_baseline_rich.py --split "$SPLIT" --img_root "$MNT"
echo "Done: $(date)"
