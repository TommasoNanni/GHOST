#!/bin/bash -l
#SBATCH --job-name=diag_tri_joints
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"

MODALITY="${1:-mhr70}"
ALL_FLAG="${2:-}"   # pass "--all" as second arg for all-scenes mode

pixi run python -m scripts.diag_triangulate_joints \
    --modality "$MODALITY" \
    $ALL_FLAG

echo "Done: $(date)"
