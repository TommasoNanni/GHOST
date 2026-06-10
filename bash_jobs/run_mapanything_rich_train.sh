#!/bin/bash -l
#SBATCH --job-name=run_mapanything
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

# Mount RICH train squashfs on compute node (unmount stale mount if present)
cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

pixi run python preprocessing/run_mapanything.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
    --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich/centered_train \
    --batch_size        8 \
    --device            cuda

echo ""
echo "Done: $(date)"
