#!/bin/bash -l
#SBATCH --job-name=vggt_rich_test_centered
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

pixi run python scripts/rerun_vggt_only.py \
    --vggt-weights  checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --rich-data-root /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \
    --output-dir    /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test

echo ""
echo "Done: $(date)"