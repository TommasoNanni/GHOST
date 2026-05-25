#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH --account=a144
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --job-name=single_point_overfitting
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

cd /users/tnanni/ghost

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $SLURM_GPUS"
echo "Start:        $(date)"
echo ""

pixi run python -m scripts.single_point_overfitting

echo ""
echo "Done: $(date)"
