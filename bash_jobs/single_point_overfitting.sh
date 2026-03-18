#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32G
#SBATCH --job-name=single_point_overfitting
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

cd /cluster/project/cvg/students/tnanni/ghost

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
