#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=rich_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0  # disable core dumps — they fill up home quota

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo ""

srun pixi run python -m scripts.rich_pipeline

echo ""
echo "Done: $(date)"