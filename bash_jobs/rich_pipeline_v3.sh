#!/bin/bash -l
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=rich_pipeline_v3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

cd /users/tnanni/ghost
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_OFFLINE=1
ulimit -c 0  # disable core dumps — they fill up home quota

# Mount the test-set SquashFS archive so it looks like a normal directory.
SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/train_dataset.sqsh
MOUNT=/tmp/rich_train
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
echo "Mounted $SQSH → $MOUNT"

# Unmount on exit (success or failure).
trap "fusermount -u '$MOUNT' && echo 'Unmounted $MOUNT'" EXIT

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo ""

pixi run python -m scripts.rich_pipeline_v3

echo ""
echo "Done: $(date)"
