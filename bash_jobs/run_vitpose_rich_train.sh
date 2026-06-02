#!/bin/bash -l
#SBATCH --job-name=run_vitpose
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

# Mount RICH train squashfs on compute node
mkdir -p /tmp/rich_train
squashfuse /capstor/scratch/cscs/tnanni/datasets/rich/train_dataset.sqsh /tmp/rich_train

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

pixi run python scripts/run_vitpose.py \
    --vitpose-weights checkpoints/vitpose/torch/coco/vitpose-b-coco.pth \
    --model-name      b \
    --device          cuda:0 \
    --data-root       /tmp/rich_train

echo ""
echo "Done: $(date)"
