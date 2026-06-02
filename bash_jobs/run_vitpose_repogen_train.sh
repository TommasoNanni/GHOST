#!/bin/bash -l
#SBATCH --job-name=vitpose_repogen_train
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

fusermount -u /tmp/rich_train 2>/dev/null || true
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
    --vitpose-weights checkpoints/vitpose/VitPose-s_RePoGen.pth \
    --model-name      s \
    --device          cuda:0 \
    --data-root       /tmp/rich_train

echo ""
echo "Done: $(date)"
