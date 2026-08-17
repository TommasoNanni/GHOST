#!/bin/bash -l
#SBATCH --job-name=runtime_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token)
# Single-GPU benchmark: the node exposes all 4 GPUs regardless of --gpus-per-node,
# and ParametersExtractor fans out over every visible device — pin to one.
export CUDA_VISIBLE_DEVICES=0

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"

# Undistorted EgoHumans frames live on iopsstor (no sqsh mount needed):
# <root>/03_fencing/media/.../03_fencing/005_fencing/exo/<cam>/images_undistorted/frames/
DATA_ROOT="/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted"
SCENES_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
WORK_DIR="/iopsstor/scratch/cscs/tnanni/ghost_outputs/runtime_benchmark"

mkdir -p logs

pixi run python scripts/runtime_benchmark.py \
    --data-root   "$DATA_ROOT" \
    --scenes-root "$SCENES_ROOT" \
    --scene       "03_fencing/005_fencing" \
    --work-dir    "$WORK_DIR" \
    --frames 60 \
    --max-shift 10 \
    --seed 42

echo "Job finished: $(date)"
