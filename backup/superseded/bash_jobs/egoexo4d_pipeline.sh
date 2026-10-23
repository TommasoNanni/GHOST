#!/bin/bash -l
#SBATCH --job-name=egoexo4d_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"
echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

mkdir -p logs

# Paths come from configuration/config.yaml (egoexo4d_frames_root / egoexo4d_output_dir).
# Override here if needed:
#   --frames-root /capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames
#   --output-dir  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d

EXTRA_ARGS=""
# Uncomment to restrict to a single take:
# EXTRA_ARGS="--take cmu_soccer06_3"
# Uncomment to process a slice:
# EXTRA_ARGS="--scene-start 0 --scene-end 10"

pixi run python scripts/egoexo4d_pipeline.py $EXTRA_ARGS

echo ""
echo "Done: $(date)"
