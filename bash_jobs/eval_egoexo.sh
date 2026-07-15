#!/bin/bash -l
#SBATCH --job-name=eval_egoexo
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

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

pixi run python evaluation/evaluate_egoexo.py \
    --ghost_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \
    --gt_root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \
    --smplx_model body_models/SMPLX_NEUTRAL.pkl \
    --checkpoint  checkpoints/fusion_module/best.pt \
    --scale       "${SCALE_MODE:-pred}" \
    --reid_map    manual_reid.json

echo ""
echo "Done: $(date)"
