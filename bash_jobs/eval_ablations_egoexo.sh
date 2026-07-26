#!/bin/bash -l
#SBATCH --job-name=abl_egoexo
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

# EgoExo4D progressive oracle ablations (error attribution).
#   M2 = pred-cam + GT-scale + pred-pose
#   M3 = GT-cam   + GT-scale + pred-pose
# There is no M4: EgoExo4D has no GT body-model parameters (see the script docstring).
# Production (non-oracle) number: bash_jobs/eval_egoexo.sh
#
# Env:
#   MODALITIES  comma-separated rungs, default "2,3"
#   SCENE       optional single scene name
#   MAX_SCENES  optional cap (debugging)

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

ARGS=(--ghost_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d
      --gt_root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt
      --smplx_model body_models/SMPLX_NEUTRAL.pkl
      --checkpoint  checkpoints/fusion_module/best.pt
      --modalities  "${MODALITIES:-2,3}")
[ -n "${SCENE:-}" ]      && ARGS+=(--scene "$SCENE")
[ -n "${MAX_SCENES:-}" ] && ARGS+=(--max_scenes "$MAX_SCENES")

pixi run python evaluation/ablations_egoexo.py "${ARGS[@]}"

echo ""
echo "Done: $(date)"
