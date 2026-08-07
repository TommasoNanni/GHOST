#!/bin/bash -l
#SBATCH --job-name=eval_egoexo_median
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

# EgoExo4D val evaluation with GEODESIC-MEDIAN FUSION — the shipped fusion rule.
#
# evaluation/evaluate_egoexo_median.py is a copy of evaluate_egoexo.py with the
# learned PoseFusionModule swapped for the geodesic median of the per-camera SAM3D
# poses: the L1 estimator on SO(3), solved by Weiszfeld/IRLS from a chordal-mean
# seed. No checkpoint is loaded. Everything else is unchanged: same auto-matched
# GT subject, same Procrustes-DLT placement, same MapAnything baseline scale, same
# W-MPJPE†/PA-MPJPE protocol and EXCLUDED_TAKES split — so the delta against
# evaluate_egoexo.py is attributable to the fusion rule alone.
#
# ZERO-SHOT: the estimator was selected on RICH's train pool and is applied here
# with no per-dataset tuning.
#
# Counterpart of bash_jobs/eval_rich_test_median.sh (RICH test).
#
#   sbatch bash_jobs/eval_egoexo_median.sh

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Fusion: geodesic median over cameras, L1/Weiszfeld (no checkpoint)"
echo ""

pixi run python evaluation/evaluate_egoexo_median.py \
    --ghost_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \
    --gt_root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \
    --smplx_model body_models/SMPLX_NEUTRAL.pkl \
    --scale       "${SCALE_MODE:-baseline}" \
    --reid_map    manual_reid.json

echo ""
echo "Done: $(date)"
