#!/bin/bash -l
#SBATCH --job-name=eval_egoexo_mean
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

# EgoExo4D val evaluation with CHORDAL-MEAN FUSION — the naive multi-view baseline.
#
# evaluation/evaluate_egoexo_mean.py is a copy of evaluate_egoexo.py with the
# learned PoseFusionModule swapped for the unweighted chordal mean of the
# per-camera SAM3D poses (arithmetic mean of the rotation matrices, SVD-projected
# back onto SO(3)). No checkpoint is loaded. Everything else is unchanged:
# same auto-matched GT subject, same Procrustes-DLT placement, same MapAnything
# baseline scale, same W-MPJPE†/PA-MPJPE protocol and EXCLUDED_TAKES split — so
# the delta against evaluate_egoexo.py is attributable to the fusion model alone.
#
# Counterpart of bash_jobs/eval_rich_test_mean.sh (RICH test).
#
#   sbatch bash_jobs/eval_egoexo_mean.sh

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Fusion: unweighted chordal mean over cameras (no checkpoint)"
echo ""

pixi run python evaluation/evaluate_egoexo_mean.py \
    --ghost_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \
    --gt_root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \
    --smplx_model body_models/SMPLX_NEUTRAL.pkl \
    --scale       "${SCALE_MODE:-baseline}" \
    --reid_map    manual_reid.json

echo ""
echo "Done: $(date)"
