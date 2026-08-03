#!/bin/bash -l
#SBATCH --job-name=eval_rich_mean
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

# RICH test evaluation with CHORDAL-MEAN FUSION — the naive multi-view baseline.
#
# evaluation/evaluate_rich_mean.py is a copy of evaluate_rich.py with the learned
# PoseFusionModule swapped for the unweighted chordal mean of the per-camera SAM3D
# poses (arithmetic mean of the rotation matrices, SVD-projected back onto SO(3)).
# No checkpoint is loaded. Everything else (mean-SAM3D betas, Procrustes-DLT
# placement, MapAnything baseline scale, M10 metric protocol) is unchanged, so the
# delta against the production run is attributable to the fusion model alone.
#
# centered_test.sqsh must be mounted node-local and passed via --centered_root:
# Lustre forbids FUSE mounts under rich_root, and without crop_meta.json the
# script silently falls back to offsets=0 (the bug behind every historical RICH
# number).
#
#   sbatch bash_jobs/eval_rich_test_mean.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh
MOUNT=/tmp/centered_test_mean
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH -> $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Fusion: unweighted chordal mean over cameras (no checkpoint)"
echo ""

pixi run python evaluation/evaluate_rich_mean.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --centered_root     "$MOUNT" \
    --device            cuda \
    --gt_split          test \
    --max_scenes        "${MAX_SCENES:-52}" \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-none}"

echo ""
echo "Done: $(date)"
