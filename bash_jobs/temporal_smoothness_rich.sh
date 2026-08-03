#!/bin/bash -l
#SBATCH --job-name=temporal_smooth
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

# Does the fusion module's edge on temporally-integrated metrics come from
# temporal smoothing? Measure angular velocity / acceleration of the predicted
# pose sequences and compare against ground truth.
#
#   A  chordal rotation mean over visible views (no fusion module)
#   B  PoseFusionModule output
#   C  RICH ground-truth SMPL-X pose
#
# Body joints 1..21 only, global orientation excluded (it comes from the placing
# stage, not the fusion module). Same scenes, frames and joints for all three;
# the script asserts the triplet sets are identical.
#
# Fused sequences are persisted to CACHE_DIR so later experiments can reuse them
# without re-running inference.
#
#   sbatch bash_jobs/temporal_smoothness_rich.sh
#   MIN_VIEWS=2 sbatch --export=ALL,MIN_VIEWS bash_jobs/temporal_smoothness_rich.sh
#
# SCENES restricts to a comma-separated subset (default: every scene),
# MAX_SCENES caps the count:
#   sbatch --export=ALL,SCENES="Gym_010_cooking1,Gym_011_burpee2" \
#          bash_jobs/temporal_smoothness_rich.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi -L
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Checkpoint: ${CKPT:-checkpoints/fusion_module/best.pt}  (joint_conf OFF)"
echo "Scenes: ${SCENES:-<all>}"
echo ""

EXTRA=()
[ -n "${SCENES:-}" ]     && EXTRA+=(--scenes     "$SCENES")
[ -n "${MAX_SCENES:-}" ] && EXTRA+=(--max_scenes "$MAX_SCENES")

pixi run python evaluation/temporal_smoothness_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_data_root    /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \
    --rich_gt_dir       /capstor/scratch/cscs/tnanni/datasets/rich \
    --checkpoint        "${CKPT:-checkpoints/fusion_module/best.pt}" \
    --body_split        "${BODY_SPLIT:-test_body}" \
    --device            cuda \
    --min_views         "${MIN_VIEWS:-1}" \
    --min_run           "${MIN_RUN:-5}" \
    --cache_dir         "${CACHE_DIR:-/iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test}" \
    --out               "${OUT:-eval_explainability/temporal_smoothness_rich.json}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "Done: $(date)"
