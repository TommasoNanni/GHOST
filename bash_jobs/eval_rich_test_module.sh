#!/bin/bash -l
#SBATCH --job-name=eval_rich_module
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

# RICH test evaluation with the LEARNED FUSION MODULE — evaluation/evaluate_rich.py,
# the flagship M10 protocol (neutral pred vs gendered GT, pred camera + pred scale).
#
# Which module is loaded is read from the checkpoint: evaluate_rich.py dispatches
# on model_config, building PoseFusionModuleV3 when it carries residual_head /
# centered_input and PoseFusionModule (v2) otherwise. v2 and v3 have IDENTICAL
# state_dict shapes, so strict=True cannot catch a mix-up — check the
# "V3-residual" / "V2-direct" line in the log before trusting the numbers.
#
# centered_test.sqsh must be mounted node-local and passed via --centered_root:
# Lustre forbids FUSE mounts under rich_root, and without crop_meta.json the
# script silently falls back to offsets=0 (the bug behind every historical RICH
# number).
#
# SCALE_SMOOTH defaults to `median` here, NOT `none`: the published comparison
# table (mean 47.6/67.7/26.5 vs v2 module 50.4/70.4/30.4) was run at smooth=median,
# and mixing smoothing settings is exactly what made the old 70.4-vs-84.0 gap fake.
#
#   CKPT=checkpoints/fusion_v3/best.pt sbatch bash_jobs/eval_rich_test_module.sh
#   CKPT=checkpoints/fusion_module/best.pt sbatch bash_jobs/eval_rich_test_module.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh
MOUNT=/tmp/centered_test_module_${SLURM_JOB_ID}
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH -> $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Fusion: LEARNED MODULE  ckpt=${CKPT:-checkpoints/fusion_module/best.pt}"
echo ""

pixi run python evaluation/evaluate_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \
    --checkpoint        "${CKPT:-checkpoints/fusion_module/best.pt}" \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --centered_root     "$MOUNT" \
    --device            cuda \
    --gt_split          test \
    --max_scenes        "${MAX_SCENES:-52}" \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-median}"

echo ""
echo "Done: $(date)"
