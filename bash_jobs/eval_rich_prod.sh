#!/bin/bash -l
#SBATCH --job-name=eval_rich_prod
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

# Production RICH number (M10: neutral pred FK vs gendered GT, no oracle swaps).
# This is the M1 rung of the ablation ladder in bash_jobs/eval_ablations.sh —
# evaluation/evaluate_rich.py and evaluation/ablations.py share a byte-identical
# core, so the two are comparable as long as BOTH are run with the same
# --scale_smooth and the same crop_meta availability.
#
# Defaults to --scale_smooth median (one constant scale per scene) to match the
# constant GT scalar the M2/M3/M4 rungs substitute. Without it, M1 -> M2 would
# conflate scale-magnitude correction with per-frame jitter removal.
#
#   sbatch bash_jobs/eval_rich_prod.sh
#   sbatch --export=ALL,SCALE_SMOOTH=none bash_jobs/eval_rich_prod.sh

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

# The mount point MUST be node-local (/tmp). squashfuse cannot mount onto
# ${RICH_ROOT}/centered_<split> itself: capstor is Lustre and FUSE refuses with
#   fusermount: mounting over filesystem type 0x0bd00bd0 is forbidden
# Hence --centered_root, which points the eval at the /tmp mount. Without it the
# placer silently falls back to offsets = 0 (as SLURM job 2768757 did).
RICH_ROOT=/capstor/scratch/cscs/tnanni/datasets/rich
CENTERED_SQSH="${RICH_ROOT}/centered_${GT_SPLIT:-test}.sqsh"
CENTERED_MNT="/tmp/centered_${GT_SPLIT:-test}_${SLURM_JOB_ID}"

mkdir -p "$CENTERED_MNT"
squashfuse "$CENTERED_SQSH" "$CENTERED_MNT"
trap 'fusermount -u "$CENTERED_MNT" 2>/dev/null || true; rmdir "$CENTERED_MNT" 2>/dev/null || true' EXIT
echo "[sqsh] mounted $CENTERED_SQSH -> $CENTERED_MNT"

N_META=$(find "$CENTERED_MNT" -maxdepth 2 -name crop_meta.json | wc -l)
echo "[sqsh] scenes with crop_meta.json: $N_META"
if [ "$N_META" -eq 0 ]; then
    echo "ERROR: no crop_meta.json under $CENTERED_MNT — aborting (would run with offsets=0)"
    exit 1
fi
echo ""

# JOINT_CONF=1 feeds the per-joint confidence channel (pred_joint_confidence) to
# the fusion model as joint_mask. The model was trained with it but no evaluation
# script has ever passed it. Unset => legacy behaviour, numbers unchanged.
#   sbatch --export=ALL,JOINT_CONF=1 bash_jobs/eval_rich_prod.sh
JOINT_CONF_FLAG=""
if [ "${JOINT_CONF:-0}" = "1" ]; then
    JOINT_CONF_FLAG="--joint_conf"
fi

pixi run python evaluation/evaluate_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         "$RICH_ROOT" \
    --checkpoint        checkpoints/fusion_module/best.pt \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --device            cuda \
    --gt_split          "${GT_SPLIT:-test}" \
    --centered_root     "$CENTERED_MNT" \
    --max_scenes        52 \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-median}" \
    ${JOINT_CONF_FLAG}

echo ""
echo "Done: $(date)"
