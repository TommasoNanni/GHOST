#!/bin/bash -l
#SBATCH --job-name=eval_ablations
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

# Progressive oracle ablations on RICH test (M2 -> M3 -> M4).
# Run ONE modality per job: all three on 52 scenes exceed the 1:30 debug limit.
#   sbatch --export=ALL,MOD=2 bash_jobs/eval_ablations.sh
#   sbatch --export=ALL,MOD=3 bash_jobs/eval_ablations.sh
#   sbatch --export=ALL,MOD=4 bash_jobs/eval_ablations.sh

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Modality: M${MOD:-2}  |  Start: $(date)"
echo ""

# ── Mount centered_test.sqsh ─────────────────────────────────────────────────
# The placer reads <rich_root>/centered_<split>/<scene>/crop_meta.json to undo the
# principal-point crop offset on SAM3D 2D keypoints (ablations.py:696 — path is
# derived from rich_root, so the mount point is fixed). Without it the placer
# silently falls back to offsets = 0 and the DLT inherits the crop bias.
#
# The mount point MUST be node-local (/tmp). squashfuse cannot mount onto
# ${RICH_ROOT}/centered_<split> itself: capstor is Lustre and FUSE refuses with
#   fusermount: mounting over filesystem type 0x0bd00bd0 is forbidden
# Hence --centered_root, which points the eval at the /tmp mount.
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

pixi run python evaluation/ablations.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         "$RICH_ROOT" \
    --checkpoint        checkpoints/fusion_module/best.pt \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --device            cuda \
    --gt_split          "${GT_SPLIT:-test}" \
    --centered_root     "$CENTERED_MNT" \
    --max_scenes        52 \
    --modalities        "${MOD:-2}" \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-none}"

echo ""
echo "Done: $(date)"
