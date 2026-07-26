#!/bin/bash -l
#SBATCH --job-name=eval_ablations_placing
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

# Placement-method comparison on RICH test (P1 = mean-SAM3D-root, P2 = depth-readout).
# Eval core is identical to eval_ablations.sh / eval_rich_prod.sh; only the
# placement step differs. Baseline to compare against = prod M10 (evaluate_rich.py):
# WA-MPJPE-100 50.4 / W-MPJPE-100 70.4 / RTE 1.00.
#
# Run ONE method per job: both on 52 scenes exceed the 1:30 debug limit.
#   sbatch --export=ALL,MOD=1 bash_jobs/eval_ablations_placing.sh   # mean-SAM3D-root
#   sbatch --export=ALL,MOD=2 bash_jobs/eval_ablations_placing.sh   # depth-readout

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Method: P${MOD:-1}  |  Start: $(date)"
echo ""

# Mount centered_<split>.sqsh node-local (/tmp). squashfuse CANNOT mount onto
# ${RICH_ROOT}/centered_<split> — capstor is Lustre and FUSE refuses with
#   fusermount: mounting over filesystem type 0x0bd00bd0 is forbidden
# --centered_root points the eval at the /tmp mount; without it the placer
# silently falls back to crop offsets = 0.
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

pixi run python evaluation/ablations_placing.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         "$RICH_ROOT" \
    --checkpoint        checkpoints/fusion_module/best.pt \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --device            cuda \
    --gt_split          "${GT_SPLIT:-test}" \
    --centered_root     "$CENTERED_MNT" \
    --max_scenes        52 \
    --modalities        "${MOD:-1}" \
    --scenes            "${SCENES:-}" \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-median}"

echo ""
echo "Done: $(date)"
