#!/bin/bash -l
#SBATCH --job-name=eval_rich_sync
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

# STEP 2 of the through-sync RICH evaluation (evaluation/evaluate_rich_sync.py):
# fusion + placement + CHROMM metrics over the trial dirs built by STEP 1
# (bash_jobs/sync_inject_rich.sh). Cheap compared to STEP 1 — no VGGT/
# MapAnything rerun, just reads the precomputed npz per trial — but still
# needs a centered_test mount for crop_meta.json (BodyPlacer's SAM3D-kp2d
# offset correction), same squashfuse pattern as eval_rich_prod.sh.
#
#   sbatch bash_jobs/eval_rich_sync.sh                      # full sync_root
#   sbatch --export=ALL,MAX_TRIALS=1 bash_jobs/eval_rich_sync.sh   # smoke test

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

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

MAX_TRIALS_FLAG=""
if [ -n "${MAX_TRIALS:-}" ]; then
    MAX_TRIALS_FLAG="--max_trials ${MAX_TRIALS}"
fi

pixi run python evaluation/evaluate_rich_sync.py \
    --sync_root     /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test_sync \
    --rich_root     "$RICH_ROOT" \
    --centered_root "$CENTERED_MNT" \
    --gt_split      "${GT_SPLIT:-test}" \
    --smplx_model   body_models/SMPLX_NEUTRAL.pkl \
    --device        cuda \
    ${MAX_TRIALS_FLAG}

echo ""
echo "Done: $(date)"
