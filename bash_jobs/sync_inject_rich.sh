#!/bin/bash -l
#SBATCH --job-name=sync_inject_rich
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# STEP 1 of the through-sync RICH evaluation (evaluation/sync_inject_rich.py):
# inject desync -> estimate it -> rerun VGGT + MapAnything on the estimated
# alignment. Reruns VGGT (multi-GPU: uses all --gpus-per-node visible devices,
# T frames split round-robin per preprocessing/run_vggt.py's multi-worker
# path) + MapAnything (single-GPU only, no multi-device path in
# preprocessing/run_mapanything.py) for every scene x trial, so this is heavy
# — expect the debug partition's 1:30:00 cap to not be enough for the full
# 52-scene RICH test set. The script is resumable (skips trial dirs that
# already have sync_meta.json, and whole scenes that are fully built), so
# just `sbatch` this again if it times out; no separate driver needed for now.
#
#   sbatch bash_jobs/sync_inject_rich.sh
#   sbatch --export=ALL,N_TRIALS=3 bash_jobs/sync_inject_rich.sh
#   sbatch --export=ALL,MAX_SCENES=5 bash_jobs/sync_inject_rich.sh   # smoke test

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0  # disable core dumps — they fill up home quota

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

# Same mount pattern as bash_jobs/eval_rich_prod.sh: capstor is Lustre and
# refuses a FUSE mount over itself, so the centered-crop archive must land
# node-local (/tmp). sync_inject_rich.py needs the actual JPEGs (not just
# crop_meta.json), since it re-runs VGGT + MapAnything on them.
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
    echo "ERROR: no crop_meta.json under $CENTERED_MNT — aborting (mount empty/wrong?)"
    exit 1
fi
echo ""

MAX_SCENES_FLAG=""
if [ -n "${MAX_SCENES:-}" ]; then
    MAX_SCENES_FLAG="--max_scenes ${MAX_SCENES}"
fi

pixi run python evaluation/sync_inject_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         "$RICH_ROOT" \
    --centered_root     "$CENTERED_MNT" \
    --gt_split          "${GT_SPLIT:-test}" \
    --sync_output_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test_sync \
    --vggt_weights      checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --max_shift         "${MAX_SHIFT:-45}" \
    --n_trials          "${N_TRIALS:-1}" \
    ${MAX_SCENES_FLAG}

echo ""
echo "Done: $(date)"
