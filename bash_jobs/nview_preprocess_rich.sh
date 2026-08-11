#!/bin/bash -l
#SBATCH --job-name=nview_prep_rich
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# N-view RICH ablation, stage 1: re-run VGGT + MapAnything on the first N cameras.
#
# Writes ONLY to  ghost_outputs/rich_${SPLIT}_nview${N}  — the production root
# ghost_outputs/rich_${SPLIT} is opened read-only (the script refuses to run if
# the output root sits inside it).
#
# Images live inside centered_${SPLIT}.sqsh, so the archive is squashfuse-mounted
# node-local for the duration of the job.
#
#   N=2 sbatch bash_jobs/nview_preprocess_rich.sh
#   N=3 sbatch bash_jobs/nview_preprocess_rich.sh
#
# The script is resumable per scene (a scene is skipped once its cameras npz,
# scale npy and body_data exist), so the 1:30 debug limit can be absorbed by
# simply resubmitting. For a single shot instead:
#
#   N=2 sbatch --partition=normal --time=06:00:00 bash_jobs/nview_preprocess_rich.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

N="${N:?set N=2 or N=3}"
SPLIT="${SPLIT:-test}"

SRC_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_${SPLIT}"
OUT_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_${SPLIT}_nview${N}"
SQSH="/capstor/scratch/cscs/tnanni/datasets/rich/centered_${SPLIT}.sqsh"

MNT="/tmp/centered_${SPLIT}_nview${N}_${SLURM_JOB_ID:-$$}"
mkdir -p "$MNT"
cleanup() { fusermount -u "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT
squashfuse "$SQSH" "$MNT"
echo "Mounted $SQSH -> $MNT ($(ls "$MNT" | wc -l) scenes)"

echo "=== GPU STATUS ==="
nvidia-smi
echo "=================="
echo "N=$N  SPLIT=$SPLIT  job=${SLURM_JOB_ID:-local}  start=$(date)"
echo "src (read-only): $SRC_ROOT"
echo "out            : $OUT_ROOT"
echo ""

pixi run python scripts/nview_preprocess_rich.py \
    --n_views  "$N" \
    --img_root "$MNT" \
    --src_root "$SRC_ROOT" \
    --out_root "$OUT_ROOT" \
    --batch_size "${BATCH_SIZE:-8}"

echo ""
echo "Done: $(date)"
