#!/bin/bash -l
#SBATCH --job-name=egohumans_new
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# One scene of the EgoHumans balance=0 re-run.
# Required env: ACT (e.g. 02_lego), SCENE (e.g. 001_legoassemble).
# Stage A: mount sqsh RO -> prep (balance=0 undistort+resize -> temp) -> pipeline -> unmount.

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${ACT:?set ACT}"; : "${SCENE:?set SCENE}"

SQSH_ROOT="/capstor/scratch/cscs/tnanni/datasets"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
TEMP="/iopsstor/scratch/cscs/tnanni/temp_egohumans"
OUT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
MNT_BASE="/tmp/egonew_${ACT}_${SLURM_JOB_ID}"

mkdir -p logs "$TEMP" "$OUT"

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $ACT/$SCENE | $(date)"

MOUNTS=()
trap 'for m in "${MOUNTS[@]:-}"; do fusermount -u "$m" 2>/dev/null || true; rmdir "$m" 2>/dev/null || true; done' EXIT

# Activities already extracted (unsquashed) in /iopsstor — read directly, no mount.
declare -A EXTRACTED_ROOTS=(
    [06_badminton]="/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans/06_badminton"
)

# Mount the activity sqsh (handles the badminton _a/_b split) and find the scene.
RAW_ROOT=""
try_mount() {  # $1 = sqsh path, $2 = mountpoint
    [ -f "$1" ] || return 1
    mkdir -p "$2"; MOUNTS+=("$2")
    squashfuse "$1" "$2" || return 1
    [ -d "$2/$INNER/$ACT/$SCENE" ]
}

if [ -n "${EXTRACTED_ROOTS[$ACT]:-}" ] && [ -d "${EXTRACTED_ROOTS[$ACT]}/$INNER/$ACT/$SCENE" ]; then
    RAW_ROOT="${EXTRACTED_ROOTS[$ACT]}"; echo "Using extracted data (no mount): $RAW_ROOT"
elif try_mount "$SQSH_ROOT/egohumans_${ACT}.sqsh" "${MNT_BASE}"; then
    RAW_ROOT="${MNT_BASE}"
else
    for SUF in _a _b; do
        if try_mount "$SQSH_ROOT/egohumans_${ACT}${SUF}.sqsh" "${MNT_BASE}${SUF}"; then
            RAW_ROOT="${MNT_BASE}${SUF}"; break
        fi
    done
fi
[ -n "$RAW_ROOT" ] || { echo "ERROR: scene $ACT/$SCENE not found in any sqsh"; exit 1; }
echo "Scene found in: $RAW_ROOT"

# 1) balance=0 undistort + resize -> temp data_root (idempotent)
pixi run python scripts/prep_undistort_egohumans.py \
    --raw-root "$RAW_ROOT" --activity "$ACT" --seq "$SCENE" \
    --out-root "$TEMP" --max-side 1440

# 2) full pipeline on the re-undistorted scene (seg -> SAM3D -> ReID -> VGGT -> MapAnything)
pixi run python scripts/egohumans_pipeline.py \
    --data-root  "$TEMP" \
    --output-dir "$OUT" \
    --activity   "$ACT" \
    --seq        "$SCENE"

echo "Done $ACT/$SCENE: $(date)"
