#!/bin/bash -l
#SBATCH --job-name=reid_cams
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144

# Stage-A probe for ReID v7: run the camera pass (preprocessing/reid_cameras.py)
# on a few ladder scenes. Generic via env vars:
#   OUTPUT_DIR   ghost output tree containing <scene>/<cam>/body_data
#   FRAMES_ROOT  root with <scene>/<cam>/ image folders
#   FRAMES_SQSH  optional squashfs to mount as FRAMES_ROOT
#   SCENES       space-separated scene names (default: all)
#   MOVING_CAMS  space-separated moving camera names (default: none)
# Example (RICH):
#   OUTPUT_DIR=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
#   FRAMES_SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh \
#   SCENES="ParkingLot1_007_overfence2 Gym_010_cooking1" \
#   sbatch bash_jobs/reid_cameras_probe.sh

set -euo pipefail
cd /users/tnanni/ghost

OUTPUT_DIR="${OUTPUT_DIR:?}"
SCENES="${SCENES:-}"
MOVING_CAMS="${MOVING_CAMS:-}"

if [[ -n "${FRAMES_SQSH:-}" ]]; then
    MNT="/tmp/reid_cams_${SLURM_JOB_ID}"; mkdir -p "$MNT"
    cleanup() { fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null || true; }
    trap cleanup EXIT
    squashfuse "$FRAMES_SQSH" "$MNT"
    FRAMES_ROOT="$MNT"
fi
FRAMES_ROOT="${FRAMES_ROOT:?set FRAMES_ROOT or FRAMES_SQSH}"

ARGS=(--output_dir "$OUTPUT_DIR" --frames_root "$FRAMES_ROOT")
[[ -n "$SCENES" ]] && ARGS+=(--scenes $SCENES)
[[ -n "$MOVING_CAMS" ]] && ARGS+=(--moving_cams $MOVING_CAMS)

pixi run python -m preprocessing.reid_cameras "${ARGS[@]}"
echo "Done: $(date)"
