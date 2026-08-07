#!/bin/bash -l
#SBATCH --job-name=audit_reid_gt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=a144

# Audit ghost person tracks + manual_reid groups against EgoHumans 2D GT.
# CPU only (no model, no GPU): undistorts GT keypoints and IoU-matches boxes.
#
# Env:
#   ACTIVITY   e.g. 07_tennis            (subdir of ghost_root and camera_ready)
#   GT_SQSH    sqsh holding this activity's GT (mounted RO), e.g.
#              /capstor/scratch/cscs/tnanni/datasets/egohumans_07_tennis_new.sqsh
#   GT_DIR     alternative: plain camera_ready/<ACTIVITY> dir
#   SCENE      optional single scene (e.g. 010_tennis)
#   OUT        report json (default eval_explainability/audit_reid_<ACTIVITY>.json)

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

ACTIVITY="${ACTIVITY:?set ACTIVITY, e.g. 07_tennis}"
GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/${ACTIVITY}"
INNER="${GT_INNER-media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready}"
OUT="${OUT:-eval_explainability/audit_reid_${ACTIVITY}.json}"

echo "Job ${SLURM_JOB_ID:-local}  activity=$ACTIVITY  start=$(date)"

cleanup() { [ -n "${MNT:-}" ] && fusermount -u "$MNT" 2>/dev/null && rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT

if [ -n "${GT_SQSH:-}" ]; then
    MNT="/tmp/gt_${SLURM_JOB_ID:-$$}"; mkdir -p "$MNT"
    squashfuse "$GT_SQSH" "$MNT"
    GT_ROOT="$MNT/${INNER:+$INNER/}$ACTIVITY"
    echo "mounted $GT_SQSH -> $GT_ROOT"
else
    GT_ROOT="${GT_DIR:?set GT_SQSH or GT_DIR}"
fi

ARGS=(--ghost_root "$GHOST_ROOT" --gt_root "$GT_ROOT" --out "$OUT")
[ -n "${SCENE:-}" ] && ARGS+=(--scene "$SCENE")

pixi run python debug/audit_egohumans_reid_gt.py "${ARGS[@]}"

echo "done $(date)"
