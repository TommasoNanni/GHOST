#!/bin/bash -l
#SBATCH --job-name=eval_egohumans
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

# Stage A (GPU): dump per-scene pred/gt joints for EgoHumans, resumable.
# Env:
#   ACTIVITY   e.g. 06_badminton   (subdir of ghost_root and camera_ready)
#   GT_SQSH    path to the sqsh holding this activity's GT  (mounted RO)
#   GT_DIR     alternative: a plain dir camera_ready/<ACTIVITY> (e.g. the badminton backup)
#   SCENE      optional single scene (e.g. 031_badminton)
#   SCALE      pred | triangulated   (default pred)
#   DUMP_DIR   default eval_egohumans/dumps

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

ACTIVITY="${ACTIVITY:?set ACTIVITY, e.g. 06_badminton}"
SCALE="${SCALE:-pred}"
DUMP_DIR="${DUMP_DIR:-eval_egohumans/dumps/${ACTIVITY}}"
GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/${ACTIVITY}"
INNER="${GT_INNER-media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready}"

echo "Job $SLURM_JOB_ID  activity=$ACTIVITY  scale=$SCALE  start=$(date)"
nvidia-smi -L

cleanup() { [ -n "${MNT:-}" ] && fusermount -u "$MNT" 2>/dev/null && rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT

if [ -n "${GT_SQSH:-}" ]; then
    MNT="/tmp/gt_${SLURM_JOB_ID}"; mkdir -p "$MNT"
    squashfuse "$GT_SQSH" "$MNT"
    GT_ROOT="$MNT/${INNER:+$INNER/}$ACTIVITY"
    echo "mounted $GT_SQSH -> $GT_ROOT"
else
    GT_ROOT="${GT_DIR:?set GT_SQSH or GT_DIR}"
fi

ARGS=(--ghost_root "$GHOST_ROOT" --gt_root "$GT_ROOT"
      --checkpoint checkpoints/fusion_module/best.pt
      --scale "$SCALE" --dump_dir "$DUMP_DIR")
[ -n "${SCENE:-}" ] && ARGS+=(--scene "$SCENE")
[ -n "${TEMPORAL:-}" ] && ARGS+=(--temporal)

pixi run python evaluation/evaluate_egohumans.py "${ARGS[@]}"

echo "=== Stage B: aggregate ==="
pixi run python evaluation/evaluate_egohumans.py --metrics_only --dump_dir "$DUMP_DIR"
echo "Done: $(date)"
