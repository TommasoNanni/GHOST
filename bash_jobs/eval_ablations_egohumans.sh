#!/bin/bash -l
#SBATCH --job-name=abl_egohumans
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

# EgoHumans progressive oracle ablations (error attribution), resumable.
#   M2 = pred-cam + GT-scale + pred-pose
#   M3 = GT-cam   + GT-scale + pred-pose
#   M4 = GT-cam   + GT-scale + GT-pose   (pose oracle; shape stays predicted)
# Production (non-oracle) number: bash_jobs/eval_egohumans.sh
#
# One rung per job (that is how the driver submits them): the dump dir is keyed
# by rung, so the resume check never mistakes another rung's dump for this one.
# Passing several rungs at once also works and shares a single fusion pass, but
# then DUMP_DIR must be set explicitly. Stage A dumps per scene and skips
# existing dumps -> relaunch until the activity is done; Stage B aggregates.
#
# GT_DIR must be the copy that carries processed_data/smpl. egohumans_gt_full is
# the only one that does: the activity sqsh images and
# capstor/.../datasets/egohumans have EMPTY smpl dirs, which silently costs M4.
#
# Scenes without colmap GT cameras are skipped whole — every rung needs GT camera
# centres, M2 included (the GT scale is derived from them). As of 2026-07-24 that
# is 040-061_badminton and 010_basketball.
#
# Env:
#   ACTIVITY    e.g. 06_badminton   (subdir of ghost_root and of GT_DIR's parent)
#   GT_DIR      default /iopsstor/scratch/cscs/tnanni/egohumans_gt_full/$ACTIVITY
#   MODALITIES  comma-separated rungs, default "2" (one rung per job)
#   SCENE       optional single scene (e.g. 031_badminton)
#   TEMPORAL    set to any value to use temporal fusion (default: per-frame)
#   DUMP_DIR    default eval_ablations_egohumans/dumps/$ACTIVITY/m<MODALITIES>

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

ACTIVITY="${ACTIVITY:?set ACTIVITY, e.g. 06_badminton}"
MODALITIES="${MODALITIES:-2}"
# Rung-keyed dump dir: two rungs sharing one dir would make the resume check skip
# scenes already dumped by the OTHER rung, silently labelling M2 numbers as M3.
DUMP_DIR="${DUMP_DIR:-eval_ablations_egohumans/dumps/${ACTIVITY}/m${MODALITIES//,/_}}"
GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/${ACTIVITY}"
GT_DIR="${GT_DIR:-/iopsstor/scratch/cscs/tnanni/egohumans_gt_full/${ACTIVITY}}"

echo "Job $SLURM_JOB_ID  activity=$ACTIVITY  modalities=$MODALITIES  start=$(date)"
nvidia-smi -L

if [ ! -d "$GT_DIR" ]; then
    echo "GT_DIR does not exist: $GT_DIR" >&2
    exit 1
fi
# Fail fast rather than silently losing the M4 rung on an smpl-less GT copy.
if ! ls "$GT_DIR"/*/processed_data/smpl/*.npy >/dev/null 2>&1; then
    echo "No processed_data/smpl/*.npy under $GT_DIR — that copy has no SMPL GT," >&2
    echo "so the M4 rung would be dropped. Use egohumans_gt_full." >&2
    exit 1
fi

ARGS=(--ghost_root "$GHOST_ROOT" --gt_root "$GT_DIR"
      --checkpoint checkpoints/fusion_module/best.pt
      --modalities "$MODALITIES" --dump_dir "$DUMP_DIR")
[ -n "${SCENE:-}" ]    && ARGS+=(--scene "$SCENE")
[ -n "${TEMPORAL:-}" ] && ARGS+=(--temporal)

pixi run python evaluation/ablations_egohumans.py "${ARGS[@]}"

echo "=== Stage B: aggregate ==="
pixi run python evaluation/ablations_egohumans.py --metrics_only --dump_dir "$DUMP_DIR"
echo "Done: $(date)"
