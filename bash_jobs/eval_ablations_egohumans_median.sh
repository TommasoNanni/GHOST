#!/bin/bash -l
#SBATCH --job-name=abl_egohumans_median
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

# EgoHumans progressive oracle ablations (error attribution), resumable,
# GEODESIC-MEDIAN FUSION — the shipped fusion rule, no checkpoint.
# evaluation/ablations_egohumans_median.py is a copy of
# evaluation/ablations_egohumans.py with the learned PoseFusionModule swapped
# for the geodesic median of the per-camera SAM3D poses (median_fuse, identical
# to evaluation/evaluate_egohumans_median.py). No --temporal flag: the median
# has no cross-frame coupling, so the checkpoint's TEMPORAL_PAD OOD workaround
# does not apply here.
#   M2 = pred-cam + GT-scale + pred-pose
#   M3 = GT-cam   + GT-scale + pred-pose
#   M4 = GT-cam   + GT-scale + GT-pose   (pose oracle; shape stays predicted)
# Production (non-oracle) number: bash_jobs/eval_egohumans_median.sh
#
# One rung per job (mirrors the v2 driver pattern): the dump dir is keyed by
# rung, so the resume check never mistakes another rung's dump for this one.
# Passing several rungs at once also works and shares a single fusion pass, but
# then DUMP_DIR must be set explicitly. Stage A dumps per scene and skips
# existing dumps -> relaunch until the activity is done; Stage B aggregates.
#
# GT_DIR must be the copy that carries processed_data/smpl. egohumans_gt_full is
# the only one that does: the activity sqsh images and
# capstor/.../datasets/egohumans have EMPTY smpl dirs, which silently costs M4.
#
# Scenes without colmap GT cameras are skipped whole — every rung needs GT camera
# centres, M2 included (the GT scale is derived from them).
#
# Env:
#   ACTIVITY    e.g. 06_badminton   (subdir of ghost_root and of GT_DIR's parent)
#   GT_DIR      default /iopsstor/scratch/cscs/tnanni/egohumans_gt_full/$ACTIVITY
#   MODALITIES  comma-separated rungs, default "2" (one rung per job)
#   SCENE       optional single scene (e.g. 031_badminton)
#   DUMP_DIR    default eval_ablations_egohumans/dumps_median/$ACTIVITY/m<MODALITIES>
#
# NOTE on sbatch --export: pass MODALITIES as a single value per job
# (--export=ALL,MODALITIES=2), not a comma list (--export=ALL,MODALITIES=2,3) —
# sbatch splits on every comma in --export, so a comma-containing value silently
# truncates to its first component.

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

ACTIVITY="${ACTIVITY:?set ACTIVITY, e.g. 06_badminton}"
MODALITIES="${MODALITIES:-2}"
# Rung-keyed dump dir: two rungs sharing one dir would make the resume check skip
# scenes already dumped by the OTHER rung, silently labelling M2 numbers as M3.
DUMP_DIR="${DUMP_DIR:-eval_ablations_egohumans/dumps_median/${ACTIVITY}/m${MODALITIES//,/_}}"
GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/${ACTIVITY}"
GT_DIR="${GT_DIR:-/iopsstor/scratch/cscs/tnanni/egohumans_gt_full/${ACTIVITY}}"

echo "Job $SLURM_JOB_ID  activity=$ACTIVITY  modalities=$MODALITIES  fusion=geodesic-median  start=$(date)"
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
      --modalities "$MODALITIES" --dump_dir "$DUMP_DIR")
[ -n "${SCENE:-}" ] && ARGS+=(--scene "$SCENE")

pixi run python evaluation/ablations_egohumans_median.py "${ARGS[@]}"

echo "=== Stage B: aggregate ==="
pixi run python evaluation/ablations_egohumans_median.py --metrics_only --dump_dir "$DUMP_DIR"
echo "Done: $(date)"
