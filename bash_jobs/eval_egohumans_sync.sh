#!/bin/bash -l
#SBATCH --job-name=eval_egohumans_sync
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

# STEP 2 of the through-sync EgoHumans evaluation (evaluation/evaluate_egohumans_sync.py):
# median fusion + placement + CHROMM metrics over every trial in the manifest.
#
# EXACT trials read the ORIGINAL scene dir sliced to their window (nothing was
# rebuilt); INEXACT trials read the trial dirs from bash_jobs/sync_build_egohumans.sh.
#
# Must run on the debug partition, NOT the login node: BodyPlacer's Procrustes DLT
# is CPU-bound for minutes and the login-node watchdog SIGKILLs it (see
# [[login-node-watchdog-sigkill]] -- observed on this exact code path).
#
# Resumable -- a trial with an existing dump is skipped:
#   sbatch bash_jobs/eval_egohumans_sync.sh
#   sbatch --export=ALL,MAX_TRIALS=2 bash_jobs/eval_egohumans_sync.sh    # smoke test
#   sbatch --export=ALL,SCENE=01_tagging/008_tagging bash_jobs/eval_egohumans_sync.sh
#
# Stage B (aggregate only, no GPU needed):
#   pixi run python evaluation/evaluate_egohumans_sync.py --metrics_only \
#       --manifest ... --sync_root ... --gt_root ... --dump_dir ...

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

MANIFEST="${MANIFEST:-eval_egohumans/sync_manifest_s75.json}"
SYNC_ROOT="${SYNC_ROOT:-/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_sync_s75}"
GT_ROOT="${GT_ROOT:-/iopsstor/scratch/cscs/tnanni/egohumans_gt_full}"
DUMP_DIR="${DUMP_DIR:-eval_egohumans/dumps_sync_s75}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest $MANIFEST not found -- run evaluation/sync_dryrun_egohumans.py first"
    exit 1
fi

ARGS=(--manifest "$MANIFEST" --sync_root "$SYNC_ROOT" --gt_root "$GT_ROOT"
      --dump_dir "$DUMP_DIR" --smplx_model body_models/SMPLX_NEUTRAL.pkl)
[ -n "${SCENE:-}" ]      && ARGS+=(--scene "$SCENE")
[ -n "${MAX_TRIALS:-}" ] && ARGS+=(--max_trials "$MAX_TRIALS")
[ -n "${OVERWRITE:-}" ]  && ARGS+=(--overwrite)
# BASELINE=1 runs the PAIRED CONTROL: each inexact trial scored on the TRUE
# alignment over the same window. Point DUMP_DIR at a separate directory.
[ -n "${BASELINE:-}" ]   && ARGS+=(--baseline)

pixi run python evaluation/evaluate_egohumans_sync.py "${ARGS[@]}"

echo ""
echo "Done: $(date)"
