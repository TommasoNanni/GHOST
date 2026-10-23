#!/bin/bash -l
#SBATCH --job-name=sync_build_egohumans
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

# STEP 1 of the through-sync EgoHumans evaluation (evaluation/sync_build_egohumans.py):
# rerun VGGT + MapAnything on the ESTIMATED alignment for the INEXACT trials only.
#
# Trials whose sync estimate was exact are deliberately NOT built: their corrected
# read index collapses to the same real frame in every camera, so the scene's
# existing vggt_cameras_centered.npz / vggt_depth_centered.npz /
# mapanything_scale_baseline.npy are already valid on that window and STEP 2 reads
# them in place. Only the 62 inexact trials cost any compute.
#
# Body parameters are never re-estimated -- person_*.npz are copied with their
# frame_indices windowed and relabeled. SAM3D ran per camera on footage the desync
# does not alter; what changes is the multi-view reconstruction.
#
# No squashfuse mount: EgoHumans images live on iopsstor as a plain directory tree
# (unlike RICH, whose centered crops are a .sqsh archive).
#
# Resumable -- a trial dir with sync_meta.json is skipped, so just resubmit after
# the 1:30 debug cap:
#   sbatch bash_jobs/sync_build_egohumans.sh
#   sbatch --export=ALL,MAX_SCENES=2 bash_jobs/sync_build_egohumans.sh   # smoke test

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0  # disable core dumps -- they fill up home quota

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

MANIFEST="${MANIFEST:-eval_egohumans/sync_manifest_s75.json}"
UNDIST_ROOT="${UNDIST_ROOT:-/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted}"
SYNC_ROOT="${SYNC_ROOT:-/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_sync_s75}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest $MANIFEST not found -- run evaluation/sync_dryrun_egohumans.py first"
    exit 1
fi

MAX_SCENES_FLAG=""
if [ -n "${MAX_SCENES:-}" ]; then
    MAX_SCENES_FLAG="--max_scenes ${MAX_SCENES}"
fi

pixi run python evaluation/sync_build_egohumans.py \
    --manifest         "$MANIFEST" \
    --undist_root      "$UNDIST_ROOT" \
    --sync_output_root "$SYNC_ROOT" \
    --vggt_weights     checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --min_error        "${MIN_ERROR:-1}" \
    ${MAX_SCENES_FLAG}

echo ""
echo "Done: $(date)"
