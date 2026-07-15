#!/bin/bash -l
#SBATCH --job-name=probe_fencing_vggt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=0:45:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"

ACTIVITY="03_fencing"
SQSH="/capstor/scratch/cscs/tnanni/datasets/egohumans_${ACTIVITY}.sqsh"
WEIGHTS="checkpoints/vggt_omega/vggt_omega_1b_512.pt"
OUT="/users/tnanni/ghost/temp_egohumans"
MNT="/tmp/fencing_mnt_${SLURM_JOB_ID}"

mkdir -p logs "$OUT" "$MNT"

# Mount the activity sqsh READ-ONLY (source data / sqsh never modified).
echo "Mounting $SQSH -> $MNT"
squashfuse "$SQSH" "$MNT"
# Always unmount on exit (success or failure).
trap 'fusermount -u "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true' EXIT

# 1) re-undistort (balance=0, no borders) + run VGGT only, into temp dir
pixi run python scripts/probe_fencing_undistort_vggt.py \
    --raw-root "$MNT" \
    --activity "$ACTIVITY" \
    --out      "$OUT" \
    --weights  "$WEIGHTS" \
    --device   cuda:0 \
    --stride   10 \
    --max-frames 60

# 2) eval the fixed cameras vs COLMAP GT (GT read from the mount, vggt from temp)
echo ""
echo "================ POST-FIX (balance=0) camera eval ================"
pixi run python scripts/eval_vggt_cameras_egohumans.py \
    --activity "$ACTIVITY" \
    --raw-root "$MNT" \
    --out-root "$OUT"

echo "Job finished: $(date)"
