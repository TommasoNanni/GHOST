#!/bin/bash -l
#SBATCH --job-name=probe_ma
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144

set -euo pipefail
cd /users/tnanni/ghost

SCENE="${SCENE:-031_badminton}"
ACTIVITY="${ACTIVITY:-06_badminton}"
GT_SQSH="${GT_SQSH:?}"
FRAME="${FRAME:-400}"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

MNT="/tmp/probe_${SLURM_JOB_ID}"; mkdir -p "$MNT"
cleanup() { fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT
squashfuse "$GT_SQSH" "$MNT"

pixi run python scripts/probe_mapanything_standalone.py \
    --scene "$SCENE" --activity "$ACTIVITY" \
    --gt_root "$MNT/$INNER/$ACTIVITY" --frame "$FRAME"
echo "Done: $(date)"
