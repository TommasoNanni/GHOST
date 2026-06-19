#!/bin/bash -l
#SBATCH --job-name=egohumans_pipeline
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

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"

SQSH_ROOT="/capstor/scratch/cscs/tnanni/datasets"
MOUNT="/tmp/egohumans"
OUTPUT_DIR="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans"

ACTIVITIES="01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis"

mkdir -p "$MOUNT"
for ACT in $ACTIVITIES; do
    SQSH="${SQSH_ROOT}/egohumans_${ACT}.sqsh"
    if [ -f "$SQSH" ]; then
        mkdir -p "${MOUNT}/${ACT}" 2>/dev/null || true
        if squashfuse "$SQSH" "${MOUNT}/${ACT}"; then
            echo "Mounted $SQSH → ${MOUNT}/${ACT}"
        else
            echo "WARNING: failed to mount $SQSH, skipping."
        fi
    else
        echo "WARNING: $SQSH not found, skipping."
    fi
done

trap 'for ACT in '"$ACTIVITIES"'; do fusermount -u "${MOUNT}/${ACT}" 2>/dev/null && echo "Unmounted ${MOUNT}/${ACT}"; done' EXIT

mkdir -p logs "$OUTPUT_DIR"

ACTIVITY_ARG=""
if [ -n "${ACTIVITY:-}" ]; then
    ACTIVITY_ARG="--activity $ACTIVITY"
fi

pixi run python scripts/egohumans_pipeline.py \
    --data-root  "$MOUNT" \
    --output-dir "$OUTPUT_DIR" \
    $ACTIVITY_ARG

echo "Job finished: $(date)"
