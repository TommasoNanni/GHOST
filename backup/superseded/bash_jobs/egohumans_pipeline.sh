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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"

SQSH_ROOT="/capstor/scratch/cscs/tnanni/datasets"
DATA_ROOT="/capstor/scratch/cscs/tnanni/datasets/egohumans"
OUTPUT_DIR="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans"

ACTIVITIES="01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis"

INNER_PATH="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

for ACT in $ACTIVITIES; do
    SQSH="${SQSH_ROOT}/egohumans_${ACT}.sqsh"
    MNT="${DATA_ROOT}/${ACT}"
    if [ -f "$SQSH" ]; then
        mkdir -p "$MNT" 2>/dev/null || true
        if squashfuse "$SQSH" "$MNT"; then
            echo "Mounted $SQSH → $MNT"
        else
            echo "WARNING: failed to mount $SQSH, skipping."
        fi
    else
        # Check for split squash files (_a, _b)
        SPLIT_FOUND=0
        for SUFFIX in _a _b; do
            SQSH_SPLIT="${SQSH_ROOT}/egohumans_${ACT}${SUFFIX}.sqsh"
            if [ -f "$SQSH_SPLIT" ]; then
                MNT_SPLIT="${DATA_ROOT}/${ACT}${SUFFIX}"
                mkdir -p "$MNT_SPLIT" 2>/dev/null || true
                if squashfuse "$SQSH_SPLIT" "$MNT_SPLIT"; then
                    echo "Mounted $SQSH_SPLIT → $MNT_SPLIT"
                    SPLIT_FOUND=1
                    # Symlink scenes into unified dir
                    SEQ_DIR="${MNT_SPLIT}/${INNER_PATH}/${ACT}"
                    UNIFIED="${MNT}/${INNER_PATH}/${ACT}"
                    mkdir -p "$UNIFIED"
                    for SEQ in "$SEQ_DIR"/*/; do
                        [ -d "$SEQ" ] && ln -sfn "$SEQ" "${UNIFIED}/$(basename $SEQ)"
                    done
                else
                    echo "WARNING: failed to mount $SQSH_SPLIT, skipping."
                fi
            fi
        done
        if [ "$SPLIT_FOUND" -eq 0 ]; then
            if [ -d "$MNT" ]; then
                echo "Using raw dir $MNT (already in place)"
            else
                echo "WARNING: no sqsh or raw dir found for $ACT, skipping."
            fi
        fi
    fi
done

trap '
for ACT in '"$ACTIVITIES"'; do
    fusermount -u "${DATA_ROOT}/${ACT}" 2>/dev/null || true
    for SUFFIX in _a _b; do
        fusermount -u "${DATA_ROOT}/${ACT}${SUFFIX}" 2>/dev/null || true
    done
done' EXIT

mkdir -p logs "$OUTPUT_DIR"

ACTIVITY_ARG=""
if [ -n "${ACTIVITY:-}" ]; then
    ACTIVITY_ARG="--activity $ACTIVITY"
fi

pixi run python scripts/egohumans_pipeline.py \
    --data-root  "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    $ACTIVITY_ARG

echo "Job finished: $(date)"
