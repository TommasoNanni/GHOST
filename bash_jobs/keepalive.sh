#!/usr/bin/env bash
# Touches all files in scratch directories to reset deletion timer.
# Run manually every ~15 days: bash bash_jobs/keepalive.sh

DIRS=(
    /iopsstor/scratch/cscs/tnanni/ghost_outputs
    /capstor/scratch/cscs/tnanni/datasets/rich
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "WARNING: $dir not found, skipping"
        continue
    fi
    echo "Touching $dir ..."
    find "$dir" -exec touch {} +
    echo "  done."
done

echo "Keepalive complete."
