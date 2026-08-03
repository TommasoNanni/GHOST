#!/usr/bin/env bash
# Touches all files in scratch directories to reset deletion timer.
# Run manually every ~15 days: bash bash_jobs/keepalive.sh

DIRS=(
    /iopsstor/scratch/cscs/tnanni
    /capstor/scratch/cscs/tnanni
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "WARNING: $dir not found, skipping"
        continue
    fi
    echo "Touching $dir ..."
    # Read-only mounts (squashfuse .sqsh) fail per-file; drop that noise, keep real errors.
    find "$dir" -exec touch {} + 2> >(grep -v 'Read-only file system' >&2)
    echo "  done."
done

echo "Keepalive complete."
