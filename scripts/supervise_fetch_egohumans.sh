#!/bin/bash -l
# Keep scripts/fetch_egohumans_inexact.py alive until it reports completion.
#
# The Clariden login node SIGKILLs long CPU-bound python (~9-15 min) and that
# kill survives nohup/setsid, so an unattended overnight fetch can die halfway
# with no one to restart it. The fetch driver is resumable (a scene with a
# complete images_undistorted/ for every kept camera is skipped), so restarting
# it costs one directory listing per finished scene.
#
# Run detached:
#   setsid nohup bash scripts/supervise_fetch_egohumans.sh \
#       > logs/supervise_fetch.log 2>&1 < /dev/null &
#
# Stop it: kill the supervisor first, then the driver, or it will restart one.
set -uo pipefail
cd /users/tnanni/ghost

MANIFEST=${MANIFEST:-eval_egohumans/sync_manifest_s75.json}
DATA_ROOT=${DATA_ROOT:-/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted}
LOG=${LOG:-logs/fetch_egohumans_inexact.log}
WORKERS=${WORKERS:-16}
POLL=${POLL:-120}
MAX_RESTARTS=${MAX_RESTARTS:-40}

# Matches only the driver's own python process. The supervisor's command line is
# "bash scripts/supervise_fetch_egohumans.sh", so it never matches itself.
PATTERN="fetch_egohumans_inexact.py --manifest"

restarts=0
echo "supervisor START $(date) host=$(hostname) poll=${POLL}s"

while true; do
    if grep -q "^Done\. ok=" "$LOG" 2>/dev/null; then
        echo "supervisor: driver reported completion -- exiting"
        grep "^Done\. ok=" "$LOG" | tail -1
        break
    fi

    if pgrep -f "$PATTERN" > /dev/null 2>&1; then
        sleep "$POLL"
        continue
    fi

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "supervisor: $restarts restarts reached, giving up $(date)"
        break
    fi
    restarts=$((restarts + 1))
    echo "supervisor: driver not running -- restart #$restarts $(date)"
    # Append, never truncate: the log is the only record of what already ran.
    setsid nohup pixi run python scripts/fetch_egohumans_inexact.py \
        --manifest "$MANIFEST" --data_root "$DATA_ROOT" --workers "$WORKERS" \
        < /dev/null >> "$LOG" 2>&1 &
    disown
    sleep 30
done

echo "supervisor END $(date) restarts=$restarts"
