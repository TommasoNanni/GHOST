#!/bin/bash
# Login-node driver for bash_jobs/egohumans_prod_node.sh: keeps exactly ONE node-job
# queued/running and resubmits it until every scene in the raw tree is done.
#
# Launch detached:
#   nohup bash bash_jobs/egohumans_prod_driver.sh > logs/egohumans_prod_driver.log 2>&1 < /dev/null &
# Stop: pkill -f egohumans_prod_driver   (running node-job keeps going; scancel it separately)
set -uo pipefail
cd /users/tnanni/ghost

RAW="/iopsstor/scratch/cscs/tnanni/egohumans_raw"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
OUT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_prod"
JOB="bash_jobs/egohumans_prod_node.sh"
NAME="egohumans_prod_node"
POLL=300
MAX_SUBMITS="${MAX_SUBMITS:-10}"    # safety: stop if scenes keep failing

log() { echo "$(date '+%F %T') $*"; }

remaining() {
    local n=0
    for d in "$RAW/$INNER"/*/*/; do
        local s="$(basename "$(dirname "$d")")/$(basename "$d")"
        [ -f "$OUT/$s/gt_reid.json" ] && [ -f "$OUT/$s/vggt_cameras_centered.npz" ] \
            && [ -f "$OUT/$s/mapanything_scale_baseline.npy" ] || n=$((n + 1))
    done
    echo "$n"
}

submits=0
while :; do
    left=$(remaining)
    if (( left == 0 )); then log "all scenes done — exiting"; break; fi
    if squeue -u "$USER" -h -n "$NAME" | grep -q .; then
        log "$left scenes left, node-job active"
    elif (( submits >= MAX_SUBMITS )); then
        log "$left scenes left but $MAX_SUBMITS submissions used — stopping, check logs/egohumans_prod/"
        break
    else
        jid=$(sbatch --parsable "$JOB" 2>&1) && submits=$((submits + 1))
        log "$left scenes left — submitted $jid (submission $submits/$MAX_SUBMITS)"
    fi
    sleep "$POLL"
done
