#!/bin/bash
# Sequential per-scene driver for 06_badminton.
# Submits one job at a time; waits for it to finish; retries if scene incomplete.

set -euo pipefail
cd /users/tnanni/ghost

DATA_ROOT="/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans"
OUTPUT_DIR="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans/06_badminton"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/06_badminton"
SLURM_SCRIPT="bash_jobs/egohumans_badminton_scene.sh"

mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a logs/egohumans_driver_badminton.log; }

scene_done() {
    local scene="$1"
    local scene_dir="${OUTPUT_DIR}/${scene}"
    [ -f "${scene_dir}/cross_view_reid.json" ] && \
    [ -f "${scene_dir}/vggt_cameras_centered.npz" ] && \
    [ -f "${scene_dir}/mapanything_scale_centered.npy" ]
}

wait_for_job() {
    local jobid="$1"
    while squeue -j "$jobid" -h 2>/dev/null | grep -q "$jobid"; do
        sleep 60
    done
}

# All scenes sorted
ALL_SCENES=$(ls "$DATA_ROOT/06_badminton/$INNER/" | sort)

for scene in $ALL_SCENES; do
    if scene_done "$scene"; then
        log "$scene: already done, skipping."
        continue
    fi

    log "$scene: starting."

    while ! scene_done "$scene"; do
        JOBID=$(sbatch --parsable --job-name="badminton_${scene}" \
            --export=SCENE="$scene" \
            "$SLURM_SCRIPT")
        log "$scene: submitted job $JOBID"
        wait_for_job "$JOBID"
        if scene_done "$scene"; then
            log "$scene: complete."
        else
            log "$scene: incomplete after job $JOBID — resubmitting."
        fi
    done
done

log "All 06_badminton scenes done — driver exiting."
