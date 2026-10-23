#!/bin/bash
# Re-download the full EgoHumans raw tree from Drive into iopsstor.
#
# Replaces the capstor .sqsh archives as the source of truth. Fetches every activity
# with scripts/download_egohumans_gt.py --full, so each scene lands as
#
#   $DEST/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/<act>/<seq>/
#       exo/<cam>/images/        raw 3840x2160 distorted frames
#       colmap/workplace/        cameras.txt + images.txt (OPENCV_FISHEYE)
#       processed_data/          GT: poses2d (per-camera 2D + aria ids), smpl, poses3d
#
# which is exactly what bash_jobs/egohumans_new_scene.sh expects at RAW_DL, and what
# scripts/egohumans_pipeline_production.py needs for --gt-root.
#
# ego/ is excluded: the pipeline runs on exo cameras only. Excluding does NOT save
# bandwidth (a .tar.gz is sequential, the bytes are pulled either way) — only disk.
#
# Resumable: download_egohumans_gt.py writes a .full_done sentinel per scene and skips
# finished ones, so restarting costs one directory listing per completed scene.
#
# Why the restart loop: the Clariden login node SIGKILLs long CPU-bound python after
# ~9-15 min and that kill survives nohup/setsid. A multi-day fetch therefore dies
# repeatedly; this wrapper just starts it again until every activity reports done.
#
# Launch detached:
#   setsid nohup bash bash_jobs/download_egohumans_all.sh \
#       > logs/download_egohumans_all.log 2>&1 < /dev/null &
# One activity only:
#   setsid nohup bash bash_jobs/download_egohumans_all.sh 06_badminton > ... &
# Stop: pkill -f download_egohumans_all
set -uo pipefail
cd /users/tnanni/ghost

DEST="${DEST:-/iopsstor/scratch/cscs/tnanni/egohumans_raw}"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
ALL_ACTS=(01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis)
ACTS=("${1:-}"); [ -z "${ACTS[0]}" ] && ACTS=("${ALL_ACTS[@]}")

MAX_RESTARTS="${MAX_RESTARTS:-200}"   # per activity
RETRIES="${RETRIES:-3}"               # passed through to the fetcher, per scene

mkdir -p logs "$DEST"
log() { echo "$(date '+%F %T') $*"; }

log "dest=$DEST  activities=${ACTS[*]}"

for ACT in "${ACTS[@]}"; do
    ACT_DEST="$DEST/$INNER/$ACT"
    mkdir -p "$ACT_DEST"

    n=0
    while (( n < MAX_RESTARTS )); do
        n=$((n + 1))
        log "=== $ACT: attempt $n ==="

        pixi run python scripts/download_egohumans_gt.py \
            --activity   "$ACT" \
            --dest-root  "$ACT_DEST" \
            --full \
            --exclude    ego \
            --retries    "$RETRIES"
        rc=$?

        if (( rc == 0 )); then
            log "$ACT: fetcher exited clean"
            break
        fi
        # rc 137/143 = SIGKILL/SIGTERM from the login-node watchdog: just go again.
        log "$ACT: fetcher exited rc=$rc — restarting"
        sleep 30
    done

    done_n=$(find "$ACT_DEST" -maxdepth 2 -name ".full_done" 2>/dev/null | wc -l)
    scenes_n=$(find "$ACT_DEST" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    log "$ACT: $done_n/$scenes_n scenes carry .full_done"
done

log "=== ALL ACTIVITIES ATTEMPTED ==="
log "Verify before deleting any .sqsh:  bash bash_jobs/verify_egohumans_download.sh"
