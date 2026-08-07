#!/bin/bash
# Regenerate the wiped RICH TRAIN ghost outputs (body_data + masks + ReID).
#
# Why: ghost_outputs/rich_train lost every person_*.npz and mask_data.npz — only
# json_data (empty), vggt_cameras_centered.npz and the MapAnything scales survived.
# That blocks retraining and the RICH val split. The source images are still in
# datasets/rich/train_dataset.sqsh, so Stage 1+2 can be re-run.
#
# What actually re-runs per scene: segmentation -> body estimation -> cross-view ReID.
# Step 6 (VGGT) skips itself because vggt_cameras_centered.npz already exists, and
# --skip-mapanything avoids reloading MapAnything since the scales are already on disk.
#
# The debug partition caps jobs at 1:30 and the QOS allows 1 running + 1 queued, so a
# single job cannot cover 62 scenes. Each submission processes as many scenes as fit
# and skips the ones already finished (the pipeline's own scene-level skip keys on
# cross_view_reid.json), so repeated submission converges.
#
#   nohup bash bash_jobs/rich_train_preprocess_driver.sh > logs/rich_train_prep_driver.log 2>&1 < /dev/null &
#   stop: pkill -f rich_train_preprocess_driver

set -uo pipefail
cd /users/tnanni/ghost

OUT_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train"
SCENE_JOB="bash_jobs/rich_pipeline_v3.sh"
JOB_NAME="rtprep"
POLL=180
MAX_ROUNDS="${MAX_ROUNDS:-400}"
# Progress is measured by scenes finished AND person_*.npz written, because a single
# scene takes far longer than a few polls: counting only finished scenes would look
# like a stall during normal processing. npz files are saved per person as body
# estimation proceeds, so they climb continuously within a scene.
STALL_LIMIT="${STALL_LIMIT:-20}"    # rounds with NEITHER counter moving -> give up

mkdir -p logs
log() { echo "$(date '+%F %T') $*"; }

n_total() { find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l; }
# The pipeline's own done-marker, so driver and pipeline agree on "finished".
n_done()  { ls "$OUT_ROOT"/*/cross_view_reid.json 2>/dev/null | wc -l; }
n_body()  { find "$OUT_ROOT" -name 'person_*.npz' -path '*body_data*' 2>/dev/null | wc -l; }
inflight() { squeue -u "$USER" -h -n "$JOB_NAME" 2>/dev/null | grep -q .; }

TOTAL=$(n_total)
log "rich_train: $TOTAL scenes, $(n_done) already complete, $(n_body) person npz on disk"

prev_sig=""
stall=0
for round in $(seq 1 "$MAX_ROUNDS"); do
    done_now=$(n_done)
    body_now=$(n_body)
    if [ "$done_now" -ge "$TOTAL" ]; then
        log "all $TOTAL scenes complete"; break
    fi

    sig="${done_now}:${body_now}"
    if [ "$sig" = "$prev_sig" ]; then
        stall=$((stall + 1))
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
            log "no progress for $STALL_LIMIT rounds (scenes=$done_now/$TOTAL, npz=$body_now) — giving up"
            break
        fi
    else
        stall=0
    fi
    prev_sig=$sig

    if ! inflight; then
        jid=$(sbatch --parsable --job-name="$JOB_NAME" \
                --export=ALL,PIPELINE_ARGS="--skip-mapanything" "$SCENE_JOB" 2>&1) \
            && log "submitted $jid  ($done_now/$TOTAL scenes done, $(n_body) npz)" \
            || log "sbatch failed: $jid"
    fi

    sleep "$POLL"
done

log "final: $(n_done)/$TOTAL scenes, $(n_body) person npz"
