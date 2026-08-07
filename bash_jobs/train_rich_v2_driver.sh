#!/bin/bash
# Drive fusion training to completion through 1:30 debug jobs.
#
# Training needs ~3-4 h but the debug partition caps jobs at 1:30 and the QOS allows
# 1 running + 1 queued, so the run must be chopped into resumable chunks.
# `last.pt` is written after every epoch, so a kill loses at most one epoch.
#
# IMPORTANT: fusion/trainer_v2.py raises FileExistsError if best.pt exists and --resume
# was not passed. So the FIRST submission goes without --resume and every later one WITH
# it; the driver decides by looking for an existing checkpoint.
#
#   nohup bash bash_jobs/train_rich_v2_driver.sh > logs/train_driver.log 2>&1 < /dev/null &
#   stop: pkill -f train_rich_v2_driver
#
# Env: MAX_EPOCHS (default from config), CKPT_DIR, MAX_ROUNDS, STALL_LIMIT

set -uo pipefail
cd /users/tnanni/ghost

CKPT_DIR="${CKPT_DIR:-checkpoints/fusion_module_new}"
SCENE_JOB="bash_jobs/train_rich_v2.sh"
JOB_NAME="trrich"
POLL=180
MAX_ROUNDS="${MAX_ROUNDS:-800}"
STALL_LIMIT="${STALL_LIMIT:-15}"    # rounds with no epoch progress -> give up

# max_epochs is the only key of that name in the config; it sits well below `fusion:`
# so an -A window around that anchor misses it.
MAX_EPOCHS="${MAX_EPOCHS:-$(grep -m1 'max_epochs:' configuration/config.yaml | tr -dc '0-9')}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"

mkdir -p logs
log() { echo "$(date '+%F %T') $*"; }

inflight() { squeue -u "$USER" -h -n "$JOB_NAME" 2>/dev/null | grep -q .; }

# Highest epoch reached, read from the tqdm descriptions in this job family's logs.
cur_epoch() {
    grep -ho "Epoch [0-9]\{4\}" logs/${JOB_NAME}_*.err 2>/dev/null \
        | awk '{print $2+0}' | sort -n | tail -1
}

log "target ${MAX_EPOCHS} epochs, checkpoints -> ${CKPT_DIR}"

prev=-1
stall=0
for round in $(seq 1 "$MAX_ROUNDS"); do
    ep=$(cur_epoch); ep="${ep:-0}"

    if [ "$ep" -ge "$((MAX_EPOCHS - 1))" ]; then
        log "reached epoch $ep / $MAX_EPOCHS — training complete"; break
    fi

    if [ "$ep" -eq "$prev" ]; then
        stall=$((stall + 1))
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
            log "no epoch progress for $STALL_LIMIT rounds (stuck at $ep) — giving up"
            break
        fi
    else
        stall=0
    fi
    prev=$ep

    if ! inflight; then
        # Resume whenever a checkpoint already exists, otherwise the trainer refuses to start.
        if [ -f "${CKPT_DIR}/best.pt" ] || [ -f "${CKPT_DIR}/last.pt" ]; then
            ARGS="--resume"
        else
            ARGS=""
            log "no checkpoint in ${CKPT_DIR} — starting a FRESH run"
        fi
        jid=$(sbatch --parsable --job-name="$JOB_NAME" \
                --export=ALL,TRAIN_ARGS="$ARGS" "$SCENE_JOB" 2>&1) \
            && log "submitted $jid  (epoch $ep/$MAX_EPOCHS, args='${ARGS:-none}')" \
            || log "sbatch failed: $jid"
    fi

    sleep "$POLL"
done

log "final: epoch $(cur_epoch)/$MAX_EPOCHS, checkpoints in ${CKPT_DIR}"
