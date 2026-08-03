#!/bin/bash
# Drive the V3 residual-fusion training to completion through 1:30 debug jobs.
#
# V3 = the R2 objective, unchanged, on the residual architecture
# (fusion/fusion_module_v3.py): predict a correction to the chordal mean rather
# than the pose itself. Joint-confidence bias stays OFF.
#
# The debug partition caps jobs at 1:30 and the QOS allows 1 running + 1 queued,
# so the run is chopped into resumable chunks. `last.pt` is written after every
# epoch, so a kill loses at most one epoch.
#
# IMPORTANT: fusion/trainer_v2.py raises FileExistsError if best.pt exists and
# --resume was not passed. So the FIRST submission goes without --resume and every
# later one WITH it; the driver decides by looking for an existing checkpoint.
#
#   nohup bash bash_jobs/train_rich_v3_driver.sh > logs/train_v3_driver.log 2>&1 < /dev/null &
#   stop: pkill -f train_rich_v3_driver
#
# Env: MAX_EPOCHS (default from config), CKPT_DIR, MAX_ROUNDS, STALL_LIMIT, RUN

set -uo pipefail
cd /users/tnanni/ghost

RUN="${RUN:-V3}"
CKPT_DIR="${CKPT_DIR:-checkpoints/fusion_${RUN,,}}"
SCENE_JOB="bash_jobs/train_rich_v3.sh"
JOB_NAME="tr${RUN}"
POLL=180
MAX_ROUNDS="${MAX_ROUNDS:-800}"
STALL_LIMIT="${STALL_LIMIT:-15}"    # rounds with no epoch progress -> give up

# max_epochs is the only key of that name in the config; it sits well below `fusion:`
# so an -A window around that anchor misses it.
MAX_EPOCHS="${MAX_EPOCHS:-300}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"

mkdir -p logs
log() { echo "$(date '+%F %T') $*"; }

inflight() { squeue -u "$USER" -h -n "$JOB_NAME" 2>/dev/null | grep -q .; }

# Highest epoch reached, read from the tqdm descriptions in this job family's logs.
cur_epoch() {
    grep -ho "Epoch [0-9]\{4\}" logs/${JOB_NAME}_*.err 2>/dev/null \
        | awk '{print $2+0}' | sort -n | tail -1
}

log "RUN=$RUN  target ${MAX_EPOCHS} epochs, checkpoints -> ${CKPT_DIR}"

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
            RESUME_FLAG=1
        else
            RESUME_FLAG=""
            log "no checkpoint in ${CKPT_DIR} — starting a FRESH run"
        fi
        jid=$(sbatch --parsable --job-name="$JOB_NAME" \
                --export=ALL,RUN="$RUN",CKPT_DIR="$CKPT_DIR",RESUME="$RESUME_FLAG" \
                "$SCENE_JOB" 2>&1) \
            && log "submitted $jid  (epoch $ep/$MAX_EPOCHS, resume='${RESUME_FLAG:-no}')" \
            || log "sbatch failed: $jid"
    fi

    sleep "$POLL"
done

log "final: epoch $(cur_epoch)/$MAX_EPOCHS, checkpoints in ${CKPT_DIR}"
