#!/bin/bash
# Overnight chain: wait for preprocessing -> GT relabel -> verify -> launch training.
#
# Sequence
#   1. wait until preprocessing is done OR has stalled (no new completed scene for
#      STALL_MIN minutes). 3 scenes are known to hang on cam_06 body estimation, so a
#      stall is expected and must NOT block the chain forever.
#   2. run scripts/gt_reid_rich.py --apply so any newly finished scene gets the same
#      GT-based id correction as the other 58 (appearance ReID mislabels the subject in
#      views where its time-offset estimate aliases — see the sync note).
#   3. GATE: count scenes RICHFusionDatapoint can actually use (frames>0 and persons>0).
#      This measures training-data health directly, unlike the projection-distance check
#      which false-alarms on 2-person scenes.
#   4. if the gate passes, stop the preprocessing driver (debug QOS allows only
#      1 running + 1 queued, so they compete) and launch the training driver.
#
# Anything unexpected -> log it and STOP, leaving the state for inspection.
#
#   nohup bash bash_jobs/overnight_chain.sh > logs/overnight_chain.log 2>&1 < /dev/null &

set -uo pipefail
cd /users/tnanni/ghost

GHOST=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
RICH=/capstor/scratch/cscs/tnanni/datasets/rich
CENTERED=/tmp/ct_ver
MIN_USABLE="${MIN_USABLE:-45}"      # gate: scenes training can consume
STALL_MIN="${STALL_MIN:-45}"        # minutes with no new completed scene = stalled
MAX_WAIT_MIN="${MAX_WAIT_MIN:-420}" # hard cap on the wait
POLL=300

log() { echo "$(date '+%F %T') | $*"; }
n_done() { ls "$GHOST"/*/cross_view_reid.json 2>/dev/null | wc -l; }

log "=== overnight chain start (host $(hostname)) ==="
log "gate: >= ${MIN_USABLE} usable scenes; stall = ${STALL_MIN} min without a new scene"

# ── 1. wait for preprocessing ────────────────────────────────────────────────
prev=$(n_done); last_change=$(date +%s); start=$(date +%s)
while :; do
    sleep "$POLL"
    cur=$(n_done); now=$(date +%s)
    if [ "$cur" -ne "$prev" ]; then
        log "preprocessing: $cur/62 complete"
        prev=$cur; last_change=$now
    fi
    if [ "$cur" -ge 62 ]; then log "preprocessing COMPLETE ($cur/62)"; break; fi
    if [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then
        log "preprocessing STALLED at $cur/62 for ${STALL_MIN} min — proceeding with what exists"
        break
    fi
    if [ $(( (now - start) / 60 )) -ge "$MAX_WAIT_MIN" ]; then
        log "hit MAX_WAIT_MIN=${MAX_WAIT_MIN} at $cur/62 — proceeding"
        break
    fi
done

# ── 2. GT-based id correction on any scene that finished since the last pass ──
if ! mountpoint -q "$CENTERED"; then
    mkdir -p "$CENTERED"
    squashfuse "$RICH/centered_train.sqsh" "$CENTERED" || { log "FATAL: cannot mount centered_train.sqsh"; exit 1; }
    log "mounted centered_train.sqsh -> $CENTERED"
fi

log "--- running GT-based ReID relabelling (--apply) ---"
pixi run python scripts/gt_reid_rich.py \
    --ghost_root "$GHOST" --rich_root "$RICH" --centered_root "$CENTERED" \
    --body_split train_body --apply 2>&1 | grep -viE "^ WARN|fsspec" | tail -40

# ── 3. gate: how many scenes can training actually use? ──────────────────────
log "--- counting usable scenes ---"
USABLE=$(pixi run python - <<'PY' 2>/dev/null | tail -1
import sys, logging
sys.path.insert(0, '.')
logging.disable(logging.WARNING)
from pathlib import Path
from data.fusion_dataset import RICHFusionDatapoint
R = Path('/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train')
RICH = '/capstor/scratch/cscs/tnanni/datasets/rich'
SKIP = {'Pavallion_013_plankjack', 'Pavallion_013_phonesiteat'}
ok = 0
for s in sorted(d for d in R.iterdir() if d.is_dir()):
    if not (s / 'cross_view_reid.json').exists() or s.name in SKIP:
        continue
    try:
        dp = RICHFusionDatapoint(scene_dir=s, rich_data_root=RICH, rich_gt_dir=RICH,
                                 body_split='train_body')
        if dp.num_frames > 0 and dp.max_persons > 0 and dp.has_gt:
            ok += 1
    except Exception:
        pass
print(ok)
PY
)
USABLE="${USABLE:-0}"
log "usable scenes: ${USABLE} (gate ${MIN_USABLE})"

if ! [ "$USABLE" -ge "$MIN_USABLE" ] 2>/dev/null; then
    log "GATE FAILED — not launching training. Inspect in the morning."
    exit 1
fi

# ── 4. hand the queue over to training ───────────────────────────────────────
log "--- gate passed: stopping preprocessing driver ---"
PIDS=$(ps -eo pid,cmd | grep "[b]ash bash_jobs/rich_train_preprocess_driver.sh" | awk '{print $1}')
[ -n "$PIDS" ] && kill -9 $PIDS && log "killed preprocessing driver: $PIDS"
scancel -u "$USER" -n rtprep 2>/dev/null
scancel -u "$USER" -n rtprep_dbg 2>/dev/null
sleep 10

if [ -f checkpoints/fusion_module_new/best.pt ] || [ -f checkpoints/fusion_module_new/last.pt ]; then
    log "NOTE: checkpoints/fusion_module_new already has a checkpoint — driver will --resume"
fi

log "--- launching training driver ---"
MAX_ROUNDS=1400 nohup bash bash_jobs/train_rich_v2_driver.sh >> logs/train_driver.log 2>&1 &
disown
sleep 30
log "training driver pid $(pgrep -f 'bash bash_jobs/train_rich_v2_driver.sh' | head -1)"
squeue -u "$USER" -h -o "  %.10i %.12j %.9T %.8M" | while read -r l; do log "queue: $l"; done
log "=== chain done: training is running ==="
