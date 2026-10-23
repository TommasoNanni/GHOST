#!/bin/bash -l
# Drive the through-sync EgoHumans evaluation to completion on the debug partition.
#
# The debug QOS allows one running job plus one queued, and caps walltime at
# 1:30, while both stages need several hours. Both stage scripts are resumable
# (STEP 1 skips a trial dir that already has sync_meta.json, STEP 2 skips a trial
# that already has a dump), so the work only needs something to keep resubmitting
# until the outputs are all there. That is this driver: it counts what is left,
# keeps the queue fed, and stops when the count reaches zero -- no manual
# relaunching, and no spare "insurance" link that wakes up only to reload model
# weights and exit.
#
# Run detached (survives ssh teardown; the login-node watchdog does not touch it,
# it only sleeps and calls sbatch):
#   setsid nohup bash scripts/drive_sync_egohumans.sh \
#       > logs/drive_sync_egohumans.log 2>&1 < /dev/null &
#
#   STAGE=build bash scripts/drive_sync_egohumans.sh   # drive STEP 1 instead
#
# Stop it: kill this script first, then scancel -- otherwise it resubmits.
set -uo pipefail
cd /users/tnanni/ghost

STAGE="${STAGE:-eval}"                       # eval | build
MANIFEST="${MANIFEST:-eval_egohumans/sync_manifest_s75.json}"
SYNC_ROOT="${SYNC_ROOT:-/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_sync_s75}"
DUMP_DIR="${DUMP_DIR:-eval_egohumans/dumps_sync_s75}"
POLL="${POLL:-120}"
MAX_SUBMITS="${MAX_SUBMITS:-30}"

case "$STAGE" in
    eval)  JOB=bash_jobs/eval_egohumans_sync.sh;  NAME=eval_egohumans_sync ;;
    build) JOB=bash_jobs/sync_build_egohumans.sh; NAME=sync_build_egohumans ;;
    *) echo "STAGE must be eval or build"; exit 2 ;;
esac

# How many outputs the stage owes us, and how many exist. Counted from the
# manifest rather than from a hardcoded number so a re-partition cannot leave the
# driver waiting for trials that no longer exist (or stopping early when new ones
# appear). sync_failure trials are excluded: STEP 0 recorded them as failures and
# neither stage produces an output for them.
counts() {
    MANIFEST="$MANIFEST" SYNC_ROOT="$SYNC_ROOT" DUMP_DIR="$DUMP_DIR" STAGE="$STAGE" \
    BASELINE="${BASELINE:-}" \
    pixi run python - <<'PY' 2>/dev/null
import json, os
from pathlib import Path
M = json.loads(Path(os.environ["MANIFEST"]).read_text())
stage, sync_root, dump_dir = os.environ["STAGE"], Path(os.environ["SYNC_ROOT"]), Path(os.environ["DUMP_DIR"])
baseline = bool(os.environ.get("BASELINE"))
want = have = 0
for key, e in M["scenes"].items():
    act, sc = key.split("/", 1)
    for t in e["trials"]:
        if t.get("sync_failure"):
            continue
        if stage == "build":
            if t["exact"]:
                continue                      # exact trials are never rebuilt
            want += 1
            have += (sync_root / act / sc / f"trial{t['trial']}" / "sync_meta.json").exists()
        else:
            # --baseline is the paired control and only scores INEXACT trials: an
            # exact trial's control IS the trial itself. Counting all 114 here would
            # leave the driver resubmitting forever against a target it cannot reach.
            if baseline and t["exact"]:
                continue
            want += 1
            have += (dump_dir / f"{act}__{sc}__trial{t['trial']}.npz").exists()
print(want, have)
PY
}

submits=0
echo "driver START $(date) stage=$STAGE host=$(hostname)"

while true; do
    read -r WANT HAVE <<< "$(counts)"
    if [ -z "${WANT:-}" ]; then
        echo "$(date +%H:%M:%S) could not count outputs (manifest unreadable?) -- retrying"
        sleep "$POLL"; continue
    fi
    if [ "$HAVE" -ge "$WANT" ]; then
        echo "$(date +%H:%M:%S) COMPLETE: $HAVE/$WANT outputs after $submits submit(s)"
        break
    fi

    RUNNING=$(squeue -h -u "$USER" -o "%j" | grep -c "$NAME")
    if [ "$RUNNING" -ge 1 ]; then
        echo "$(date +%H:%M:%S) $HAVE/$WANT done, $RUNNING job(s) in queue -- waiting"
        sleep "$POLL"; continue
    fi

    if [ "$submits" -ge "$MAX_SUBMITS" ]; then
        echo "$(date +%H:%M:%S) GIVING UP at $HAVE/$WANT after $MAX_SUBMITS submits"
        break
    fi
    submits=$((submits + 1))
    JOBID=$(sbatch --parsable "$JOB")
    echo "$(date +%H:%M:%S) $HAVE/$WANT done -- submitted $JOBID (submit #$submits)"
    sleep 30
done

echo "driver END $(date) submits=$submits"
