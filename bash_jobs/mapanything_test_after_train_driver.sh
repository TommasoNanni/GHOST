#!/bin/bash
# Login-node supervisor: MapAnything scale on the TEST split, AFTER MapAnything
# on the TRAIN split is finished (sequencing to avoid GPU/partition contention).
# Test VGGT is already complete, so the only gate is MapAnything-train.
#
# Phase 1 — wait until every TRAIN scene has mapanything_scale_centered.npy.
# Phase 2 — resubmit the 1.5h MapAnything test job (ONE at a time) until every
#           TEST scene has mapanything_scale_centered.npy, then exit.
#
# Not a SLURM job (no 1.5h limit); mostly sleeps. Launch detached:
#   nohup bash bash_jobs/mapanything_test_after_train_driver.sh > logs/map_test_driver.log 2>&1 < /dev/null &
# Stop early: kill the printed PID (or: pkill -f mapanything_test_after_train_driver)
set -uo pipefail
cd /users/tnanni/ghost

TRAIN_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
TEST_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
MAP_JOB=bash_jobs/run_mapanything_rich_test.sh    # ghost_output_root=rich_test, rich_root=centered_test
MAP_JOBNAME=run_mapanything_test                  # its #SBATCH --job-name
POLL=120

remaining() {   # $1 = output root → scene dirs lacking the MapAnything scale file
  local n=0 d
  for d in "$1"/*/; do
    [[ -f "$d/mapanything_scale_centered.npy" ]] || n=$((n + 1))
  done
  echo "$n"
}

echo "$(date '+%F %T') mapanything-test supervisor up (pid $$)"

# ── Phase 1: wait for MapAnything-train ─────────────────────────────────────
while :; do
  TR=$(remaining "$TRAIN_OUT")
  if (( TR == 0 )); then
    echo "$(date '+%F %T') MapAnything train complete — starting test loop"
    break
  fi
  echo "$(date '+%F %T') waiting on MapAnything train ($TR scene(s) left)"
  sleep "$POLL"
done

# ── Phase 2: MapAnything test resubmit loop ─────────────────────────────────
while :; do
  REM=$(remaining "$TEST_OUT")
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all test scenes have MapAnything scale — supervisor exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$MAP_JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$MAP_JOB")
    echo "$(date '+%F %T') map-test: ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') map-test: ${REM} scene(s) left, a job is RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
