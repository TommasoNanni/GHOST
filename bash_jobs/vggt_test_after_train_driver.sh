#!/bin/bash
# Login-node supervisor: run the VGGT centered re-run on the TEST split, but
# only AFTER the TRAIN split is fully done.
#
# Phase 1 — wait until all train scenes have centered VGGT outputs (polls the
#           train output dir; coordinates with the train driver via the
#           filesystem, no signaling needed).
# Phase 2 — resubmit the 1.5h test job (ONE at a time) until every test scene
#           has centered outputs, then exit.
#
# Not a SLURM job (no 1.5h limit); mostly sleeps. Launch detached so it
# survives closing VSCode / SSH / the PC:
#   nohup bash bash_jobs/vggt_test_after_train_driver.sh > logs/vggt_test_driver.log 2>&1 < /dev/null &
#
# Stop early:  kill the printed PID  (or: pkill -f vggt_test_after_train_driver)
set -uo pipefail
cd /users/tnanni/ghost

TRAIN_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
TEST_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
TEST_JOB=bash_jobs/run_vggt_rich_test.sh     # points at centered_test + rich_test
TEST_JOBNAME=vggt_rich_test_centered         # its #SBATCH --job-name
POLL=120

count_remaining() {   # $1 = output root → number of scene dirs missing centered outputs
  local n=0 d
  for d in "$1"/*/; do
    [[ -f "$d/vggt_cameras_centered.npz" && -f "$d/vggt_depth_centered.npz" ]] || n=$((n + 1))
  done
  echo "$n"
}

echo "$(date '+%F %T') test-supervisor up (pid $$)"

# ── Phase 1: wait for TRAIN to finish ───────────────────────────────────────
while :; do
  TR=$(count_remaining "$TRAIN_OUT")
  if (( TR == 0 )); then
    echo "$(date '+%F %T') train split complete — starting test loop"
    break
  fi
  echo "$(date '+%F %T') waiting on train ($TR scene(s) left)"
  sleep "$POLL"
done

# ── Phase 2: resubmit test job until the test split is done ──────────────────
while :; do
  REM=$(count_remaining "$TEST_OUT")
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all test scenes have centered VGGT outputs — supervisor exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$TEST_JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$TEST_JOB")
    echo "$(date '+%F %T') test: ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') test: ${REM} scene(s) left, a job is RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
