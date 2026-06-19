#!/bin/bash
# Login-node driver for the VGGT centered-train re-run.
#
# Keeps resubmitting the 1.5h SLURM job (ONE at a time) until all 62
# centered-train scenes have both vggt_cameras_centered.npz and
# vggt_depth_centered.npz. The driver itself is a plain login-node shell loop
# (not a SLURM job, so no 1.5h limit) that mostly sleeps.
#
# Launch detached so it survives closing VSCode / SSH / the PC:
#   nohup bash bash_jobs/vggt_train_nohup_driver.sh > logs/vggt_driver.log 2>&1 < /dev/null &
#
# Stop early:  kill the printed PID  (or: pkill -f vggt_train_nohup_driver)
set -uo pipefail
cd /users/tnanni/ghost

OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
JOB=bash_jobs/rerun_vggt_bbq_pavallion.sh   # the existing 1.5h all-scenes job
JOBNAME=vggt_rich_train_centered            # its #SBATCH --job-name
POLL=120                                    # seconds between checks

count_remaining() {
  local n=0 d
  for d in "$OUT"/*/; do
    [[ -f "$d/vggt_cameras_centered.npz" && -f "$d/vggt_depth_centered.npz" ]] || n=$((n + 1))
  done
  echo "$n"
}

echo "$(date '+%F %T') driver up (pid $$) — polling every ${POLL}s"
while :; do
  REM=$(count_remaining)
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all 62 scenes have centered VGGT outputs — driver exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$JOB")
    echo "$(date '+%F %T') ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') ${REM} scene(s) left, a job is already RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
