#!/bin/bash
# Login-node driver: MapAnything scale on the TRAIN split.
#
# Phase 1 — wait until VGGT-train is fully done (all scene dirs have both
#           vggt_cameras_centered.npz + vggt_depth_centered.npz AND no VGGT
#           job is still queued/running), so MapAnything reads correct VGGT
#           outputs and doesn't fight VGGT for the debug partition.
# Phase 2 — resubmit the 1.5h MapAnything job (ONE at a time) until every
#           train scene has mapanything_scale_baseline.npy, then exit.
#
# Not a SLURM job (no 1.5h limit); mostly sleeps. Launch detached:
#   nohup bash bash_jobs/mapanything_train_driver.sh > logs/map_train_driver.log 2>&1 < /dev/null &
# Stop early: kill the printed PID (or: pkill -f mapanything_train_driver)
set -uo pipefail
cd /users/tnanni/ghost

OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
MAP_JOB=bash_jobs/run_mapanything_rich_train.sh   # ghost_output_root=rich_train, rich_root=centered_train
MAP_JOBNAME=run_mapanything                       # its #SBATCH --job-name
VGGT_JOBNAME=vggt_rich_train_centered             # VGGT train job name (for the gate)
POLL=120

vggt_ready() {   # true once all VGGT outputs exist and no VGGT job is active
  local miss=0 d
  for d in "$OUT"/*/; do
    [[ -f "$d/vggt_cameras_centered.npz" && -f "$d/vggt_depth_centered.npz" ]] || miss=$((miss + 1))
  done
  local active; active=$(squeue -u "$USER" -h -n "$VGGT_JOBNAME" 2>/dev/null | wc -l)
  (( miss == 0 && active == 0 ))
}

map_remaining() {   # scene dirs lacking the MapAnything scale file
  local n=0 d
  for d in "$OUT"/*/; do
    [[ -f "$d/mapanything_scale_baseline.npy" ]] || n=$((n + 1))
  done
  echo "$n"
}

echo "$(date '+%F %T') mapanything-train driver up (pid $$) — waiting for VGGT train"
while :; do
  vggt_ready && { echo "$(date '+%F %T') VGGT train ready — starting MapAnything loop"; break; }
  sleep "$POLL"
done

while :; do
  REM=$(map_remaining)
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all train scenes have MapAnything scale — driver exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$MAP_JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$MAP_JOB")
    echo "$(date '+%F %T') map-train: ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') map-train: ${REM} scene(s) left, a job is RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
