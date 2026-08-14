#!/bin/bash
# Login-node driver: Sapiens 2D keypoints on the TRAIN split.
#
# Phase 1 — wait until MapAnything-train is fully done (all 62 train scenes
#           have mapanything_scale_baseline.npy), so we don't fight MapAnything
#           for the debug partition.
# Phase 2 — resubmit the 1.5h Sapiens train job (ONE at a time) until every
#           train scene is fully done, then exit.
#
# "Fully done" for a scene = every person_*.npz in every cam's body_data/ has
# a matching sapiens_centered_kps_person_<pid>.npz (mirrors _scene_fully_done
# in scripts/run_sapiens_all_scenes.py).
#
# Not a SLURM job (no 1.5h limit); mostly sleeps. Launch detached:
#   nohup bash bash_jobs/sapiens_train_driver.sh > logs/sapiens_train_driver.log 2>&1 < /dev/null &
# Stop early: kill the printed PID (or: pkill -f sapiens_train_driver)
set -uo pipefail
cd /users/tnanni/ghost

OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
SAP_JOB=bash_jobs/run_sapiens_rich_train.sh
SAP_JOBNAME=sapiens_rich_train
POLL=120

map_ready() {
  local miss=0 d
  for d in "$OUT"/*/; do
    [[ -f "$d/mapanything_scale_baseline.npy" ]] || miss=$((miss + 1))
  done
  (( miss == 0 ))
}

sap_remaining() {
  python3 - "$OUT" <<'PYEOF'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
n = 0
for scene_dir in sorted(root.iterdir()):
    if not scene_dir.is_dir():
        continue
    cam_dirs = [d for d in scene_dir.iterdir()
                if d.is_dir() and (d / "body_data").is_dir()]
    if not cam_dirs:
        continue
    done = True
    for cam_dir in cam_dirs:
        pids = [int(p.stem.split("_")[1])
                for p in (cam_dir / "body_data").glob("person_*.npz")]
        if not pids:
            continue
        if not all(
            (cam_dir / f"sapiens_centered_kps_person_{pid}.npz").exists()
            for pid in pids
        ):
            done = False
            break
    if not done:
        n += 1
print(n)
PYEOF
}

echo "$(date '+%F %T') sapiens-train driver up (pid $$) — waiting for MapAnything train"

# ── Phase 1: wait for MapAnything-train ─────────────────────────────────────
while :; do
  map_ready && { echo "$(date '+%F %T') MapAnything train ready — starting Sapiens loop"; break; }
  echo "$(date '+%F %T') waiting on MapAnything train"
  sleep "$POLL"
done

# ── Phase 2: Sapiens train resubmit loop ────────────────────────────────────
while :; do
  REM=$(sap_remaining)
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all train scenes have Sapiens keypoints — driver exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$SAP_JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$SAP_JOB" 2>&1) || { echo "$(date '+%F %T') sbatch failed: $JID — will retry"; sleep "$POLL"; continue; }
    echo "$(date '+%F %T') sapiens-train: ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') sapiens-train: ${REM} scene(s) left, a job is RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
