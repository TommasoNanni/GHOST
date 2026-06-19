#!/bin/bash
# Login-node supervisor: Sapiens 2D keypoints on the TEST split, AFTER
# Sapiens on the TRAIN split is finished (to avoid GPU/partition contention).
#
# Phase 1 — wait until every TRAIN scene is fully done (all sapiens outputs).
# Phase 2 — resubmit the 1.5h Sapiens test job (ONE at a time) until every
#           TEST scene is fully done, then exit.
#
# Not a SLURM job (no 1.5h limit); mostly sleeps. Launch detached:
#   nohup bash bash_jobs/sapiens_test_after_train_driver.sh > logs/sapiens_test_driver.log 2>&1 < /dev/null &
# Stop early: kill the printed PID (or: pkill -f sapiens_test_after_train_driver)
set -uo pipefail
cd /users/tnanni/ghost

TRAIN_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
TEST_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
SAP_JOB=bash_jobs/run_sapiens_rich_test.sh
SAP_JOBNAME=sapiens_rich_test
POLL=120

sap_remaining() {
  python3 - "$1" <<'PYEOF'
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

echo "$(date '+%F %T') sapiens-test supervisor up (pid $$)"

# ── Phase 1: wait for Sapiens-train ─────────────────────────────────────────
while :; do
  TR=$(sap_remaining "$TRAIN_OUT")
  if (( TR == 0 )); then
    echo "$(date '+%F %T') Sapiens train complete — starting test loop"
    break
  fi
  echo "$(date '+%F %T') waiting on Sapiens train ($TR scene(s) left)"
  sleep "$POLL"
done

# ── Phase 2: Sapiens test resubmit loop ─────────────────────────────────────
while :; do
  REM=$(sap_remaining "$TEST_OUT")
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all test scenes have Sapiens keypoints — supervisor exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$SAP_JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(sbatch --parsable "$SAP_JOB" 2>&1) || { echo "$(date '+%F %T') sbatch failed: $JID — will retry"; sleep "$POLL"; continue; }
    echo "$(date '+%F %T') sapiens-test: ${REM} scene(s) left, no active job — submitted $JID"
  else
    echo "$(date '+%F %T') sapiens-test: ${REM} scene(s) left, a job is RUNNING/PENDING — waiting"
  fi
  sleep "$POLL"
done
