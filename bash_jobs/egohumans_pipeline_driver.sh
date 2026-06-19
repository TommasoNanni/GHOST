#!/bin/bash
# Login-node driver for egohumans_pipeline.
#
# Resubmits bash_jobs/egohumans_pipeline.sh (ONE job at a time) until all
# scenes for the given activity have been fully processed.
#
# Launch detached:
#   nohup bash bash_jobs/egohumans_pipeline_driver.sh 02_lego > logs/egohumans_driver_02_lego.log 2>&1 < /dev/null &
# To process all activities, omit the argument.
# Stop early: pkill -f egohumans_pipeline_driver
set -uo pipefail
cd /users/tnanni/ghost

ACTIVITY="${1:-}"   # optional: e.g. "02_lego"
OUT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans"
JOB="bash_jobs/egohumans_pipeline.sh"
JOBNAME="egohumans_pipeline"
POLL=120

count_remaining() {
  python3 - "$OUT" "$ACTIVITY" <<'PYEOF'
import pathlib, sys, subprocess, tempfile, os

SQSH_ROOT  = pathlib.Path("/capstor/scratch/cscs/tnanni/datasets")
OUT_ROOT   = pathlib.Path(sys.argv[1])
FILTER_ACT = sys.argv[2]
INNER      = pathlib.Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")
ALL_ACTS   = ["01_tagging","02_lego","03_fencing","04_basketball",
              "05_volleyball","06_badminton","07_tennis"]
ACTIVITIES = [FILTER_ACT] if FILTER_ACT else ALL_ACTS

remaining = 0
for activity in ACTIVITIES:
    sqsh = SQSH_ROOT / f"egohumans_{activity}.sqsh"
    if not sqsh.exists():
        continue
    # Mount squash temporarily to count scenes
    mnt = pathlib.Path(tempfile.mkdtemp())
    try:
        r = subprocess.run(["squashfuse", str(sqsh), str(mnt)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            continue
        cam_ready = mnt / INNER / activity
        if not cam_ready.is_dir():
            continue
        for seq_dir in sorted(cam_ready.iterdir()):
            if not seq_dir.is_dir():
                continue
            out_seq = OUT_ROOT / activity / seq_dir.name
            done = (
                (out_seq / "cross_view_reid.json").exists()
                and (out_seq / "vggt_cameras_centered.npz").exists()
                and (out_seq / "mapanything_scale_centered.npy").exists()
            )
            if not done:
                remaining += 1
    finally:
        subprocess.run(["fusermount", "-u", str(mnt)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mnt.rmdir()
print(remaining)
PYEOF
}

mkdir -p logs
LABEL="${ACTIVITY:-all}"
echo "$(date '+%F %T') egohumans driver up (pid $$, activity=${LABEL}) — polling every ${POLL}s"

while :; do
  REM=$(count_remaining)
  if (( REM == 0 )); then
    echo "$(date '+%F %T') all scenes done for activity=${LABEL} — driver exiting"
    break
  fi
  ACTIVE=$(squeue -u "$USER" -h -n "$JOBNAME" 2>/dev/null | wc -l)
  if (( ACTIVE == 0 )); then
    JID=$(ACTIVITY="$ACTIVITY" sbatch --parsable --export=ALL "$JOB" 2>&1) || {
      echo "$(date '+%F %T') sbatch failed: $JID — will retry in ${POLL}s"
      sleep "$POLL"
      continue
    }
    echo "$(date '+%F %T') ${REM} scene(s) remaining — submitted job $JID"
  else
    echo "$(date '+%F %T') ${REM} scene(s) remaining — job already RUNNING/PENDING, waiting"
  fi
  sleep "$POLL"
done
