#!/bin/bash
# Per-scene driver for the EgoHumans balance=0 re-run (Stage A).
# Generalises egohumans_badminton_driver.sh: enumerates scenes from the activity
# sqsh, submits one egohumans_new_scene.sh per scene (capped parallelism),
# resubmits incomplete scenes, exits when the activity is fully processed.
#
# Launch detached, one activity:
#   nohup bash bash_jobs/egohumans_new_driver.sh 03_fencing > logs/egonew_driver_03_fencing.log 2>&1 < /dev/null &
# All activities (sequential over activities, parallel within):
#   nohup bash bash_jobs/egohumans_new_driver.sh > logs/egonew_driver_all.log 2>&1 < /dev/null &
# Stop: pkill -f egohumans_new_driver

set -uo pipefail
cd /users/tnanni/ghost

SQSH_ROOT="/capstor/scratch/cscs/tnanni/datasets"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
OUT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
SCENE_JOB="bash_jobs/egohumans_new_scene.sh"
# Activities already extracted (unsquashed) in /iopsstor — enumerated from disk, no mount.
declare -A EXTRACTED_ROOTS=(
    [06_badminton]="/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans/06_badminton"
)
MAXJOBS="${MAXJOBS:-6}"     # concurrent scene jobs
POLL=120

ALL_ACTS=(01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis)
ACTS=("${1:-}"); [ -z "${ACTS[0]}" ] && ACTS=("${ALL_ACTS[@]}")

mkdir -p logs "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a logs/egonew_driver.log; }

# List scene names for an activity by mounting its sqsh (plain or _a/_b) read-only.
list_scenes() {
    local act="$1"
    local ext="${EXTRACTED_ROOTS[$act]:-}"
    if [ -n "$ext" ] && [ -d "$ext/$INNER/$act" ]; then
        for d in "$ext/$INNER/$act"/*/; do [ -d "$d" ] && basename "$d"; done
        return
    fi
    python3 - "$act" <<'PYEOF'
import os, sys, subprocess, tempfile, pathlib
act=sys.argv[1]
SQSH="/capstor/scratch/cscs/tnanni/datasets"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
cands=[f"egohumans_{act}.sqsh", f"egohumans_{act}_a.sqsh", f"egohumans_{act}_b.sqsh"]
scenes=set()
for c in cands:
    p=os.path.join(SQSH,c)
    if not os.path.exists(p): continue
    mnt=tempfile.mkdtemp()
    try:
        if subprocess.run(["squashfuse",p,mnt],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0: continue
        cr=pathlib.Path(mnt)/INNER/act
        if cr.is_dir():
            scenes.update(s.name for s in cr.iterdir() if s.is_dir())
    finally:
        subprocess.run(["fusermount","-u",mnt],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        os.rmdir(mnt)
print("\n".join(sorted(scenes)))
PYEOF
}

scene_done() {  # $1=act $2=scene
    local d="${OUT}/$1/$2"
    [ -f "${d}/cross_view_reid.json" ] && \
    [ -f "${d}/vggt_cameras_centered.npz" ] && \
    [ -f "${d}/mapanything_scale_baseline.npy" ]
}

JOB_PREFIX="egn"
# count this driver's in-flight jobs (any egn_* name)
n_active()       { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^${JOB_PREFIX}_"; }
# authoritative: is a job for THIS scene already queued/running? (race-free, survives restarts)
scene_inflight() { squeue -u "$USER" -h -n "${JOB_PREFIX}_$1" 2>/dev/null | grep -q .; }

for ACT in "${ACTS[@]}"; do
    log "=== activity $ACT ==="
    mapfile -t SCENES < <(list_scenes "$ACT")
    if [ "${#SCENES[@]}" -eq 0 ]; then log "$ACT: no scenes found (sqsh missing?) — skip"; continue; fi
    log "$ACT: ${#SCENES[@]} scenes"

    while :; do
        remaining=0
        for s in "${SCENES[@]}"; do scene_done "$ACT" "$s" || remaining=$((remaining+1)); done
        if (( remaining == 0 )); then log "$ACT: all scenes done"; break; fi

        for s in "${SCENES[@]}"; do
            scene_done "$ACT" "$s" && continue
            scene_inflight "$s" && continue   # already queued/running → never duplicate
            (( $(n_active) >= MAXJOBS )) && break
            jid=$(sbatch --parsable --job-name="${JOB_PREFIX}_$s" \
                    --export=ALL,ACT="$ACT",SCENE="$s" "$SCENE_JOB" 2>&1) \
                && log "$ACT/$s: submitted $jid" \
                || log "$ACT/$s: sbatch failed: $jid"
        done
        log "$ACT: $remaining remaining, $(n_active) active"
        sleep "$POLL"
    done
done

log "Driver exiting — all requested activities done."
