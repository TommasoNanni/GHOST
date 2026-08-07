#!/bin/bash
# Driver for the EgoHumans oracle ablations, GEODESIC-MEDIAN FUSION: one job
# per (activity, rung). Mirrors eval_ablations_egohumans_driver.sh (the
# v2-checkpoint driver) exactly, pointed at the median job script and a
# separate dump root so the two fusion methods' resumable dumps never mix.
#
# Stage A jobs are resumable (a scene with an existing dump is skipped), so an
# activity that does not fit the 1:30 debug limit simply gets resubmitted until
# it is complete.
#
# Launch detached, all activities and rungs:
#   nohup bash bash_jobs/eval_ablations_egohumans_median_driver.sh > logs/abl_egohumans_median_driver.log 2>&1 < /dev/null &
# One activity:
#   nohup bash bash_jobs/eval_ablations_egohumans_median_driver.sh 06_badminton > logs/abl_median_driver_badminton.log 2>&1 < /dev/null &
# Subset of rungs:
#   MODALITIES="3,4" nohup bash bash_jobs/eval_ablations_egohumans_median_driver.sh ... &
# Stop: pkill -f eval_ablations_egohumans_median_driver
#
# Env:
#   MODALITIES  space/comma separated rungs to run, default "2 3 4"
#   MAXJOBS     concurrent jobs, default 2 (debug QOS allows 1 running + 1 queued)
#   GT_ROOT     parent of the per-activity GT dirs, default egohumans_gt_full

set -uo pipefail
cd /users/tnanni/ghost

GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
GT_ROOT="${GT_ROOT:-/iopsstor/scratch/cscs/tnanni/egohumans_gt_full}"
DUMP_ROOT="${DUMP_ROOT:-eval_ablations_egohumans/dumps_median}"
SCENE_JOB="bash_jobs/eval_ablations_egohumans_median.sh"
MAXJOBS="${MAXJOBS:-2}"
POLL=180

ALL_ACTS=(01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis)
ACTS=("${1:-}"); [ -z "${ACTS[0]}" ] && ACTS=("${ALL_ACTS[@]}")
read -r -a MODS <<< "${MODALITIES:-2 3 4}"
MODS=("${MODS[@]//,/ }"); read -r -a MODS <<< "${MODS[*]}"

mkdir -p logs
log() { echo "$(date '+%F %T') $*" | tee -a logs/abl_egohumans_median_driver.log; }

# Distinct prefix from the v2 driver's "abl_" so the two can never miscount
# each other's jobs against their own MAXJOBS if both happen to run at once.
JOB_PREFIX="ablmed"
n_active()     { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^${JOB_PREFIX}_"; }
job_inflight() { squeue -u "$USER" -h -n "$1" 2>/dev/null | grep -q .; }

# Scenes that the ladder can actually score: ghost output present AND GT with
# colmap cameras (every rung needs GT camera centres — the GT scale comes from
# them) AND GT smpl (M4). A scene missing those is not "pending", it is out of
# scope, and counting it as pending would make the driver loop forever.
eligible_scenes() {   # $1=activity
    local act="$1" s name
    [ -d "${GHOST_ROOT}/${act}" ] || return 0
    for s in "${GHOST_ROOT}/${act}"/*/; do
        [ -d "$s" ] || continue
        name=$(basename "$s")
        [ -f "${s}/vggt_cameras_centered.npz" ] || continue
        [ -f "${GT_ROOT}/${act}/${name}/colmap/workplace/images.txt" ] || continue
        [ -f "${GT_ROOT}/${act}/${name}/colmap/workplace/colmap_from_aria_transforms.pkl" ] || continue
        ls "${GT_ROOT}/${act}/${name}/processed_data/smpl/"*.npy >/dev/null 2>&1 || continue
        echo "$name"
    done
}

pending() {   # $1=activity $2=modality -> number of eligible scenes without a dump
    local act="$1" m="$2" n=0 s
    while read -r s; do
        [ -z "$s" ] && continue
        [ -f "${DUMP_ROOT}/${act}/m${m}/${s}.npz" ] || n=$((n+1))
    done < <(eligible_scenes "$act")
    echo "$n"
}

if [ ! -d "$GT_ROOT" ]; then
    log "GT_ROOT does not exist: $GT_ROOT"; exit 1
fi

log "activities: ${ACTS[*]}   rungs: ${MODS[*]}   maxjobs=$MAXJOBS   fusion=geodesic-median"
for ACT in "${ACTS[@]}"; do
    n_elig=$(eligible_scenes "$ACT" | grep -c .)
    log "$ACT: $n_elig eligible scene(s)"
done

while :; do
    total_remaining=0
    for ACT in "${ACTS[@]}"; do
        for M in "${MODS[@]}"; do
            rem=$(pending "$ACT" "$M")
            total_remaining=$((total_remaining + rem))
            (( rem == 0 )) && continue

            JOB="${JOB_PREFIX}_${ACT}_m${M}"
            job_inflight "$JOB" && continue          # queued/running → never duplicate
            (( $(n_active) >= MAXJOBS )) && continue

            jid=$(sbatch --parsable --job-name="$JOB" \
                    --export=ALL,ACTIVITY="$ACT",MODALITIES="$M",GT_DIR="${GT_ROOT}/${ACT}",\
DUMP_DIR="${DUMP_ROOT}/${ACT}/m${M}" "$SCENE_JOB" 2>&1) \
                && log "$ACT m$M: submitted $jid ($rem scene(s) pending)" \
                || log "$ACT m$M: sbatch failed: $jid"
        done
    done

    if (( total_remaining == 0 )); then
        log "all activities x rungs complete"
        break
    fi
    log "$total_remaining scene-rung(s) remaining, $(n_active) job(s) active"
    sleep "$POLL"
done

echo "=== final aggregate per activity ==="
for ACT in "${ACTS[@]}"; do
    echo "--- $ACT ---"
    pixi run python evaluation/ablations_egohumans_median.py --metrics_only \
        --dump_dir "${DUMP_ROOT}/${ACT}"
done
log "Driver exiting."
