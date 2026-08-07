#!/bin/bash
# EgoHumans eval with GEODESIC-MEDIAN fusion (no checkpoint, no joint_conf).
# Mirrors bash_jobs/eval_egohumans_jconf_driver.sh: submit, wait, resubmit what is
# unfinished (Stage A is resumable — a scene with a dump is skipped).
#
# Writes to a SEPARATE dump dir so the published module dumps (dumps_smpl24, 132
# scenes) survive untouched, giving a clean A/B on identical scenes.
#
# ZERO-SHOT: the estimator was selected on RICH's 12-scene train pool and applied
# here with no per-dataset tuning.
#
#   nohup bash bash_jobs/eval_egohumans_median_driver.sh > logs/egohumans_median_driver.log 2>&1 < /dev/null &
#   stop: pkill -f eval_egohumans_median_driver

set -uo pipefail
cd /users/tnanni/ghost

GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
GT_ROOT="${GT_ROOT:-/iopsstor/scratch/cscs/tnanni/egohumans_gt_full}"
DUMP_ROOT="${DUMP_ROOT:-eval_egohumans/dumps_smpl24_median}"
SCENE_JOB="bash_jobs/eval_egohumans_median.sh"
MAXJOBS="${MAXJOBS:-2}"     # debug QOS: 1 running + 1 queued
POLL=180
# 132 scenes at ~3 min each, serialised by the debug QOS (1 running + 1 queued),
# is ~6 h of wall clock = ~120 rounds at POLL=180. The jconf driver's 40 was sized
# for a partial re-run and would give up with most of EgoHumans undone.
MAX_ROUNDS="${MAX_ROUNDS:-200}"

ALL_ACTS=(01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis)
ACTS=("${1:-}"); [ -z "${ACTS[0]}" ] && ACTS=("${ALL_ACTS[@]}")

mkdir -p logs
log() { echo "$(date '+%F %T') $*"; }

n_active()     { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^md_"; }
job_inflight() { squeue -u "$USER" -h -n "$1" 2>/dev/null | grep -q .; }

# Same eligibility contract as the ablations driver: ghost output present AND GT
# with colmap cameras AND GT smpl. Scenes failing this can never produce a dump,
# so counting them as pending would loop forever.
eligible_scenes() {
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

pending() {
    local act="$1" n=0 s
    while read -r s; do
        [ -z "$s" ] && continue
        [ -f "${DUMP_ROOT}/${act}/${s}.npz" ] || n=$((n+1))
    done < <(eligible_scenes "$act")
    echo "$n"
}

for ACT in "${ACTS[@]}"; do
    log "$ACT: $(eligible_scenes "$ACT" | grep -c .) eligible scene(s)"
done

round=0
while :; do
    round=$((round + 1))
    (( round > MAX_ROUNDS )) && { log "hit MAX_ROUNDS=$MAX_ROUNDS, giving up"; break; }

    total=0
    for ACT in "${ACTS[@]}"; do
        rem=$(pending "$ACT")
        total=$((total + rem))
        (( rem == 0 )) && continue

        JOB="md_${ACT}"
        job_inflight "$JOB" && continue
        (( $(n_active) >= MAXJOBS )) && continue

        jid=$(sbatch --parsable --job-name="$JOB" \
                --export=ALL,ACTIVITY="$ACT",SCALE=baseline,\
GT_DIR="${GT_ROOT}/${ACT}",DUMP_DIR="${DUMP_ROOT}/${ACT}" "$SCENE_JOB" 2>&1) \
            && log "$ACT: submitted $jid ($rem scene(s) pending)" \
            || log "$ACT: sbatch failed: $jid"
    done

    (( total == 0 )) && { log "all activities complete"; break; }
    log "round $round: $total scene(s) remaining, $(n_active) job(s) active"
    sleep "$POLL"
done

echo "=== final aggregate per activity (GEODESIC MEDIAN) ==="
for ACT in "${ACTS[@]}"; do
    echo "--- $ACT ---"
    pixi run python evaluation/evaluate_egohumans_median.py --metrics_only \
        --dump_dir "${DUMP_ROOT}/${ACT}"
done
log "Driver exiting."
