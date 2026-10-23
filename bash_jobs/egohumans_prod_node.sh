#!/bin/bash -l
#SBATCH --job-name=egohumans_prod_node
#SBATCH --account=a0287
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=800G
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# EgoHumans production preprocessing, 4 scenes in parallel per node (one per GPU).
#
# Each of the 4 workers is pinned to one GPU (CUDA_VISIBLE_DEVICES) and pulls the next
# unfinished scene from the shared list of every scene in the downloaded raw tree.
# Per scene it runs exactly the two steps of egohumans_new_scene.sh:
#   1) prep_undistort_egohumans.py  (balance=0 undistort + resize to 1440 -> TEMP)
#   2) egohumans_pipeline_production.py (SAM3 -> SAM3D -> GT ReID -> VGGT -> MapAnything)
#
# A scene is done when gt_reid.json, vggt_cameras_centered.npz and
# mapanything_scale_baseline.npy exist (same test as egohumans_new_driver.sh).
# Claims are atomic mkdir locks under $OUT/.claims, wiped at job start, so a scene
# left half-done by a killed job is redone by the next one. Resubmit until all done
# (bash_jobs/egohumans_prod_driver.sh does that, one node at a time).
set -uo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The env carries a hand-installed CUDA torch; never let pixi resync it to the lockfile.
export PIXI_FROZEN=true

RAW="/iopsstor/scratch/cscs/tnanni/egohumans_raw"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
TEMP="/iopsstor/scratch/cscs/tnanni/temp_egohumans"
OUT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_prod"
CLAIMS="$OUT/.claims"
SLOG="logs/egohumans_prod"
N_GPUS=4

mkdir -p "$TEMP" "$OUT" "$SLOG"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"     # one node-job at a time: old claims are stale

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | start $(date)"
nvidia-smi -L

SCENES=()
for d in "$RAW/$INNER"/*/*/; do
    act=$(basename "$(dirname "$d")"); seq=$(basename "$d")
    SCENES+=("$act/$seq")
done
echo "${#SCENES[@]} scenes in $RAW"

scene_done() {
    local d="$OUT/$1"
    [ -f "$d/gt_reid.json" ] && [ -f "$d/vggt_cameras_centered.npz" ] \
        && [ -f "$d/mapanything_scale_baseline.npy" ]
}

worker() {
    local gpu=$1
    for s in "${SCENES[@]}"; do
        scene_done "$s" && continue
        mkdir "$CLAIMS/${s//\//__}" 2>/dev/null || continue     # claimed by another worker
        # Separate statements: bash expands every word of one `local` before assigning,
        # so log= on the same line would see the global act/seq (the last scene).
        local act=${s%%/*} seq=${s##*/}
        local log="$SLOG/${act}__${seq}.log"
        echo "$(date '+%T') gpu$gpu START $s"
        {
            echo "=== job $SLURM_JOB_ID gpu $gpu $(date) ==="
            CUDA_VISIBLE_DEVICES=$gpu pixi run python scripts/prep_undistort_egohumans.py \
                --raw-root "$RAW" --activity "$act" --seq "$seq" \
                --out-root "$TEMP" --max-side 1440 \
            && CUDA_VISIBLE_DEVICES=$gpu pixi run python scripts/egohumans_pipeline_production.py \
                --data-root  "$TEMP" \
                --gt-root    "$RAW" \
                --output-dir "$OUT" \
                --activity   "$act" \
                --seq        "$seq" \
                --max-side   1440
        } >> "$log" 2>&1
        if scene_done "$s"; then
            echo "$(date '+%T') gpu$gpu DONE  $s"
        else
            echo "$(date '+%T') gpu$gpu FAIL  $s  (see $log)"
        fi
    done
    echo "$(date '+%T') gpu$gpu no scenes left"
}

for g in $(seq 0 $((N_GPUS - 1))); do
    worker "$g" &
    sleep 20      # stagger model loading
done
wait

n_done=0
for s in "${SCENES[@]}"; do scene_done "$s" && n_done=$((n_done + 1)); done
echo "end $(date) | scenes done: $n_done / ${#SCENES[@]}"
