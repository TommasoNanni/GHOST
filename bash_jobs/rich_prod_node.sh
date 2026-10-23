#!/bin/bash -l
#SBATCH --job-name=rich_prod_node
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

# RICH test production preprocessing on one node, 4 scenes in parallel (one per GPU).
#
#   0) scenes = every scene in test_raw whose cameras all carry the same frame count
#      (+-2) — skips the scene a truncated download cut in half
#   1) resize test_raw -> test_840           (scripts/resize_rich_frames.py, max side 840)
#   2) PP-centre test_840 -> centered_test   (utilities/center_images.py; plain dir, so its
#                                             squashfuse mount check is bypassed)
#   3) 4 GPU workers pull scenes and run scripts/rich_pipeline_production.py:
#      SAM3 -> GT ReID -> SAM3D (GT people only) -> VGGT + MapAnything on centered frames
# A scene is done when gt_reid.json, vggt_cameras_centered.npz and
# mapanything_scale_baseline.npy exist. Resubmit to resume.
set -uo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIXI_FROZEN=true          # never resync the hand-installed CUDA torch

R=/iopsstor/scratch/cscs/tnanni/datasets/rich
RAW=$R/test_raw
D840=$R/test_840
CENT=$R/centered_test
OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
CLAIMS="$OUT/.claims"
SLOG=logs/rich_prod
N_GPUS=4
mkdir -p "$D840" "$CENT" "$OUT" "$SLOG"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | start $(date)"
nvidia-smi -L

# --- 0) complete scenes only ---------------------------------------------------------
SCENES=()
for sd in "$RAW"/*/; do
    s=$(basename "$sd")
    counts=$(for c in "$sd"cam_*/; do ls "$c" | wc -l; done | sort -n)
    lo=$(echo "$counts" | head -1); hi=$(echo "$counts" | tail -1)
    if (( hi - lo > 2 )); then echo "SKIP $s: incomplete cameras ($lo..$hi frames)"; continue; fi
    [ -d "$R/test_body/$s" ] || { echo "SKIP $s: no GT"; continue; }
    SCENES+=("$s")
done
echo "${#SCENES[@]} complete scenes"

# --- 1) resize to 840 ------------------------------------------------------------------
for s in "${SCENES[@]}"; do
    pixi run python scripts/resize_rich_frames.py --data_root "$RAW" --output_dir "$D840" \
        --max_side 840 --workers 64 --skip_existing --scene "$s" > /dev/null 2>> "$SLOG/resize.err" \
        || echo "resize FAILED for $s (see $SLOG/resize.err)"
done
echo "$(date '+%T') resize done"

# --- 2) PP-centre (writes <scene>/crop_meta.json) ---------------------------------------
pixi run python - "$D840" "$R/scan_calibration" "$CENT" "${SCENES[@]}" <<'EOF'
import sys
import utilities.center_images as C
C.ensure_mounted = lambda *a: True          # input is a plain directory, not a sqsh mount
d840, calib, cent, *scenes = sys.argv[1:]
sys.argv = ["center_images.py", "--mount", d840, "--calib_root", calib,
            "--output", cent, "--workers", "64", "--scenes", *scenes]
C.main()
EOF
echo "$(date '+%T') centering done: $(ls "$CENT"/*/crop_meta.json 2>/dev/null | wc -l) crop_meta.json"

# Build the frames/ symlink dirs once, before 4 workers race to create them.
pixi run python -c "from data.video_dataset import RichDataset; RichDataset(data_root='$D840', max_side=None, frames_base_dir=None)" > /dev/null

# --- 3) pipeline, 4 scenes in parallel -------------------------------------------------
scene_done() {
    local d="$OUT/$1"
    [ -f "$d/gt_reid.json" ] && [ -f "$d/vggt_cameras_centered.npz" ] \
        && [ -f "$d/mapanything_scale_baseline.npy" ]
}
worker() {
    local gpu=$1
    for s in "${SCENES[@]}"; do
        scene_done "$s" && continue
        mkdir "$CLAIMS/$s" 2>/dev/null || continue
        echo "$(date '+%T') gpu$gpu START $s"
        CUDA_VISIBLE_DEVICES=$gpu pixi run python scripts/rich_pipeline_production.py \
            --data-root     "$D840" \
            --gt-root       "$R" \
            --raw-root      "$RAW" \
            --centered-root "$CENT" \
            --output-dir    "$OUT" \
            --scene         "$s" >> "$SLOG/$s.log" 2>&1
        if scene_done "$s"; then echo "$(date '+%T') gpu$gpu DONE  $s"
        else echo "$(date '+%T') gpu$gpu FAIL  $s  (see $SLOG/$s.log)"; fi
    done
}
for g in $(seq 0 $((N_GPUS - 1))); do worker "$g" & sleep 20; done
wait

n=0; for s in "${SCENES[@]}"; do scene_done "$s" && n=$((n + 1)); done
echo "end $(date) | scenes done: $n / ${#SCENES[@]}"
