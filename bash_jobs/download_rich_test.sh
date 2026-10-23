#!/bin/bash -l
#SBATCH --job-name=rich_dl_test
#SBATCH --account=a0287
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=normal

# Download all 52 RICH test scenes, convert BMP→JPEG, resize to 840px max side
# (the resolution used for all published RICH test numbers). Plain directories
# on iopsstor, no squash. Resumable: resubmit if it hits the time limit.
#
# Submit with credentials:
#   sbatch --export=ALL,RICH_USER=youruser,RICH_PASS=yourpass bash_jobs/download_rich_test.sh

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "Start:   $(date)"
echo ""

# ── Credentials ───────────────────────────────────────────────────────────────
USERNAME="${RICH_USER:-}"
PASSWORD="${RICH_PASS:-}"
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "ERROR: RICH_USER and RICH_PASS must be set."
    echo "Submit with: sbatch --export=ALL,RICH_USER=<user>,RICH_PASS=<pass> $0"
    exit 1
fi

urle() {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}
USERNAME=$(urle "$USERNAME")
PASSWORD=$(urle "$PASSWORD")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Raw extracted + BMP→JPEG converted scenes (original resolution)
STAGING=/iopsstor/scratch/cscs/tnanni/datasets/rich/test_raw
# Resized to 840px max side — input to center_images.py / the pipeline
STAGING_RESIZED=/iopsstor/scratch/cscs/tnanni/datasets/rich/test_840

mkdir -p "$STAGING" "$STAGING_RESIZED"

# ── All 52 test scenes ────────────────────────────────────────────────────────
SCENES=(
    "Gym_010_cooking1"
    "Gym_010_dips1"
    "Gym_010_dips2"
    "Gym_010_lunge1"
    "Gym_010_lunge2"
    "Gym_010_mountainclimber1"
    "Gym_010_mountainclimber2"
    "Gym_010_pushup1"
    "Gym_010_pushup2"
    "Gym_011_burpee2"
    "Gym_011_cooking1"
    "Gym_011_cooking2"
    "Gym_011_dips1"
    "Gym_011_dips2"
    "Gym_011_dips3"
    "Gym_011_dips4"
    "Gym_011_pushup1"
    "Gym_011_pushup2"
    "Gym_012_cooking2"
    "Gym_012_lunge1"
    "Gym_012_lunge2"
    "Gym_012_pushup2"
    "Gym_013_burpee4"
    "Gym_013_dips1"
    "Gym_013_dips2"
    "Gym_013_dips3"
    "Gym_013_lunge1"
    "Gym_013_lunge2"
    "Gym_013_pushup1"
    "Gym_013_pushup2"
    "LectureHall_009_021_reparingprojector1"
    "LectureHall_009_sidebalancerun1"
    "LectureHall_010_plankjack1"
    "LectureHall_010_sidebalancerun1"
    "LectureHall_019_wipingchairs1"
    "LectureHall_021_plankjack1"
    "LectureHall_021_sidebalancerun1"
    "ParkingLot2_009_burpeejump1"
    "ParkingLot2_009_burpeejump2"
    "ParkingLot2_009_impro1"
    "ParkingLot2_009_impro2"
    "ParkingLot2_009_impro5"
    "ParkingLot2_009_overfence1"
    "ParkingLot2_009_overfence2"
    "ParkingLot2_009_spray1"
    "ParkingLot2_017_burpeejump1"
    "ParkingLot2_017_burpeejump2"
    "ParkingLot2_017_eating1"
    "ParkingLot2_017_overfence1"
    "ParkingLot2_017_overfence2"
    "ParkingLot2_017_pushup1"
    "ParkingLot2_017_pushup2"
)

# ── Step 1: Download + convert + resize ───────────────────────────────────────
for scene_name in "${SCENES[@]}"; do
    dest_dir="$STAGING/$scene_name"
    # Marker written only after a full extract, so a job killed mid-extract
    # re-downloads the scene instead of keeping a partial directory.
    if [ -f "$dest_dir/.extracted" ]; then
        echo "Skipping download for $scene_name (already staged)"
    else
        tarfile="${scene_name}.tar.gz"
        echo ""
        echo "══════════════════════════════════════════════"
        echo "  Scene: $scene_name"
        echo "══════════════════════════════════════════════"
        rm -rf "$dest_dir"

        echo "[$(date '+%H:%M:%S')] Downloading ..."
        curl -L \
            --data "username=${USERNAME}&password=${PASSWORD}" \
            "https://download.is.tue.mpg.de/download.php?domain=rich&sfile=test/${tarfile}&resume=1" \
            -o "$STAGING/$tarfile" \
            --insecure

        echo "[$(date '+%H:%M:%S')] Extracting ..."
        tar -xzf "$STAGING/$tarfile" -C "$STAGING" --strip-components=5
        rm "$STAGING/$tarfile"
        touch "$dest_dir/.extracted"
    fi

    # Idempotent: only touches remaining .bmp files (BMPs are deleted on success)
    echo "[$(date '+%H:%M:%S')] Converting BMP → JPEG ..."
    CONDA_OVERRIDE_CUDA=12.6 pixi run python -m utilities.convert_rich_bmp_to_jpeg \
        --root "$dest_dir" \
        --quality 92 \
        --workers 16

    echo "[$(date '+%H:%M:%S')] Resizing $scene_name to 840px ..."
    CONDA_OVERRIDE_CUDA=12.6 pixi run python -m scripts.resize_rich_frames \
        --data_root "$STAGING" \
        --output_dir "$STAGING_RESIZED" \
        --max_side 840 \
        --workers 16 \
        --skip_existing \
        --scene "$scene_name"
done

# ── Step 2: Frame count report (raw vs resized) ───────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Frame counts per camera (raw / resized)"
echo "══════════════════════════════════════════════"
for scene_name in "${SCENES[@]}"; do
    echo "Scene: $scene_name"
    for cam_dir in "$STAGING/$scene_name"/cam_*/; do
        [ -d "$cam_dir" ] || continue
        cam=$(basename "$cam_dir")
        n_raw=$(find "$cam_dir" -type f -name "*.jpg" | wc -l)
        n_res=$(find "$STAGING_RESIZED/$scene_name/$cam" -type f -name "*.jpg" 2>/dev/null | wc -l)
        flag=""; [ "$n_raw" -ne "$n_res" ] && flag="  <-- MISMATCH"
        echo "  ${cam}: ${n_raw} / ${n_res}${flag}"
    done
done

echo ""
echo "[$(date '+%H:%M:%S')] All done. Resized frames in $STAGING_RESIZED"
