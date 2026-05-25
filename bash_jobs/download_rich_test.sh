#!/bin/bash -l
#SBATCH --job-name=rich_dl_test
#SBATCH --account=a0185
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

# Download missing RICH test scenes, convert BMP→JPEG, resize to 1440px,
# then rebuild test_dataset.sqsh.
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
STAGING=/capstor/scratch/cscs/tnanni/rich_test_staging
# Resized to 1440px — this is what goes into the squash
STAGING_RESIZED=/capstor/scratch/cscs/tnanni/rich_test_staging_1440
OLD_SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/test_dataset.sqsh
NEW_SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/test_dataset_new.sqsh
FINAL_SQSH="$OLD_SQSH"

mkdir -p "$STAGING" "$STAGING_RESIZED"

# ── Scenes already in the existing squash (skip re-download) ──────────────────
ALREADY_DONE=(
    "Gym_010_cooking1"
    "Gym_010_dips1"
    "Gym_010_dips2"
    "Gym_010_lunge1"
    "Gym_010_lunge2"
    "Gym_010_mountainclimber1"
    "Gym_010_mountainclimber2"
)

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

# ── Step 1: Download + convert missing scenes ─────────────────────────────────
for scene_name in "${SCENES[@]}"; do
    if printf '%s\n' "${ALREADY_DONE[@]}" | grep -qx "$scene_name"; then
        echo "Skipping download for $scene_name (already in old squash)"
        continue
    fi

    dest_dir="$STAGING/$scene_name"
    if [ -d "$dest_dir" ]; then
        echo "Skipping download for $scene_name (already staged)"
    else
        tarfile="${scene_name}.tar.gz"
        echo ""
        echo "══════════════════════════════════════════════"
        echo "  Scene: $scene_name"
        echo "══════════════════════════════════════════════"

        echo "[$(date '+%H:%M:%S')] Downloading ..."
        curl -L \
            --data "username=${USERNAME}&password=${PASSWORD}" \
            "https://download.is.tue.mpg.de/download.php?domain=rich&sfile=test/${tarfile}&resume=1" \
            -o "$STAGING/$tarfile" \
            --insecure

        echo "[$(date '+%H:%M:%S')] Extracting ..."
        tar -xzf "$STAGING/$tarfile" -C "$STAGING" --strip-components=5
        rm "$STAGING/$tarfile"

        echo "[$(date '+%H:%M:%S')] Converting BMP → JPEG ..."
        CONDA_OVERRIDE_CUDA=12.6 pixi run python -m utilities.convert_rich_bmp_to_jpeg \
            --root "$dest_dir" \
            --quality 92 \
            --workers 16
    fi

    # Resize to 1440px (skip if already done)
    if [ ! -d "$STAGING_RESIZED/$scene_name" ]; then
        echo "[$(date '+%H:%M:%S')] Resizing $scene_name to 1440px ..."
        CONDA_OVERRIDE_CUDA=12.6 pixi run python -m scripts.resize_rich_frames \
            --data_root "$STAGING" \
            --output_dir "$STAGING_RESIZED" \
            --max_side 1440 \
            --workers 16 \
            --scene "$scene_name"
    else
        echo "Skipping resize for $scene_name (already resized)"
    fi
done

# ── Step 2: Copy the 7 existing scenes from the old squash + resize ──────────
echo ""
echo "Copying existing scenes from old squash ..."
OLD_MOUNT=/tmp/rich_test_old
mkdir -p "$OLD_MOUNT"
squashfuse "$OLD_SQSH" "$OLD_MOUNT"
trap "fusermount -u '$OLD_MOUNT' 2>/dev/null || true" EXIT

for scene_name in "${ALREADY_DONE[@]}"; do
    if [ ! -d "$STAGING/$scene_name" ]; then
        echo "  Copying $scene_name from old squash → staging ..."
        cp -r "$OLD_MOUNT/$scene_name" "$STAGING/$scene_name"
    fi
    if [ ! -d "$STAGING_RESIZED/$scene_name" ]; then
        echo "  Resizing $scene_name to 1440px ..."
        CONDA_OVERRIDE_CUDA=12.6 pixi run python -m scripts.resize_rich_frames \
            --data_root "$STAGING" \
            --output_dir "$STAGING_RESIZED" \
            --max_side 1440 \
            --workers 16 \
            --scene "$scene_name"
    fi
done

fusermount -u "$OLD_MOUNT"
trap - EXIT

# ── Step 3: Rebuild squash from resized scenes ────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Building new squash from $STAGING_RESIZED ..."
mksquashfs "$STAGING_RESIZED" "$NEW_SQSH" \
    -comp lz4 -noI -noX -b 1M -processors 16

echo "[$(date '+%H:%M:%S')] Replacing old squash ..."
mv "$NEW_SQSH" "$FINAL_SQSH"

echo ""
echo "[$(date '+%H:%M:%S')] All done. Squash rebuilt at $FINAL_SQSH"
echo "Staging dirs can be deleted once the squash is verified:"
echo "  rm -rf $STAGING $STAGING_RESIZED"
