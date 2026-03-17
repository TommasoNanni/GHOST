#!/bin/bash
#SBATCH --job-name=rich_download
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/tnanni/rich_download_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#
# Download RICH train scenes, extract archives, convert images to JPEG,
# and report frame counts per camera for every downloaded scene.
#
# Submit with credentials (never hardcode them in this file):
#   sbatch --export=ALL,RICH_USER=youruser,RICH_PASS=yourpass bash_jobs/download_and_process_rich.sh
set -euo pipefail

TARGET_DIR="/cluster/project/cvg/data/rich"
RICH_TRAIN_ROOT="/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
GHOST_DIR="/cluster/project/cvg/students/tnanni/ghost"

# ── Credentials ──────────────────────────────────────────────────────────────
# Pass via sbatch: --export=ALL,RICH_USER=youruser,RICH_PASS=yourpass
USERNAME="${RICH_USER:-}"
PASSWORD="${RICH_PASS:-}"
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "ERROR: RICH_USER and RICH_PASS must be set."
    echo "Submit with: sbatch --export=ALL,RICH_USER=<user>,RICH_PASS=<pass> $0"
    exit 1
fi

# URL-encode a string (same helper as in download_rich_train.sh)
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

# ── Scenes to download ───────────────────────────────────────────────────────
# Each entry is the filename that will be saved in TARGET_DIR.
# To add/remove scenes, comment/uncomment lines below (mirrors download_rich_train.sh).
SCENES=(
    # "BBQ_001_guitar.tar.gz" # DOWNLOADED
    # "BBQ_001_juggle.tar.gz" # DOWNLOADED
    # "LectureHall_018_wipingchairs1.tar.gz" # DOWNLOADED
    #"LectureHall_018_wipingspray1.tar.gz" # DOWNLOADED
    #"LectureHall_020_wipingtable1.tar.gz" # DOWNLOADED
    "ParkingLot1_002_burpee3.tar.gz"
    # "ParkingLot1_002_overfence1.tar.gz"
    # "ParkingLot1_002_overfence2.tar.gz"
    # "ParkingLot1_002_pushup1.tar.gz"
    # "ParkingLot1_002_stretching1.tar.gz"
    # "ParkingLot1_004_005_greetingchattingeating1.tar.gz"
    # "ParkingLot1_004_burpeejump1.tar.gz"
    # "ParkingLot1_004_eating1.tar.gz"
    # "ParkingLot1_004_pushup2.tar.gz"
    # "ParkingLot1_004_takingphotos1.tar.gz"
    # "ParkingLot1_005_burpeejump2.tar.gz"
    # "ParkingLot1_005_overfence1.tar.gz"
    # "ParkingLot1_005_pushup2.tar.gz"
    # "ParkingLot1_005_pushup3.tar.gz"
    # "ParkingLot1_007_eating1.tar.gz"
    # "ParkingLot1_007_eating2.tar.gz"
    # "ParkingLot1_007_overfence2.tar.gz"
    # "ParkingLot2_008_burpeejump1.tar.gz"
    # "ParkingLot2_008_eating1.tar.gz"
    # "ParkingLot2_008_overfence1.tar.gz"
    # "ParkingLot2_008_overfence3.tar.gz"
    # "ParkingLot2_008_phonetalk1.tar.gz"
    # "ParkingLot2_008_pushup1.tar.gz"
    "ParkingLot2_008_pushup2.tar.gz"
    # "ParkingLot2_014_burpeejump1.tar.gz"
    # "ParkingLot2_014_burpeejump2.tar.gz"
    # "ParkingLot2_014_overfence3.tar.gz"
    # "ParkingLot2_014_phonetalk2.tar.gz"
    # "ParkingLot2_014_pushup2.tar.gz"
    # "ParkingLot2_014_takingphotos2.tar.gz"
    # "ParkingLot2_015_burpeejump2.tar.gz"
    # "ParkingLot2_015_overfence1.tar.gz"
    "ParkingLot2_015_pushup1.tar.gz"
    # "ParkingLot2_016_burpeejump2.tar.gz"
    # "ParkingLot2_016_overfence2.tar.gz"
    # "ParkingLot2_016_pushup1.tar.gz"
    # "ParkingLot2_016_pushup2.tar.gz"
    # "ParkingLot2_016_stretching1.tar.gz"
    "Pavallion_000_phonesiteat.tar.gz"
    # "Pavallion_000_plankjack.tar.gz"
    # "Pavallion_000_sidebalancerun.tar.gz"
    # "Pavallion_000_yoga2.tar.gz"
    # "Pavallion_002_phonesiteat.tar.gz"
    # "Pavallion_002_plankjack.tar.gz"
    #"Pavallion_003_018_tossball.tar.gz" # DOWNLOADED
    # "Pavallion_003_phonesiteat.tar.gz"
    # "Pavallion_003_plankjack.tar.gz"
    # "Pavallion_003_sidebalancerun.tar.gz"
    # "Pavallion_006_phonesiteat.tar.gz"
    # "Pavallion_013_phonesiteat.tar.gz"
    # "Pavallion_013_plankjack.tar.gz"
    "Pavallion_013_yoga2.tar.gz"
    # "ParkingLot2_015_eating2.tar.gz"
    # "ParkingLot1_004_phonetalk1.tar.gz"
    # "Pavallion_006_plankjack.tar.gz"
    # "Pavallion_006_sidebalancerun.tar.gz"
    # "ParkingLot2_008_overfence2.tar.gz"
)

# ── Step 1: Download ──────────────────────────────────────────────────────────
cd "$TARGET_DIR"
echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 1 — Downloading scenes"
echo "═══════════════════════════════════════════════════"

for tarfile in "${SCENES[@]}"; do
    sfile="${tarfile}"   # filename equals the sfile query param for train/
    echo "[$(date '+%H:%M:%S')] Downloading $tarfile ..."
    wget \
        --post-data "username=${USERNAME}&password=${PASSWORD}" \
        "https://download.is.tue.mpg.de/download.php?domain=rich&sfile=train/${sfile}&resume=1" \
        -O "$tarfile" \
        --no-check-certificate \
        --continue
    echo "[$(date '+%H:%M:%S')] Done: $tarfile"
done

# ── Step 2: Extract and remove archives ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 2 — Extracting archives"
echo "═══════════════════════════════════════════════════"

for tarfile in "${SCENES[@]}"; do
    if [ -f "$tarfile" ]; then
        echo "[$(date '+%H:%M:%S')] Extracting $tarfile ..."
        tar -xzf "$tarfile"
        echo "[$(date '+%H:%M:%S')] Removing $tarfile ..."
        rm "$tarfile"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: $tarfile not found, skipping extraction."
    fi
done

# ── Step 3: BMP → JPEG conversion ────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 3a — Converting BMP → JPEG"
echo "═══════════════════════════════════════════════════"
cd "$GHOST_DIR"

CONDA_OVERRIDE_CUDA=12.6 pixi run python -m utilities.convert_rich_bmp_to_jpeg \
    --root "$RICH_TRAIN_ROOT" \
    --quality 92 \
    --workers 16

echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 3b — Converting PNG → JPEG"
echo "═══════════════════════════════════════════════════"

CONDA_OVERRIDE_CUDA=12.6 pixi run python - <<'PYEOF'
import os, sys
from pathlib import Path
from multiprocessing import Pool

root = Path(os.environ.get("RICH_TRAIN_ROOT", "/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"))
png_files = sorted(root.rglob("*.png"))

if not png_files:
    print("No .png files found — skipping.")
    sys.exit(0)

print(f"Found {len(png_files)} PNG files")

def convert_one(p):
    from PIL import Image
    jpg = p.with_suffix(".jpg")
    try:
        Image.open(p).save(jpg, "JPEG", quality=92, subsampling=0)
        p.unlink()
        return str(p), True, ""
    except Exception as e:
        return str(p), False, str(e)

ok = fail = 0
with Pool(processes=16) as pool:
    for i, (path, success, err) in enumerate(pool.imap_unordered(convert_one, png_files, chunksize=20), 1):
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAILED {path}: {err}")
        if i % 500 == 0 or i == len(png_files):
            print(f"  {i}/{len(png_files)}  ok={ok}  fail={fail}")

print(f"\nDone. Converted: {ok}  Failed: {fail}")
if fail:
    sys.exit(1)
PYEOF

# Export for the heredoc's Python subprocess
export RICH_TRAIN_ROOT

# ── Step 4: Frame count report ────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 4 — Frame counts per camera"
echo "═══════════════════════════════════════════════════"

for tarfile in "${SCENES[@]}"; do
    # Strip .tar.gz suffix to get the scene name
    scene_name="${tarfile%.tar.gz}"
    scene_path="${RICH_TRAIN_ROOT}/${scene_name}"

    if [ ! -d "$scene_path" ]; then
        echo "Scene: $scene_name  (directory not found at $scene_path)"
        continue
    fi

    echo "Scene: $scene_name"
    found_cameras=0
    for cam_dir in "${scene_path}"/cam_*/; do
        if [ -d "$cam_dir" ]; then
            cam=$(basename "$cam_dir")
            # Count image files directly inside the camera directory (not in subdirs like frames/)
            count=$(find "$cam_dir" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.png" \) | wc -l)
            echo "  ${cam}: ${count} frames"
            found_cameras=1
        fi
    done
    if [ "$found_cameras" -eq 0 ]; then
        echo "  (no cam_* subdirectories found)"
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] All done."
