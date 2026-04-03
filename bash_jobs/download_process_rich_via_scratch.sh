#!/bin/bash
#SBATCH --job-name=rich_dl_scratch
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/tnanni/rich_dl_scratch_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#
# Download RICH train scenes to scratch, extract and convert images there,
# then move the processed scene directory to the final destination on project.
#
# Submit with credentials:
#   sbatch --export=ALL,RICH_USER=youruser,RICH_PASS=yourpass bash_jobs/download_process_rich_via_scratch.sh
set -euo pipefail

RICH_TRAIN_ROOT="/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
SCRATCH_DIR="/cluster/scratch/tnanni/rich_staging"
GHOST_DIR="/cluster/project/cvg/students/tnanni/ghost"

mkdir -p "$SCRATCH_DIR"

# ── Credentials ──────────────────────────────────────────────────────────────
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

# ── Scenes to download ───────────────────────────────────────────────────────
SCENES=(
    # "BBQ_001_guitar.tar.gz" # DOWNLOADED
    # "BBQ_001_juggle.tar.gz" # DOWNLOADED
    # "LectureHall_018_wipingchairs1.tar.gz" # DOWNLOADED
    # "LectureHall_018_wipingspray1.tar.gz" # DOWNLOADED
    # "LectureHall_020_wipingtable1.tar.gz" # DOWNLOADED
    # "ParkingLot1_002_burpee3.tar.gz" # DOWNLOADED
    "ParkingLot1_002_overfence1.tar.gz"
    "ParkingLot1_002_overfence2.tar.gz"
    "ParkingLot1_002_pushup1.tar.gz"
    "ParkingLot1_002_stretching1.tar.gz"
    "ParkingLot1_004_005_greetingchattingeating1.tar.gz"
    "ParkingLot1_004_burpeejump1.tar.gz"
    "ParkingLot1_004_eating1.tar.gz"
    "ParkingLot1_004_pushup2.tar.gz"
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
    # "ParkingLot2_008_pushup2.tar.gz" # DOWNLOADED
    # "ParkingLot2_014_burpeejump1.tar.gz"
    # "ParkingLot2_014_burpeejump2.tar.gz"
    # "ParkingLot2_014_overfence3.tar.gz"
    # "ParkingLot2_014_phonetalk2.tar.gz"
    # "ParkingLot2_014_pushup2.tar.gz"
    # "ParkingLot2_014_takingphotos2.tar.gz"
    # "ParkingLot2_015_burpeejump2.tar.gz"
    # "ParkingLot2_015_overfence1.tar.gz"
    # "ParkingLot2_015_pushup1.tar.gz" # DOWNLOADED
    # "ParkingLot2_016_burpeejump2.tar.gz"
    # "ParkingLot2_016_overfence2.tar.gz"
    # "ParkingLot2_016_pushup1.tar.gz"
    # "ParkingLot2_016_pushup2.tar.gz"
    # "ParkingLot2_016_stretching1.tar.gz"
    # "Pavallion_000_phonesiteat.tar.gz" # DOWNLOADED
    # "Pavallion_000_plankjack.tar.gz" # DOWNLOADED
    # "Pavallion_000_sidebalancerun.tar.gz" # DOWNLOADED
    # "Pavallion_000_yoga2.tar.gz" # DOWNLOADED
    # "Pavallion_002_phonesiteat.tar.gz" # DOWNLOADED
    # "Pavallion_002_plankjack.tar.gz" # DOWNLOADED
    # "Pavallion_003_018_tossball.tar.gz" # DOWNLOADED
    # "Pavallion_003_phonesiteat.tar.gz" # DOWNLOADED
    # "Pavallion_003_plankjack.tar.gz" # DOWNLOADED
    # "Pavallion_003_sidebalancerun.tar.gz" # DOWNLOADED
    # "Pavallion_006_phonesiteat.tar.gz" # DOWNLOADED
    # "Pavallion_013_phonesiteat.tar.gz" # DOWNLOADED
    # "Pavallion_013_plankjack.tar.gz" # DOWNLOADED
    # "Pavallion_013_yoga2.tar.gz" # DOWNLOADED
    # "ParkingLot2_015_eating2.tar.gz"
    # "ParkingLot1_004_phonetalk1.tar.gz"
    # "Pavallion_006_plankjack.tar.gz"
    # "Pavallion_006_sidebalancerun.tar.gz"
    # "ParkingLot2_008_overfence2.tar.gz"
)

# ── Per-scene pipeline ────────────────────────────────────────────────────────
for tarfile in "${SCENES[@]}"; do
    scene_name="${tarfile%.tar.gz}"
    scratch_scene="$SCRATCH_DIR/$scene_name"
    dest_dir="$RICH_TRAIN_ROOT/$scene_name"

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Scene: $scene_name"
    echo "═══════════════════════════════════════════════════"

    # ── 1. Download to scratch ─────────────────────────────
    echo "[$(date '+%H:%M:%S')] Downloading $tarfile to scratch ..."
    wget \
        --post-data "username=${USERNAME}&password=${PASSWORD}" \
        "https://download.is.tue.mpg.de/download.php?domain=rich&sfile=train/${tarfile}&resume=1" \
        -O "$SCRATCH_DIR/$tarfile" \
        --no-check-certificate \
        --continue

    # ── 2. Extract on scratch ──────────────────────────────
    echo "[$(date '+%H:%M:%S')] Extracting ..."
    tar -xzf "$SCRATCH_DIR/$tarfile" -C "$SCRATCH_DIR" --strip-components=5
    echo "[$(date '+%H:%M:%S')] Removing archive ..."
    rm "$SCRATCH_DIR/$tarfile"

    # ── 3a. Convert BMP → JPEG on scratch ─────────────────
    echo "[$(date '+%H:%M:%S')] Converting BMP → JPEG ..."
    cd "$GHOST_DIR"
    CONDA_OVERRIDE_CUDA=12.6 pixi run python -m utilities.convert_rich_bmp_to_jpeg \
        --root "$scratch_scene" \
        --quality 92 \
        --workers 16

    # ── 3b. Convert PNG → JPEG on scratch ─────────────────
    echo "[$(date '+%H:%M:%S')] Converting PNG → JPEG ..."
    SCENE_DIR="$scratch_scene" CONDA_OVERRIDE_CUDA=12.6 pixi run python - <<'PYEOF'
import os, sys
from pathlib import Path
from multiprocessing import Pool

root = Path(os.environ["SCENE_DIR"])
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

    # ── 4. Move to final destination ───────────────────────
    echo "[$(date '+%H:%M:%S')] Moving $scene_name to $RICH_TRAIN_ROOT ..."
    mkdir -p "$RICH_TRAIN_ROOT"
    mv "$scratch_scene" "$dest_dir"
    echo "[$(date '+%H:%M:%S')] Done: $scene_name"
    echo "Scene $scene_name fully processed and moved to $dest_dir" | mail -s "[SLURM] rich_dl_scratch: $scene_name done" tnanni@ethz.ch
done

rmdir "$SCRATCH_DIR" 2>/dev/null || true

# ── Frame count report ────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Frame counts per camera"
echo "═══════════════════════════════════════════════════"

for tarfile in "${SCENES[@]}"; do
    scene_name="${tarfile%.tar.gz}"
    scene_path="$RICH_TRAIN_ROOT/$scene_name"

    if [ ! -d "$scene_path" ]; then
        echo "Scene: $scene_name  (directory not found)"
        continue
    fi

    echo "Scene: $scene_name"
    found_cameras=0
    for cam_dir in "${scene_path}"/cam_*/; do
        if [ -d "$cam_dir" ]; then
            cam=$(basename "$cam_dir")
            count=$(find "$cam_dir" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
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
