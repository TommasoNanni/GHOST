#!/bin/bash
#SBATCH --job-name=rich_process
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/tnanni/rich_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#
# For each tar.gz already sitting in TARGET_DIR:
#   1. Move tar.gz → scratch
#   2. Extract on scratch
#   3. Convert BMP/PNG → JPEG on scratch
#   4. Move extracted scene dir → RICH_TRAIN_ROOT on project
#   5. Clean up scratch
#
set -euo pipefail

TARGET_DIR="/cluster/project/cvg/data/rich"
RICH_TRAIN_ROOT="/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
SCRATCH_DIR="/cluster/scratch/tnanni/rich_staging"
GHOST_DIR="/cluster/project/cvg/students/tnanni/ghost"

mkdir -p "$SCRATCH_DIR"

for tarfile in "$TARGET_DIR"/*.tar.gz; do
    [ -f "$tarfile" ] || continue
    fname=$(basename "$tarfile")
    scene_name="${fname%.tar.gz}"
    dest_dir="${RICH_TRAIN_ROOT}/${scene_name}"

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Scene: $scene_name"
    echo "═══════════════════════════════════════════════════"
    df -h "$TARGET_DIR" | tail -1

    # ── 1. Move tar.gz to scratch ──────────────────────────
    echo "[$(date '+%H:%M:%S')] Moving $fname to scratch ..."
    mv "$tarfile" "$SCRATCH_DIR/$fname"

    # ── 2. Extract on scratch ──────────────────────────────
    echo "[$(date '+%H:%M:%S')] Extracting ..."
    tar -xzf "$SCRATCH_DIR/$fname" -C "$SCRATCH_DIR" --strip-components=5
    echo "[$(date '+%H:%M:%S')] Removing archive from scratch ..."
    rm "$SCRATCH_DIR/$fname"

    # ── 3a. Convert BMP → JPEG on scratch ─────────────────
    echo "[$(date '+%H:%M:%S')] Converting BMP → JPEG ..."
    cd "$GHOST_DIR"
    CONDA_OVERRIDE_CUDA=12.6 pixi run python -m utilities.convert_rich_bmp_to_jpeg \
        --root "$SCRATCH_DIR/$scene_name" \
        --quality 92 \
        --workers 4

    # ── 3b. Convert PNG → JPEG on scratch ─────────────────
    echo "[$(date '+%H:%M:%S')] Converting PNG → JPEG ..."
    SCENE_DIR="$SCRATCH_DIR/$scene_name" CONDA_OVERRIDE_CUDA=12.6 pixi run python - <<'PYEOF'
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

    # ── 4. Move processed scene to project ────────────────
    echo "[$(date '+%H:%M:%S')] Moving $scene_name to project ..."
    mkdir -p "$RICH_TRAIN_ROOT"
    mv "$SCRATCH_DIR/$scene_name" "$dest_dir"

    echo "[$(date '+%H:%M:%S')] Done: $scene_name"
    df -h "$TARGET_DIR" | tail -1
done

rmdir "$SCRATCH_DIR" 2>/dev/null || true

echo ""
echo "[$(date '+%H:%M:%S')] All scenes processed."
