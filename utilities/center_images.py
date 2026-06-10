#!/usr/bin/env python3
"""
Crop RICH training images so that the GT principal point lands at image center.

VGGT assumes a centered principal point.  This script reads images from a
squashfuse-mounted .sqsh archive, crops each camera's frames symmetrically
around its calibrated (cx, cy), and writes the results preserving the
<scene>/<cam_XX>/<frame> folder structure.

Intrinsics in scan_calibration XMLs are for the original 4012×3008 resolution.
The script reads the actual image size from the first frame of each camera and
scales (cx, cy) accordingly before cropping.

Usage (typical):
    python utilities/center_images.py \\
        --sqsh /capstor/scratch/cscs/tnanni/datasets/rich/train_dataset.sqsh \\
        --mount /tmp/rich_sqsh \\
        --calib_root /capstor/scratch/cscs/tnanni/datasets/rich/scan_calibration \\
        --output /capstor/scratch/cscs/tnanni/datasets/rich/centered_train

    # If the sqsh is already mounted at <path>, skip --sqsh and use --mount only:
    python utilities/center_images.py \\
        --mount /tmp/rich_sqsh \\
        --calib_root ... --output ...
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Native resolution at which GT intrinsics were measured.
ORIG_W, ORIG_H = 4012, 3008

_IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqsh", default=None,
                   help="Path to .sqsh file.  If omitted, --mount must already be mounted.")
    p.add_argument("--mount", default="/tmp/rich_sqsh",
                   help="Mount point for squashfuse (created if absent).")
    p.add_argument("--calib_root",
                   default="/capstor/scratch/cscs/tnanni/datasets/rich/scan_calibration",
                   help="Root of scan_calibration directory.")
    p.add_argument("--output",
                   default="/capstor/scratch/cscs/tnanni/datasets/rich/centered_train",
                   help="Output root directory.")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel camera threads.")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG output quality (0-100).")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Process only these scene names (default: all).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Squashfuse helpers
# ---------------------------------------------------------------------------

def ensure_mounted(sqsh_path: str | None, mount_point: str) -> bool:
    mp = Path(mount_point)
    mp.mkdir(parents=True, exist_ok=True)

    # Check if already mounted.
    if subprocess.run(["mountpoint", "-q", str(mp)]).returncode == 0:
        print(f"[mount] {mount_point} already mounted — reusing")
        return True

    if sqsh_path is None:
        print(f"[error] {mount_point} is not mounted and --sqsh was not provided.", file=sys.stderr)
        return False

    result = subprocess.run(["squashfuse", sqsh_path, str(mp)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] squashfuse failed:\n{result.stderr}", file=sys.stderr)
        return False

    print(f"[mount] Mounted {sqsh_path} → {mount_point}")
    return True


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def scene_stem(scene_name: str) -> str:
    """'BBQ_001_guitar' → 'BBQ',  'LectureHall_018_wipe' → 'LectureHall'."""
    m = re.match(r'^(.+?)_\d{3}_', scene_name)
    return m.group(1) if m else scene_name


def parse_intrinsics(xml_path: Path) -> np.ndarray | None:
    """Return the (3, 3) camera intrinsic matrix from an OpenCV XML file, or None."""
    if not xml_path.exists():
        return None
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    node = root.find("Intrinsics")
    if node is None:
        return None
    rows = int(node.findtext("rows", default="3"))
    cols = int(node.findtext("cols", default="3"))
    data = list(map(float, node.findtext("data", default="").split()))
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def scale_intrinsics(K: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Scale K defined at (ORIG_W, ORIG_H) to match (img_w, img_h)."""
    K = K.copy()
    K[0] *= img_w / ORIG_W  # scales fx and cx
    K[1] *= img_h / ORIG_H  # scales fy and cy
    return K


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def compute_crop(img_h: int, img_w: int, cx: float, cy: float) -> tuple[int, int, int, int]:
    """Return (y1, y2, x1, x2) so that (cx, cy) lands exactly at the crop center.

    Uses the largest symmetric window around the rounded principal point that
    stays within image bounds.
    """
    cx_i = round(cx)
    cy_i = round(cy)
    # Maximum half-width / half-height that fits symmetrically around (cx_i, cy_i).
    margin_x = min(cx_i, img_w - cx_i)
    margin_y = min(cy_i, img_h - cy_i)
    return (cy_i - margin_y, cy_i + margin_y,
            cx_i - margin_x, cx_i + margin_x)


# ---------------------------------------------------------------------------
# Per-camera processing
# ---------------------------------------------------------------------------

def process_camera(
    cam_dir: Path,
    out_cam_dir: Path,
    K_orig: np.ndarray,
    quality: int,
) -> tuple[int, int]:
    """Crop and save all frames for one camera.

    Returns (frames_written, frames_skipped).
    """
    images = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        return 0, 0

    # Derive actual image resolution from the first frame.
    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"  [warn] cannot read {images[0]}", file=sys.stderr)
        return 0, len(images)

    img_h, img_w = first.shape[:2]
    K = scale_intrinsics(K_orig, img_w, img_h)
    cx, cy = K[0, 2], K[1, 2]
    y1, y2, x1, x2 = compute_crop(img_h, img_w, cx, cy)

    out_cam_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    written = skipped = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        cropped = img[y1:y2, x1:x2]
        out_path = out_cam_dir / (img_path.stem + ".jpg")
        cv2.imwrite(str(out_path), cropped, encode_params)
        written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not ensure_mounted(args.sqsh, args.mount):
        sys.exit(1)

    mount = Path(args.mount)
    calib_root = Path(args.calib_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover scenes (top-level dirs in the sqsh mount).
    all_scene_dirs = sorted(p for p in mount.iterdir() if p.is_dir())
    if args.scenes:
        scene_set = set(args.scenes)
        all_scene_dirs = [d for d in all_scene_dirs if d.name in scene_set]
    print(f"Scenes to process: {len(all_scene_dirs)}")

    # Build the list of (cam_dir, out_cam_dir, K_orig) tasks.
    tasks: list[tuple[Path, Path, np.ndarray]] = []

    for scene_dir in all_scene_dirs:
        scene_name = scene_dir.name
        stem = scene_stem(scene_name)
        calib_dir = calib_root / stem / "calibration"

        cam_dirs = sorted(p for p in scene_dir.iterdir()
                          if p.is_dir() and p.name.startswith("cam"))
        if not cam_dirs:
            print(f"  [skip] {scene_name}: no cam_* subdirectories found")
            continue

        for cam_dir in cam_dirs:
            m = re.search(r"\d+", cam_dir.name)
            cam_num = int(m.group()) if m else 0
            xml_path = calib_dir / f"{cam_num:03d}.xml"

            K = parse_intrinsics(xml_path)
            if K is None:
                print(f"  [warn] {scene_name}/{cam_dir.name}: missing calibration {xml_path.name}, skipping")
                continue

            out_cam_dir = output_root / scene_name / cam_dir.name
            tasks.append((cam_dir, out_cam_dir, K))

    print(f"Total cameras: {len(tasks)}  |  workers: {args.workers}")

    total_written = total_skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_camera, cam_dir, out_cam_dir, K, args.quality): (cam_dir,)
            for cam_dir, out_cam_dir, K in tasks
        }
        for fut in as_completed(futures):
            (cam_dir,) = futures[fut]
            try:
                written, skipped = fut.result()
            except Exception as exc:
                print(f"  [error] {cam_dir}: {exc}", file=sys.stderr)
                continue
            total_written += written
            total_skipped += skipped
            scene_cam = f"{cam_dir.parent.name}/{cam_dir.name}"
            print(f"  {scene_cam}: {written} frames written" +
                  (f", {skipped} skipped" if skipped else ""))

    print(f"\nDone — {total_written} frames written, {total_skipped} skipped → {args.output}")


if __name__ == "__main__":
    main()
