"""
Undistort EgoExo4D exo-camera frames in-place using Kannala-Brandt calibration.

Reads gopro_calibs.csv for each take and overwrites the extracted JPEGs
(cam*/frame_*.jpg) with undistorted versions at the original resolution.
The pipeline then reads those files directly and handles resizing via the
existing _maybe_resize convention (Video max_side=1440).

Run this ONCE before the pipeline, after utilities/extract_egoexo4d_frames.py.

Usage:
    pixi run python utilities/undistort_egoexo4d.py \\
        --frames-root /capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames \\
        --gt-dir      /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \\
        --workers     16
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SENTINEL   = ".undistorted"   # written to cam_dir after undistortion completes


def _parse_calibration(csv_path: Path) -> dict[str, dict]:
    """Return {cam_uid: {K, D, W, H}} from gopro_calibs.csv."""
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            uid = row["cam_uid"].strip()
            if uid in rows:
                continue
            K = np.array([
                [float(row["intrinsics_0"]), 0, float(row["intrinsics_2"])],
                [0, float(row["intrinsics_1"]), float(row["intrinsics_3"])],
                [0, 0, 1],
            ], dtype=np.float64)
            D = np.array(
                [[float(row[f"intrinsics_{i}"])] for i in range(4, 8)],
                dtype=np.float64,
            )
            rows[uid] = {
                "K": K, "D": D,
                "W": int(row["image_width"]),
                "H": int(row["image_height"]),
            }
    return rows


def _build_maps(K: np.ndarray, D: np.ndarray, W: int, H: int) -> tuple:
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2
    )
    return map1, map2, K_new


def _process_camera(args: tuple) -> str:
    cam_dir, csv_path = args
    take_name = cam_dir.parent.name
    cam_name  = cam_dir.name

    sentinel = cam_dir / _SENTINEL
    if sentinel.exists():
        return f"  {take_name}/{cam_name}: already undistorted, skipping"

    src_frames = sorted(p for p in cam_dir.iterdir()
                        if p.suffix.lower() in _IMAGE_EXTS)
    if not src_frames:
        return f"  {take_name}/{cam_name}: no frames found"

    calib = _parse_calibration(csv_path)
    if cam_name not in calib:
        return f"  {take_name}/{cam_name}: WARNING — not in gopro_calibs.csv, skipping"

    c = calib[cam_name]
    map1, map2, _ = _build_maps(c["K"], c["D"], c["W"], c["H"])

    for src in src_frames:
        img = cv2.imread(str(src))
        if img is None:
            continue
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(src), und, [cv2.IMWRITE_JPEG_QUALITY, 95])

    sentinel.touch()
    return f"  {take_name}/{cam_name}: {len(src_frames)} frame(s) undistorted in-place"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--take", default=None, help="Filter by take name substring")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    gt_dir      = Path(args.gt_dir)

    tasks: list[tuple] = []
    for take_dir in sorted(frames_root.iterdir()):
        if not take_dir.is_dir():
            continue
        if args.take and args.take not in take_dir.name:
            continue
        csv_path = gt_dir / take_dir.name / "gopro_calibs.csv"
        if not csv_path.exists():
            print(f"WARNING: no gopro_calibs.csv for {take_dir.name}")
            continue
        for cam_dir in sorted(p for p in take_dir.iterdir() if p.is_dir()):
            tasks.append((cam_dir, csv_path))

    print(f"Undistorting {len(tasks)} camera(s) with {args.workers} worker(s)...")
    if args.workers <= 1:
        for t in tasks:
            print(_process_camera(t))
    else:
        with mp.Pool(args.workers) as pool:
            for msg in pool.imap_unordered(_process_camera, tasks):
                print(msg)
    print("Done.")


if __name__ == "__main__":
    main()
