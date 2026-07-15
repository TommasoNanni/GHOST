#!/usr/bin/env python3
"""
Undistort EgoHumans exo-camera images using COLMAP OPENCV_FISHEYE calibration.

For each sequence, reads intrinsics from colmap/workplace/cameras.txt and the
cam_name→camera_id mapping from colmap/workplace/images.txt, then undistorts
every frame in exo/<cam>/images/ and writes to exo/<cam>/images_undistorted/.

Already-processed cameras are skipped (presence of images_undistorted/ with
the same frame count).

Usage:
    python utilities/undistort_egohumans.py \\
        --data_root /capstor/scratch/cscs/tnanni/datasets/egohumans \\
        --workers 8

    # Single activity:
    python utilities/undistort_egohumans.py \\
        --data_root /capstor/scratch/cscs/tnanni/datasets/egohumans \\
        --activity 01_tagging
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}


def _parse_colmap_calibration(seq_dir: Path) -> dict[str, dict]:
    """Return {cam_name: {K, D, W, H}} parsed from COLMAP txt files."""
    cameras_txt = seq_dir / "colmap" / "workplace" / "cameras.txt"
    images_txt  = seq_dir / "colmap" / "workplace" / "images.txt"

    # camera_id -> intrinsics
    cam_params: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # OPENCV_FISHEYE: ID MODEL W H fx fy cx cy k1 k2 k3 k4
            cid = int(parts[0])
            W, H = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = (float(x) for x in parts[4:8])
            k1, k2, k3, k4 = (float(x) for x in parts[8:12])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
            cam_params[cid] = {"K": K, "D": D, "W": W, "H": H}

    # cam_name -> camera_id (first occurrence in images.txt)
    name_to_cid: dict[str, int] = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(parts) < 10:
                continue
            name = parts[9]
            cam_name = name.split("/")[0]
            if cam_name.startswith("cam") and cam_name not in name_to_cid:
                name_to_cid[cam_name] = int(parts[8])

    result: dict[str, dict] = {}
    for cam_name, cid in name_to_cid.items():
        if cid in cam_params:
            result[cam_name] = cam_params[cid]
    return result


def _undistort_camera(args: tuple) -> str:
    seq_dir, cam_name, calib = args
    images_dir = seq_dir / "exo" / cam_name / "images"
    out_dir    = seq_dir / "exo" / cam_name / "images_undistorted"

    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        return f"SKIP {seq_dir.name}/{cam_name}: no frames"

    # Skip if already done
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == len(frames):
        return f"SKIP {seq_dir.name}/{cam_name}: already done"

    out_dir.mkdir(exist_ok=True)

    K, D, W, H = calib["K"], calib["D"], calib["W"], calib["H"]
    # Compute new camera matrix. balance=0.0 crops to the fully-valid region so the
    # output has NO black borders (balance=1.0 keeps full FOV but leaves black
    # half-moon borders that corrupt VGGT/SAM3D). Raise toward ~0.3 to trade a
    # little border back for more FOV.
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2
    )

    import json
    for frame in frames:
        img = cv2.imread(str(frame))
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / frame.name), undistorted)

    calib_out = {
        "K": K_new.tolist(),
        "width": W,
        "height": H,
    }
    with open(seq_dir / "exo" / cam_name / "calibration.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    return f"OK {seq_dir.name}/{cam_name}: {len(frames)} frames"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root of EgoHumans dataset")
    parser.add_argument("--activity", default=None,
                        help="Process only this activity (e.g. 01_tagging)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    activities = [args.activity] if args.activity else sorted(KEEP_CAMS.keys())

    tasks: list[tuple] = []
    for activity in activities:
        keep = KEEP_CAMS[activity]
        cam_ready = data_root / activity / INNER / activity
        for seq_dir in sorted(cam_ready.iterdir()):
            if not seq_dir.is_dir():
                continue
            calib = _parse_colmap_calibration(seq_dir)
            for cam_name in keep:
                if cam_name not in calib:
                    print(f"WARN: {activity}/{seq_dir.name}/{cam_name} missing calibration")
                    continue
                tasks.append((seq_dir, cam_name, calib[cam_name]))

    print(f"Tasks: {len(tasks)} camera×sequence pairs")

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_undistort_camera, tasks):
            print(result)


if __name__ == "__main__":
    main()
