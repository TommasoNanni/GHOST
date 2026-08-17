"""Reproject EgoExo4D GT joints into every gopro frame and save an overlay.

Visual check for the three takes excluded from the EgoExo4D tables
(``evaluation/ablations_egoexo.py::EXCLUDED_TAKES``): does the annotated GT
skeleton actually land on a person in each calibrated camera?

    pixi run python scripts/egoexo_gt_reproject_overlay.py
    pixi run python scripts/egoexo_gt_reproject_overlay.py cmu_soccer16_2

Per camera it writes ``renders/egoexo_test/<take>__<cam>.jpg`` (frame resized to
1440x810, the space ``pred_keypoints_2d`` live in) and prints how many joints sit
in front of the camera, how many fall inside the image, and the depth spread.
A joint with camera-space depth <= 0 is drawn as a red cross labelled BEHIND.

Self-contained: the camera loading and projection are copied from
``scripts/egoexo_confirm_frame_fix.py`` rather than imported.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

GT_ROOT     = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt")
FRAMES_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames")
OUT_ROOT    = Path(__file__).resolve().parents[1] / "renders" / "egoexo_test"

# The takes dropped from the tables for broken GT / calibration.
DEFAULT_TAKES = ["cmu_soccer16_2", "uniandes_dance_002_2", "uniandes_dance_002_11"]

# The 12 body joints the eval scores (GT COCO naming).
JOINTS = ["left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
          "left-wrist", "right-wrist", "left-hip", "right-hip",
          "left-knee", "right-knee", "left-ankle", "right-ankle"]

LIMBS = [("left-shoulder", "right-shoulder"), ("left-shoulder", "left-elbow"),
         ("left-elbow", "left-wrist"), ("right-shoulder", "right-elbow"),
         ("right-elbow", "right-wrist"), ("left-shoulder", "left-hip"),
         ("right-shoulder", "right-hip"), ("left-hip", "right-hip"),
         ("left-hip", "left-knee"), ("left-knee", "left-ankle"),
         ("right-hip", "right-knee"), ("right-knee", "right-ankle")]

VIEW_W, VIEW_H = 1440, 810


def _load_cameras(gopro_csv: Path) -> dict:
    """Undistorted-pinhole cameras keyed by cam_uid, at the 1440-wide frame scale."""
    cams: dict[str, dict] = {}
    for row in csv.DictReader(open(gopro_csv)):
        K = np.array([[float(row["intrinsics_0"]), 0, float(row["intrinsics_2"])],
                      [0, float(row["intrinsics_1"]), float(row["intrinsics_3"])],
                      [0, 0, 1]])
        D = np.array([[float(row[f"intrinsics_{i}"])] for i in range(4, 8)])
        W, H = int(row["image_width"]), int(row["image_height"])
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=0.0)
        t = np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"])
        Rwc = R.from_quat([float(row[f"q{a}_world_cam"]) for a in ["x", "y", "z", "w"]]).as_matrix()
        cams[row["cam_uid"]] = {"Knew": Knew, "Rwc": Rwc, "t": t,
                                "sx": VIEW_W / W, "sy": VIEW_H / H}
    return cams


def _project(cam: dict, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (pixel, depth). Depth <= 0 means the point is behind the camera."""
    Xc = cam["Rwc"].T @ (X - cam["t"])
    z = float(Xc[2])
    if abs(z) < 1e-9:
        z = 1e-9
    uv = np.array([(cam["Knew"][0, 0] * Xc[0] / z + cam["Knew"][0, 2]) * cam["sx"],
                   (cam["Knew"][1, 1] * Xc[1] / z + cam["Knew"][1, 2]) * cam["sy"]])
    return uv, z


def _draw(img: np.ndarray, uv: dict, depth: dict, header: str) -> np.ndarray:
    for a, b in LIMBS:
        if a not in uv or b not in uv or depth[a] <= 0 or depth[b] <= 0:
            continue
        cv2.line(img, tuple(np.round(uv[a]).astype(int)),
                 tuple(np.round(uv[b]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    for n, p in uv.items():
        x, y = int(round(p[0])), int(round(p[1]))
        if depth[n] <= 0:
            cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
            cv2.putText(img, "BEHIND", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, f"{depth[n]:.1f}m", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (VIEW_W, 34), (0, 0, 0), -1)
    cv2.putText(img, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_take(take: str) -> None:
    gt_dir = GT_ROOT / take
    kp_path, cal_path = gt_dir / "keypoints_gt.json", gt_dir / "gopro_calibs.csv"
    if not kp_path.exists() or not cal_path.exists():
        print(f"{take}: missing GT files")
        return

    frame_key, joints = next(iter(json.load(open(kp_path)).items()))
    frame_idx = int(frame_key)
    G = {n: np.array([joints[n]["x"], joints[n]["y"], joints[n]["z"]])
         for n in JOINTS if n in joints and joints[n].get("num_views_for_3d", 0) > 0}
    print(f"\n=== {take}  frame {frame_idx}  ({len(G)}/12 body joints with 3D)")
    if not G:
        return
    nviews = {n: joints[n]["num_views_for_3d"] for n in G}
    print(f"    num_views_for_3d: {sorted(set(nviews.values()))}  "
          f"joints at 2 views: {sum(v <= 2 for v in nviews.values())}/{len(G)}")

    cams = _load_cameras(cal_path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for cam_id in sorted(cams):
        cam = cams[cam_id]
        frame_path = FRAMES_ROOT / take / cam_id / "frames" / f"frame_{frame_idx:06d}.jpg"
        if not frame_path.exists():
            print(f"    {cam_id}: no frame on disk ({frame_path.name})")
            continue
        img = cv2.resize(cv2.imread(str(frame_path)), (VIEW_W, VIEW_H))

        uv, depth = {}, {}
        for n, X in G.items():
            uv[n], depth[n] = _project(cam, X)
        front = [n for n in G if depth[n] > 0]
        inside = [n for n in front if 0 <= uv[n][0] < VIEW_W and 0 <= uv[n][1] < VIEW_H]
        zs = np.array([depth[n] for n in front]) if front else np.array([np.nan])

        header = (f"{take}  {cam_id}  frame {frame_idx}   front {len(front)}/{len(G)}"
                  f"   in-image {len(inside)}/{len(G)}   z {np.nanmin(zs):.1f}-{np.nanmax(zs):.1f}m")
        out = OUT_ROOT / f"{take}__{cam_id}.jpg"
        cv2.imwrite(str(out), _draw(img, uv, depth, header))
        behind = [n for n in G if depth[n] <= 0]
        print(f"    {cam_id}: front {len(front):2d}/{len(G)}  in-image {len(inside):2d}/{len(G)}"
              f"  z {np.nanmin(zs):6.1f}..{np.nanmax(zs):6.1f}m"
              + (f"  BEHIND: {','.join(behind)}" if behind else "")
              + f"  -> {out.name}")


def main() -> None:
    for take in (sys.argv[1:] or DEFAULT_TAKES):
        run_take(take)
    print(f"\nimages in {OUT_ROOT}")


if __name__ == "__main__":
    main()
