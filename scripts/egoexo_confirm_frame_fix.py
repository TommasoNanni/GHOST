"""Confirm the EgoExo4D wrong-frame fix, no ground-truth-image needed.

For each take: triangulate the GT subject's 3D from the pipeline's own
``pred_keypoints_2d`` (across the gopro GT cameras) and compare to GT-3D.

  offset  = mean ||triangulated3D - GT3D||   (cm)
  residual= mean multi-view reprojection residual (px) — sanity that the cams
            agree with each other (should be a few px)

Before the fix, dynamic takes read tens of cm (soccer06_3 ~59cm) because the
extracted image was a keyframe, not the annotated frame. After re-extracting the
exact frame and rerunning the pipeline, the offset should collapse to the
static-take floor (~4-10cm = triangulation + GT noise). Run it on a take once its
pipeline outputs have been regenerated:

    pixi run python scripts/egoexo_confirm_frame_fix.py cmu_soccer06_3 cmu_soccer12_2

Self-contained (copies the triangulation logic rather than importing eval code).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

GT_ROOT   = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt")
GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d")

# GT COCO joint name -> MHR70 index (pred_keypoints_2d ordering), from placer.py.
GT_TO_MHR70 = {
    "left-shoulder": 5, "right-shoulder": 6, "left-elbow": 7, "right-elbow": 8,
    "left-wrist": 62, "right-wrist": 41, "left-hip": 9, "right-hip": 10,
    "left-knee": 11, "right-knee": 12, "left-ankle": 13, "right-ankle": 14,
}
_MATCH_THRESH_PX = 80.0   # a camera's best person must be within this to be trusted
_GOOD_CM = 15.0           # offset below this = frame is correct (eval-ready)


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
        sx, sy = 1440 / W, 810 / H
        P = (np.diag([sx, sy, 1.0]) @ Knew) @ np.hstack([Rwc.T, (-Rwc.T @ t).reshape(3, 1)])
        cams[row["cam_uid"]] = {"P": P, "Knew": Knew, "Rwc": Rwc, "t": t, "sx": sx, "sy": sy}
    return cams


def _project(cam: dict, X: np.ndarray) -> np.ndarray:
    Xc = cam["Rwc"].T @ (X - cam["t"])
    if Xc[2] <= 1e-6:
        return np.array([1e9, 1e9])
    return np.array([(cam["Knew"][0, 0] * Xc[0] / Xc[2] + cam["Knew"][0, 2]) * cam["sx"],
                     (cam["Knew"][1, 1] * Xc[1] / Xc[2] + cam["Knew"][1, 2]) * cam["sy"]])


def _dlt(cams: list[dict], uvs: list[np.ndarray]) -> np.ndarray:
    A = []
    for cam, uv in zip(cams, uvs):
        P = cam["P"]
        A += [uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def confirm(take: str) -> str:
    gt_dir, ghost_dir = GT_ROOT / take, GHOST_ROOT / take
    kp_path, cal_path = gt_dir / "keypoints_gt.json", gt_dir / "gopro_calibs.csv"
    if not kp_path.exists() or not cal_path.exists():
        return f"{take}: no GT files"
    if not ghost_dir.exists():
        return f"{take}: no pipeline output (rerun first)"

    frame_key, joints = next(iter(json.load(open(kp_path)).items()))
    frame_idx = int(frame_key)
    G = {n: np.array([joints[n]["x"], joints[n]["y"], joints[n]["z"]])
         for n in GT_TO_MHR70 if n in joints and joints[n].get("num_views_for_3d", 0) > 0}
    if len(G) < 8:
        return f"{take}: hand/partial GT ({len(G)} body joints)"

    cams = _load_cameras(cal_path)

    # best-matching detected person per camera (nearest to GT reprojection)
    picks: dict[str, tuple[float, dict]] = {}
    for cam_id, cam in cams.items():
        body_dir = ghost_dir / cam_id / "body_data"
        if not body_dir.exists():
            continue
        gt_uv = {n: _project(cam, G[n]) for n in G}
        best = None
        for npz in sorted(body_dir.glob("person_*.npz")):
            d = np.load(npz, allow_pickle=False)
            idx = np.where(d["frame_indices"].astype(int) == frame_idx)[0]
            if not idx.size:
                continue
            k = d["pred_keypoints_2d"][idx[0]]
            err = np.mean([np.linalg.norm(gt_uv[n] - k[GT_TO_MHR70[n]]) for n in G])
            if best is None or err < best[0]:
                best = (err, {n: k[GT_TO_MHR70[n]] for n in G})
        if best:
            picks[cam_id] = best

    good = [c for c in picks if picks[c][0] < _MATCH_THRESH_PX]
    if len(good) < 2:
        return f"{take}: <2 confident cams (matches {[(c, round(picks[c][0])) for c in picks]})"

    res, dg = [], []
    for n in G:
        uvs = [picks[c][1][n] for c in good]
        cs = [cams[c] for c in good]
        X = _dlt(cs, uvs)
        res.append(np.mean([np.linalg.norm(_project(cams[c], X) - picks[c][1][n]) for c in good]))
        dg.append(np.linalg.norm(X - G[n]) * 100)
    offset = float(np.mean(dg))
    verdict = "OK (eval-ready)" if offset < _GOOD_CM else "STILL WRONG-FRAME"
    return (f"{take:30s} cams={len(good)} residual={np.mean(res):4.0f}px  "
            f"offset={offset:5.0f}cm  -> {verdict}")


def main():
    takes = sys.argv[1:]
    if not takes:
        print("usage: egoexo_confirm_frame_fix.py <take> [<take> ...]")
        raise SystemExit(1)
    for t in takes:
        print(confirm(t))


if __name__ == "__main__":
    main()
