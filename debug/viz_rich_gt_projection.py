#!/usr/bin/env python
"""Check the RICH GT projection used by scripts/rich_pipeline_production.py.

Draws the GT SMPL-X body joints of one frame into every camera of a scene, using the
production script's OWN functions (load_calibration, load_gt_joints, calibration_size,
scale_K, project) — so what you see is exactly what the GT ReID matches against.
Each raw 4112x3008 frame is resized to --max-side (840 = SAM3 input) and the joints are
projected with K scaled to that size, i.e. the pipeline's pixel space.

    pixi run python debug/viz_rich_gt_projection.py \\
        --raw-root /iopsstor/scratch/cscs/tnanni/datasets/rich/test_raw \\
        --gt-root  /iopsstor/scratch/cscs/tnanni/datasets/rich \\
        --scene Gym_010_dips1 --out logs/debug_viz/rich_Gym_010_dips1.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import rich_pipeline_production as P  # noqa: E402

# SMPL-X body kinematic tree (joints 0..21)
BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
         (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18),
         (17, 19), (18, 20), (19, 21)]
COLORS = [(0, 0, 255), (0, 200, 0), (255, 80, 0), (0, 220, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--frame", type=int, default=None, help="GT frame index; default = middle")
    ap.add_argument("--max-side", type=int, default=840)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    joints = P.load_gt_joints(a.gt_root, a.split, a.scene, _REPO / "body_models",
                              _REPO / "resource" / "rich_gender.json")
    frames = sorted(set.intersection(*[set(f) for f in joints.values()]))
    fidx = a.frame if a.frame is not None else frames[len(frames) // 2]
    print(f"{a.scene}: GT subjects {sorted(joints)}  frame {fidx}")

    panels = []
    for cam_dir in sorted((a.raw_root / a.scene).glob("cam_*")):
        cam = cam_dir.name
        img_path = next((p for p in cam_dir.iterdir()
                         if P.frame_index(p.stem) == fidx), None)
        calib = P.load_calibration(a.gt_root, a.scene, cam)
        if img_path is None or calib is None:
            print(f"  {cam}: missing image or calibration"); continue
        raw = cv2.imread(str(img_path))
        H0, W0 = raw.shape[:2]
        s = a.max_side / max(W0, H0)
        W, H = int(round(W0 * s)), int(round(H0 * s))
        img = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
        cw, ch = P.calibration_size(a.raw_root, a.scene, cam, W, H)
        K = P.scale_K(calib["K"], cw, ch, W, H)
        for k, (pid, fr) in enumerate(sorted(joints.items())):
            if fidx not in fr:
                continue
            uv, front = P.project(fr[fidx], calib, K)
            col = COLORS[k % len(COLORS)]
            for i, j in BONES:
                if front[i] and front[j]:
                    cv2.line(img, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), col, 2)
            for u, v in uv[front]:
                cv2.circle(img, (int(u), int(v)), 3, col, -1)
            cv2.putText(img, f"GT {pid}", tuple(np.int32(uv[15]) + [5, -10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(img, f"{cam} {img_path.name} calib {cw}x{ch} -> {W}x{H}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(f"  {cam}: image {W0}x{H0} -> {W}x{H}  calib {cw}x{ch}  "
              f"PP at ({K[0, 2] / W:.2f}, {K[1, 2] / H:.2f}) of frame")
        panels.append(img)

    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
    grid = np.concatenate(panels, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
