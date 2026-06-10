"""Quick visual comparison: ViTPose-B (current) vs RePoGen ViTPose-S on bad frames.

Reads existing vitpose_kps_person_1.npz for the B baseline, runs the RePoGen
checkpoint live on the same frames, and writes side-by-side renders.

Usage:
    pixi run python approaches/test_repogen.py \
        --repogen-weights checkpoints/vitpose/VitPose-s_RePoGen.pth \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_vitpose import _load_vitpose_model, _run_vitpose_on_bbox

SCENE  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/ParkingLot2_008_eating1")
OFFSET = 180   # global_frame = video_frame + OFFSET
BAD_FRAMES = [523, 535, 592, 595, 672, 807]
CAMS = ["cam_00", "cam_01", "cam_02", "cam_03", "cam_04"]

COCO_SKEL = [
    (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


def draw_kps(img: np.ndarray, kps: np.ndarray, color: tuple, thr: float = 0.1) -> None:
    h, w = img.shape[:2]
    for a, b in COCO_SKEL:
        if kps[a, 2] >= thr and kps[b, 2] >= thr:
            xa, ya = int(kps[a, 0]), int(kps[a, 1])
            xb, yb = int(kps[b, 0]), int(kps[b, 1])
            if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                cv2.line(img, (xa, ya), (xb, yb), color, 2)
    for j in range(17):
        if kps[j, 2] >= thr:
            x, y = int(kps[j, 0]), int(kps[j, 1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.putText(img, str(j), (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repogen-weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="renders/repogen_vs_b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RePoGen ViTPose-S from {args.repogen_weights} ...")
    model_s, target_size = _load_vitpose_model(args.repogen_weights, "s", args.device)
    print("Model loaded.")

    for cam_name in CAMS:
        cam = SCENE / cam_name
        bd_path = cam / "body_data" / "person_1.npz"
        vp_b_path = cam / "vitpose_kps_person_1.npz"
        vid_path = cam / f"{cam_name}_segmentation_reid.mp4"
        if not bd_path.exists() or not vp_b_path.exists() or not vid_path.exists():
            continue

        bd = np.load(bd_path)
        fi_arr = bd["frame_indices"].astype(int)
        bbox_arr = bd["bbox"]
        vp_b = np.load(vp_b_path)["keypoints"]   # (T, 17, 3) existing ViTPose-B
        fi_to_lt = {int(f): i for i, f in enumerate(fi_arr)}

        cap = cv2.VideoCapture(str(vid_path))

        for gf in BAD_FRAMES:
            if gf not in fi_to_lt:
                continue
            lt = fi_to_lt[gf]
            cap.set(cv2.CAP_PROP_POS_FRAMES, gf - OFFSET)
            ret, frame = cap.read()
            if not ret:
                continue

            bbox = bbox_arr[lt]
            b = bbox.astype(int)

            # Left panel: ViTPose-B (existing saved kps)
            left = frame.copy()
            cv2.rectangle(left, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            draw_kps(left, vp_b[lt], color=(0, 255, 0))
            cv2.putText(left, "ViTPose-B (current)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Right panel: RePoGen ViTPose-S (live inference)
            right = frame.copy()
            cv2.rectangle(right, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kps_s = _run_vitpose_on_bbox(model_s, target_size, args.device, img_rgb, bbox)
            if kps_s is not None:
                draw_kps(right, kps_s, color=(0, 200, 255))
            cv2.putText(right, "RePoGen-S (new)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            combined = np.concatenate([left, right], axis=1)
            out_path = out_dir / f"{cam_name}_fr{gf}.jpg"
            cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])

        cap.release()
        print(f"  {cam_name}: done")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
