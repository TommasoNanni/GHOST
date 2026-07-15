#!/usr/bin/env python
"""Render EgoHumans GT **2D** keypoint overlays with the correct aria IDs.

Sibling of ``render_egohumans_gt.py`` but consumes the ground-truth 2D
annotations (``processed_data/poses2d/<cam>/rgb/<frame>.npy``) instead of
projecting the 3D SMPL fit. No COLMAP / 3D needed — the keypoints already live
in the raw fisheye image pixel space.

Each ``poses2d/<cam>/rgb/<frame>.npy`` is an object array of per-person dicts:
    {'bbox': [x1,y1,x2,y2,score], 'human_name': 'aria01', 'human_id': 0,
     'keypoints': (17,3) COCO [x,y,conf], 'is_valid': bool, ...}

Colours are keyed by aria ID and fixed across all scenes, matching the 3D
renderer (aria01=red, aria02=green, aria03=blue, aria04=yellow).

Example:
    pixi run python utilities/render_egohumans_gt_2d.py \
        --cam_ready_root <mnt>/media/.../camera_ready/06_badminton \
        --activity 06_badminton --out_dir figures/egohumans_gt_ids/06_badminton_2d
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

# Curated exo cameras per activity (same set the 3D renderer uses).
KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}

# Fixed BGR colours per aria ID (identical to render_egohumans_gt.py).
ID_COLORS: dict[str, tuple[int, int, int]] = {
    "aria01": (60, 60, 230),    # red
    "aria02": (60, 200, 60),    # green
    "aria03": (230, 130, 40),   # blue
    "aria04": (40, 220, 220),   # yellow
}
DEFAULT_COLOR = (200, 200, 200)

# COCO-17 skeleton edges.
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]
KP_CONF_THR = 0.3


def _load_people(npy_path: Path) -> list[dict]:
    """Return the list of valid person dicts in one poses2d frame file."""
    if not npy_path.exists():
        return []
    arr = np.load(str(npy_path), allow_pickle=True)
    out = []
    for person in arr:
        if not isinstance(person, dict):
            continue
        if not person.get("is_valid", True):
            continue
        if "keypoints" not in person or "human_name" not in person:
            continue
        out.append(person)
    return out


def _frame_indices(rgb_dir: Path) -> list[int]:
    idxs = []
    for p in rgb_dir.glob("*.npy"):
        try:
            idxs.append(int(p.stem))
        except ValueError:
            continue
    return sorted(idxs)


def _draw_person(img, person, color) -> tuple[int, int]:
    """Draw bbox + COCO skeleton for one person. Returns label anchor (x, y)."""
    kps = np.asarray(person["keypoints"], np.float32)  # (17,3)
    bbox = np.asarray(person.get("bbox", []), np.float32)

    # bounding box
    if bbox.shape[0] >= 4:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    vis = kps[:, 2] > KP_CONF_THR
    # skeleton edges
    for a, b in COCO_EDGES:
        if vis[a] and vis[b]:
            pa = (int(round(kps[a, 0])), int(round(kps[a, 1])))
            pb = (int(round(kps[b, 0])), int(round(kps[b, 1])))
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    # joints
    for j in range(kps.shape[0]):
        if vis[j]:
            cv2.circle(img, (int(round(kps[j, 0])), int(round(kps[j, 1]))),
                       4, color, -1, cv2.LINE_AA)

    # label anchor: top of the bbox (fall back to highest visible joint)
    if bbox.shape[0] >= 4:
        return int(round((bbox[0] + bbox[2]) / 2)), int(round(bbox[1]))
    if vis.any():
        top = kps[vis][np.argmin(kps[vis][:, 1])]
        return int(round(top[0])), int(round(top[1]))
    return 0, 24


def render_scene(seq_dir: Path, cam_list: list[str], out_dir: Path) -> str:
    poses2d = seq_dir / "processed_data" / "poses2d"
    if not poses2d.is_dir():
        return f"SKIP {seq_dir.name}: no poses2d dir"

    # For each cam gather frame list; target the middle frame of the richest cam.
    cam_frames = {}
    for cam in cam_list:
        rgb = poses2d / cam / "rgb"
        if rgb.is_dir():
            fr = _frame_indices(rgb)
            if fr:
                cam_frames[cam] = fr
    if not cam_frames:
        return f"SKIP {seq_dir.name}: no poses2d for cams {cam_list}"

    ref_frames = max(cam_frames.values(), key=len)
    mid = ref_frames[len(ref_frames) // 2]

    # Around the middle, find the (cam, frame) with the most valid people.
    best = None  # (nvalid, -dist_to_mid, cam, frame, people)
    for cam, frames in cam_frames.items():
        # sample up to ~60 frames spread across the sequence, biased to middle
        step = max(1, len(frames) // 60)
        cands = sorted(set(frames[::step] + [f for f in frames
                       if abs(f - mid) <= 15]))
        for f in cands:
            people = _load_people(poses2d / cam / "rgb" / f"{f:05d}.npy")
            if not people:
                continue
            key = (len(people), -abs(f - mid))
            if best is None or key > best[0]:
                best = (key, cam, f, people)
    if best is None:
        return f"SKIP {seq_dir.name}: no valid poses2d people among {cam_list}"

    _, cam_name, frame_idx, people = best
    img_path = seq_dir / "exo" / cam_name / "images" / f"{frame_idx:05d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return f"SKIP {seq_dir.name}: unreadable {img_path}"
    H_img, W_img = img.shape[:2]

    labels = []
    names_present = sorted({p["human_name"] for p in people})
    for person in sorted(people, key=lambda p: p["human_name"]):
        name = person["human_name"]
        color = ID_COLORS.get(name, DEFAULT_COLOR)
        ax, ay = _draw_person(img, person, color)
        labels.append((name, (ax, ay), color))

    for name, (ax, ay), color in labels:
        ay = max(ay - 12, 24)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x0 = int(np.clip(ax - tw // 2, 2, W_img - tw - 2))
        cv2.rectangle(img, (x0 - 4, ay - th - 6), (x0 + tw + 4, ay + 6), color, -1)
        cv2.putText(img, name, (x0, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    hdr = (f"{seq_dir.name}  {cam_name}  frame {frame_idx:05d}  "
           f"({len(names_present)} GT people)  [2D]")
    cv2.rectangle(img, (0, 0), (W_img, 34), (0, 0, 0), -1)
    cv2.putText(img, hdr, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    ly = 60
    for name in names_present:
        color = ID_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(img, (8, ly - 14), (28, ly + 4), color, -1)
        cv2.putText(img, name, (34, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        ly += 26

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_dir.name}_{cam_name}_2d.jpg"
    cv2.imwrite(str(out_path), img)
    names = "+".join(names_present)
    return (f"OK {seq_dir.name}: {cam_name} frame {frame_idx:05d}, "
            f"{len(names_present)} people [{names}] -> {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--cam_ready_root", default=None,
                    help="camera_ready/<activity> directory directly")
    ap.add_argument("--activity", required=True, help="e.g. 06_badminton")
    ap.add_argument("--cam", default=None, help="Force a single exo camera")
    ap.add_argument("--seq", default=None, help="Render only this sequence")
    ap.add_argument("--out_dir", default="figures/egohumans_gt_ids_2d")
    args = ap.parse_args()

    if args.cam:
        cam_list = [args.cam]
    elif args.activity in KEEP_CAMS:
        cam_list = KEEP_CAMS[args.activity]
    else:
        ap.error(f"no KEEP_CAMS for {args.activity}; pass --cam explicitly")

    if args.cam_ready_root:
        cam_ready = Path(args.cam_ready_root)
    elif args.data_root:
        cam_ready = Path(args.data_root) / args.activity / INNER / args.activity
    else:
        ap.error("provide --data_root or --cam_ready_root")

    if not cam_ready.is_dir():
        ap.error(f"camera_ready dir not found: {cam_ready}")

    out_dir = Path(args.out_dir)
    seqs = [cam_ready / args.seq] if args.seq else sorted(
        d for d in cam_ready.iterdir() if d.is_dir())

    for seq_dir in seqs:
        try:
            print(render_scene(seq_dir, cam_list, out_dir), flush=True)
        except Exception as e:
            print(f"ERR {seq_dir.name}: {e}", flush=True)


if __name__ == "__main__":
    main()
