"""Visualise SMPL-X body estimates overlaid on video frames.

Reads per-person `body_data/person_<id>.npz` files and overlays the
reconstructed body on the original frames.  Rendering uses `pred_keypoints_2d`
(already in pixel coordinates) and draws OpenPose-style coloured capsule limbs
with alpha blending — similar in style to visualize_segmented_reids.py.

No GPU required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm


# ── Colour palette (BGR, one per person, cycles) ─────────────────────────────
_PALETTE: list[tuple[int, int, int]] = [
    ( 60,  80, 220),  # red
    ( 60, 200,  60),  # green
    (220,  80,  60),  # blue
    ( 40, 210, 210),  # yellow
    (210,  60, 210),  # magenta
    (210, 210,  40),  # cyan
    ( 40, 140, 220),  # orange
    (160,  60, 160),  # purple
]


def _color(person_id: int) -> tuple[int, int, int]:
    return _PALETTE[person_id % len(_PALETTE)]


# ── SMPL-X 22 body joint indices ─────────────────────────────────────────────
# 0 pelvis  1 L_hip    2 R_hip    3 spine1
# 4 L_knee  5 R_knee   6 spine2   7 L_ankle
# 8 R_ankle 9 spine3  10 L_foot  11 R_foot
# 12 neck   13 L_collar 14 R_collar 15 head
# 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow
# 20 L_wrist    21 R_wrist

BODY_EDGES = [
    # spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # pelvis → hips
    (0, 1), (0, 2),
    # neck → collars → shoulders
    (12, 13), (12, 14), (13, 16), (14, 17),
    # arms
    (16, 18), (17, 19), (18, 20), (19, 21),
    # legs
    (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_persons(body_dir: Path) -> Dict[int, dict]:
    body_dir = Path(body_dir)
    persons: Dict[int, dict] = {}
    for p in body_dir.glob("person_*.npz"):
        try:
            with np.load(str(p)) as f:
                data = dict(f)
            pid = int(p.stem.split("_")[-1])
            if "frame_indices" not in data or "pred_keypoints_2d" not in data:
                continue
            # reverse lookup: frame_idx value → row in arrays
            data["_frame_to_row"] = {int(v): i for i, v in enumerate(data["frame_indices"])}
            persons[pid] = data
        except Exception:
            continue
    return persons


def _parse_frame_num(path: Path) -> int:
    """Parse the frame index from a filename stem.

    Uses the same logic as parameters_extraction.py: ``int(stem)``.
    Python's int() treats underscores as digit separators, so RICH-style
    filenames like '00042_00' → int('00042_00') = 4200, which matches the
    value stored in frame_indices (the mask JSON stem goes through the same
    int() call).  EgoExo-style '000042' → 42 as expected.
    """
    try:
        return int(path.stem.split("_")[0])
    except ValueError:
        raise ValueError(f"Cannot parse frame number from {path.name}")


# ── Rendering ─────────────────────────────────────────────────────────────────

def _capsule(img: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             radius: int, color: tuple) -> None:
    """Draw a filled capsule (thick rounded line) between two 2D points."""
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
             color, radius * 2 + 1, cv2.LINE_AA)
    cv2.circle(img, tuple(p1.astype(int)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(p2.astype(int)), radius, color, -1, cv2.LINE_AA)


def draw_body(frame_bgr: np.ndarray,
              persons_this_frame: List[tuple[int, np.ndarray, np.ndarray]],
              alpha: float = 0.55) -> np.ndarray:
    """Overlay body estimates on *frame_bgr* (in-place copy).

    Parameters
    ----------
    persons_this_frame : list of (person_id, joints2d_22, bbox)
        joints2d_22  : (22, 2) – first 22 SMPL-X body joints in pixel coords
        bbox         : (4,)  – [x1, y1, x2, y2]
    alpha : float
        Opacity of the body overlay (0 = invisible, 1 = fully opaque).
    """
    overlay = frame_bgr.copy()

    for pid, joints2d, bbox in persons_this_frame:
        color = _color(pid)
        body_h = max(1, float(bbox[3] - bbox[1]))
        radius = max(4, int(body_h / 22))

        # ── limb capsules ────────────────────────────────────────────────────
        for a, b in BODY_EDGES:
            if a < len(joints2d) and b < len(joints2d):
                _capsule(overlay, joints2d[a], joints2d[b], radius, color)

        # ── joint dots ───────────────────────────────────────────────────────
        for j, (x, y) in enumerate(joints2d):
            cv2.circle(overlay, (int(x), int(y)), max(3, radius // 2),
                       color, -1, cv2.LINE_AA)

        # ── head circle (joint 15 = head) ────────────────────────────────────
        if len(joints2d) > 15:
            neck  = joints2d[12]
            head  = joints2d[15]
            head_r = max(radius, int(np.linalg.norm(head - neck) * 0.6))
            cv2.circle(overlay, tuple(head.astype(int)), head_r, color, -1, cv2.LINE_AA)

    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)


# ── Core visualize function ───────────────────────────────────────────────────

def visualize(
    body_dir: Path,
    frames_dir: Path,
    out_video: Path,
    n_frames: int | None = None,
    fps: int = 30,
) -> None:
    persons = load_persons(body_dir)
    if not persons:
        raise RuntimeError(f"No person_*.npz files found in {body_dir}")

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    if not frame_paths:
        raise RuntimeError(f"No images found in {frames_dir}")

    writer = None
    for loop_i, fp in enumerate(tqdm(frame_paths, desc=f"  {Path(frames_dir).parent.name}", leave=False)):
        if n_frames is not None and loop_i >= n_frames:
            break

        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            continue
        h, w = frame_bgr.shape[:2]

        if writer is None and out_video is not None:
            out_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (w, h))

        try:
            frame_num = _parse_frame_num(fp)
        except ValueError:
            frame_num = loop_i

        persons_this_frame = []
        for pid, data in persons.items():
            row = data["_frame_to_row"].get(frame_num)
            if row is None:
                continue
            joints2d = data["pred_keypoints_2d"][row]   # (70, 2) pixel coords
            bbox     = data["bbox"][row]                 # [x1, y1, x2, y2]
            persons_this_frame.append((pid, joints2d[:22], bbox))

        out = draw_body(frame_bgr, persons_this_frame)
        if writer is not None:
            writer.write(out)

    if writer is not None:
        writer.release()


# ── Scene-level helper ────────────────────────────────────────────────────────

def visualize_scene(
    scene_output_dir: Path,
    data_root: Path,
    n_frames: int | None = None,
    fps: int = 30,
) -> None:
    """Run visualize() for every camera in a scene output directory.

    Saves ``smplx_video.mp4`` inside each camera subdirectory.
    Expected layout mirrors visualize_segmented_reids.py conventions::

        scene_output_dir/<cam_id>/body_data/person_*.npz
        data_root/<scene>/<cam_id>/         (frames or frames/ subdir)
    """
    scene_output_dir = Path(scene_output_dir)
    data_root        = Path(data_root)
    scene_name       = scene_output_dir.name
    scene_data       = data_root / scene_name

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    cam_dirs = sorted(
        d for d in scene_output_dir.iterdir()
        if d.is_dir() and (d / "body_data").is_dir()
    )
    if not cam_dirs:
        print(f"No camera directories with body_data/ found in {scene_output_dir}")
        return

    for cam_dir in cam_dirs:
        cam_id    = cam_dir.name
        body_dir  = cam_dir / "body_data"
        out_video = cam_dir / "smplx_video.mp4"

        # Find source frames
        frames_dir = None
        for candidate in [scene_data / cam_id / "frames", scene_data / cam_id]:
            if candidate.is_dir() and any(
                p.suffix.lower() in _IMG_EXTS for p in candidate.iterdir()
            ):
                frames_dir = candidate
                break

        if frames_dir is None:
            print(f"  {cam_id}: no source frames found, skipping")
            continue

        print(f"  {cam_id}: generating smplx_video.mp4 …")
        try:
            visualize(body_dir, frames_dir, out_video, n_frames, fps)
            print(f"  {cam_id}: saved → {out_video}")
        except Exception as exc:
            print(f"  {cam_id}: failed — {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise SMPL-X body estimates overlaid on video frames."
    )
    parser.add_argument("--scene-dir",  type=Path, help="Scene output directory (e.g. test_outputs/.../BBQ_001_guitar)")
    parser.add_argument("--data-root",  type=Path, help="Data root with source frames (required with --scene-dir)")
    parser.add_argument("--frames-dir", type=Path, help="Frames directory for single-camera mode")
    parser.add_argument("--body-dir",   type=Path, help="body_data/ directory for single-camera mode")
    parser.add_argument("--out-video",  type=Path, help="Output mp4 path (single-camera mode)")
    parser.add_argument("--frames",     type=int,  help="Max number of frames to process")
    parser.add_argument("--fps",        type=int,  default=30)
    args = parser.parse_args()

    if args.scene_dir is not None:
        if args.data_root is None:
            parser.error("--data-root is required with --scene-dir")
        visualize_scene(args.scene_dir, args.data_root, args.frames, args.fps)
    else:
        if args.body_dir is None or args.frames_dir is None:
            parser.error("--body-dir and --frames-dir are required in single-camera mode")
        visualize(args.body_dir, args.frames_dir, args.out_video, args.frames, args.fps)


if __name__ == "__main__":
    main()
