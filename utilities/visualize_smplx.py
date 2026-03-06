"""Visualise SMPL-X body estimates overlaid on video frames.

Mirrors ``visualize_segmented_reids.py`` exactly:
  - drives frame iteration from ``json_data/*.json``
  - loads frames via the same filename-match logic
  - overlays coloured translucent masks (from ``mask_data.npz``)
  - draws bounding boxes + ID labels
  - draws SMPL-X 2D skeleton capsules on top

No GPU required.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

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


# ── SMPL-X body joint skeleton (first 22 joints) ─────────────────────────────
# 0 pelvis  1 L_hip    2 R_hip    3 spine1   4 L_knee   5 R_knee
# 6 spine2  7 L_ankle  8 R_ankle  9 spine3  10 L_foot  11 R_foot
# 12 neck  13 L_collar 14 R_collar 15 head
# 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow
# 20 L_wrist    21 R_wrist
BODY_EDGES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),   # spine + head
    (0, 1), (0, 2),                                # pelvis → hips
    (12, 13), (12, 14), (13, 16), (14, 17),        # neck → collars → shoulders
    (16, 18), (17, 19), (18, 20), (19, 21),        # arms
    (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),  # legs
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_persons(body_dir: Path) -> dict[int, dict]:
    """Load all person_*.npz files; build frame_idx → row lookup."""
    persons: dict[int, dict] = {}
    for p in body_dir.glob("person_*.npz"):
        try:
            with np.load(str(p)) as f:
                data = dict(f)
            pid = int(p.stem.split("_")[-1])
            if "frame_indices" not in data or "pred_keypoints_2d" not in data:
                continue
            data["_frame_to_row"] = {int(v): i for i, v in enumerate(data["frame_indices"])}
            persons[pid] = data
        except Exception:
            continue
    return persons


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _capsule(img: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             radius: int, color: tuple) -> None:
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
             color, radius * 2 + 1, cv2.LINE_AA)
    cv2.circle(img, tuple(p1.astype(int)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(p2.astype(int)), radius, color, -1, cv2.LINE_AA)


def _draw_skeleton(overlay: np.ndarray, joints2d: np.ndarray,
                   bbox: np.ndarray, color: tuple) -> None:
    """Draw capsule skeleton + head circle onto *overlay* (in-place)."""
    body_h = max(1, float(bbox[3] - bbox[1]))
    radius = max(3, int(body_h / 24))

    for a, b in BODY_EDGES:
        if a < len(joints2d) and b < len(joints2d):
            _capsule(overlay, joints2d[a], joints2d[b], radius, color)

    for x, y in joints2d:
        cv2.circle(overlay, (int(x), int(y)), max(2, radius // 2),
                   color, -1, cv2.LINE_AA)

    if len(joints2d) > 15:
        neck  = joints2d[12]
        head  = joints2d[15]
        head_r = max(radius, int(np.linalg.norm(head - neck) * 0.6))
        cv2.circle(overlay, tuple(head.astype(int)), head_r, color, -1, cv2.LINE_AA)


# ── Core visualize function ───────────────────────────────────────────────────

def visualize_smplx(
    video_dir: Path,
    fps: int = 30,
    frames_dir: Path | None = None,
) -> Path:
    """Render an mp4 with coloured masks, bboxes, labels, and SMPL-X skeleton.

    Mirrors ``visualize_reid`` from ``visualize_segmented_reids.py``.

    Parameters
    ----------
    video_dir : Path
        Root output directory for one video (contains mask_data.npz,
        json_data/, body_data/).
    fps : int
        Frame rate of the output mp4.
    frames_dir : Path | None
        Directory with extracted JPEG frames.  Falls back to
        ``video_dir/frames/`` when not provided.
    """
    video_dir = Path(video_dir)
    frame_dir = frames_dir if frames_dir is not None else video_dir / "frames"
    npz_path  = video_dir / "mask_data.npz"
    json_dir  = video_dir / "json_data"
    body_dir  = video_dir / "body_data"

    # ── Load person body data ─────────────────────────────────────────────────
    persons = _load_persons(body_dir)
    if not persons:
        print(f"  No person_*.npz files found in {body_dir}, skipping")
        return video_dir / "smplx_video.mp4"

    # ── Collect sorted frame list (same as reid viz) ──────────────────────────
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON metadata files found in {json_dir}")

    if not npz_path.exists():
        raise FileNotFoundError(f"mask_data.npz not found at {npz_path}")

    # ── Determine output dimensions from first readable frame ─────────────────
    _FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def _find_frame(fi_str: str) -> Path | None:
        for ext in _FRAME_EXTS:
            p = frame_dir / f"{fi_str}{ext}"
            if p.exists():
                return p
        return None

    H, W = 0, 0
    for jf in json_files:
        fi_str = jf.stem.replace("mask_", "")
        p = _find_frame(fi_str)
        if p is not None:
            sample = cv2.imread(str(p))
            if sample is not None:
                H, W = sample.shape[:2]
                break
    if H == 0:
        raise FileNotFoundError(f"No readable frames found in {frame_dir}")

    out_path = video_dir / "smplx_video.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H),
    )

    with zipfile.ZipFile(str(npz_path), "r") as zf:
        npz_keys = set(zf.namelist())

        for json_path in tqdm(json_files, desc=f"Rendering {video_dir.name}", leave=False):
            fi_str = json_path.stem.replace("mask_", "")
            frame_path = _find_frame(fi_str)
            if frame_path is None:
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            with open(json_path) as f:
                labels: dict = json.load(f).get("labels", {})

            # frame_idx for body data lookup (same split logic as extraction)
            frame_idx = int(fi_str.split("_")[0])

            # Load per-frame mask
            mask_key = json_path.stem + ".npy"
            mask_img: np.ndarray | None = None
            if mask_key in npz_keys:
                with zf.open(mask_key) as mf:
                    mask_img = np.load(io.BytesIO(mf.read()))

            # 1. Coloured mask overlay
            if mask_img is not None:
                overlay = frame.copy()
                for str_id in labels:
                    canon_id = int(str_id)
                    person_mask = mask_img == canon_id
                    if not person_mask.any():
                        continue
                    if int(person_mask.sum()) > 0.80 * H * W:
                        continue
                    color = np.array(_color(canon_id), dtype=np.float32)
                    overlay[person_mask] = (
                        0.5 * overlay[person_mask] + 0.5 * color
                    ).astype(np.uint8)
                frame = overlay

            # 2. SMPL-X skeleton overlay (semi-transparent)
            skel_overlay = frame.copy()
            for pid, data in persons.items():
                row = data["_frame_to_row"].get(frame_idx)
                if row is None:
                    continue
                joints2d = data["pred_keypoints_2d"][row]   # (≥22, 2) pixel coords
                bbox     = data["bbox"][row]                 # [x1, y1, x2, y2]
                _draw_skeleton(skel_overlay, joints2d[:22], bbox, _color(pid))
            frame = cv2.addWeighted(frame, 0.45, skel_overlay, 0.55, 0)

            # 3. Bounding boxes and ID labels
            for str_id, info in labels.items():
                canon_id = int(str_id)
                color    = _color(canon_id)
                x1, y1   = int(info["x1"]), int(info["y1"])
                x2, y2   = int(info["x2"]), int(info["y2"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"P{canon_id}"
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness  = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                ty = max(y1 - 6, th + 4)
                cv2.rectangle(
                    frame,
                    (x1, ty - th - 4), (x1 + tw + 6, ty + 2),
                    color, cv2.FILLED,
                )
                cv2.putText(
                    frame, label,
                    (x1 + 3, ty - 1),
                    font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA,
                )

            writer.write(frame)

    writer.release()
    print(f"  SMPL-X visualisation saved: {out_path}")
    return out_path


# ── Scene-level helper ────────────────────────────────────────────────────────

def visualize_scene(
    scene_output_dir: Path,
    data_root: Path,
    fps: int = 30,
) -> None:
    """Run visualize_smplx() for every camera in a scene output directory."""
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
        cam_id = cam_dir.name

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
            visualize_smplx(cam_dir, fps=fps, frames_dir=frames_dir)
        except Exception as exc:
            print(f"  {cam_id}: failed — {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise SMPL-X body estimates overlaid on video frames."
    )
    parser.add_argument("--scene-dir",  type=Path, help="Scene output directory (e.g. test_outputs/.../BBQ_001_guitar)")
    parser.add_argument("--data-root",  type=Path, help="Data root with source frames (required with --scene-dir)")
    parser.add_argument("--video-dir",  type=Path, help="Single-camera output directory (single-camera mode)")
    parser.add_argument("--frames-dir", type=Path, help="Source frames directory (overrides auto-detection)")
    parser.add_argument("--fps",        type=int,  default=30)
    args = parser.parse_args()

    if args.scene_dir is not None:
        if args.data_root is None:
            parser.error("--data-root is required with --scene-dir")
        visualize_scene(args.scene_dir, args.data_root, args.fps)
    elif args.video_dir is not None:
        visualize_smplx(args.video_dir, fps=args.fps, frames_dir=args.frames_dir)
    else:
        parser.error("Provide --scene-dir or --video-dir")


if __name__ == "__main__":
    main()
