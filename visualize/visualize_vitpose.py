"""Visualize ViTPose COCO-17 keypoints overlaid on the original images.

Reads ``vitpose_kps_person_<pid>.npz`` from every camera directory under
``ghost_output_root/<scene>/cam_XX/`` and draws the COCO-17 skeleton.

Output layout (same convention as visualize_2d_kps.py)::

    renders/vitpose/<scene_name>/<frame_idx:05d>/cam_XX.jpg

Usage
-----
    pixi run python visualize/visualize_vitpose.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_data_root    /tmp/rich_train \\
        --scene             BBQ_001_guitar \\
        --frames            0 47 94 141    \\
        --out_dir           renders/vitpose \\
        --conf_threshold    0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 skeleton
# ---------------------------------------------------------------------------

_COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# (idx_a, idx_b, BGR_color)
_SKELETON: list[tuple[int, int, tuple[int, int, int]]] = [
    # face
    (0, 1,  (255, 153, 51)), (0, 2,  (255, 153, 51)),
    (1, 3,  (255, 153, 51)), (2, 4,  (255, 153, 51)),
    (3, 5,  (255, 153, 51)), (4, 6,  (255, 153, 51)),
    # torso
    (5,  6,  (255, 153, 51)),
    (5,  11, (255, 153, 51)), (6, 12, (255, 153, 51)),
    (11, 12, (255, 153, 51)),
    # left arm (green)
    (5, 7,   (0, 255, 0)), (7, 9,   (0, 255, 0)),
    # right arm (blue-orange)
    (6, 8,   (0, 128, 255)), (8, 10, (0, 128, 255)),
    # left leg (green)
    (11, 13, (0, 255, 0)), (13, 15, (0, 255, 0)),
    # right leg (blue-orange)
    (12, 14, (0, 128, 255)), (14, 16, (0, 128, 255)),
]

_PERSON_PALETTE = [
    (255,   0,   0),
    (  0, 200,   0),
    (  0,   0, 255),
    (255, 165,   0),
    (128,   0, 128),
    (  0, 200, 200),
    (220,  20, 220),
    (255, 215,   0),
]


def person_color(pid: int) -> tuple[int, int, int]:
    return _PERSON_PALETTE[pid % len(_PERSON_PALETTE)]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def find_image(rich_data_root: Path, scene: str, cam: str, frame_idx: int) -> Path | None:
    base = rich_data_root / scene / cam
    cam_idx = int(cam.split("_")[-1]) if "_" in cam else 0
    for stem in (f"{frame_idx:05d}_{cam_idx:02d}", f"{frame_idx:05d}_00"):
        for ext in (".jpg", ".jpeg", ".bmp", ".png"):
            p = base / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vitpose(
    ghost_output_root: Path,
    scene: str,
    cam: str,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{pid: {frame_idx: (kps (17,2), conf (17,))}}``.  [x, y] original pixels."""
    cam_dir  = ghost_output_root / scene / cam
    body_dir = cam_dir / "body_data"
    if not body_dir.is_dir():
        return {}
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        pid     = int(npz_path.stem.split("_")[1])
        vp_path = cam_dir / f"vitpose_kps_person_{pid}.npz"
        if not vp_path.exists():
            continue
        fi_arr = np.load(npz_path, allow_pickle=False)["frame_indices"].astype(int)
        vp     = np.load(vp_path)["keypoints"]   # (T_local, 17, 3)
        per_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for local_t, global_t in enumerate(fi_arr):
            per_frame[int(global_t)] = (vp[local_t, :, :2].copy(), vp[local_t, :, 2].copy())
        result[pid] = per_frame
    return result


# MHR70 joint subset used for visualization (SMPL-X index → MHR70 index)
_MHR70_JOINTS = {1: 9, 2: 10, 4: 11, 5: 12, 7: 13, 8: 14,
                 16: 5, 17: 6, 18: 7, 19: 8, 20: 62, 21: 41}
# (mhr70_a, mhr70_b) bone pairs for the subset above
_MHR70_SKEL = [
    (9, 10), (9, 11), (10, 12), (11, 13), (12, 14),   # hips/legs
    (5, 6),  (5, 9),  (6, 10),                         # shoulders ↔ hips
    (5, 7),  (7, 62), (6, 8),  (8, 41),                # arms
]


def load_mhr70(
    ghost_output_root: Path,
    scene: str,
    cam: str,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{pid: {frame_idx: (kps (70,2), conf (70,))}}``.  [x, y] original pixels."""
    cam_dir  = ghost_output_root / scene / cam
    body_dir = cam_dir / "body_data"
    if not body_dir.is_dir():
        return {}
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        pid = int(npz_path.stem.split("_")[1])
        d   = np.load(npz_path, allow_pickle=False)
        if "pred_keypoints_2d" not in d.files:
            continue
        fi_arr = d["frame_indices"].astype(int)
        kps    = d["pred_keypoints_2d"]              # (T_local, 70, 2)
        conf_key = "pred_joint_confidence"
        conf   = (d[conf_key] if conf_key in d.files
                  else np.ones((len(fi_arr), 70), dtype=np.float32))
        per_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for local_t, global_t in enumerate(fi_arr):
            per_frame[int(global_t)] = (kps[local_t].copy(), conf[local_t].copy())
        result[pid] = per_frame
    return result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_vitpose(
    img: np.ndarray,
    kps: np.ndarray,            # (17, 2) [x, y] in image pixels
    conf: np.ndarray,           # (17,)
    color: tuple[int, int, int],
    conf_threshold: float,
    dot_radius: int = 5,
    bone_thickness: int = 2,
) -> None:
    h, w = img.shape[:2]

    def ok(j: int) -> bool:
        if conf[j] < conf_threshold:
            return False
        x, y = kps[j]
        return 0 <= x < w and 0 <= y < h

    for ia, ib, bone_color in _SKELETON:
        if ok(ia) and ok(ib):
            xa, ya = int(round(kps[ia, 0])), int(round(kps[ia, 1]))
            xb, yb = int(round(kps[ib, 0])), int(round(kps[ib, 1]))
            cv2.line(img, (xa, ya), (xb, yb), bone_color, bone_thickness, cv2.LINE_AA)

    for j in range(17):
        if ok(j):
            x, y = int(round(kps[j, 0])), int(round(kps[j, 1]))
            # dot colored by confidence: bright = high conf
            alpha = float(conf[j])
            dot_color = tuple(int(c * alpha) for c in color)
            cv2.circle(img, (x, y), dot_radius, dot_color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), dot_radius, (255, 255, 255), 1, cv2.LINE_AA)
            # joint index label
            cv2.putText(img, str(j), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def draw_pid_label(img: np.ndarray, kps: np.ndarray, conf: np.ndarray,
                   pid: int, color: tuple[int, int, int], conf_threshold: float) -> None:
    for anchor in (5, 6, 0):
        if conf[anchor] >= conf_threshold:
            x, y = int(kps[anchor, 0]), int(kps[anchor, 1])
            cv2.putText(img, f"P{pid}", (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return


def draw_mhr70(
    img: np.ndarray,
    kps: np.ndarray,    # (70, 2) [x, y]
    conf: np.ndarray,   # (70,)
    color: tuple[int, int, int],
    conf_threshold: float,
    dot_radius: int = 5,
) -> None:
    h, w = img.shape[:2]

    def ok(j: int) -> bool:
        return (j < len(conf) and conf[j] >= conf_threshold
                and 0 <= kps[j, 0] < w and 0 <= kps[j, 1] < h)

    for ja, jb in _MHR70_SKEL:
        if ok(ja) and ok(jb):
            cv2.line(img, (int(kps[ja,0]), int(kps[ja,1])),
                     (int(kps[jb,0]), int(kps[jb,1])), color, 2, cv2.LINE_AA)

    for j in _MHR70_JOINTS.values():
        if ok(j):
            alpha = float(conf[j])
            dot_color = tuple(int(c * alpha) for c in color)
            cv2.circle(img, (int(kps[j,0]), int(kps[j,1])), dot_radius, dot_color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(kps[j,0]), int(kps[j,1])), dot_radius, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(j), (int(kps[j,0])+4, int(kps[j,1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def process_scene(
    ghost_output_root: Path,
    rich_data_root: Path,
    scene: str,
    frames: list[int] | None,
    out_dir: Path,
    conf_threshold: float,
    jpeg_quality: int,
    modality: str = "vitpose",   # "vitpose" | "mhr70" | "both"
) -> None:
    print(f"\n=== {scene} ===")
    scene_dir = ghost_output_root / scene
    if not scene_dir.is_dir():
        print(f"  Scene dir not found: {scene_dir}")
        return

    cam_names = sorted(d.name for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("cam"))

    for cam in cam_names:
        vp_data  = load_vitpose(ghost_output_root, scene, cam) if modality in ("vitpose", "both") else {}
        mhr_data = load_mhr70(ghost_output_root, scene, cam)   if modality in ("mhr70",  "both") else {}

        all_frames = sorted({fi for pd in (vp_data or mhr_data).values() for fi in pd})
        if not all_frames:
            print(f"  {cam}: no data — skipping")
            continue
        target_frames = frames if frames is not None else all_frames

        saved = missing = 0
        for frame_idx in target_frames:
            if frame_idx not in all_frames:
                continue

            img_path = find_image(rich_data_root, scene, cam, frame_idx)
            if img_path is None:
                missing += 1
                continue
            base_img = cv2.imread(str(img_path))
            if base_img is None:
                missing += 1
                continue

            if modality == "both":
                # side-by-side: left=ViTPose, right=MHR70
                left  = base_img.copy()
                right = base_img.copy()
                all_pids = sorted(set(vp_data) | set(mhr_data))
                for pid in all_pids:
                    color = person_color(pid)
                    if pid in vp_data and frame_idx in vp_data[pid]:
                        kps, conf = vp_data[pid][frame_idx]
                        draw_vitpose(left, kps, conf, color, conf_threshold)
                        draw_pid_label(left, kps, conf, pid, color, conf_threshold)
                    if pid in mhr_data and frame_idx in mhr_data[pid]:
                        kps, conf = mhr_data[pid][frame_idx]
                        draw_mhr70(right, kps, conf, color, conf_threshold)
                cv2.putText(left,  "ViTPose (RePoGen-S)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(right, "MHR70 (SAM3D)",      (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
                img = np.concatenate([left, right], axis=1)
            else:
                img = base_img.copy()
                pid_data = vp_data if modality == "vitpose" else mhr_data
                for pid, pdata in sorted(pid_data.items()):
                    if frame_idx not in pdata:
                        continue
                    kps, conf = pdata[frame_idx]
                    color = person_color(pid)
                    if modality == "vitpose":
                        draw_vitpose(img, kps, conf, color, conf_threshold)
                        draw_pid_label(img, kps, conf, pid, color, conf_threshold)
                    else:
                        draw_mhr70(img, kps, conf, color, conf_threshold)

            save_path = out_dir / scene / f"{frame_idx:05d}" / f"{cam}.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            saved += 1

        msg = f"  {cam}: saved {saved}"
        if missing:
            msg += f"  ({missing} missing images)"
        print(msg + f" → {out_dir / scene}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ViTPose COCO-17 keypoints on RICH images."
    )
    parser.add_argument("--ghost_output_root", required=True)
    parser.add_argument("--rich_data_root",    required=True)
    parser.add_argument("--scene",             required=True)
    parser.add_argument("--frames",  nargs="+", type=int, default=None,
                        help="Frame indices to render. Default: all frames.")
    parser.add_argument("--out_dir", default="renders/vitpose")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--jpeg_quality",   type=int,   default=92)
    parser.add_argument("--modality", default="vitpose",
                        choices=["vitpose", "mhr70", "both"],
                        help="vitpose=RePoGen-S only, mhr70=SAM3D only, both=side-by-side")
    args = parser.parse_args()

    process_scene(
        ghost_output_root=Path(args.ghost_output_root),
        rich_data_root=Path(args.rich_data_root),
        scene=args.scene,
        frames=args.frames,
        out_dir=Path(args.out_dir),
        conf_threshold=args.conf_threshold,
        jpeg_quality=args.jpeg_quality,
        modality=args.modality,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
