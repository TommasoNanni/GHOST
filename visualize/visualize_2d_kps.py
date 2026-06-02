"""Visualize predicted 2D keypoints overlaid on the original images.

For each scene, reads ``body_data/person_*.npz`` from every camera directory
under ``ghost_output_root/<scene>/cam_XX/`` and draws the MHR-70 skeleton on
the corresponding source images.

Output layout::

    renders/2d_kps/<scene_name>/<frame_idx:05d>/cam_XX.jpg

Usage
-----
    pixi run python visualize/visualize_2d_kps.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --rich_data_root    /tmp/rich_train \\
        --scene             Gym_010_lunge2  \\
        --out_dir           renders/2d_kps  \\
        --max_frames        50

Arguments
---------
ghost_output_root : str
    Root of the ghost pipeline outputs (contains scene sub-directories).
rich_data_root : str
    Root of the RICH dataset (mounted squashfs or plain directory).
    Frames are looked up as ``<rich_data_root>/<scene>/cam_XX/<idx:05d>_00.{jpg,bmp}``.
scene : str, optional
    Process only this scene.  Default: all scenes found in ghost_output_root.
out_dir : str
    Output root directory.  Default: ``renders/2d_kps``.
max_frames : int, optional
    Limit the number of frames processed per camera (useful for quick checks).
conf_threshold : float
    Minimum joint confidence to render a keypoint (0 = show all).  Default 0.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MHR-70 skeleton definition
# ---------------------------------------------------------------------------
# Index → joint name (from mhr70.py original_keypoint_info)
JOINT_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_hip",
    10: "right_hip",
    11: "left_knee",
    12: "right_knee",
    13: "left_ankle",
    14: "right_ankle",
    15: "left_big_toe_tip",
    16: "left_small_toe_tip",
    17: "left_heel",
    18: "right_big_toe_tip",
    19: "right_small_toe_tip",
    20: "right_heel",
    # right hand (21-41)
    21: "right_thumb_tip",
    22: "right_thumb_first_joint",
    23: "right_thumb_second_joint",
    24: "right_thumb_third_joint",
    25: "right_index_tip",
    26: "right_index_first_joint",
    27: "right_index_second_joint",
    28: "right_index_third_joint",
    29: "right_middle_tip",
    30: "right_middle_first_joint",
    31: "right_middle_second_joint",
    32: "right_middle_third_joint",
    33: "right_ring_tip",
    34: "right_ring_first_joint",
    35: "right_ring_second_joint",
    36: "right_ring_third_joint",
    37: "right_pinky_tip",
    38: "right_pinky_first_joint",
    39: "right_pinky_second_joint",
    40: "right_pinky_third_joint",
    41: "right_wrist",
    # left hand (42-62)
    42: "left_thumb_tip",
    43: "left_thumb_first_joint",
    44: "left_thumb_second_joint",
    45: "left_thumb_third_joint",
    46: "left_index_tip",
    47: "left_index_first_joint",
    48: "left_index_second_joint",
    49: "left_index_third_joint",
    50: "left_middle_tip",
    51: "left_middle_first_joint",
    52: "left_middle_second_joint",
    53: "left_middle_third_joint",
    54: "left_ring_tip",
    55: "left_ring_first_joint",
    56: "left_ring_second_joint",
    57: "left_ring_third_joint",
    58: "left_pinky_tip",
    59: "left_pinky_first_joint",
    60: "left_pinky_second_joint",
    61: "left_pinky_third_joint",
    62: "left_wrist",
    # extras (63-69)
    63: "left_olecranon",
    64: "right_olecranon",
    65: "left_cubital_fossa",
    66: "right_cubital_fossa",
    67: "left_acromion",
    68: "right_acromion",
    69: "neck",
}

# Name → index lookup (built once)
_NAME_TO_IDX: dict[str, int] = {v: k for k, v in JOINT_NAMES.items()}

# skeleton_info from mhr70.py translated to (idx_a, idx_b, BGR_color)
# fmt: off
_SKEL_RAW: list[tuple[str, str, tuple[int, int, int]]] = [
    # body
    ("left_ankle",    "left_knee",      (0, 255, 0)),
    ("left_knee",     "left_hip",       (0, 255, 0)),
    ("right_ankle",   "right_knee",     (0, 128, 255)),
    ("right_knee",    "right_hip",      (0, 128, 255)),
    ("left_hip",      "right_hip",      (255, 153, 51)),
    ("left_shoulder", "left_hip",       (255, 153, 51)),
    ("right_shoulder","right_hip",      (255, 153, 51)),
    ("left_shoulder", "right_shoulder", (255, 153, 51)),
    ("left_shoulder", "left_elbow",     (0, 255, 0)),
    ("right_shoulder","right_elbow",    (0, 128, 255)),
    ("left_elbow",    "left_wrist",     (0, 255, 0)),
    ("right_elbow",   "right_wrist",    (0, 128, 255)),
    # face
    ("left_eye",      "right_eye",      (255, 153, 51)),
    ("nose",          "left_eye",       (255, 153, 51)),
    ("nose",          "right_eye",      (255, 153, 51)),
    ("left_eye",      "left_ear",       (255, 153, 51)),
    ("right_eye",     "right_ear",      (255, 153, 51)),
    ("left_ear",      "left_shoulder",  (255, 153, 51)),
    ("right_ear",     "right_shoulder", (255, 153, 51)),
    # feet
    ("left_ankle",  "left_big_toe_tip",   (0, 255, 0)),
    ("left_ankle",  "left_small_toe_tip", (0, 255, 0)),
    ("left_ankle",  "left_heel",          (0, 255, 0)),
    ("right_ankle", "right_big_toe_tip",  (0, 128, 255)),
    ("right_ankle", "right_small_toe_tip",(0, 128, 255)),
    ("right_ankle", "right_heel",         (0, 128, 255)),
    # left hand
    ("left_wrist", "left_thumb_third_joint",  (0, 128, 255)),
    ("left_thumb_third_joint","left_thumb_second_joint",(0,128,255)),
    ("left_thumb_second_joint","left_thumb_first_joint",(0,128,255)),
    ("left_thumb_first_joint","left_thumb_tip",(0,128,255)),
    ("left_wrist","left_index_third_joint",    (255,153,255)),
    ("left_index_third_joint","left_index_second_joint",(255,153,255)),
    ("left_index_second_joint","left_index_first_joint",(255,153,255)),
    ("left_index_first_joint","left_index_tip",(255,153,255)),
    ("left_wrist","left_middle_third_joint",   (255,178,102)),
    ("left_middle_third_joint","left_middle_second_joint",(255,178,102)),
    ("left_middle_second_joint","left_middle_first_joint",(255,178,102)),
    ("left_middle_first_joint","left_middle_tip",(255,178,102)),
    ("left_wrist","left_ring_third_joint",     (51,51,255)),
    ("left_ring_third_joint","left_ring_second_joint",(51,51,255)),
    ("left_ring_second_joint","left_ring_first_joint",(51,51,255)),
    ("left_ring_first_joint","left_ring_tip",  (51,51,255)),
    ("left_wrist","left_pinky_third_joint",    (0,255,0)),
    ("left_pinky_third_joint","left_pinky_second_joint",(0,255,0)),
    ("left_pinky_second_joint","left_pinky_first_joint",(0,255,0)),
    ("left_pinky_first_joint","left_pinky_tip",(0,255,0)),
    # right hand
    ("right_wrist","right_thumb_third_joint",  (0,128,255)),
    ("right_thumb_third_joint","right_thumb_second_joint",(0,128,255)),
    ("right_thumb_second_joint","right_thumb_first_joint",(0,128,255)),
    ("right_thumb_first_joint","right_thumb_tip",(0,128,255)),
    ("right_wrist","right_index_third_joint",  (255,153,255)),
    ("right_index_third_joint","right_index_second_joint",(255,153,255)),
    ("right_index_second_joint","right_index_first_joint",(255,153,255)),
    ("right_index_first_joint","right_index_tip",(255,153,255)),
    ("right_wrist","right_middle_third_joint", (255,178,102)),
    ("right_middle_third_joint","right_middle_second_joint",(255,178,102)),
    ("right_middle_second_joint","right_middle_first_joint",(255,178,102)),
    ("right_middle_first_joint","right_middle_tip",(255,178,102)),
    ("right_wrist","right_ring_third_joint",   (51,51,255)),
    ("right_ring_third_joint","right_ring_second_joint",(51,51,255)),
    ("right_ring_second_joint","right_ring_first_joint",(51,51,255)),
    ("right_ring_first_joint","right_ring_tip",(51,51,255)),
    ("right_wrist","right_pinky_third_joint",  (0,255,0)),
    ("right_pinky_third_joint","right_pinky_second_joint",(0,255,0)),
    ("right_pinky_second_joint","right_pinky_first_joint",(0,255,0)),
    ("right_pinky_first_joint","right_pinky_tip",(0,255,0)),
]
# fmt: on

# Resolve to index pairs (skip any link whose joint isn't in our 70-joint set)
SKELETON: list[tuple[int, int, tuple[int, int, int]]] = []
for _a, _b, _c in _SKEL_RAW:
    if _a in _NAME_TO_IDX and _b in _NAME_TO_IDX:
        SKELETON.append((_NAME_TO_IDX[_a], _NAME_TO_IDX[_b], _c))

# Subset of joints shown with --key_joints
# 5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip,
# 13=left_ankle, 14=right_ankle
KEY_JOINTS: frozenset[int] = frozenset({5, 6, 9, 10, 13, 14})

# ---------------------------------------------------------------------------
# Per-person color palette (BGR)
# ---------------------------------------------------------------------------
_PERSON_PALETTE = [
    (255,   0,   0),   # blue
    (  0, 200,   0),   # green
    (  0,   0, 255),   # red
    (255, 165,   0),   # orange
    (128,   0, 128),   # purple
    (  0, 200, 200),   # cyan
    (220,  20, 220),   # magenta
    (255, 215,   0),   # gold
]


def person_color(pid: int) -> tuple[int, int, int]:
    return _PERSON_PALETTE[pid % len(_PERSON_PALETTE)]


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def find_image(rich_data_root: Path, scene: str, cam: str, frame_idx: int) -> Path | None:
    """Return the Path to the source frame for a given camera & frame index.

    RICH frame filenames follow ``<frame_idx:05d>_<cam_idx:02d>.<ext>``
    (e.g. ``cam_04`` → ``00001_04.jpeg``).  Falls back to ``_00`` suffix
    for datasets that don't embed the camera index in the filename.
    """
    base = rich_data_root / scene / cam
    # Parse camera index from name, e.g. "cam_04" → 4
    cam_idx = int(cam.split("_")[-1]) if "_" in cam else 0
    stems = [f"{frame_idx:05d}_{cam_idx:02d}", f"{frame_idx:05d}_00"]
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".bmp", ".png"):
            p = base / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Body-data loading
# ---------------------------------------------------------------------------

def load_body_data(
    ghost_output_root: Path,
    scene: str,
    cam: str,
) -> dict[int, dict[int, np.ndarray]]:
    """Return ``{pid: {frame_idx: kps2d (70, 2)}}``.

    Also returns confidence if available (shape 55); stored separately.
    """
    body_dir = ghost_output_root / scene / cam / "body_data"
    if not body_dir.is_dir():
        return {}

    result: dict[int, dict[int, np.ndarray]] = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        pid = int(npz_path.stem.split("_")[1])
        data = np.load(npz_path, allow_pickle=False)

        frame_indices = data["frame_indices"]          # (T,) int32
        kps2d = data["pred_keypoints_2d"]               # (T, 70, 2)
        conf = data.get("pred_joint_confidence", None)  # (T, 55) or None

        per_frame: dict[int, np.ndarray] = {}
        for t, fi in enumerate(frame_indices):
            fi = int(fi)
            kp = kps2d[t]  # (70, 2)
            if conf is not None:
                # Attach confidence as a (70,) array; pad with 1.0 for last 15 joints
                c = conf[t]  # (55,)
                full_conf = np.ones(70, dtype=np.float32)
                full_conf[:55] = c
                per_frame[fi] = (kp, full_conf)
            else:
                per_frame[fi] = (kp, np.ones(70, dtype=np.float32))

        result[pid] = per_frame
    return result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton_on_image(
    img: np.ndarray,
    kps: np.ndarray,              # (70, 2) pixel coords
    conf: np.ndarray,             # (70,)   confidence in [0, 1]
    color: tuple[int, int, int],
    conf_threshold: float = 0.0,
    dot_radius: int = 4,
    bone_thickness: int = 2,
    joint_mask: frozenset[int] | None = None,
) -> None:
    """Draw MHR-70 skeleton in-place on *img* (BGR).

    joint_mask : if given, only joints whose index is in the set are drawn,
                 and only bones whose both endpoints are in the set.
    """
    h, w = img.shape[:2]

    def valid(idx: int) -> bool:
        if joint_mask is not None and idx not in joint_mask:
            return False
        x, y = kps[idx]
        return (conf[idx] >= conf_threshold
                and 0 <= x < w and 0 <= y < h)

    # Bones
    for ia, ib, bone_color in SKELETON:
        if valid(ia) and valid(ib):
            xa, ya = int(round(kps[ia, 0])), int(round(kps[ia, 1]))
            xb, yb = int(round(kps[ib, 0])), int(round(kps[ib, 1]))
            cv2.line(img, (xa, ya), (xb, yb), bone_color, bone_thickness, cv2.LINE_AA)

    # Keypoint dots (drawn on top of bones)
    for j in range(70):
        if valid(j):
            x, y = int(round(kps[j, 0])), int(round(kps[j, 1]))
            cv2.circle(img, (x, y), dot_radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), dot_radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pid_label(
    img: np.ndarray,
    kps: np.ndarray,
    pid: int,
    color: tuple[int, int, int],
) -> None:
    """Put person ID label near the top keypoint (head area)."""
    # Use neck (69) or nose (0) as anchor
    for anchor in [69, 0, 5, 6]:
        x, y = kps[anchor]
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.putText(
                img, f"P{pid}",
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA,
            )
            return


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def iter_cameras(ghost_output_root: Path, scene: str) -> Iterator[str]:
    """Yield camera directory names (e.g. 'cam_00') sorted."""
    scene_dir = ghost_output_root / scene
    if not scene_dir.is_dir():
        return
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("cam"):
            yield d.name


def process_scene(
    ghost_output_root: Path,
    rich_data_root: Path,
    scene: str,
    out_dir: Path,
    max_frames: int | None,
    conf_threshold: float,
    jpeg_quality: int,
    joint_mask: frozenset[int] | None = None,
    frames: list[int] | None = None,
) -> None:
    print(f"\n=== Scene: {scene} ===")

    # Collect all body data: cam → pid → {frame_idx: (kps2d, conf)}
    cam_data: dict[str, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
    for cam in iter_cameras(ghost_output_root, scene):
        body = load_body_data(ghost_output_root, scene, cam)
        if body:
            cam_data[cam] = body
            total = sum(len(v) for v in body.values())
            print(f"  {cam}: {len(body)} person(s), {total} tracked frames")

    if not cam_data:
        print("  No body data found — skipping.")
        return

    # Determine frame universe per camera
    for cam, pid_data in cam_data.items():
        all_frames = sorted({fi for pdata in pid_data.values() for fi in pdata})
        if frames is not None:
            all_frames = [f for f in all_frames if f in frames]
        elif max_frames is not None:
            all_frames = all_frames[:max_frames]

        saved = 0
        missing = 0
        for frame_idx in all_frames:
            # --- load image ---
            img_path = find_image(rich_data_root, scene, cam, frame_idx)
            if img_path is None:
                missing += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [warn] cannot read {img_path}")
                continue

            # --- draw each person ---
            for pid, pdata in sorted(pid_data.items()):
                if frame_idx not in pdata:
                    continue
                kps, conf = pdata[frame_idx]
                color = person_color(pid)
                draw_skeleton_on_image(img, kps, conf, color, conf_threshold,
                                       joint_mask=joint_mask)
                draw_pid_label(img, kps, pid, color)

            # --- save ---
            save_path = out_dir / scene / f"{frame_idx:05d}" / f"{cam}.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(save_path), img,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            saved += 1

        if missing:
            print(f"  {cam}: saved {saved} frames ({missing} missing images) → {out_dir / scene}")
        else:
            print(f"  {cam}: saved {saved} frames → {out_dir / scene}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize predicted 2D keypoints on RICH images."
    )
    parser.add_argument(
        "--ghost_output_root", required=True,
        help="Root directory of ghost outputs (contains scene sub-dirs).",
    )
    parser.add_argument(
        "--rich_data_root", required=True,
        help="Root of the RICH dataset (mounted squashfs or plain dir).",
    )
    parser.add_argument(
        "--scene", default=None,
        help="Process only this scene name. Default: all scenes.",
    )
    parser.add_argument(
        "--out_dir", default="renders/2d_kps",
        help="Output root. Default: renders/2d_kps",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Limit frames per camera (useful for quick checks).",
    )
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None,
        help="Only render these specific frame indices.",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.0,
        help="Min joint confidence to render a keypoint (default 0 = show all).",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=92,
        help="JPEG output quality [0-100]. Default: 92.",
    )
    parser.add_argument(
        "--key_joints", action="store_true",
        help=(
            "Draw only the 6 key body joints: "
            "5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip, "
            "13=left_ankle, 14=right_ankle."
        ),
    )
    args = parser.parse_args()

    ghost_output_root = Path(args.ghost_output_root)
    rich_data_root = Path(args.rich_data_root)
    out_dir = Path(args.out_dir)

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted(
            d.name
            for d in ghost_output_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    joint_mask = KEY_JOINTS if args.key_joints else None

    print(f"ghost_output_root : {ghost_output_root}")
    print(f"rich_data_root    : {rich_data_root}")
    print(f"out_dir           : {out_dir}")
    print(f"Scenes            : {len(scenes)}")
    print(f"key_joints only   : {args.key_joints}")

    for scene in scenes:
        process_scene(
            ghost_output_root=ghost_output_root,
            rich_data_root=rich_data_root,
            scene=scene,
            out_dir=out_dir,
            max_frames=args.max_frames,
            conf_threshold=args.conf_threshold,
            jpeg_quality=args.jpeg_quality,
            joint_mask=joint_mask,
            frames=args.frames,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
