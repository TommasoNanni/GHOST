"""Visualize Sapiens 308-keypoint predictions overlaid on RICH centered images.

Reads ``sapiens_centered_kps_person_<pid>.npz`` from every camera directory under
``ghost_output_root/<scene>/cam_XX/`` and draws the skeleton.
Keypoints are in centered_train image space — ``rich_data_root`` must point to
the RICH ``centered_train`` directory (e.g. ``.../rich/centered_train``).

Sapiens COCO-WholeBody layout:
    0-16:   COCO body (same order as ViTPose COCO-17)
    17-22:  feet  (left big/small/heel, right big/small/heel)
    23-90:  face  (68 landmarks)
    91-111: left hand  (21 keypoints)
    112-132: right hand (21 keypoints)

Output layout::

    renders/sapiens/<scene_name>/<frame_idx:05d>/cam_XX.jpg

Usage
-----
    pixi run python visualize/visualize_sapiens.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_data_root    /capstor/scratch/cscs/tnanni/datasets/rich/centered_train \\
        --scene             BBQ_001_guitar \\
        --frames            0 47 94 141    \\
        --out_dir           renders/sapiens \\
        --conf_threshold    0.3 \\
        --parts             body hands face
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Skeleton definitions
# ---------------------------------------------------------------------------

# Body: COCO-17 (indices 0-16)
# User-specified connections for Sapiens body.
# Uses both COCO wrists (j9=l_wrist, j10=r_wrist) and Sapiens hand wrists (j62=l_wrist, j41=r_wrist).
# Left side:  0→1→3→5→7→62  and  5→9→11→13
# Right side: 14→12→10→6→8→41  and  6→4→2→0
_BODY_SKELETON: list[tuple[int, int, tuple[int, int, int]]] = [
    # left face+arm+body (green)
    (0,  1,  (0, 220, 0)),
    (1,  3,  (0, 220, 0)),
    (3,  5,  (0, 220, 0)),
    (5,  7,  (0, 220, 0)),
    (7,  62, (0, 220, 0)),   # l_elbow → l_sapiens_wrist
    (5,  9,  (0, 220, 0)),   # l_shoulder → l_coco_wrist
    (9,  11, (0, 220, 0)),   # l_coco_wrist → l_hip
    (11, 13, (0, 220, 0)),   # l_hip → l_knee
    # cross bar
    (9,  10, (255, 153, 51)),
    # right face+arm+body (orange)
    (14, 12, (0, 128, 255)),
    (12, 10, (0, 128, 255)),
    (10, 6,  (0, 128, 255)), # r_coco_wrist → r_shoulder
    (6,  8,  (0, 128, 255)),
    (8,  41, (0, 128, 255)), # r_elbow → r_sapiens_wrist
    (6,  4,  (0, 128, 255)),
    (4,  2,  (0, 128, 255)),
    (2,  0,  (0, 128, 255)),
]

# Feet: 17=left_heel, 18=right_big_toe, 19=right_small_toe, 20=right_heel
_FEET_SKELETON: list[tuple[int, int, tuple[int, int, int]]] = [
    (15, 17, (0, 200, 100)),                             # l_ankle → l_heel
    (16, 18, (0, 100, 255)), (16, 19, (0, 100, 255)),   # r_ankle → r_big/small toe
    (16, 20, (0, 100, 255)),                             # r_ankle → r_heel
]

# Goliath 308 keypoint layout — source: goliath.py in facebookresearch/sapiens
#   0-16:  COCO body (nose … r_ankle)
#   17-20: feet  (left_heel=17, right_big_toe=18, right_small_toe=19, right_heel=20)
#   21-41: right hand  — fingertips first, wrist LAST (right_wrist=41)
#          offsets: thumb4=0,thumb3=1,thumb2=2,thumb_mcp=3,
#                   ff4=4,ff3=5,ff2=6,ff_mcp=7, mf4=8..mf_mcp=11,
#                   rf4=12..rf_mcp=15, pf4=16..pf_mcp=19, wrist=20
#   42-62: left hand   — same layout (left_wrist=62)
#   63-69: extra body  (l/r olecranon=63-64, l/r cubital_fossa=65-66,
#                       l/r acromion=67-68, neck=69)
#   70-307: face (center_of_glabella … ear anatomy — 238 joints)
_RH = 21   # right hand base; wrist = _RH + 20 = 41
_LH = 42   # left  hand base; wrist = _LH + 20 = 62

# Offsets within each hand: wrist=20, MCP joints=3,7,11,15,19; tips=0,4,8,12,16
_HAND_CONNECTIONS = [
    # wrist → each MCP
    (20, 3), (20, 7), (20, 11), (20, 15), (20, 19),
    # thumb:      MCP→prox→mid→tip
    (3, 2),  (2, 1),  (1, 0),
    # forefinger: MCP→prox→mid→tip
    (7, 6),  (6, 5),  (5, 4),
    # middle:     MCP→prox→mid→tip
    (11, 10), (10, 9), (9, 8),
    # ring:       MCP→prox→mid→tip
    (15, 14), (14, 13), (13, 12),
    # pinky:      MCP→prox→mid→tip
    (19, 18), (18, 17), (17, 16),
    # palm knuckle bar
    (3, 7), (7, 11), (11, 15), (15, 19),
]
_RIGHT_HAND_SKELETON = [(_RH + a, _RH + b, (0, 100, 255))  for a, b in _HAND_CONNECTIONS]
_LEFT_HAND_SKELETON  = [(_LH + a, _LH + b, (0, 220, 0))    for a, b in _HAND_CONNECTIONS]

# Face: j70-307 — dots only; j63-69 are extra body (olecranon/acromion/neck)
_FACE_START, _FACE_END = 70, 308

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

def load_sapiens(
    ghost_output_root: Path,
    scene: str,
    cam: str,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{pid: {frame_idx: (kps (308,2), conf (308,))}}``.

    kps in original image pixels [x, y].
    """
    cam_dir  = ghost_output_root / scene / cam
    body_dir = cam_dir / "body_data"
    if not body_dir.is_dir():
        return {}
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        pid     = int(npz_path.stem.split("_")[1])
        sp_path = cam_dir / f"sapiens_centered_kps_person_{pid}.npz"
        if not sp_path.exists():
            continue
        d      = np.load(sp_path)
        kps_all = d["keypoints"]       # (T, 308, 3)  [x, y, conf]
        fi_arr  = d["frame_indices"].astype(int)
        per_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for local_t, global_t in enumerate(fi_arr):
            per_frame[int(global_t)] = (
                kps_all[local_t, :, :2].copy(),   # (308, 2)
                kps_all[local_t, :, 2].copy(),    # (308,)
            )
        result[pid] = per_frame
    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_bones(
    img: np.ndarray,
    kps: np.ndarray,
    conf: np.ndarray,
    skeleton: list[tuple[int, int, tuple[int, int, int]]],
    conf_thr: float,
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    for ia, ib, color in skeleton:
        if (ia >= len(conf) or ib >= len(conf)
                or conf[ia] < conf_thr or conf[ib] < conf_thr):
            continue
        xa, ya = int(round(kps[ia, 0])), int(round(kps[ia, 1]))
        xb, yb = int(round(kps[ib, 0])), int(round(kps[ib, 1]))
        if not (0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h):
            continue
        cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def _draw_dots(
    img: np.ndarray,
    kps: np.ndarray,
    conf: np.ndarray,
    indices: range | list[int],
    base_color: tuple[int, int, int],
    conf_thr: float,
    radius: int = 4,
    label: bool = False,
) -> None:
    h, w = img.shape[:2]
    for j in indices:
        if j >= len(conf) or conf[j] < conf_thr:
            continue
        x, y = int(round(kps[j, 0])), int(round(kps[j, 1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        alpha = float(conf[j])
        dot_color = tuple(int(c * alpha) for c in base_color)
        cv2.circle(img, (x, y), radius, dot_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)
        if label:
            cv2.putText(img, str(j), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)


def draw_sapiens(
    img: np.ndarray,
    kps: np.ndarray,        # (308, 2)
    conf: np.ndarray,       # (308,)
    color: tuple[int, int, int],
    conf_thr: float,
    parts: set[str],
    label_joints: bool = False,
) -> None:
    if "body" in parts:
        _draw_bones(img, kps, conf, _BODY_SKELETON, conf_thr)
        _draw_dots(img, kps, conf, range(15), color, conf_thr, radius=5, label=label_joints)

    if "hands" in parts:
        _draw_bones(img, kps, conf, _RIGHT_HAND_SKELETON, conf_thr, thickness=1)
        _draw_bones(img, kps, conf, _LEFT_HAND_SKELETON,  conf_thr, thickness=1)
        _draw_dots(img, kps, conf, range(_RH, _RH + 21), (0, 100, 255), conf_thr, radius=3, label=label_joints)
        _draw_dots(img, kps, conf, range(_LH, _LH + 21), (0, 220, 0),   conf_thr, radius=3, label=label_joints)

    if "face" in parts:
        _draw_dots(img, kps, conf, range(_FACE_START, _FACE_END), (200, 200, 50), conf_thr,
                   radius=2, label=label_joints)


def draw_pid_label(
    img: np.ndarray,
    kps: np.ndarray,
    conf: np.ndarray,
    pid: int,
    color: tuple[int, int, int],
    conf_thr: float,
) -> None:
    for anchor in (5, 6, 0):
        if conf[anchor] >= conf_thr:
            x, y = int(kps[anchor, 0]), int(kps[anchor, 1])
            cv2.putText(img, f"P{pid}", (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def process_scene(
    ghost_output_root: Path,
    rich_data_root: Path,
    scene: str,
    frames: list[int] | None,
    out_dir: Path,
    conf_thr: float,
    jpeg_quality: int,
    parts: set[str],
    label_joints: bool,
) -> None:
    print(f"\n=== {scene} ===")
    scene_dir = ghost_output_root / scene
    if not scene_dir.is_dir():
        print(f"  Scene dir not found: {scene_dir}")
        return

    cam_names = sorted(d.name for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("cam"))

    for cam in cam_names:
        sp_data = load_sapiens(ghost_output_root, scene, cam)
        if not sp_data:
            print(f"  {cam}: no sapiens data — skipping")
            continue

        all_frames = sorted({fi for pd in sp_data.values() for fi in pd})
        target_frames = frames if frames is not None else all_frames

        saved = missing = 0
        for frame_idx in target_frames:
            if frame_idx not in all_frames:
                continue

            img_path = find_image(rich_data_root, scene, cam, frame_idx)
            if img_path is None:
                missing += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                missing += 1
                continue

            for pid, pdata in sorted(sp_data.items()):
                if frame_idx not in pdata:
                    continue
                kps, conf = pdata[frame_idx]
                color = person_color(pid)
                draw_sapiens(img, kps, conf, color, conf_thr, parts, label_joints)
                draw_pid_label(img, kps, conf, pid, color, conf_thr)

            # Overlay label
            parts_str = "+".join(sorted(parts))
            cv2.putText(img, f"Sapiens ({parts_str})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)

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
        description="Visualize Sapiens 308-keypoint predictions on RICH images."
    )
    parser.add_argument("--ghost_output_root", required=True)
    parser.add_argument("--rich_data_root",    required=True)
    parser.add_argument("--scene",             required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Frame indices to render. Default: all frames.")
    parser.add_argument("--out_dir", default="renders/sapiens")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--jpeg_quality",   type=int,   default=92)
    parser.add_argument("--parts", nargs="+",
                        choices=["body", "hands", "face"], default=["body"],
                        help="Which body parts to draw (default: body only)")
    parser.add_argument("--label_joints", action="store_true",
                        help="Draw joint index numbers next to each keypoint")
    args = parser.parse_args()

    process_scene(
        ghost_output_root=Path(args.ghost_output_root),
        rich_data_root=Path(args.rich_data_root),
        scene=args.scene,
        frames=args.frames,
        out_dir=Path(args.out_dir),
        conf_thr=args.conf_threshold,
        jpeg_quality=args.jpeg_quality,
        parts=set(args.parts),
        label_joints=args.label_joints,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
