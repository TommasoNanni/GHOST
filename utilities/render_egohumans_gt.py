#!/usr/bin/env python3
"""
Render one GT-overlay frame per scene of an EgoHumans activity, showing who is
who with the correct (aria) person IDs.

For each sequence of the requested activity, this:
  1. picks a single frame where the most people are present (nearest the middle),
  2. loads that frame's raw fisheye exo image (default cam05 for fencing),
  3. projects every GT-SMPL person's vertices into the image using the COLMAP
     OPENCV_FISHEYE calibration, and
  4. draws each person as a translucent coloured point cloud with a bold label
     carrying its aria ID.

Colours are keyed by aria ID and fixed across all scenes, so the same colour
always means the same participant slot:
    aria01=red  aria02=green  aria03=blue  aria04=yellow

GT-SMPL vertices (and all EgoHumans 3D annotations) are expressed in the primary
Aria's reference SLAM frame — aria01.  They are lifted to the COLMAP world frame
(the frame the exo extrinsics live in) with the SINGLE anchor similarity
colmap_from_aria["aria01"] from colmap/workplace/colmap_from_aria_transforms.pkl,
applied to EVERY person (NOT a per-person transform):
    X_colmap = T_colmap_from_aria["aria01"] @ [X_aria01_world; 1].
Verified against the dataset's own per-camera 2D GT (processed_data/poses2d/camXX):
the anchor transform reprojects every person to within ~30-60 px of their GT box;
using a per-person transform mis-places the non-primary subjects by 400+ px.

Layout (identical to utilities/undistort_egohumans.py):
    <data_root>/<activity>/<INNER>/<activity>/<seq>/
        exo/<cam>/images/<frame:05d>.jpg          raw fisheye frames
        colmap/workplace/{cameras.txt,images.txt} calibration + extrinsics
        processed_data/smpl/<frame:05d>.npy        GT SMPL params per person

Usage:
    # data_root is the dir that holds the activity mount (e.g. a squashfuse mount
    # or the extracted dataset), exactly like undistort_egohumans.py.
    pixi run python utilities/render_egohumans_gt.py \\
        --data_root /capstor/scratch/cscs/tnanni/datasets/egohumans \\
        --activity 03_fencing \\
        --out_dir figures/egohumans_gt_ids/03_fencing
    # --cam omitted -> best of KEEP_CAMS[activity] auto-picked per scene.
    # Pass --cam cam05 to force a single camera.

    # If your mount point IS the activity root (single-activity sqsh mounted at
    # <mnt>/media/.../camera_ready/03_fencing), pass --cam_ready_root directly:
    pixi run python utilities/render_egohumans_gt.py \\
        --cam_ready_root <mnt>/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/03_fencing \\
        --activity 03_fencing --cam cam05 --out_dir figures/egohumans_gt_ids
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

# Curated exo cameras per activity (same set the pipeline undistorts). When no
# --cam is given, the best of these per scene is auto-selected (most GT people
# visible), so a single occluding view never wrecks the overlay.
KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}

# Fixed BGR colours per aria ID so the same colour == same participant slot in
# every rendered scene.
ID_COLORS: dict[str, tuple[int, int, int]] = {
    "aria01": (60, 60, 230),    # red
    "aria02": (60, 200, 60),    # green
    "aria03": (230, 130, 40),   # blue
    "aria04": (40, 220, 220),   # yellow
}
DEFAULT_COLOR = (200, 200, 200)


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _parse_cameras(cameras_txt: Path) -> dict[int, dict]:
    """camera_id -> {K, D, W, H} from a COLMAP OPENCV_FISHEYE cameras.txt."""
    cam: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cid = int(p[0])
            W, H = int(p[2]), int(p[3])
            fx, fy, cx, cy = (float(v) for v in p[4:8])
            k1, k2, k3, k4 = (float(v) for v in p[8:12])
            cam[cid] = {
                "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
                "D": np.array([[k1], [k2], [k3], [k4]], dtype=np.float64),
                "W": W, "H": H,
            }
    return cam


def _parse_images_for_cam(images_txt: Path, cam_name: str) -> tuple[int, dict[str, tuple]]:
    """Return (camera_id, {image_basename: (R, t)}) for one exo cam.

    R, t map COLMAP world -> camera:  X_cam = R @ X_world + t.
    """
    camera_id = -1
    poses: dict[str, tuple] = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10:
                continue
            name = p[9]
            if name.split("/")[0] != cam_name:
                continue
            q = np.array([float(v) for v in p[1:5]], dtype=np.float64)  # qw qx qy qz
            t = np.array([float(v) for v in p[5:8]], dtype=np.float64)
            R = _quat_wxyz_to_R(q)
            poses[Path(name).name] = (R, t)   # key by "00001.jpg"
            camera_id = int(p[8])
    return camera_id, poses


def _pick_frame(smpl_dir: Path) -> tuple[int, dict]:
    """Pick the frame with the most GT people, nearest the sequence middle.

    Returns (frame_idx, {person_name: params_dict}).
    """
    files = sorted(smpl_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no smpl npy in {smpl_dir}")
    mid = len(files) / 2.0
    best = None  # (num_people, -dist_to_mid, idx, path)
    for i, f in enumerate(files):
        data = np.load(str(f), allow_pickle=True).item()
        people = {k: v for k, v in data.items() if isinstance(v, dict)}
        key = (len(people), -abs(i - mid))
        if best is None or key > best[0]:
            best = (key, int(f.stem), people)
    _, frame_idx, people = best
    return frame_idx, people


def _project(points_w: np.ndarray, R: np.ndarray, t: np.ndarray,
             K: np.ndarray, D: np.ndarray, W: int, H: int) -> np.ndarray:
    """Project world points onto a fisheye image; return in-bounds (u, v) ints."""
    Xc = points_w.astype(np.float64) @ R.T + t          # (N, 3) camera frame
    Xc = Xc[Xc[:, 2] > 1e-3]                             # drop points behind cam
    if Xc.shape[0] == 0:
        return np.empty((0, 2), dtype=int)
    uv, _ = cv2.fisheye.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[m].astype(int)


def render_scene(seq_dir: Path, cam_list: list[str], out_dir: Path) -> str:
    smpl_dir    = seq_dir / "processed_data" / "smpl"
    cameras_txt = seq_dir / "colmap" / "workplace" / "cameras.txt"
    images_txt  = seq_dir / "colmap" / "workplace" / "images.txt"
    tf_path     = seq_dir / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    if not smpl_dir.is_dir():
        return f"SKIP {seq_dir.name}: no smpl dir"
    if not (cameras_txt.exists() and images_txt.exists()):
        return f"SKIP {seq_dir.name}: no colmap calibration"
    if not tf_path.exists():
        return f"SKIP {seq_dir.name}: no colmap_from_aria_transforms.pkl"
    import pickle
    with open(tf_path, "rb") as f:
        colmap_from_aria: dict[str, np.ndarray] = pickle.load(f)
    # All 3D GT lives in the primary Aria's frame; use its single anchor transform
    # for every person. Fall back to the lowest-numbered aria if aria01 is absent.
    anchor_key = "aria01" if "aria01" in colmap_from_aria else (
        sorted(colmap_from_aria)[0] if colmap_from_aria else None)
    if anchor_key is None:
        return f"SKIP {seq_dir.name}: empty colmap_from_aria transform"
    T_anchor = np.asarray(colmap_from_aria[anchor_key], dtype=np.float64)

    frame_idx, people = _pick_frame(smpl_dir)
    cams_meta = _parse_cameras(cameras_txt)

    def _view(cam_name: str):
        img_path = seq_dir / "exo" / cam_name / "images" / f"{frame_idx:05d}.jpg"
        if not img_path.exists():
            return None
        cam_id, poses = _parse_images_for_cam(images_txt, cam_name)
        if cam_id not in cams_meta or not poses:
            return None
        R, t = poses.get(f"{frame_idx:05d}.jpg") or next(iter(poses.values()))
        return {"cam": cam_name, "img_path": img_path, "R": R, "t": t,
                **cams_meta[cam_id]}

    def _persons_uv(view) -> dict[str, np.ndarray]:
        out = {}
        for name in people:
            v = np.asarray(people[name]["vertices"], np.float64)
            v = v @ T_anchor[:3, :3].T + T_anchor[:3, 3]   # aria01 frame -> colmap world
            out[name] = _project(v, view["R"], view["t"], view["K"], view["D"],
                                 view["W"], view["H"])
        return out

    # Auto-select the camera showing the most GT people (tie: most in-bounds verts).
    best = None
    for cam_name in cam_list:
        view = _view(cam_name)
        if view is None:
            continue
        uv = _persons_uv(view)
        nvis = sum(1 for n in people if uv[n].shape[0] >= 50)
        tot  = int(sum(uv[n].shape[0] for n in people))
        if best is None or (nvis, tot) > best[0]:
            best = ((nvis, tot), view, uv)
    if best is None:
        return f"SKIP {seq_dir.name}: no usable camera among {cam_list} at frame {frame_idx:05d}"
    _, view, persons_uv = best
    cam_name = view["cam"]
    img = cv2.imread(str(view["img_path"]))
    if img is None:
        return f"SKIP {seq_dir.name}: unreadable {view['img_path']}"
    H_img, W_img = img.shape[:2]

    overlay = img.copy()
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for name in sorted(people):
        uv = persons_uv[name]
        if uv.shape[0] == 0:
            continue
        color = ID_COLORS.get(name, DEFAULT_COLOR)
        for u, v in uv:
            cv2.circle(overlay, (u, v), 1, color, -1)
        # label anchor: top of the projected point cloud (head-ish)
        top = uv[uv[:, 1].argmin()]
        cx = int(np.median(uv[:, 0]))
        labels.append((name, (cx, int(top[1])), color))

    out = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    for name, (ax, ay), color in labels:
        ay = max(ay - 12, 24)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x0 = int(np.clip(ax - tw // 2, 2, W_img - tw - 2))
        cv2.rectangle(out, (x0 - 4, ay - th - 6), (x0 + tw + 4, ay + 6), color, -1)
        cv2.putText(out, name, (x0, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # header + legend
    hdr = f"{seq_dir.name}  {cam_name}  frame {frame_idx:05d}  ({len(people)} GT people)"
    cv2.rectangle(out, (0, 0), (W_img, 34), (0, 0, 0), -1)
    cv2.putText(out, hdr, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ly = 60
    for name in sorted(people):
        color = ID_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(out, (8, ly - 14), (28, ly + 4), color, -1)
        cv2.putText(out, name, (34, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ly += 26

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_dir.name}_{cam_name}.jpg"
    cv2.imwrite(str(out_path), out)
    names = "+".join(sorted(people))
    return f"OK {seq_dir.name}: {cam_name} frame {frame_idx:05d}, {len(people)} people [{names}] -> {out_path.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=None,
                    help="Root holding <activity>/INNER/<activity> (as in undistort_egohumans.py)")
    ap.add_argument("--cam_ready_root", default=None,
                    help="Direct path to .../camera_ready/<activity> (overrides --data_root)")
    ap.add_argument("--activity", required=True, help="e.g. 03_fencing")
    ap.add_argument("--cam", default=None,
                    help="Force a single exo camera (e.g. cam05). "
                         "Omit to auto-pick the best of KEEP_CAMS[activity] per scene.")
    ap.add_argument("--out_dir", default="figures/egohumans_gt_ids")
    ap.add_argument("--seq", default=None, help="Render only this sequence (e.g. 007_fencing)")
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
            print(render_scene(seq_dir, cam_list, out_dir))
        except Exception as e:
            print(f"ERR {seq_dir.name}: {e}")


if __name__ == "__main__":
    main()
