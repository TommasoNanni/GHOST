"""Stage-A gate for ReID v7: score reid_cameras.npz consensus rigs vs GT.

Compares the background-estimated camera rig (preprocessing/reid_cameras.py)
against GT calibration, after re-rooting both to a common reference camera so
the arbitrary world frames cancel.  Reports per camera:

  - rotation error (degrees)
  - camera-centre direction error (degrees)

Both metrics are scale-free (the v7 world frame deliberately stays in
normalized VGGT units).  Stage-A acceptance gate: rot ≤ 3° AND dir ≤ 5° on
≥ 80% of cameras.

GT sources
----------
- ``--dataset rich``      : RICH scan_calibration XMLs (<CameraMatrix> node).
                            Camera names are derived from the XML filename's
                            integer → ``cam_XX``; verify once against a scene
                            (the sorted-index ↔ cam mapping bit us before).
- ``--dataset egohumans`` : COLMAP ``images.txt`` (exo cams, first registered
                            frame; convention: line = ID QW QX QY QZ TX TY TZ
                            CAM NAME → world-to-cam R, t; centre C = -Rᵀt).

Usage
-----
pixi run python scripts/eval_reid_cameras.py --dataset egohumans \
    --output_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new \
    --raw_root /iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans \
    --scenes 06_badminton/031_badminton
pixi run python scripts/eval_reid_cameras.py --dataset rich \
    --output_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root /capstor/scratch/cscs/tnanni/datasets/rich --scenes <scene>
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# inner path from a raw EgoHumans activity root to the camera_ready tree
EGOHUMANS_INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

ROT_GATE_DEG = 3.0
DIR_GATE_DEG = 5.0
GATE_FRACTION = 0.80


# ── shared helpers (copied from scripts/eval_vggt_cameras_egohumans.py) ───────

def quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def reroot(pose_dict: dict, ref: str) -> dict:
    R0, t0 = pose_dict[ref]
    out = {}
    for name, (Rk, tk) in pose_dict.items():
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        out[name] = (R_rel, t_rel)
    return out


def cam_center(R, t):
    return -R.T @ t


def rot_err_deg(R1, R2):
    cos = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ── GT loaders ────────────────────────────────────────────────────────────────

def parse_colmap_images(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{cam_name: (R, t)} for static exo cams (first registered frame)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(images_txt) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or not ln.strip():
            i += 1
            continue
        p = ln.split()
        name = p[9]
        cam = name.split("/")[0]
        if cam.startswith("cam") and cam not in out:
            qw, qx, qy, qz = map(float, p[1:5])
            tx, ty, tz = map(float, p[5:8])
            out[cam] = (quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz], np.float64))
        i += 2  # skip the POINTS2D line
    return out


def find_colmap(raw_roots: list[Path], scene_rel: str) -> Path | None:
    """scene_rel = '06_badminton/055_badminton' → images.txt path or None."""
    activity, seq = scene_rel.split("/")
    rel = f"{EGOHUMANS_INNER}/{activity}/{seq}/colmap/workplace/images.txt"
    for root in raw_roots:
        for cand in (root / rel, root / activity / rel):
            if cand.exists():
                return cand
    return None


def load_rich_gt(rich_root: Path, scene_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{cam_XX: (R, t)} from RICH scan_calibration XMLs (CameraMatrix node)."""
    location = scene_name.split("_")[0]
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not calib_dir.is_dir():
        return out
    for xml_path in sorted(calib_dir.glob("*.xml")):
        m = re.search(r"\d+", xml_path.stem)
        if m is None:
            continue
        cam_name = f"cam_{int(m.group(0)):02d}"
        tree = ET.parse(xml_path)
        node = tree.getroot().find("CameraMatrix")
        if node is None:
            continue
        vals = list(map(float, node.find("data").text.split()))
        E = np.array(vals, dtype=np.float64).reshape(3, 4)
        out[cam_name] = (E[:3, :3], E[:3, 3])
    return out


# ── evaluation ────────────────────────────────────────────────────────────────

def load_pred(scene_dir: Path) -> tuple[dict, dict] | None:
    """{cam: (R, t)} + {cam: inlier_frac} from reid_cameras.npz (static only)."""
    npz = scene_dir / "reid_cameras.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["camera_names"]]
    ext = d["extrinsics_static"].astype(np.float64)
    moving = d["is_moving"]
    frac = d["cam_inlier_frac"]
    pred, fracs = {}, {}
    for k, name in enumerate(names):
        if moving[k] or not np.isfinite(ext[k]).all():
            continue
        pred[name] = (ext[k, :3, :3], ext[k, :3, 3])
        fracs[name] = float(frac[k])
    return pred, fracs


def evaluate_scene(scene_dir: Path, gt: dict) -> list[dict] | None:
    loaded = load_pred(scene_dir)
    if loaded is None:
        print(f"  {scene_dir.name}: no reid_cameras.npz")
        return None
    pred, fracs = loaded
    common = [c for c in pred if c in gt]
    if len(common) < 2:
        print(f"  {scene_dir.name}: <2 common cams (pred={sorted(pred)}, gt={sorted(gt)})")
        return None
    ref = common[0]
    pred_rel = reroot({c: pred[c] for c in common}, ref)
    gt_rel = reroot({c: gt[c] for c in common}, ref)
    rows = []
    for c in common[1:]:
        R_p, t_p = pred_rel[c]
        R_g, t_g = gt_rel[c]
        rows.append({
            "scene": scene_dir.name,
            "cam": c,
            "rot_err": rot_err_deg(R_p, R_g),
            "dir_err": angle_between(cam_center(R_g, t_g), cam_center(R_p, t_p)),
            "inlier_frac": fracs[c],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-A gate: reid_cameras vs GT")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--dataset", choices=["rich", "egohumans"], required=True)
    ap.add_argument("--rich_root", type=Path, default=None)
    ap.add_argument("--raw_root", action="append", default=[],
                    help="EgoHumans raw/colmap root(s); repeatable")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="rich: scene names; egohumans: activity/scene")
    args = ap.parse_args()

    if args.scenes:
        scene_rels = args.scenes
    elif args.dataset == "egohumans":
        scene_rels = sorted(
            f"{a.name}/{s.name}"
            for a in args.output_dir.iterdir() if a.is_dir()
            for s in a.iterdir() if s.is_dir()
        )
    else:
        scene_rels = sorted(p.name for p in args.output_dir.iterdir() if p.is_dir())

    all_rows = []
    for scene_rel in scene_rels:
        scene_dir = args.output_dir / scene_rel
        if args.dataset == "egohumans":
            images_txt = find_colmap([Path(r) for r in args.raw_root], scene_rel)
            if images_txt is None:
                print(f"  {scene_rel}: no colmap GT found")
                continue
            gt = parse_colmap_images(images_txt)
        else:
            if args.rich_root is None:
                ap.error("--rich_root required for --dataset rich")
            gt = load_rich_gt(args.rich_root, scene_dir.name)
            if not gt:
                print(f"  {scene_rel}: no RICH calibration found")
                continue
        rows = evaluate_scene(scene_dir, gt)
        if rows:
            all_rows.extend(rows)
            rot = [r["rot_err"] for r in rows]
            dr = [r["dir_err"] for r in rows]
            print(f"  {scene_rel:<36} cams={len(rows)}  "
                  f"rot_mean={np.mean(rot):5.2f}° rot_max={np.max(rot):5.2f}°  "
                  f"dir_mean={np.nanmean(dr):5.2f}° dir_max={np.nanmax(dr):5.2f}°")

    if not all_rows:
        print("No data collected.")
        return

    rot = np.array([r["rot_err"] for r in all_rows])
    dr = np.array([r["dir_err"] for r in all_rows])
    ok = (rot <= ROT_GATE_DEG) & (np.nan_to_num(dr, nan=1e9) <= DIR_GATE_DEG)
    frac = float(ok.mean())
    print(f"\n{'=' * 64}")
    print(f"Cameras: {len(all_rows)}   rot median {np.median(rot):.2f}°  "
          f"dir median {np.nanmedian(dr):.2f}°")
    print(f"Gate (rot≤{ROT_GATE_DEG}° ∧ dir≤{DIR_GATE_DEG}°): "
          f"{frac * 100:.0f}% of cams  →  STAGE A "
          + ("PASS" if frac >= GATE_FRACTION else "FAIL"))
    worst = sorted(all_rows, key=lambda r: -(r["rot_err"] + np.nan_to_num(r["dir_err"])))[:8]
    print("Worst cams:")
    for r in worst:
        print(f"  {r['scene']}/{r['cam']}: rot={r['rot_err']:.2f}° "
              f"dir={r['dir_err']:.2f}° inlier_frac={r['inlier_frac']:.2f}")


if __name__ == "__main__":
    main()
