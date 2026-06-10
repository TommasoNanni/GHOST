"""assessment/check_scales.py

Compare per-scene MapAnything predicted scale vs. GT scale derived from RICH
calibration XMLs.

GT scale definition (same as debug_scene.py):

    gt_scale = median_k( |t_gt_k| / |t_pred_k| )  for k = 1 … K-1

where:
  t_pred_k = VGGT predicted translation of camera k (median over valid frames T)
  t_gt_k   = GT camera translation in the normalised frame where cam_00 = [I|0]

Usage:
    pixi run python assessment/check_scales.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich

    # single scene
    pixi run python assessment/check_scales.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \\
        --scenes            BBQ_001_guitar
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    """'BBQ_001_guitar' → 'BBQ', 'LectureHall_018_wipingchairs1' → 'LectureHall'."""
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def _load_gt_extrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,4) [R|t] world-to-camera matrices from RICH XMLs."""
    location  = _scene_to_location(scene_name)
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    exts: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        root_node = ET.parse(xml_path).getroot()
        cam_node  = root_node.find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts if exts else None


def compute_gt_scale(
    scene_name: str,
    vggt_exts:  np.ndarray,   # (T, K, 3, 4)
    vggt_valid: np.ndarray,   # (T, K) bool
    rich_root:  Path,
) -> float | None:
    """GT scale = median_k( |t_gt_k| / |t_pred_k| ) for k=1..K-1."""
    gt_exts_list = _load_gt_extrinsics(scene_name, rich_root)
    if not gt_exts_list:
        return None

    T, K = vggt_exts.shape[:2]

    # Median VGGT extrinsic per camera over valid frames
    pred_t = np.zeros((K, 3), dtype=np.float64)
    for ki in range(K):
        valid_t = np.where(vggt_valid[:, ki])[0]
        if len(valid_t) == 0:
            continue
        pred_t[ki] = np.median(vggt_exts[valid_t, ki, :3, 3], axis=0)

    # Normalise GT cameras so cam_00 = [I|0]
    E0    = np.array(gt_exts_list[0], dtype=np.float64)
    R0, t0 = E0[:3, :3], E0[:3, 3]

    gt_t = np.zeros((K, 3), dtype=np.float64)
    for ki in range(min(K, len(gt_exts_list))):
        Ek = np.array(gt_exts_list[ki], dtype=np.float64)
        gt_t[ki] = Ek[:3, 3] - Ek[:3, :3] @ R0.T @ t0

    ratios = []
    for ki in range(1, min(K, len(gt_exts_list))):
        pn = np.linalg.norm(pred_t[ki])
        gn = np.linalg.norm(gt_t[ki])
        if pn > 1e-6:
            ratios.append(gn / pn)

    return float(np.median(ratios)) if ratios else None


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def check_scene(
    scene_dir: Path,
    rich_root:  Path,
) -> dict | None:
    cam_path   = scene_dir / "vggt_cameras.npz"
    scale_path = scene_dir / "mapanything_scale.npy"

    if not cam_path.exists():
        return None

    cam_npz    = np.load(cam_path, mmap_mode="r")
    vggt_exts  = cam_npz["extrinsics"].astype(np.float64)   # (T, K, 3, 4)
    vggt_valid = cam_npz["valid"]                            # (T, K) bool
    T, K       = vggt_exts.shape[:2]

    gt_scale = compute_gt_scale(scene_dir.name, vggt_exts, vggt_valid, rich_root)

    if scale_path.exists():
        ma_scale = np.load(scale_path).astype(np.float32)
        ma_mean   = float(np.mean(ma_scale))
        ma_median = float(np.median(ma_scale))
        ma_std    = float(np.std(ma_scale))
    else:
        ma_mean = ma_median = ma_std = float("nan")

    err_pct = (
        (ma_median - gt_scale) / gt_scale * 100
        if gt_scale and np.isfinite(ma_median)
        else float("nan")
    )

    return dict(
        scene     = scene_dir.name,
        T         = T,
        K         = K,
        gt_scale  = gt_scale,
        ma_mean   = ma_mean,
        ma_median = ma_median,
        ma_std    = ma_std,
        err_pct   = err_pct,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ghost_output_root", required=True, type=Path)
    p.add_argument("--rich_root",         required=True, type=Path)
    p.add_argument("--scenes", default="",
                   help="Comma-separated scene names; empty = all scenes with vggt_cameras.npz")
    args = p.parse_args()

    if args.scenes:
        scene_dirs = [args.ghost_output_root / s.strip() for s in args.scenes.split(",")]
    else:
        scene_dirs = sorted(
            d for d in args.ghost_output_root.iterdir()
            if d.is_dir() and (d / "vggt_cameras.npz").exists()
        )

    rows = []
    for sd in scene_dirs:
        r = check_scene(sd, args.rich_root)
        if r:
            rows.append(r)

    if not rows:
        print("No scenes found.")
        return

    # ── Print table ────────────────────────────────────────────────────────────
    hdr = f"{'Scene':<45} {'T':>5} {'K':>3}  {'gt_scale':>9}  {'ma_med':>9}  {'ma_mean':>9}  {'ma_std':>7}  {'err%':>7}"
    print(hdr)
    print("-" * len(hdr))

    errors = []
    for r in rows:
        gt  = f"{r['gt_scale']:.4f}"  if r['gt_scale'] is not None else "  N/A  "
        med = f"{r['ma_median']:.4f}" if np.isfinite(r['ma_median']) else "  N/A  "
        mn  = f"{r['ma_mean']:.4f}"   if np.isfinite(r['ma_mean'])   else "  N/A  "
        sd  = f"{r['ma_std']:.4f}"    if np.isfinite(r['ma_std'])    else "  N/A  "
        err = f"{r['err_pct']:+.1f}%" if np.isfinite(r['err_pct'])   else "  N/A  "
        print(f"{r['scene']:<45} {r['T']:>5} {r['K']:>3}  {gt:>9}  {med:>9}  {mn:>9}  {sd:>7}  {err:>7}")
        if np.isfinite(r['err_pct']):
            errors.append(r['err_pct'])

    if errors:
        arr = np.array(errors)
        print("-" * len(hdr))
        print(f"\nScenes with both GT and MA scale: {len(errors)}/{len(rows)}")
        print(f"  Mean abs error : {np.abs(arr).mean():.2f}%")
        print(f"  Median abs error: {np.median(np.abs(arr)):.2f}%")
        print(f"  Mean signed err: {arr.mean():+.2f}%")
        print(f"  Std of errors  : {arr.std():.2f}%")


if __name__ == "__main__":
    main()
