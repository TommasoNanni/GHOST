"""
Evaluate VGGT camera predictions vs GT calibration across all training scenes.

For each non-reference camera, computes:
  - rotation error (degrees)
  - translation direction error (degrees, angle between camera centers)
  - scale ratio  ||C_gt|| / ||C_vggt||  (>1 means VGGT under-estimates distance)
  - translation error (m): ||C_vggt * scene_scale - C_gt||
    where scene_scale = median GT/VGGT ratio across all cameras in the scene

GT extrinsics are re-rooted to cam_00 (VGGT reference convention).
VGGT extrinsics are averaged across valid frames before comparison (static scene).
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR   = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test")
CALIB_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/rich/scan_calibration")
LOCATIONS  = ["BBQ", "Gym", "LectureHall", "ParkingLot1", "ParkingLot2", "Pavallion"]


# ── helpers ──────────────────────────────────────────────────────────────────

def scene_location(scene_name):
    for loc in LOCATIONS:
        if scene_name.startswith(loc):
            return loc
    raise ValueError(f"Unknown location for scene: {scene_name}")


def load_gt_ext(calib_dir, cam_name):
    cam_idx = int(cam_name.split("_")[1])
    xml_path = calib_dir / f"{cam_idx:03d}.xml"
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    vals = list(map(float, root.find("CameraMatrix/data").text.split()))
    return np.array(vals, dtype=np.float64).reshape(3, 4)


def reroot(ext_dict, ref_name):
    """Return extrinsics expressed relative to ref_name."""
    R0, t0 = ext_dict[ref_name][:, :3], ext_dict[ref_name][:, 3]
    out = {}
    for name, E in ext_dict.items():
        Rk, tk = E[:, :3], E[:, 3]
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


# ── per-scene evaluation ──────────────────────────────────────────────────────

def evaluate_scene(scene_name):
    scene_dir  = BASE_DIR / scene_name
    npz_path   = scene_dir / "vggt_cameras.npz"
    if not npz_path.exists():
        return None

    d          = np.load(npz_path)
    extrinsics = d["extrinsics"].astype(np.float64)   # (T, K, 3, 4)
    valid      = d["valid"]                            # (T, K)
    cam_names  = [n.decode() for n in d["camera_names"]]
    K          = len(cam_names)

    calib_dir  = CALIB_ROOT / scene_location(scene_name) / "calibration"

    # --- load GT and re-root to cam_00 ---
    gt_raw = {}
    for name in cam_names:
        E = load_gt_ext(calib_dir, name)
        if E is not None:
            gt_raw[name] = E

    ref_name = cam_names[0]
    if ref_name not in gt_raw:
        return None   # can't establish a common reference

    gt_rel = reroot(gt_raw, ref_name)

    # --- average VGGT extrinsics across valid frames (static scene) ---
    vggt_avg = {}
    for k, name in enumerate(cam_names):
        frames = np.where(valid[:, k])[0]
        if len(frames) == 0:
            continue
        E_mean = extrinsics[frames, k].mean(axis=0)   # (3,4)
        # re-orthogonalise rotation via SVD
        U, _, Vt = np.linalg.svd(E_mean[:, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        vggt_avg[name] = (R, E_mean[:, 3])

    # --- collect per-camera data ---
    cam_data = {}
    for name in cam_names[1:]:   # skip reference
        if name not in gt_rel or name not in vggt_avg:
            continue
        R_gt, t_gt = gt_rel[name]
        R_vg, t_vg = vggt_avg[name]
        C_gt = cam_center(R_gt, t_gt)
        C_vg = cam_center(R_vg, t_vg)
        d_gt = np.linalg.norm(C_gt)
        d_vg = np.linalg.norm(C_vg)
        cam_data[name] = (R_gt, R_vg, C_gt, C_vg, d_gt, d_vg)

    if not cam_data:
        return None

    # per-scene scale: median GT/VGGT ratio across all non-reference cameras
    scale_ratios = [d_gt / d_vg for _, _, _, _, d_gt, d_vg in cam_data.values() if d_vg > 1e-9]
    scene_scale  = float(np.median(scale_ratios)) if scale_ratios else np.nan

    rows = []
    for name, (R_gt, R_vg, C_gt, C_vg, d_gt, d_vg) in cam_data.items():
        scale_ratio  = d_gt / d_vg if d_vg > 1e-9 else np.nan
        transl_err_m = float(np.linalg.norm(C_vg * scene_scale - C_gt)) if not np.isnan(scene_scale) else np.nan
        rows.append({
            "scene":        scene_name,
            "cam":          name,
            "rot_err":      rot_err_deg(R_vg, R_gt),
            "dir_err":      angle_between(C_gt, C_vg),
            "scale_ratio":  scale_ratio,
            "transl_err_m": transl_err_m,
            "d_gt_m":       d_gt,
        })

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    scenes = sorted(p.name for p in BASE_DIR.iterdir() if p.is_dir())
    print(f"Evaluating {len(scenes)} scenes...\n")

    all_rows = []
    skipped  = []
    for scene in scenes:
        rows = evaluate_scene(scene)
        if rows is None:
            skipped.append(scene)
            continue
        all_rows.extend(rows)
        rot_vals   = [r["rot_err"]      for r in rows]
        scale_vals = [r["scale_ratio"]  for r in rows if not np.isnan(r["scale_ratio"])]
        terr_vals  = [r["transl_err_m"] for r in rows if not np.isnan(r["transl_err_m"])]
        print(f"  {scene:<45}  cams={len(rows)}  "
              f"rot={np.mean(rot_vals):.2f}°  scale={np.mean(scale_vals):.3f}×  "
              f"terr={np.mean(terr_vals)*100:.1f}cm")

    if not all_rows:
        print("No data collected.")
        return

    rot_all   = np.array([r["rot_err"]      for r in all_rows])
    dir_all   = np.array([r["dir_err"]      for r in all_rows])
    scale_all = np.array([r["scale_ratio"]  for r in all_rows if not np.isnan(r["scale_ratio"])])
    terr_all  = np.array([r["transl_err_m"] for r in all_rows if not np.isnan(r["transl_err_m"])]) * 100  # cm

    print(f"\n{'='*60}")
    print(f"AGGREGATE  ({len(all_rows)} camera–scene pairs, {len(scenes)-len(skipped)} scenes)")
    print(f"{'='*60}")
    print(f"  Rotation error (°)         mean={rot_all.mean():.2f}  "
          f"median={np.median(rot_all):.2f}  std={rot_all.std():.2f}  "
          f"p90={np.percentile(rot_all,90):.2f}  max={rot_all.max():.2f}")
    print(f"  Transl direction error (°) mean={dir_all.mean():.2f}  "
          f"median={np.median(dir_all):.2f}  std={dir_all.std():.2f}  "
          f"p90={np.percentile(dir_all,90):.2f}  max={dir_all.max():.2f}")
    print(f"  Scale ratio (GT/VGGT)      mean={scale_all.mean():.3f}  "
          f"median={np.median(scale_all):.3f}  std={scale_all.std():.3f}  "
          f"p5={np.percentile(scale_all,5):.3f}  p95={np.percentile(scale_all,95):.3f}")
    print(f"  Transl error after scale (cm)  mean={terr_all.mean():.1f}  "
          f"median={np.median(terr_all):.1f}  std={terr_all.std():.1f}  "
          f"p90={np.percentile(terr_all,90):.1f}  max={terr_all.max():.1f}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
