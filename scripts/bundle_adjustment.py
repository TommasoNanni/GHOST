"""scripts/bundle_adjustment.py

Test whether BA on top of uncentered VGGT cameras improves camera quality.
Uses SAM3D pred_keypoints_2d (on uncropped images) as 2D observations.

Pipeline:
  1. Load vggt_cameras.npz (uncentered), average extrinsics over valid frames.
  2. Reroot to cam_0 convention (matches eval_vggt_cameras.py).
  3. Load SAM3D pred_keypoints_2d for each camera, map to VGGT pixel space.
  4. Triangulate 3D joint positions via DLT from initial cameras + 2D obs.
  5. Run L-M BA over camera poses (k=1..K-1; cam_0 fixed at [I|0]).
  6. Report rotation/translation errors before & after BA vs GT RICH calibration.
     Optionally compare against vggt_cameras_centered.npz if present.

Usage:
    pixi run python scripts/bundle_adjustment.py \\
        --scene BBQ_001_guitar \\
        --output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --calib_root /capstor/scratch/cscs/tnanni/datasets/rich/scan_calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LOCATIONS = ["BBQ", "Gym", "LectureHall", "ParkingLot1", "ParkingLot2", "Pavallion"]
_CONF_THR = 0.3         # min SAM3D joint confidence to include
_MIN_CAMS_TRI = 2       # min cameras for DLT triangulation
_MAX_REPROJ_PX = 50.0   # max initial reprojection error to accept an observation

# MHR70 hip indices used to compute pelvis proxy (midpoint subtracted for depth-invariant δX)
_HIP_L, _HIP_R = 9, 10


# ── GT helpers (same logic as eval_vggt_cameras.py) ──────────────────────────

def scene_location(scene_name: str) -> str:
    for loc in LOCATIONS:
        if scene_name.startswith(loc):
            return loc
    raise ValueError(f"Unknown location for scene: {scene_name}")


def load_gt_ext(calib_dir: Path, cam_name: str):
    cam_idx = int(cam_name.split("_")[1])
    xml_path = calib_dir / f"{cam_idx:03d}.xml"
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    vals = list(map(float, root.find("CameraMatrix/data").text.split()))
    return np.array(vals, dtype=np.float64).reshape(3, 4)


def reroot(ext_dict: dict[str, np.ndarray], ref_name: str) -> dict[str, tuple]:
    """Reroot {name: (3,4)} extrinsics to ref_name. Returns {name: (R, t)}."""
    E0 = ext_dict[ref_name]
    R0, t0 = E0[:, :3], E0[:, 3]
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
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


# ── Geometry ──────────────────────────────────────────────────────────────────

def orig_to_vggt(kp, oc, W_orig, H_orig):
    """Map 2D keypoint from original image pixels to VGGT 518×518 space."""
    x1, y1, x2, y2 = oc
    u = x1 + float(kp[0]) * (x2 - x1) / W_orig
    v = y1 + float(kp[1]) * (y2 - y1) / H_orig
    return u, v


def triangulate_dlt(observations, proj_matrices):
    """DLT triangulation. observations: list of (u,v). proj_matrices: list of (3,4)."""
    rows = []
    for (u, v), P in zip(observations, proj_matrices):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


# ── Camera loading ────────────────────────────────────────────────────────────

def load_avg_cameras(npz_path: Path):
    """Load npz and return time-averaged per-camera poses and intrinsics."""
    d = np.load(npz_path)
    extrinsics     = d["extrinsics"].astype(np.float64)      # (T, K, 3, 4)
    intrinsics     = d["intrinsics"].astype(np.float64)      # (T, K, 3, 3)
    valid          = d["valid"]                               # (T, K) bool
    original_coords = d["original_coords"].astype(np.float64) # (T, K, 4)
    original_size   = d["original_size"].astype(np.float64)   # (T, K, 2)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in d["camera_names"]]

    K_cams = len(cam_names)
    Rs, ts, Ks, ocs, oss = [], [], [], [], []
    for k in range(K_cams):
        frames = np.where(valid[:, k])[0]
        if len(frames) == 0:
            Rs.append(np.eye(3)); ts.append(np.zeros(3))
            Ks.append(np.eye(3)); ocs.append(np.zeros(4)); oss.append(np.ones(2))
            continue
        E_mean = extrinsics[frames, k].mean(axis=0)
        U, _, Vt = np.linalg.svd(E_mean[:, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        Rs.append(R)
        ts.append(E_mean[:, 3])
        Ks.append(intrinsics[frames, k].mean(axis=0))
        ocs.append(original_coords[frames, k].mean(axis=0))
        oss.append(original_size[frames, k].mean(axis=0))

    return cam_names, Rs, ts, Ks, ocs, oss


def rerooted_cameras(cam_names, Rs, ts):
    """Reroot to cam_0 → returns Rs_rr, ts_rr where cam_0 is [I|0]."""
    R0, t0 = Rs[0], ts[0]
    Rs_rr, ts_rr = [], []
    for k in range(len(cam_names)):
        R_rel = Rs[k] @ R0.T
        t_rel = ts[k] - R_rel @ t0
        Rs_rr.append(R_rel)
        ts_rr.append(t_rel)
    return Rs_rr, ts_rr


# ── Body data loading ─────────────────────────────────────────────────────────

def load_body_data(scene_dir: Path, cam_names: list[str]):
    """Load SAM3D pred_keypoints_2d for all cameras and persons.

    Returns list of K dicts: {pid: {frame_to_local: {global→local}, kp2d: array}}.
    """
    all_cam_data = []
    for cam_name in cam_names:
        body_dir = scene_dir / cam_name / "body_data"
        cam_data = {}
        if body_dir.exists():
            for npz_path in sorted(body_dir.glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                d = np.load(npz_path, allow_pickle=False)
                if "pred_keypoints_2d" not in d.files or "frame_indices" not in d.files:
                    continue
                fi = d["frame_indices"].astype(int)
                cam_data[pid] = {
                    "frame_to_local": {int(g): int(l) for l, g in enumerate(fi)},
                    "kp2d": d["pred_keypoints_2d"],  # (T_local, J, 2 or 3)
                }
        all_cam_data.append(cam_data)
    return all_cam_data


# ── Build observations ────────────────────────────────────────────────────────

def build_observations(cam_names, Rs_rr, ts_rr, Ks, ocs, oss, all_cam_data):
    """Triangulate 3D per (frame, person, joint) and collect (k, X3d, uv, w) tuples.

    Returns obs_k (N,), obs_X (N,3), obs_uv (N,2), obs_w (N,) arrays.
    """
    K_cams = len(cam_names)
    Pmats = [Ks[k] @ np.hstack([Rs_rr[k], ts_rr[k].reshape(3, 1)]) for k in range(K_cams)]

    all_pids = sorted({pid for cam in all_cam_data for pid in cam})

    obs_k_list, obs_X_list, obs_uv_list, obs_w_list = [], [], [], []
    tri_total = tri_ok = obs_total = 0

    for pid in all_pids:
        # frames where ≥ _MIN_CAMS_TRI cameras see this person
        frame_to_cams: dict[int, list[int]] = {}
        for k, cam in enumerate(all_cam_data):
            if pid not in cam:
                continue
            for g in cam[pid]["frame_to_local"]:
                frame_to_cams.setdefault(g, []).append(k)
        valid_frames = {g: ks for g, ks in frame_to_cams.items() if len(ks) >= _MIN_CAMS_TRI}

        # number of joints from first available camera
        sample_cam = next(k for k in range(K_cams) if pid in all_cam_data[k])
        J = all_cam_data[sample_cam][pid]["kp2d"].shape[1]
        has_conf = all_cam_data[sample_cam][pid]["kp2d"].shape[2] >= 3

        for global_t, vis_cams in valid_frames.items():
            for j in range(J):
                tri_total += 1
                # collect valid 2D observations for this joint across cameras
                uvs_for_tri, pmats_for_tri, all_obs = [], [], []

                for k in vis_cams:
                    cam_d = all_cam_data[k][pid]
                    local_t = cam_d["frame_to_local"][global_t]
                    kp = cam_d["kp2d"][local_t, j]
                    w = float(kp[2]) if has_conf else 1.0
                    if w < _CONF_THR:
                        continue
                    u, v = orig_to_vggt(kp, ocs[k], oss[k][0], oss[k][1])
                    if not (0.0 <= u < ocs[k][2] and 0.0 <= v < ocs[k][3]):
                        continue
                    uvs_for_tri.append((u, v))
                    pmats_for_tri.append(Pmats[k])
                    all_obs.append((k, u, v, w))

                if len(uvs_for_tri) < _MIN_CAMS_TRI:
                    continue

                try:
                    X3d = triangulate_dlt(uvs_for_tri, pmats_for_tri)
                except Exception:
                    continue
                if not np.all(np.isfinite(X3d)):
                    continue

                # filter out observations with poor initial reprojection
                good_obs = []
                for k, u, v, w in all_obs:
                    Xc = Rs_rr[k] @ X3d + ts_rr[k]
                    if Xc[2] < 1e-6:
                        continue
                    xh = Ks[k] @ Xc
                    uv_proj = xh[:2] / xh[2]
                    if np.linalg.norm(uv_proj - np.array([u, v])) > _MAX_REPROJ_PX:
                        continue
                    good_obs.append((k, u, v, w))

                if len(good_obs) < _MIN_CAMS_TRI:
                    continue

                tri_ok += 1
                for k, u, v, w in good_obs:
                    obs_k_list.append(k)
                    obs_X_list.append(X3d)
                    obs_uv_list.append([u, v])
                    obs_w_list.append(w)
                    obs_total += 1

    print(f"  triangulation: {tri_ok}/{tri_total} joints succeeded, "
          f"{obs_total} total observations")

    if not obs_X_list:
        return np.array([]), np.zeros((0, 3)), np.zeros((0, 2)), np.array([])

    return (
        np.array(obs_k_list, dtype=np.int32),
        np.array(obs_X_list, dtype=np.float64),
        np.array(obs_uv_list, dtype=np.float64),
        np.array(obs_w_list, dtype=np.float64),
    )


# ── Bundle adjustment ─────────────────────────────────────────────────────────

def _unpack_params(params, K_count, Ks_base, R0, t0):
    """Unpack flat params into (Rs, ts, Ks).

    Layout: [rv_1,t_1, ..., rv_{K-1},t_{K-1},  dcx_0,dcy_0, ..., dcx_{K-1},dcy_{K-1}]
    Poses for k=1..K-1 (cam_0 fixed). K offsets (dcx, dcy) for ALL cameras.
    f is kept fixed — VGGT gets it right.
    """
    Rs = [R0]
    ts = [t0]
    for i in range(K_count - 1):
        Rs.append(SciR.from_rotvec(params[i*6:i*6+3]).as_matrix())
        ts.append(params[i*6+3:i*6+6])

    pose_end = 6 * (K_count - 1)
    Ks_cur = []
    for k in range(K_count):
        dcx = params[pose_end + 2*k]
        dcy = params[pose_end + 2*k + 1]
        K_k = Ks_base[k].copy()
        K_k[0, 2] += dcx
        K_k[1, 2] += dcy
        Ks_cur.append(K_k)

    return Rs, ts, Ks_cur


def ba_residuals(params, Ks_base, obs_k, obs_X, obs_uv, obs_w, R0, t0):
    """Vectorised reprojection residuals. Cam 0 pose fixed; all K optimised."""
    K_count = len(Ks_base)
    Rs_cur, ts_cur, Ks_cur = _unpack_params(params, K_count, Ks_base, R0, t0)

    N = len(obs_k)
    res = np.zeros(2 * N, dtype=np.float64)

    for k in range(K_count):
        mask = obs_k == k
        if not mask.any():
            continue
        X  = obs_X[mask]
        uv = obs_uv[mask]
        w  = obs_w[mask]

        Xc    = (Rs_cur[k] @ X.T).T + ts_cur[k]
        valid = Xc[:, 2] > 1e-6
        xh    = (Ks_cur[k] @ Xc.T).T
        uv_proj = np.where(valid[:, None], xh[:, :2] / np.maximum(xh[:, 2:3], 1e-6), 0.0)
        r = w[:, None] * (uv_proj - uv)
        r[~valid] = 0.0

        idx = np.where(mask)[0]
        res[idx * 2]     = r[:, 0]
        res[idx * 2 + 1] = r[:, 1]

    return res


def run_ba(Rs_rr, ts_rr, Ks, obs_k, obs_X, obs_uv, obs_w):
    """Run L-M BA: fix cam_0 pose, optimise poses k=1..K-1 + (dcx,dcy) per camera."""
    K_count = len(Rs_rr)
    R0, t0 = Rs_rr[0], ts_rr[0]

    # initial pose params (k=1..K-1) + zero PP offsets for all cameras
    x0 = np.concatenate([
        np.concatenate([SciR.from_matrix(Rs_rr[k]).as_rotvec(), ts_rr[k]])
        for k in range(1, K_count)
    ] + [np.zeros(2 * K_count)])   # dcx=0, dcy=0 for each camera

    result = least_squares(
        ba_residuals, x0,
        args=(Ks, obs_k, obs_X, obs_uv, obs_w, R0, t0),
        method="lm",
        max_nfev=500,
        verbose=1,
    )

    Rs_new, ts_new, _ = _unpack_params(result.x, K_count, Ks, R0, t0)

    # extract estimated PP corrections in VGGT pixels
    pose_end = 6 * (K_count - 1)
    pp_corrections = [(result.x[pose_end + 2*k], result.x[pose_end + 2*k + 1])
                      for k in range(K_count)]

    return Rs_new, ts_new, pp_corrections, result


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(cam_names, Rs, ts, gt_rel):
    """Compare cameras to GT. Returns list of per-cam dicts with error metrics."""
    rows = []
    scale_samples = []
    cam_data = {}
    for k, name in enumerate(cam_names[1:], start=1):
        if name not in gt_rel:
            continue
        R_gt, t_gt = gt_rel[name]
        C_gt = cam_center(R_gt, t_gt)
        C_vg = cam_center(Rs[k], ts[k])
        d_gt, d_vg = np.linalg.norm(C_gt), np.linalg.norm(C_vg)
        cam_data[name] = (R_gt, Rs[k], C_gt, C_vg, d_gt, d_vg)
        if d_vg > 1e-9:
            scale_samples.append(d_gt / d_vg)

    scene_scale = float(np.median(scale_samples)) if scale_samples else np.nan
    for name, (R_gt, R_vg, C_gt, C_vg, d_gt, d_vg) in cam_data.items():
        t_err = float(np.linalg.norm(C_vg * scene_scale - C_gt)) if not np.isnan(scene_scale) else np.nan
        rows.append({
            "cam": name,
            "rot_err": rot_err_deg(R_vg, R_gt),
            "dir_err": angle_between(C_gt, C_vg),
            "t_err_cm": t_err * 100,
        })
    return rows


def print_results(label, rows):
    print(f"\n--- {label} ---")
    for r in rows:
        print(f"  {r['cam']}: rot={r['rot_err']:.2f}°  dir={r['dir_err']:.2f}°  "
              f"t={r['t_err_cm']:.1f}cm")
    if rows:
        rot_mean = np.mean([r['rot_err'] for r in rows])
        t_mean   = np.nanmean([r['t_err_cm'] for r in rows])
        print(f"  MEAN: rot={rot_mean:.2f}°  t={t_mean:.1f}cm")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--calib_root",  required=True, type=Path)
    args = parser.parse_args()

    scene_dir = args.output_root / args.scene
    calib_dir = args.calib_root / scene_location(args.scene) / "calibration"
    print(f"Scene : {args.scene}")
    print(f"Dir   : {scene_dir}")

    # ── VGGT uncentered cameras ───────────────────────────────────────────────
    npz_unc = scene_dir / "vggt_cameras.npz"
    if not npz_unc.exists():
        raise FileNotFoundError(f"vggt_cameras.npz not found in {scene_dir}")
    cam_names, Rs, ts, Ks, ocs, oss = load_avg_cameras(npz_unc)
    Rs_rr, ts_rr = rerooted_cameras(cam_names, Rs, ts)
    print(f"Cameras ({len(cam_names)}): {cam_names}")

    # ── GT cameras ───────────────────────────────────────────────────────────
    gt_raw = {}
    for name in cam_names:
        E = load_gt_ext(calib_dir, name)
        if E is not None:
            gt_raw[name] = E
    if cam_names[0] not in gt_raw:
        raise RuntimeError(f"Reference camera {cam_names[0]} has no GT calibration")
    gt_rel = reroot(gt_raw, cam_names[0])

    # ── Baseline: uncentered VGGT ─────────────────────────────────────────────
    rows_unc = print_results("Uncentered VGGT (baseline)", evaluate(cam_names, Rs_rr, ts_rr, gt_rel))

    # ── Optional: centered VGGT for reference ─────────────────────────────────
    rows_cen = None
    npz_cen = scene_dir / "vggt_cameras_centered.npz"
    if npz_cen.exists():
        _, Rs_c, ts_c, Ks_c, _, _ = load_avg_cameras(npz_cen)
        Rs_c_rr, ts_c_rr = rerooted_cameras(cam_names, Rs_c, ts_c)
        rows_cen = print_results("Centered VGGT (reference)", evaluate(cam_names, Rs_c_rr, ts_c_rr, gt_rel))

    # ── Body data + observations ──────────────────────────────────────────────
    print("\nLoading body data...")
    all_cam_data = load_body_data(scene_dir, cam_names)
    for k, name in enumerate(cam_names):
        pids = list(all_cam_data[k].keys())
        print(f"  {name}: PIDs {pids}")

    print("\nBuilding observations (triangulate + filter)...")
    obs_k, obs_X, obs_uv, obs_w = build_observations(
        cam_names, Rs_rr, ts_rr, Ks, ocs, oss, all_cam_data
    )
    if len(obs_k) == 0:
        print("No valid observations — cannot run BA."); return

    # ── Bundle adjustment ─────────────────────────────────────────────────────
    K_count = len(cam_names)
    x0 = np.concatenate([
        np.concatenate([SciR.from_matrix(Rs_rr[k]).as_rotvec(), ts_rr[k]])
        for k in range(1, K_count)
    ] + [np.zeros(2 * K_count)])
    res_before = ba_residuals(x0, Ks, obs_k, obs_X, obs_uv, obs_w, Rs_rr[0], ts_rr[0])
    rms_before = float(np.sqrt((res_before ** 2).mean()))

    n_pose_dof = 6 * (K_count - 1)
    n_k_dof    = 2 * K_count
    print(f"\nRunning BA over {K_count} cameras "
          f"({len(obs_k)} observations, {n_pose_dof} pose DOF + {n_k_dof} PP DOF)...")
    Rs_ba, ts_ba, pp_corrections, result = run_ba(Rs_rr, ts_rr, Ks, obs_k, obs_X, obs_uv, obs_w)
    rms_after = float(np.sqrt((result.fun ** 2).mean()))
    print(f"BA: cost {result.cost:.3f}  nfev={result.nfev}  "
          f"RMS reprojection: {rms_before:.2f}px → {rms_after:.2f}px")

    print("\nEstimated PP corrections (Δcx, Δcy in VGGT pixels):")
    for name, (dcx, dcy) in zip(cam_names, pp_corrections):
        print(f"  {name}: Δcx={dcx:+.1f}  Δcy={dcy:+.1f}")

    rows_ba = print_results("After BA (poses + K)", evaluate(cam_names, Rs_ba, ts_ba, gt_rel))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    header = f"{'Method':<30}  {'rot (°)':>8}  {'t (cm)':>8}"
    print(header)
    print("-"*55)

    def summary_row(label, rows):
        rot = np.mean([r['rot_err'] for r in rows])
        t   = np.nanmean([r['t_err_cm'] for r in rows])
        print(f"  {label:<28}  {rot:>8.2f}  {t:>8.1f}")

    summary_row("Uncentered VGGT", rows_unc)
    summary_row("After BA (poses + K)", rows_ba)
    if rows_cen:
        summary_row("Centered VGGT (reference)", rows_cen)
    print("="*55)


if __name__ == "__main__":
    main()
