"""
Debug script for BBQ_001_guitar: load all pipeline inputs and run task 1.

Saves plots to debug_outputs/BBQ_001_guitar/.

Usage:
    pixi run python debug/debug_scene.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.evaluate_on_rich_test import (
    load_scene_body_data, build_fusion_tensors, load_fusion_model,
    load_gt_extrinsics, load_gt_body_data, _6d_to_aa, _BODY_JOINT_IDX,
    metric_wa_mpjpe, metric_w_mpjpe,
    _RICH_ORIG_W, _RICH_ORIG_H,
)
from fusion.placer import BodyPlacer, _SMPLX_TO_MHR70_ALIGN, _SCALE_BONE_CANDIDATES

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────────
SCENE       = "BBQ_001_guitar"
GHOST_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT   = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
CHECKPOINT  = Path("/users/tnanni/ghost/checkpoints/fusion_module_latest/best.pt")
SMPLX_MODEL = Path("/users/tnanni/ghost/body_models/SMPLX_NEUTRAL.pkl")
DEVICE      = torch.device("cpu")
GT_SPLIT    = "train"
PERSON_ID   = 1

OUT_DIR = Path("debug_outputs") / SCENE
OUT_DIR.mkdir(parents=True, exist_ok=True)

scene_dir = GHOST_ROOT / SCENE


# ══════════════════════════════════════════════════════════════════════════════
# 1. Predicted cameras (VGGT)
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 1. Predicted cameras ===")
cam_npz    = np.load(scene_dir / "vggt_cameras.npz", mmap_mode="r")
pred_exts  = cam_npz["extrinsics"]    # (T, K, 3, 4)
pred_intrs = cam_npz["intrinsics"]    # (T, K, 3, 3)
cam_valid  = cam_npz["valid"]         # (T, K)
cam_names  = [n.decode() if isinstance(n, bytes) else n for n in cam_npz["camera_names"]]
T_full, K  = cam_valid.shape

log.info(f"  T={T_full}  K={K}  cameras={cam_names}")
log.info(f"  valid frames per cam: {cam_valid.sum(0).astype(int).tolist()}")

# Median pred extrinsic per camera (static cameras)
pred_exts_med = np.zeros((K, 3, 4), dtype=np.float64)
for ki in range(K):
    vmask = cam_valid[:, ki].astype(bool)
    if not vmask.any():
        continue
    E = pred_exts[vmask, ki].astype(np.float64)
    pred_exts_med[ki, :, :3] = np.median(E[:, :, :3], axis=0)
    pred_exts_med[ki, :,  3] = np.median(E[:, :,  3], axis=0)

log.info(f"  pred cam_00 extrinsic (should be [I|0]):\n{pred_exts_med[0].round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GT cameras
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 2. GT cameras ===")
gt_exts_list = load_gt_extrinsics(SCENE, RICH_ROOT)   # list of (3,4)
log.info(f"  {len(gt_exts_list)} GT cameras")

ref_idx  = int(re.search(r"\d+", cam_names[0]).group())   # 0 for cam_00
E0_gt    = np.array(gt_exts_list[ref_idx], dtype=np.float64)
R0_gt, t0_gt = E0_gt[:3, :3], E0_gt[:3, 3]

# Re-root GT extrinsics so cam_00 = [I|0], same convention as VGGT
gt_exts = np.zeros((K, 3, 4), dtype=np.float64)
for ki, cn in enumerate(cam_names):
    gidx = int(re.search(r"\d+", cn).group())
    if gidx >= len(gt_exts_list):
        continue
    Ek  = np.array(gt_exts_list[gidx], dtype=np.float64)
    Rk  = Ek[:3, :3] @ R0_gt.T
    tk  = Ek[:3, 3] - Ek[:3, :3] @ R0_gt.T @ t0_gt
    gt_exts[ki] = np.hstack([Rk, tk[:, None]])

log.info(f"  gt cam_00 extrinsic (should be [I|0]):\n{gt_exts[0].round(4)}")

# Camera rotation and translation errors (pred vs GT)
log.info("  Camera errors (pred vs GT, at GT scale):")
gt_scale_ratios = []
for ki in range(1, K):                        # skip cam_00 (reference = identity)
    Rp = pred_exts_med[ki, :3, :3]
    tp = pred_exts_med[ki, :3,  3]
    Rg = gt_exts[ki, :3, :3]
    tg = gt_exts[ki, :3,  3]
    R_err = Rp @ Rg.T
    ang   = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
    pn, gn = np.linalg.norm(tp), np.linalg.norm(tg)
    if pn > 1e-6:
        gt_scale_ratios.append(gn / pn)
    log.info(f"    {cam_names[ki]}: rot_err={ang:.2f}°  |t_pred|={pn:.4f}  |t_gt|={gn:.4f}m")

ratios_arr = np.array(gt_scale_ratios)
for ki in range(1, K):
    if ki - 1 < len(ratios_arr):
        log.info(f"    {cam_names[ki]}: ratio={ratios_arr[ki-1]:.4f}")
gt_scale = float(np.median(ratios_arr)) if len(ratios_arr) else float("nan")
log.info(f"  GT scale: median={gt_scale:.4f}  mean={ratios_arr.mean():.4f}  "
         f"std={ratios_arr.std():.4f}  min={ratios_arr.min():.4f}  max={ratios_arr.max():.4f}")

# Load raw XML intrinsics (sensor space, unscaled) — will be mapped to VGGT
# space per-camera in section 6 once orig_coords/orig_size are available.
location_str = re.match(r"^(.+?)_\d{3}_", SCENE).group(1)
calib_dir_gt = RICH_ROOT / "scan_calibration" / location_str / "calibration"
_gt_K_xml: list[np.ndarray] = []   # raw sensor-space K per XML file
for _xml in sorted(calib_dir_gt.glob("*.xml")):
    _inode = ET.parse(_xml).getroot().find("Intrinsics")
    if _inode is None:
        continue
    _Kraw = np.array(list(map(float, _inode.find("data").text.split())),
                     dtype=np.float64).reshape(3, 3)
    _gt_K_xml.append(_Kraw)
log.info(f"  Loaded {len(_gt_K_xml)} GT intrinsics from XML "
         f"(sensor assumed {_RICH_ORIG_W}×{_RICH_ORIG_H} — verify with raw images)")



# ══════════════════════════════════════════════════════════════════════════════
# 3. Body data → fused pose + betas
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 3. Fused pose + betas ===")
cam_dirs, raw = load_scene_body_data(scene_dir)
log.info(f"  {len(cam_dirs)} cameras with body data")

pose_t, mask_t, shape_t, all_pids, frame_start = build_fusion_tensors(raw)
T = pose_t.shape[1]
pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}
log.info(f"  frame_start={frame_start}  T={T}  pids={all_pids}")

fusion_model = load_fusion_model(CHECKPOINT, DEVICE)
with torch.no_grad():
    fused_pose_t, _ = fusion_model(
        pose_t.to(DEVICE), mask_t.to(DEVICE), shape=shape_t.to(DEVICE)
    )
fused_pose  = fused_pose_t[0].cpu().numpy()    # (T, P, 54, 6)
log.info(f"  fused_pose: {fused_pose.shape}")

# Mean SAM3D betas per pid (fallback)
sam3d_betas_acc: dict[int, list] = {}
for cam_dir in cam_dirs:
    for pid in all_pids:
        bf = cam_dir / "body_data" / f"person_{pid}.npz"
        if not bf.exists():
            continue
        d = np.load(bf, allow_pickle=False)
        if "smplx_betas" in d.files:
            sam3d_betas_acc.setdefault(pid, []).append(d["smplx_betas"].mean(0))

betas_by_pid: dict[int, np.ndarray] = {}
for pid in all_pids:
    if pid in sam3d_betas_acc:
        betas_by_pid[pid] = np.mean(sam3d_betas_acc[pid], axis=0).astype(np.float32)
    else:
        betas_by_pid[pid] = np.zeros(10, np.float32)
    log.info(f"  pid={pid} betas (first 5): {betas_by_pid[pid][:5].round(3)}")

fused_pose_by_pid = {pid: fused_pose[:, pid_to_slot[pid]] for pid in all_pids}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Scale estimation
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 4. Scale ===")
placer = BodyPlacer(scene_dir, SMPLX_MODEL)

sam3d_betas_map = {
    cam_dir / "body_data" / f"person_{pid}.npz": betas_by_pid[pid]
    for cam_dir in cam_dirs for pid in all_pids
    if (cam_dir / "body_data" / f"person_{pid}.npz").exists()
}
pred_scale = placer.load_mapanything_scale()
if pred_scale is not None:
    log.info("  Using MapAnything scale")
else:
    log.info("  MapAnything scale not found — estimating via triangulation ...")
    pred_scale = placer.estimate_scale_triangulated(
        fused_pose_by_pid=fused_pose_by_pid,
        fused_betas_map=sam3d_betas_map,
        frame_start=frame_start,
    )   # (T_full,) per-frame

# Per-bone calibration factors (L_fk / (L_vggt × gt_scale)), from compute_bone_calibration.py.
_CALIB_FACTORS = dict(zip(_SCALE_BONE_CANDIDATES,
                           [0.97, 0.96, 0.91, 0.86, 1.03, 1.04, 0.95, 1.01]))
_valid_bones = [(a, b) for a, b in _SCALE_BONE_CANDIDATES
                if a in placer._mapper._index and b in placer._mapper._index]

valid_sc = pred_scale[pred_scale > 0]
log.info(f"  scale: mean={valid_sc.mean():.4f}  std={valid_sc.std():.4f}  "
         f"min={valid_sc.min():.4f}  max={valid_sc.max():.4f}  "
         f"valid={len(valid_sc)}/{T_full}")
log.info(f"  GT scale={gt_scale:.4f}  pred mean={valid_sc.mean():.4f}  "
         f"error={( valid_sc.mean() - gt_scale) / gt_scale * 100:+.1f}%")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(pred_scale, lw=0.8, label="pred_scale (per frame)")
ax.axhline(valid_sc.mean(), color="r", ls="--", label=f"mean={valid_sc.mean():.3f}")
ax.axhline(gt_scale, color="g", ls="--", label=f"gt_scale={gt_scale:.3f}")
ax.set(xlabel="frame", ylabel="scale", title="Per-frame pred scale vs GT scale")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "scale.png", dpi=120)
plt.close()
log.info(f"  → saved {OUT_DIR / 'scale.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMPL-X 2D keypoints (pred_keypoints_2d from SAM3D, per camera)
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 5. SMPL-X 2D keypoints ===")
# pred_keypoints_2d: (N_valid, 70, 2) in original image coords (1440-wide)
# pred_joint_confidence: (N_valid, 70)
kps_2d:  dict[str, np.ndarray] = {}   # cam -> (T_full, 70, 2)
kps_conf: dict[str, np.ndarray] = {}  # cam -> (T_full, 70)

for cam_dir in cam_dirs:
    cn = cam_dir.name
    bf = cam_dir / "body_data" / f"person_{PERSON_ID}.npz"
    if not bf.exists():
        continue
    d  = np.load(bf, allow_pickle=False)
    fi     = d["frame_indices"].astype(int)
    fi_rel = fi - frame_start
    ok     = (fi_rel >= 0) & (fi_rel < T_full)

    kp_full   = np.full((T_full, 70, 2), np.nan, np.float32)
    kp_full[fi_rel[ok]] = d["pred_keypoints_2d"][ok]

    raw_conf  = d["pred_joint_confidence"] if "pred_joint_confidence" in d.files \
                else np.ones((len(fi), 70), np.float32)
    n_conf    = raw_conf.shape[1]
    conf_full = np.full((T_full, n_conf), np.nan, np.float32)
    conf_full[fi_rel[ok]] = raw_conf[ok]
    kps_2d[cn]   = kp_full
    kps_conf[cn] = conf_full

    log.info(f"  {cn}: mean_conf={np.nanmean(conf_full):.3f}  "
             f"frames={np.isfinite(kp_full[:,0,0]).sum()}/{T_full}")

log.info(f"Outputs saved to {OUT_DIR}/")
log.info("Done — task 1 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DLT reprojection error: pred cameras vs GT cameras
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 6. DLT reprojection error ===")

orig_coords = cam_npz["original_coords"]   # (T, K, 4)  [0,0,W_vggt,H_vggt]
orig_size   = cam_npz["original_size"]     # (T, K, 2)  [W_orig, H_orig]

# Build per-camera GT intrinsics in VGGT space.
# scale_x = W_vggt / W_sensor, scale_y = H_vggt / H_sensor
# W_vggt, H_vggt read from orig_coords; W_sensor=_RICH_ORIG_W, H_sensor=_RICH_ORIG_H
gt_K_vggt = np.zeros((K, 3, 3), dtype=np.float64)
for ki, cn in enumerate(cam_names):
    gidx = int(re.search(r"\d+", cn).group())
    if gidx >= len(_gt_K_xml):
        continue
    vt_ki = np.argwhere(cam_valid[:, ki])
    if len(vt_ki) == 0:
        continue
    t0 = int(vt_ki[0, 0])
    W_vggt_ki = float(orig_coords[t0, ki, 2])
    H_vggt_ki = float(orig_coords[t0, ki, 3])
    sx = W_vggt_ki / _RICH_ORIG_W
    sy = H_vggt_ki / _RICH_ORIG_H
    K_ki = _gt_K_xml[gidx].copy()
    K_ki[0] *= sx   # scale fx, cx
    K_ki[1] *= sy   # scale fy, cy
    K_ki[2]  = [0., 0., 1.]
    gt_K_vggt[ki] = K_ki
log.info(f"  GT K VGGT-space cam_00: fx={gt_K_vggt[0,0,0]:.1f}  cx={gt_K_vggt[0,0,2]:.1f}  "
         f"(scale {orig_coords[0,0,2]:.0f}/{_RICH_ORIG_W}={orig_coords[0,0,2]/_RICH_ORIG_W:.5f})")

_JOINT_NAMES = {1: "left_hip", 2: "right_hip", 12: "neck"}

def orig_to_vggt(kp, oc):
    x1, y1, x2, y2 = oc
    return x1 + kp[0] * (x2 - x1) / W_orig, y1 + kp[1] * (y2 - y1) / H_orig

def triangulate_dlt(obs, pmats):
    rows = []
    for (u, v), P in zip(obs, pmats):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def reproject(X_world, P):
    """Return (u_proj, v_proj) in VGGT pixel space."""
    x = P @ np.append(X_world, 1.0)
    return x[0] / x[2], x[1] / x[2]

# Per-frame reprojection error for each joint, with pred cams and GT cams
# repr_errs[mode][joint_name] = list of per-observation pixel errors
repr_errs = {"pred": {n: [] for n in _JOINT_NAMES.values()},
             "gt":   {n: [] for n in _JOINT_NAMES.values()}}

# Per-frame mean reprojection error (averaged over joints and cameras)
per_frame_repr = {"pred": np.full(T_full, np.nan),
                  "gt":   np.full(T_full, np.nan)}

scale_mean = float(valid_sc.mean())

for t in range(T_full):
    s_pred = float(pred_scale[t]) if pred_scale[t] > 0 else scale_mean

    frame_errs = {"pred": [], "gt": []}

    for smplx_j, mhr_j in _SMPLX_TO_MHR70_ALIGN.items():
        jname = _JOINT_NAMES[smplx_j]
        obs, Pmats_pred, Pmats_gt = [], [], []

        for ki, cn in enumerate(cam_names):
            if not cam_valid[t, ki]:
                continue
            if cn not in kps_2d:
                continue
            kp_orig = kps_2d[cn][t, mhr_j]
            if np.any(np.isnan(kp_orig)):
                continue

            oc      = orig_coords[t, ki]
            W_orig  = float(orig_size[t, ki, 0])
            H_orig  = float(orig_size[t, ki, 1])
            u, v    = orig_to_vggt(kp_orig, oc)
            if not (0 <= u < oc[2] and 0 <= v < oc[3]):
                continue

            K_intr = pred_intrs[t, ki].astype(np.float64)

            E_pred = pred_exts[t, ki].astype(np.float64).copy()
            E_pred[:, 3] *= s_pred
            Pmats_pred.append(K_intr @ E_pred)

            # GT: same VGGT-space obs, GT intrinsics mapped to VGGT space
            E_gt = gt_exts[ki].astype(np.float64)
            Pmats_gt.append(pred_intrs[t, ki].astype(np.float64) @ E_gt)

            obs.append((u, v))

        if len(obs) < 2:
            continue

        # Both sets of errors are now in VGGT px
        X_pred = triangulate_dlt(obs, Pmats_pred)
        for (u_obs, v_obs), P in zip(obs, Pmats_pred):
            u_p, v_p = reproject(X_pred, P)
            err = np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2)
            repr_errs["pred"][jname].append(err)
            frame_errs["pred"].append(err)

        X_gt = triangulate_dlt(obs, Pmats_gt)
        for (u_obs, v_obs), P in zip(obs, Pmats_gt):
            u_p, v_p = reproject(X_gt, P)
            err = np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2)
            repr_errs["gt"][jname].append(err)
            frame_errs["gt"].append(err)

    if frame_errs["pred"]:
        per_frame_repr["pred"][t] = np.mean(frame_errs["pred"])
    if frame_errs["gt"]:
        per_frame_repr["gt"][t] = np.mean(frame_errs["gt"])

# Print summary
for mode in ("pred", "gt"):
    label = "Pred cams" if mode == "pred" else "GT cams  "
    log.info(f"\n  {label}:")
    for jname in _JOINT_NAMES.values():
        errs = np.array(repr_errs[mode][jname])
        if len(errs):
            log.info(f"    {jname:<12}: mean={errs.mean():.2f}px  "
                     f"med={np.median(errs):.2f}px  "
                     f"p90={np.percentile(errs,90):.2f}px  "
                     f"max={errs.max():.2f}px  (N={len(errs)})")

# Plot 1: per-frame mean reprojection error
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(per_frame_repr["pred"], lw=0.8, label="pred cams", alpha=0.8)
ax.plot(per_frame_repr["gt"],   lw=0.8, label="GT cams",   alpha=0.8)
ax.set(xlabel="frame", ylabel="mean reprojection error (px, VGGT space)",
       title="Per-frame DLT reprojection error — pred vs GT cameras")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "reproj_per_frame.png", dpi=120)
plt.close()

# Plot 2: error distribution per joint, pred vs GT
fig, axes = plt.subplots(1, len(_JOINT_NAMES), figsize=(14, 4), sharey=True)
for ax, (smplx_j, jname) in zip(axes, _JOINT_NAMES.items()):
    ep = np.array(repr_errs["pred"][jname])
    eg = np.array(repr_errs["gt"][jname])
    ax.hist(ep, bins=50, alpha=0.6, label=f"pred (med={np.median(ep):.1f})")
    ax.hist(eg, bins=50, alpha=0.6, label=f"GT   (med={np.median(eg):.1f})")
    ax.set(title=jname, xlabel="reprojection error (px)")
    ax.legend(fontsize=7)
axes[0].set_ylabel("count")
plt.suptitle("DLT reprojection error distribution per joint")
plt.tight_layout()
plt.savefig(OUT_DIR / "reproj_distribution.png", dpi=120)
plt.close()

log.info(f"\n  → saved reproj_per_frame.png and reproj_distribution.png to {OUT_DIR}")

# ── Per-camera breakdown ──────────────────────────────────────────────────────
# repr_errs_cam[mode][cam_name][joint_name] = list of pixel errors
repr_errs_cam = {
    "pred":   {cn: {n: [] for n in _JOINT_NAMES.values()} for cn in cam_names},
    "gt":     {cn: {n: [] for n in _JOINT_NAMES.values()} for cn in cam_names},
    "hybrid": {cn: {n: [] for n in _JOINT_NAMES.values()} for cn in cam_names},
}
# hybrid = VGGT rotation + GT-scale translation (isolates rotation vs scale error)

for t in range(T_full):
    s_pred = float(pred_scale[t]) if pred_scale[t] > 0 else scale_mean

    for smplx_j, mhr_j in _SMPLX_TO_MHR70_ALIGN.items():
        jname = _JOINT_NAMES[smplx_j]
        obs, Pmats_pred, Pmats_gt, Pmats_hyb = [], [], [], []
        cam_ids = []

        for ki, cn in enumerate(cam_names):
            if not cam_valid[t, ki] or cn not in kps_2d:
                continue
            kp_orig = kps_2d[cn][t, mhr_j]
            if np.any(np.isnan(kp_orig)):
                continue
            oc     = orig_coords[t, ki]
            W_orig = float(orig_size[t, ki, 0])
            H_orig = float(orig_size[t, ki, 1])
            u, v   = orig_to_vggt(kp_orig, oc)
            if not (0 <= u < oc[2] and 0 <= v < oc[3]):
                continue

            K_intr = pred_intrs[t, ki].astype(np.float64)

            E_pred = pred_exts[t, ki].astype(np.float64).copy()
            E_pred[:, 3] *= s_pred
            Pmats_pred.append(K_intr @ E_pred)

            # GT: same VGGT-space obs, GT intrinsics mapped to VGGT space
            E_gt = gt_exts[ki].astype(np.float64)
            Pmats_gt.append(pred_intrs[t, ki].astype(np.float64) @ E_gt)

            # Hybrid: VGGT rotation + GT-scale translation
            E_hyb = pred_exts[t, ki].astype(np.float64).copy()
            E_hyb[:, 3] *= gt_scale
            Pmats_hyb.append(K_intr @ E_hyb)

            obs.append((u, v))
            cam_ids.append(ki)

        if len(obs) < 2:
            continue

        X_pred = triangulate_dlt(obs, Pmats_pred)
        X_gt   = triangulate_dlt(obs, Pmats_gt)
        X_hyb  = triangulate_dlt(obs, Pmats_hyb)

        for idx, ki in enumerate(cam_ids):
            cn = cam_names[ki]
            u_obs, v_obs = obs[idx]
            for mode, X, Pmats in [("pred",   X_pred, Pmats_pred),
                                    ("gt",     X_gt,   Pmats_gt),
                                    ("hybrid", X_hyb,  Pmats_hyb)]:
                u_p, v_p = reproject(X, Pmats[idx])
                err = np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2)
                repr_errs_cam[mode][cn][jname].append(err)

log.info(f"\nPer-camera mean reprojection error (all in VGGT px):")
log.info(f"  {'camera':<10}  {'rot_err':>8}  {'pred(VGGT)':>10}  {'hybrid(GT_sc)':>14}  {'gt(VGGT_K)':>10}")
log.info(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*8}")

# get rotation errors per camera
rot_errs_by_cam = {}
for ki in range(1, K):
    Rp = pred_exts_med[ki, :3, :3]; Rg = gt_exts[ki, :3, :3]
    R_err = Rp @ Rg.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R_err)-1)/2, -1, 1))))
    rot_errs_by_cam[cam_names[ki]] = ang

cam00_vals = []
for mode in ("pred", "hybrid", "gt"):
    errs = [e for jn in _JOINT_NAMES.values() for e in repr_errs_cam[mode]["cam_00"][jn]]
    cam00_vals.append(f"{np.mean(errs) if errs else float('nan'):>8.2f}")
log.info(f"  {'cam_00':<10}  {'(ref)':>8}  {cam00_vals[0]}  {cam00_vals[1]:>14}  {cam00_vals[2]}")

for ki in range(1, K):
    cn = cam_names[ki]
    rot_e = rot_errs_by_cam.get(cn, float("nan"))
    vals = []
    for mode in ("pred", "hybrid", "gt"):
        errs = [e for jn in _JOINT_NAMES.values() for e in repr_errs_cam[mode][cn][jn]]
        vals.append(f"{np.mean(errs) if errs else float('nan'):>8.2f}")
    log.info(f"  {cn:<10}  {rot_e:>7.2f}°  {vals[0]}  {vals[1]:>14}  {vals[2]}")

log.info("\nOverall mean across all cameras:")
for mode, label in [("pred","pred cams"), ("hybrid","hybrid(GT scale)"), ("gt","GT cams")]:
    all_e = [e for cn in cam_names for jn in _JOINT_NAMES.values()
             for e in repr_errs_cam[mode][cn][jn]]
    log.info(f"  {label:<18}: {np.mean(all_e):.2f}px  med={np.median(all_e):.2f}px")

log.info("Done.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. DLT + Procrustes with pred_scale vs gt_scale → WA/W-MPJPE
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 7. DLT + Procrustes evaluation ===")

gt_body_data = load_gt_body_data(SCENE, RICH_ROOT, split=GT_SPLIT)
if not gt_body_data:
    log.info("  No GT body data found, skipping evaluation.")
else:
    import torch
    from fusion.placer import _SMPLX_TO_MHR70
    # Only use joints the mapper actually supports
    _SMPLX_TO_MHR70_EVAL = {smplx_j: mhr_j for smplx_j, mhr_j in _SMPLX_TO_MHR70.items()
                             if mhr_j in placer._mapper._index}

    J_body   = len(_BODY_JOINT_IDX)
    pid      = PERSON_ID
    betas_p  = betas_by_pid.get(pid, np.zeros(10, np.float32))

    # ── Precompute pred FK (batched, one call) ────────────────────────────────
    log.info("  Precomputing FK for all frames …")
    _N_KPS = placer._mapper._R.shape[0]
    _body_poses = np.stack([
        _6d_to_aa(fused_pose_by_pid[pid][t, :21]).reshape(63) for t in range(T_full)
    ])  # (T, 63)
    _betas_rep = np.tile(betas_p[np.newaxis], (T_full, 1))  # (T, 10)
    _fk, _verts = placer._smplx_fk(
        _betas_rep, _body_poses, np.zeros((T_full, 3), np.float32), return_verts=True,
    )
    J_can_all  = _fk                             # (T, 55, 3)
    kp_can_all = placer._mapper.map(_verts)      # (T, N_KPS, 3)
    torch.cuda.empty_cache()

    # ── Calibration-corrected scale ───────────────────────────────────────────
    # Same logic as estimate_scale_triangulated, but:
    #   - cameras scaled by pred_scale[t] so triangulated lengths are in metres
    #   - s_sample = pred_scale[t] × L_fk / (factor[bone] × L_metric)
    # Mathematically equivalent to L_fk / (factor × L_vggt_unscaled), but uses
    # metric cameras so the triangulation geometry matches section 7's DLT.
    def _compute_calib_scale() -> np.ndarray:
        from collections import defaultdict
        frame_samples_c: dict[int, list[float]] = defaultdict(list)
        for t_rel in range(T_full):
            s_t = float(pred_scale[t_rel]) if pred_scale[t_rel] > 0 else float(valid_sc.mean())
            for bone in _valid_bones:
                mhr_a, mhr_b = bone
                pts_a: list = []; pts_b: list = []; Pmats_s: list = []; fk_lens: list = []
                for ki, cn in enumerate(cam_names):
                    if not cam_valid[t_rel, ki] or cn not in kps_2d:
                        continue
                    ka = kps_2d[cn][t_rel, mhr_a]
                    kb = kps_2d[cn][t_rel, mhr_b]
                    if not (np.isfinite(ka).all() and np.isfinite(kb).all()):
                        continue
                    oc = placer.original_coords[t_rel, ki]
                    Wo = float(placer.original_size[t_rel, ki, 0])
                    Ho = float(placer.original_size[t_rel, ki, 1])
                    ua, va = placer._orig_to_vggt(ka, oc, Wo, Ho)
                    ub, vb = placer._orig_to_vggt(kb, oc, Wo, Ho)
                    if not (placer._in_bounds(ua, va, oc[2], oc[3]) and
                            placer._in_bounds(ub, vb, oc[2], oc[3])):
                        continue
                    E_s = pred_exts[t_rel, ki].astype(np.float64).copy()
                    E_s[:3, 3] *= s_t
                    Pmats_s.append(pred_intrs[t_rel, ki].astype(np.float64) @ E_s)
                    pts_a.append((ua, va))
                    pts_b.append((ub, vb))
                    ia = placer._mapper.index(mhr_a)
                    ib = placer._mapper.index(mhr_b)
                    L = float(np.linalg.norm(kp_can_all[t_rel, ib] - kp_can_all[t_rel, ia]))
                    if L > 0.01:
                        fk_lens.append(L)
                if len(pts_a) < 2 or not fk_lens:
                    continue
                try:
                    Xa = placer._triangulate_dlt(pts_a, Pmats_s)
                    Xb = placer._triangulate_dlt(pts_b, Pmats_s)
                except Exception:
                    continue
                L_metric = float(np.linalg.norm(Xb - Xa))
                if L_metric < 1e-3:
                    continue
                L_fk = float(np.median(fk_lens))
                factor = _CALIB_FACTORS.get(bone, 1.0)
                s_corr = s_t * L_fk / (factor * L_metric)
                if 0.1 < s_corr < 100.0:
                    frame_samples_c[t_rel].append(s_corr)
        all_s = [s for sl in frame_samples_c.values() for s in sl]
        if not all_s:
            return pred_scale.copy()
        global_med = float(np.median(all_s))
        result = np.full(T_full, global_med, dtype=np.float32)
        for t_rel, slist in frame_samples_c.items():
            result[t_rel] = float(np.median(slist))
        return result

    calib_scale = _compute_calib_scale()
    valid_calib_sc = calib_scale[calib_scale > 0]
    log.info(f"  calib scale: mean={valid_calib_sc.mean():.4f}  "
             f"error={( valid_calib_sc.mean() - gt_scale) / gt_scale * 100:+.1f}%")

    # ── GT joints (reference) + GT FK (batched) ───────────────────────────────
    gt_pid    = next(iter(gt_body_data))
    gt_items  = [(fi - frame_start, p) for fi, p in gt_body_data[gt_pid].items()
                 if 0 <= fi - frame_start < T_full]
    gt_t_rels = [t for t, _ in gt_items]

    _gt_betas  = np.stack([p["betas"].reshape(-1)[:10] for _, p in gt_items])
    _gt_bp     = np.stack([p["betas"].reshape(-1)[:10] for _, p in gt_items])  # placeholder shape
    _gt_bp     = np.stack([p["body_pose"] for _, p in gt_items])
    _gt_go     = np.stack([p["global_orient"] for _, p in gt_items])
    _gt_transl = np.stack([p["transl"] for _, p in gt_items])

    # GT joints in world frame (with GT global_orient + transl)
    _fk_gt_world = placer._smplx_fk(_gt_betas, _gt_bp, _gt_go)  # (N_gt, 55, 3)
    gt_joints = np.full((T_full, 1, J_body, 3), np.nan, dtype=np.float32)
    for i, t_rel in enumerate(gt_t_rels):
        gt_joints[t_rel, 0] = (_fk_gt_world[i] + _gt_transl[i])[_BODY_JOINT_IDX]

    # GT FK with zero global_orient (for Procrustes B-side)
    _fk_gt_can, _verts_gt = placer._smplx_fk(
        _gt_betas, _gt_bp, np.zeros((len(gt_t_rels), 3), np.float32), return_verts=True,
    )
    J_can_gt_all  = np.full((T_full, 55,     3), np.nan, np.float32)
    kp_can_gt_all = np.full((T_full, _N_KPS, 3), np.nan, np.float32)
    _kp_gt = placer._mapper.map(_verts_gt)  # (N_gt, N_KPS, 3)
    for i, t_rel in enumerate(gt_t_rels):
        J_can_gt_all[t_rel]  = _fk_gt_can[i]
        kp_can_gt_all[t_rel] = _kp_gt[i]
    torch.cuda.empty_cache()

    # ── Core DLT + Procrustes loop using all _SMPLX_TO_MHR70 joints ───────────
    def _dlt_procrustes_eval(get_pmat, label, min_joints=3,
                             fk_arrays=None):
        """
        get_pmat(t, k) -> (3,4) projection matrix or None if camera invalid.
        All 11 joints in _SMPLX_TO_MHR70 are triangulated; a joint is included
        if it is visible (finite kp) in ≥ 2 cameras.
        """
        _J_can_arr, _kp_can_arr = fk_arrays if fk_arrays is not None else (J_can_all, kp_can_all)
        pred_joints = np.full((T_full, 1, J_body, 3), np.nan, dtype=np.float32)
        for t in range(T_full):
            J_can  = _J_can_arr[t]
            kp_can = _kp_can_arr[t]
            if not np.isfinite(J_can).any():
                continue

            joint_world: dict[int, np.ndarray] = {}
            for smplx_j, mhr_j in _SMPLX_TO_MHR70_EVAL.items():
                obs, pmats_t = [], []
                for k, cn in enumerate(cam_names):
                    P = get_pmat(t, k)
                    if P is None:
                        continue
                    kp = kps_2d.get(cn, np.full((T_full, 70, 2), np.nan))[t, mhr_j]
                    if not np.isfinite(kp).all():
                        continue
                    oc     = placer.original_coords[t, k]
                    W_orig = float(placer.original_size[t, k, 0])
                    H_orig = float(placer.original_size[t, k, 1])
                    u, v   = placer._orig_to_vggt(kp, oc, W_orig, H_orig)
                    if not placer._in_bounds(u, v, oc[2], oc[3]):
                        continue
                    obs.append((u, v))
                    pmats_t.append(P)
                if len(obs) >= 2:
                    joint_world[smplx_j] = placer._triangulate_dlt(obs, pmats_t, [1.0] * len(obs))

            if len(joint_world) < min_joints:
                continue
            vis    = sorted(joint_world.keys())
            A = np.stack([joint_world[j] for j in vis]).astype(np.float64)
            B = np.stack([kp_can[placer._mapper.index(_SMPLX_TO_MHR70_EVAL[j])] for j in vis]).astype(np.float64)
            A_mean, B_mean = A.mean(0), B.mean(0)
            H_svd = (B - B_mean).T @ (A - A_mean)
            U, _, Vt = np.linalg.svd(H_svd)
            R = (Vt.T @ np.diag([1., 1., np.linalg.det(Vt.T @ U.T)]) @ U.T).astype(np.float32)
            t_vec = (A_mean - R.astype(np.float64) @ B_mean).astype(np.float32)
            pelvis = (R.astype(np.float64) @ J_can[0].astype(np.float64) + t_vec.astype(np.float64)).astype(np.float32)
            J_world = (R @ (J_can - J_can[0]).T).T + pelvis
            pred_joints[t, 0] = J_world[_BODY_JOINT_IDX]

        valid = np.isfinite(pred_joints).all((-2, -1)) & np.isfinite(gt_joints).all((-2, -1))
        wa = metric_wa_mpjpe(pred_joints, gt_joints, valid)
        w  = metric_w_mpjpe( pred_joints, gt_joints, valid)
        log.info(f"  {label:<28}: WA-MPJPE={wa:.1f}mm  W-MPJPE={w:.1f}mm  "
                 f"valid={int(valid[:, 0].sum())}/{T_full}")

    # ── pred cameras + pred_scale ──────────────────────────────────────────────
    def _pred_pmat(t, k, s):
        if not cam_valid[t, k]:
            return None
        ext = pred_exts[t, k].astype(np.float64).copy()
        ext[:3, 3] *= s
        return pred_intrs[t, k].astype(np.float64) @ ext

    _dlt_procrustes_eval(
        lambda t, k: _pred_pmat(t, k, float(pred_scale[t]) if pred_scale[t] > 0 else float(valid_sc.mean())),
        f"pred_cams + pred_scale ({valid_sc.mean():.3f})",
    )
    _dlt_procrustes_eval(
        lambda t, k: _pred_pmat(t, k, gt_scale),
        f"pred_cams + gt_scale  ({gt_scale:.4f})",
    )
    _dlt_procrustes_eval(
        lambda t, k: _pred_pmat(t, k, float(calib_scale[t]) if calib_scale[t] > 0 else float(valid_calib_sc.mean())),
        f"pred_cams + calib_scale ({valid_calib_sc.mean():.3f})",
    )

    # ── pred cameras + GT pose (isolates pose error from camera error) ────────
    _dlt_procrustes_eval(
        lambda t, k: _pred_pmat(t, k, float(pred_scale[t]) if pred_scale[t] > 0 else float(valid_sc.mean())),
        f"pred_cams + gt_pose  ({valid_sc.mean():.3f})",
        fk_arrays=(J_can_gt_all, kp_can_gt_all),
    )

    # ── GT cameras (already metric, no scale) ──────────────────────────────────
    def _gt_pmat(t, k):
        if not cam_valid[t, k]:
            return None
        return pred_intrs[t, k].astype(np.float64) @ gt_exts[k].astype(np.float64)

    _dlt_procrustes_eval(
        _gt_pmat,
        "gt_cams + gt_scale (oracle)",
    )
    _dlt_procrustes_eval(
        _gt_pmat,
        "gt_cams + gt_pose (full oracle)",
        fk_arrays=(J_can_gt_all, kp_can_gt_all),
    )

# ══════════════════════════════════════════════════════════════════════════════
# 8. Bone length analysis: pred betas vs GT betas vs GT pose
#
# For each bone in _SCALE_BONE_CANDIDATES, on frames where GT is available:
#   - Triangulate endpoints from 2D MHR70 obs with UNSCALED pred cameras → L_vggt
#   - Compute FK bone length under three variants:
#       A) pred_b + pred_p  (current pipeline)
#       B) gt_b  + pred_p  (what if we knew the GT shape)
#       C) gt_b  + gt_p    ("real life" metric bone length from GT)
#   - Implied scale per variant = median(L_fk / L_vggt)
#   - L_vggt × gt_scale vs L_fk reveals systematic MHR70 landmark bias
# ══════════════════════════════════════════════════════════════════════════════
log.info("=== 8. Bone length analysis ===")

if not gt_body_data:
    log.info("  No GT body data — skipping.")
else:


    # --- GT betas + fused pose FK (variant B) --------------------------------
    _gt_betas_mean = _gt_betas.mean(0)                              # (10,)
    _gt_betas_rep  = np.tile(_gt_betas_mean[np.newaxis], (T_full, 1))
    _, _verts_gtb  = placer._smplx_fk(
        _gt_betas_rep, _body_poses, np.zeros((T_full, 3), np.float32),
        return_verts=True,
    )
    kp_can_gtb_all = placer._mapper.map(_verts_gtb)                 # (T_full, N_KPS, 3)
    torch.cuda.empty_cache()

    _BONE_NAMES = {
        (9, 13):  "L hip→ankle",  (10, 14): "R hip→ankle",
        (9, 11):  "L hip→knee",   (10, 12): "R hip→knee",
        (11, 13): "L knee→ankle", (12, 14): "R knee→ankle",
        (7, 62):  "L elbow→wrist",(8, 41):  "R elbow→wrist",
    }

    # bone_records[bone] = list of (L_vggt_unscaled, L_pred, L_gtb, L_gt)
    bone_records: dict = {b: [] for b in _valid_bones}

    for t_rel in gt_t_rels:
        for bone in _valid_bones:
            mhr_a, mhr_b = bone
            pts_a, pts_b, Pmats_u = [], [], []
            for ki, cn in enumerate(cam_names):
                if not cam_valid[t_rel, ki] or cn not in kps_2d:
                    continue
                ka = kps_2d[cn][t_rel, mhr_a]
                kb = kps_2d[cn][t_rel, mhr_b]
                if not (np.isfinite(ka).all() and np.isfinite(kb).all()):
                    continue
                oc = orig_coords[t_rel, ki]
                Wo = float(orig_size[t_rel, ki, 0])
                Ho = float(orig_size[t_rel, ki, 1])
                ua, va = placer._orig_to_vggt(ka, oc, Wo, Ho)
                ub, vb = placer._orig_to_vggt(kb, oc, Wo, Ho)
                if not (placer._in_bounds(ua, va, oc[2], oc[3]) and
                        placer._in_bounds(ub, vb, oc[2], oc[3])):
                    continue
                # unscaled: no s applied to extrinsics
                P = (pred_intrs[t_rel, ki].astype(np.float64) @
                     pred_exts[t_rel, ki].astype(np.float64))
                Pmats_u.append(P)
                pts_a.append((ua, va))
                pts_b.append((ub, vb))

            if len(pts_a) < 2:
                continue
            try:
                Xa = placer._triangulate_dlt(pts_a, Pmats_u)
                Xb = placer._triangulate_dlt(pts_b, Pmats_u)
            except Exception:
                continue
            Lv = float(np.linalg.norm(Xb - Xa))
            if Lv < 1e-4:
                continue

            ia = placer._mapper.index(mhr_a)
            ib = placer._mapper.index(mhr_b)
            Lp  = float(np.linalg.norm(kp_can_all    [t_rel, ib] - kp_can_all    [t_rel, ia]))
            Lgb = float(np.linalg.norm(kp_can_gtb_all[t_rel, ib] - kp_can_gtb_all[t_rel, ia]))
            Lgt = float(np.linalg.norm(kp_can_gt_all [t_rel, ib] - kp_can_gt_all [t_rel, ia]))
            if min(Lp, Lgb, Lgt) < 0.01:
                continue
            bone_records[bone].append((Lv, Lp, Lgb, Lgt))

    # ── Per-bone mean length table ─────────────────────────────────────────────
    # Column "L_vggt×s_gt" = triangulated length rescaled to metric by GT scale.
    # If MHR70 landmarks were perfect SMPL-X joint positions this should equal gt_b+gt_p.
    # Any gap is the systematic MHR70 surface-landmark offset.
    log.info(f"\n  Mean bone lengths (metres) — 'real life' = gt_b+gt_p column:")
    hdr = (f"  {'bone':<16}  {'L_vggt×s_gt':>12}  "
           f"{'pred_b+pred_p':>14}  {'gt_b+pred_p':>12}  {'gt_b+gt_p':>10}  N")
    log.info(hdr)
    log.info(f"  {'-'*16}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*10}  -")
    for bone in _valid_bones:
        recs = bone_records[bone]
        if not recs:
            continue
        recs_np = np.array(recs)          # (N, 4)
        bname   = _BONE_NAMES.get(bone, str(bone))
        lv_sc   = recs_np[:, 0].mean() * gt_scale   # metric triangulated
        lp_m    = recs_np[:, 1].mean()              # pred betas + pred pose
        lgb_m   = recs_np[:, 2].mean()              # gt  betas + pred pose
        lgt_m   = recs_np[:, 3].mean()              # gt  betas + gt  pose
        log.info(f"  {bname:<16}  {lv_sc:>12.4f}  "
                 f"{lp_m:>14.4f}  {lgb_m:>12.4f}  {lgt_m:>10.4f}  {len(recs)}")

    # ── Implied scale per variant ──────────────────────────────────────────────
    log.info(f"\n  Implied scale (median L_fk / L_vggt over all bones × frames):")
    for ci, label in enumerate(["pred_b+pred_p", "gt_b+pred_p ", "gt_b+gt_p  "], start=1):
        samples = [r[ci] / r[0]
                   for recs in bone_records.values() for r in recs
                   if r[0] > 1e-4]
        if samples:
            s   = np.median(samples)
            err = (s - gt_scale) / gt_scale * 100
            log.info(f"    {label}: scale={s:.4f}  err={err:+.1f}%  (N={len(samples)})")
    log.info(f"    {'GT (cameras)':<18}: scale={gt_scale:.4f}")

    # Calibration-corrected: divide each sample by the per-bone factor before median
    calib_samples = [r[1] / (r[0] * _CALIB_FACTORS.get(bone, 1.0))
                     for bone, recs in bone_records.items() for r in recs
                     if r[0] > 1e-4]
    if calib_samples:
        s_cal = np.median(calib_samples)
        err_cal = (s_cal - gt_scale) / gt_scale * 100
        log.info(f"    {'pred_b+pred_p+calib':<18}: scale={s_cal:.4f}  err={err_cal:+.1f}%  (N={len(calib_samples)})")

    # ── Per-bone systematic landmark offset ───────────────────────────────────
    # Ratio = (gt FK bone) / (triangulated metric bone).
    # > 1 means the FK landmark is farther than the MHR70 2D projection suggests.
    # This is the irreducible bias that remains even with GT betas + GT pose.
    log.info(f"\n  Systematic MHR70 landmark offset per bone  (gt_b+gt_p / L_vggt×s_gt):")
    for bone in _valid_bones:
        recs = bone_records[bone]
        if not recs:
            continue
        recs_np = np.array(recs)
        ratio   = np.mean(recs_np[:, 3] / (recs_np[:, 0] * gt_scale))
        bname   = _BONE_NAMES.get(bone, str(bone))
        log.info(f"    {bname:<16}: {ratio:.4f}  ({(ratio-1)*100:+.1f}%)")
