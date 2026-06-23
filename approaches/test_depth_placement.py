"""
Depth-cloud ICP placement: post-processing on top of the fusion+placer pipeline.

Correct pipeline (as opposed to naive cam_00-only approach):
  1. Load all cameras' body data.
  2. Run fusion model → fused pose (multi-view aggregated).
  3. Run Procrustes DLT placer → initial (R, t) per frame  [baseline M1].
  4. Build SMPL-X body mesh with fused pose + mean betas across all cameras.
     Initial placement = DLT result from step 3.
  5. Build multi-camera depth point cloud (silhouette-masked VGGT depth,
     converted to metric using the same pred_scale as the placer).
  6. Rigid ICP (SE3) to refine (R, t) so the mesh surface matches the cloud.
  7. W/WA-MPJPE for DLT baseline vs ICP-refined.

Usage:
    pixi run python approaches/test_depth_placement.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────────
SCENE       = "BBQ_001_guitar"
GHOST_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT   = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
CHECKPOINT  = Path("/users/tnanni/ghost/checkpoints/fusion_module/best.pt")
SMPLX_MODEL = Path("/users/tnanni/ghost/body_models/SMPLX_NEUTRAL.pkl")
DEVICE      = torch.device("cpu")   # no GPU needed
GT_SPLIT    = "train"
PERSON_ID   = 1   # ghost pid

CONF_THR    = 0.1
DEPTH_MIN_V = 0.3   # VGGT metres
DEPTH_MAX_V = 1.5   # VGGT metres (~9.6 real metres)
MAX_PTS_CAM = 3000  # subsample per camera
ICP_ITERS   = 10
ICP_OUTLIER = 2.0   # reject > K × median_dist

# ── shared imports from evaluation pipeline ────────────────────────────────────
from evaluation.evaluate_on_rich_test import (
    load_scene_body_data, build_fusion_tensors, load_fusion_model,
    load_gt_body_data, load_gt_extrinsics,
    _6d_to_aa, _aa_to_6d,
    metric_wa_mpjpe, metric_w_mpjpe,
    _BODY_JOINT_IDX,
)
from fusion.placer import BodyPlacer

# ── VGGT data ──────────────────────────────────────────────────────────────────
scene_dir  = GHOST_ROOT / SCENE
depth_npz  = np.load(scene_dir / "vggt_depth.npz",   mmap_mode="r")
cam_npz    = np.load(scene_dir / "vggt_cameras.npz",  mmap_mode="r")

depth_mm   = depth_npz["depth"]       # (T, K, H, W) uint16 VGGT mm
depth_conf = depth_npz["depth_conf"]  # (T, K, H, W)
cam_valid  = cam_npz["valid"]         # (T, K)
extrinsics = cam_npz["extrinsics"]    # (T, K, 3, 4)
intrinsics = cam_npz["intrinsics"]    # (T, K, 3, 3)
cam_names  = [n.decode() if isinstance(n, bytes) else n
              for n in cam_npz["camera_names"]]
T_full, K, H_d, W_d = depth_mm.shape


# ── depth point cloud ──────────────────────────────────────────────────────────

def point_cloud_world(frame_t: int, person_id: int,
                      scale: float) -> np.ndarray:
    """(N,3) person surface cloud in METRIC world frame (VGGT cam_00 = world)."""
    all_pts: list[np.ndarray] = []
    for ki in range(K):
        if not cam_valid[frame_t, ki]:
            continue
        mpath = scene_dir / cam_names[ki] / "mask_data.npz"
        if not mpath.exists():
            continue
        m   = np.load(mpath)
        key = f"mask_{frame_t:05d}_{ki:02d}"
        if key not in m:
            continue
        mask = np.array(Image.fromarray(m[key]).resize((W_d, H_d), Image.NEAREST))
        vs, us = np.where(mask == person_id)
        if len(vs) < 20:
            continue
        d_raw = depth_mm[frame_t, ki][vs, us].astype(np.float32) / 1000.0
        conf  = depth_conf[frame_t, ki][vs, us].astype(np.float32)
        ok    = (d_raw > DEPTH_MIN_V) & (d_raw < DEPTH_MAX_V) & (conf > CONF_THR)
        vs, us, d_raw = vs[ok], us[ok], d_raw[ok]
        if len(vs) < 20:
            continue
        if len(vs) > MAX_PTS_CAM:
            idx = np.random.choice(len(vs), MAX_PTS_CAM, replace=False)
            vs, us, d_raw = vs[idx], us[idx], d_raw[idx]
        Ki = intrinsics[frame_t, ki]
        fx, fy, cx, cy = Ki[0,0], Ki[1,1], Ki[0,2], Ki[1,2]
        pts_cam = np.stack([(us-cx)/fx*d_raw, (vs-cy)/fy*d_raw, d_raw], axis=1)
        E = extrinsics[frame_t, ki]
        R, t = E[:, :3], E[:, 3]
        pts_world = (pts_cam - t) @ R          # R.T @ (pts - t), world VGGT units
        all_pts.append(pts_world * scale)      # → metric
    return np.concatenate(all_pts) if all_pts else np.zeros((0,3), np.float32)


# ── rigid ICP ──────────────────────────────────────────────────────────────────

def icp_rigid(V_init: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SE(3) point-to-point ICP.

    V_init: (Nv,3) source — body mesh at initial DLT placement, metric world frame
    P:      (Np,3) target — depth point cloud, metric world frame
    Returns R (3,3), t (3,) such that (R @ V.T).T + t ≈ P
    Accumulated so that final joints = (R @ J_init.T).T + t
    """
    V = V_init.astype(np.float64)
    P = P.astype(np.float64)
    R_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)

    for _ in range(ICP_ITERS):
        tree         = cKDTree(V)
        dists, nn    = tree.query(P, k=1)
        thresh       = np.median(dists) * ICP_OUTLIER
        ok           = dists < thresh
        if ok.sum() < 10:
            break
        P_v, V_v     = P[ok], V[nn[ok]]
        mu_s, mu_t   = V_v.mean(0), P_v.mean(0)
        H            = (V_v - mu_s).T @ (P_v - mu_t)
        U, _, Vt     = np.linalg.svd(H)
        d            = float(np.sign(np.linalg.det(Vt.T @ U.T)))
        R            = Vt.T @ np.diag([1,1,d]) @ U.T
        t            = mu_t - R @ mu_s
        V            = (R @ V.T).T + t
        R_acc        = R @ R_acc
        t_acc        = R @ t_acc + t

    return R_acc.astype(np.float32), t_acc.astype(np.float32)


# ── smoothness helper ──────────────────────────────────────────────────────────

def traj_std(traj):
    ok  = ~np.any(np.isnan(traj), axis=1)
    idx = np.where(ok)[0]
    if len(idx) < 2: return float("nan")
    return float(np.std(np.linalg.norm(np.diff(traj[idx], axis=0), axis=1)))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)

    # ── 1. Load all-camera body data ──────────────────────────────────────────
    log.info("Loading scene body data …")
    cam_dirs, raw = load_scene_body_data(scene_dir)
    if not cam_dirs:
        log.error("No body data found"); return

    # ── 2. Fusion model → fused pose ─────────────────────────────────────────
    log.info("Running fusion model …")
    pose_t, mask_t, shape_t, all_pids, frame_start = build_fusion_tensors(raw)
    T       = pose_t.shape[1]
    P_count = len(all_pids)
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    fusion_model = load_fusion_model(CHECKPOINT, DEVICE)
    with torch.no_grad():
        fused_pose_t, _ = fusion_model(
            pose_t.to(DEVICE), mask_t.to(DEVICE), shape=shape_t.to(DEVICE)
        )
    fused_pose  = fused_pose_t[0].cpu().numpy()   # (T, P, 54, 6)

    _betas_lists: dict[int, list[np.ndarray]] = {}
    for cam_dir in cam_dirs:
        for pid in all_pids:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if not bf.exists(): continue
            d = np.load(bf, allow_pickle=False)
            if "smplx_betas" in d.files:
                _betas_lists.setdefault(pid, []).append(d["smplx_betas"].mean(0))
    betas_by_pid: dict[int, np.ndarray] = {
        pid: np.mean(v, axis=0).astype(np.float32)
        for pid, v in _betas_lists.items()
    }
    for pid in all_pids:
        betas_by_pid.setdefault(pid, np.zeros(10, np.float32))
    fused_pose_by_pid = {pid: fused_pose[:, pid_to_slot[pid]] for pid in all_pids}

    # ── 3. Procrustes DLT placer → initial (R, t) ────────────────────────────
    log.info("Running placer …")
    placer = BodyPlacer(scene_dir, SMPLX_MODEL)
    pred_scale = placer.estimate_scale_triangulated(
        fused_pose_by_pid=fused_pose_by_pid,
        pred_betas_map={
            cam_dir / "body_data" / f"person_{pid}.npz": betas_by_pid[pid]
            for cam_dir in cam_dirs for pid in all_pids
            if (cam_dir / "body_data" / f"person_{pid}.npz").exists()
        },
        frame_start=frame_start,
    )
    scale_val = float(np.mean(pred_scale[pred_scale > 0])) if (pred_scale > 0).any() else 6.4
    log.info(f"pred_scale = {scale_val:.4f}")

    trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
        scale=pred_scale,
        all_pids=set(all_pids),
        pred_betas_by_pid=betas_by_pid,
        fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start,
    )

    # ── 4+5+6. Per-frame: build mesh at DLT placement, ICP on depth cloud ────
    log.info("Running ICP per frame …")
    J_BODY = len(_BODY_JOINT_IDX)

    pred_dlt = np.full((T, P_count, J_BODY, 3), np.nan, np.float32)
    pred_icp = np.full((T, P_count, J_BODY, 3), np.nan, np.float32)

    for pid in all_pids:
        p_slot   = pid_to_slot[pid]
        betas_p  = betas_by_pid.get(pid, np.zeros(10, np.float32))
        frames_t = trans_dict.get(pid, {})

        for global_t, pelvis_world in sorted(frames_t.items()):
            t_rel  = int(global_t) - frame_start
            R_mat  = orient_dict.get(pid, {}).get(global_t)
            if R_mat is None or not (0 <= t_rel < T):
                continue

            body_pose_aa = _6d_to_aa(fused_pose[t_rel, p_slot, :21])  # (21, 3)

            # Canonical joints (zero global_orient, zero transl)
            J_can = placer._smplx_fk(
                betas_p[np.newaxis],
                body_pose_aa.reshape(63)[np.newaxis],
                np.zeros((1,3), np.float32),
            )[0]   # (55, 3)

            # DLT placement: rotate canonical joints around pelvis, translate
            J_dlt = (R_mat @ (J_can - J_can[0]).T).T + pelvis_world
            pred_dlt[t_rel, p_slot] = J_dlt[_BODY_JOINT_IDX]

            # Only run ICP for the target person
            if pid != PERSON_ID:
                continue

            # Mesh vertices at DLT placement (same FK call, grab verts)
            _, V_verts = placer._smplx_fk(
                betas_p[np.newaxis],
                body_pose_aa.reshape(63)[np.newaxis],
                np.zeros((1,3), np.float32),
                return_verts=True,
            )
            V_init = (R_mat @ (V_verts[0] - J_can[0]).T).T + pelvis_world

            # Depth point cloud at this frame
            P_all = point_cloud_world(int(global_t), PERSON_ID, scale_val)
            if len(P_all) < 50:
                pred_icp[t_rel, p_slot] = J_dlt[_BODY_JOINT_IDX]  # fallback to DLT
                continue

            R_icp, t_icp = icp_rigid(V_init, P_all)

            # Apply ICP correction to DLT joints
            J_icp = (R_icp @ J_dlt.T).T + t_icp
            pred_icp[t_rel, p_slot] = J_icp[_BODY_JOINT_IDX]

            if t_rel % 50 == 0:
                corr = np.linalg.norm(t_icp) * 100
                log.info(f"  pid={pid} t={t_rel:4d}  ICP correction |t|={corr:.1f}cm  N_pts={len(P_all)}")

    # ── 7. GT joints ─────────────────────────────────────────────────────────
    log.info("Loading GT …")
    gt_body = load_gt_body_data(SCENE, RICH_ROOT, split=GT_SPLIT)
    if not gt_body:
        log.error("No GT found"); return
    gt_pid  = sorted(gt_body.keys())[0]
    log.info(f"GT pid={gt_pid}  ({len(gt_body[gt_pid])} frames)")

    gt_exts = load_gt_extrinsics(SCENE, RICH_ROOT)
    ref_idx = int(re.search(r"\d+", cam_dirs[0].name).group())
    E_ref   = np.array(gt_exts[ref_idx], dtype=np.float64)
    R_w2ref, t_w2ref = E_ref[:3, :3], E_ref[:3, 3]

    gt_joints = np.full((T, P_count, J_BODY, 3), np.nan, np.float32)
    for fi, params in gt_body[gt_pid].items():
        t_rel = fi - frame_start
        if not (0 <= t_rel < T): continue
        J_gt = placer._smplx_fk(
            params["betas"].reshape(1,-1)[:,:10],
            params["body_pose"][np.newaxis],
            params["global_orient"][np.newaxis],
        )[0] + params["transl"]
        gt_joints[t_rel, pid_to_slot.get(PERSON_ID, 0)] = J_gt[_BODY_JOINT_IDX]

    # ── 8. Metrics ───────────────────────────────────────────────────────────
    pid_slot = pid_to_slot.get(PERSON_ID, 0)

    # Single-person arrays (T, 1, J, 3) for metric functions
    def s(arr): return arr[:, pid_slot:pid_slot+1]

    valid_dlt = np.isfinite(s(pred_dlt)).all((-2,-1)) & np.isfinite(s(gt_joints)).all((-2,-1))
    valid_icp = np.isfinite(s(pred_icp)).all((-2,-1)) & np.isfinite(s(gt_joints)).all((-2,-1))
    log.info(f"Valid — DLT: {valid_dlt.sum()}  ICP: {valid_icp.sum()}")

    # Smoothness
    r_dlt = s(pred_dlt)[:, 0, 0, :]
    r_icp = s(pred_icp)[:, 0, 0, :]
    print(f"\nSmoothness pelvis traj (frame-to-frame std, m):")
    print(f"  DLT: {traj_std(r_dlt):.4f}")
    print(f"  ICP: {traj_std(r_icp):.4f}")

    print("\n" + "="*60)
    print(f"W / WA-MPJPE  (mm)  —  {SCENE} [{GT_SPLIT}]")
    print("="*60)
    print(f"  {'Method':<28}  {'WA-MPJPE':>10}  {'W-MPJPE':>10}  {'valid':>6}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*6}")
    for name, pred, valid in [
        ("DLT (M1 baseline)", s(pred_dlt), valid_dlt),
        ("ICP on depth cloud",  s(pred_icp), valid_icp),
    ]:
        wa = metric_wa_mpjpe(pred, s(gt_joints), valid)
        w  = metric_w_mpjpe( pred, s(gt_joints), valid)
        print(f"  {name:<28}  {wa:>10.1f}  {w:>10.1f}  {int(valid.sum()):>6}")


if __name__ == "__main__":
    main()
