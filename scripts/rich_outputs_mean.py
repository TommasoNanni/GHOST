"""
Naive multi-view mean aggregator baseline.

For every (frame, person):
  - Joints 1-54 (relative rotations): average the 6D representations across
    cameras that observed the joint, then re-orthogonalize via Gram-Schmidt.
  - Joint 0 (global orientation, camera-local): rotate each camera estimate to
    world frame (R_w2c^T @ R_body_cam), then take the SVD geodesic mean.
  - Translation: back-project each camera's body_transl_cam_in to world frame,
    average over cameras where the person was detected.

Prints per-joint geodesic error vs RICH GT and MetricCollection scores.

Run:
    pixi run python scripts/rich_outputs_mean.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.metric import (
    MetricCollection,
    WMPJPE, GAMPJPE, PAMPJPE,
    WMPJRE, GAMPJRE, PAMPJRE,
    TranslationError, ScaledTranslationError,
    AngleError,
    RRA, CCA, ScaledCCA,
)
from utilities.smplx_utilities import get_smplx_joints

RICH_SCENE_DIR = Path(
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs"
    "/rich11_segmentation_test/Pavallion_003_phonesiteat"
)

SMPLX_JOINT_NAMES = (
    [f"body_{i:02d}" for i in range(22)]
    + ["jaw", "eye_L", "eye_R"]
    + [f"hand_{i:02d}" for i in range(30)]
)

METRIC_STRIDE = 8


def so3_mean(rotmats: torch.Tensor) -> torch.Tensor:
    """Geodesic mean of rotation matrices via SVD projection.

    Args:
        rotmats: (N, 3, 3) — N rotation matrices to average.
    Returns:
        (3, 3) — nearest rotation matrix to the element-wise mean.
    """
    R_sum = rotmats.sum(0)          # (3, 3)
    U, _, Vh = torch.linalg.svd(R_sum)
    d = torch.det(U @ Vh)
    fix = torch.ones(3, device=rotmats.device)
    fix[-1] = d.sign()
    return U @ torch.diag(fix) @ Vh


def aggregate_pose(
    pose_cam: torch.Tensor,       # (T, K, P, J, 6)  joint 0 in cam frame, 1+ are relative
    cam_quats: torch.Tensor,      # (T, K, 4)  quaternion w2c
    person_mask: torch.Tensor,    # (T, K, P)  bool — person detected in this cam/frame
    joint_mask: torch.Tensor,     # (T, K, P, J)  float — joint was observed
) -> torch.Tensor:
    """Returns (T, P, J, 6) averaged pose in world frame."""
    T, K, P, J, _ = pose_cam.shape
    device = pose_cam.device

    R_w2c = quaternion_to_matrix(cam_quats.reshape(-1, 4)).reshape(T, K, 3, 3)  # (T, K, 3, 3)
    R_c2w = R_w2c.transpose(-1, -2)                                               # (T, K, 3, 3)

    out_pose = torch.zeros(T, P, J, 6, device=device)

    for t in range(T):
        for p in range(P):
            valid_cams = person_mask[t, :, p]  # (K,)
            if not valid_cams.any():
                continue

            # ── Joint 0: global orient in cam frame → world frame → SO(3) mean ──
            r6d_j0 = pose_cam[t, :, p, 0, :]          # (K, 6)
            R_j0_cam = rotation_6d_to_matrix(r6d_j0)  # (K, 3, 3)
            R_j0_world = torch.bmm(R_c2w[t], R_j0_cam)  # (K, 3, 3)  R_c2w @ R_body_cam
            R_j0_mean = so3_mean(R_j0_world[valid_cams])
            out_pose[t, p, 0, :] = matrix_to_rotation_6d(R_j0_mean.unsqueeze(0)).squeeze(0)

            # ── Joints 1-54: frame-independent relative rotations ──
            for j in range(1, J):
                obs = joint_mask[t, :, p, j].bool() & valid_cams  # (K,)
                src = valid_cams if not obs.any() else obs
                r6d = pose_cam[t, src, p, j, :]   # (N, 6)
                out_pose[t, p, j, :] = r6d.mean(0)  # average 6D, GS re-ortho on read

    return out_pose  # (T, P, J, 6)


def aggregate_translation(
    body_transl_cam: torch.Tensor,  # (T, K, P, 3)  in cam-k local frame
    cam_quats: torch.Tensor,         # (T, K, 4)
    cam_transl_w2c: torch.Tensor,    # (T, K, 3)
    person_mask: torch.Tensor,       # (T, K, P)
) -> torch.Tensor:
    """Returns (T, P, 3) world-frame translation via mean back-projection."""
    T, K, P, _ = body_transl_cam.shape
    device = body_transl_cam.device

    R_w2c = quaternion_to_matrix(cam_quats.reshape(-1, 4)).reshape(T, K, 3, 3)
    R_c2w = R_w2c.transpose(-1, -2)

    out = torch.zeros(T, P, 3, device=device)
    for t in range(T):
        for p in range(P):
            valid = person_mask[t, :, p]  # (K,)
            if not valid.any():
                continue
            # world = R_c2w @ body_transl_cam + cam_centre_world
            # cam_centre_world = -R_c2w @ cam_transl_w2c
            bt_cam = body_transl_cam[t, valid, p, :]          # (N, 3)
            Rc2w_v = R_c2w[t, valid]                           # (N, 3, 3)
            cam_t  = cam_transl_w2c[t, valid]                  # (N, 3)

            bt_world = torch.bmm(Rc2w_v, bt_cam.unsqueeze(-1)).squeeze(-1) - \
                       torch.bmm(Rc2w_v, cam_t.unsqueeze(-1)).squeeze(-1)
            out[t, p] = bt_world.mean(0)
    return out  # (T, P, 3)


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    trace = torch.einsum("...ij,...ij->...", R_pred, R_gt).clamp(-1 + 1e-6, 3 - 1e-6)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / torch.pi)


def main():
    dp = RICHFusionDatapoint(scene_dir=RICH_SCENE_DIR, rich_data_root=CONFIG.data.rich_data_root)
    ds = RICHFusionDataset([dp])
    inputs, targets = ds[0]

    pose_cam      = inputs["pose"].float()            # (T, K, P, J, 6)
    body_t_cam    = inputs["body_transl_cam_in"].float()  # (T, K, P, 3)
    cam_vec       = inputs["camera"].float()          # (T, K, 8)
    person_mask   = inputs["person_mask"].bool()      # (T, K, P)
    joint_mask    = inputs["joint_mask"].float()      # (T, K, P, J)

    gt_pose       = targets["pose"].float()           # (T, P, J, 6)
    gt_trans      = targets["trans"].float()          # (T, P, 3)
    gt_shape      = targets["shape"].float()          # (T, P, 10)
    gt_valid      = targets["gt_valid"]               # (T, P) bool
    gt_cam        = targets["camera"].float()         # (T, K, 8)

    cam_quats     = cam_vec[..., :4]   # (T, K, 4)
    cam_transl    = cam_vec[..., 4:7]  # (T, K, 3)

    T, K, P, J, _ = pose_cam.shape
    print(f"T={T}, K={K}, P={P}, J={J}")
    print("Aggregating... (this iterates over T×P, may take a minute)")

    agg_pose  = aggregate_pose(pose_cam, cam_quats, person_mask, joint_mask)   # (T, P, J, 6)
    agg_trans = aggregate_translation(body_t_cam, cam_quats, cam_transl, person_mask)  # (T, P, 3)

    # ── Per-joint geodesic error ──────────────────────────────────────────────
    shape_4d = agg_pose.shape[:-1]   # (T, P, J)
    R_agg = rotation_6d_to_matrix(agg_pose.reshape(-1, 6)).reshape(*shape_4d, 3, 3)
    R_gt  = rotation_6d_to_matrix(gt_pose.reshape(-1, 6)).reshape(*shape_4d, 3, 3)
    err   = geodesic_deg(R_agg, R_gt)    # (T, P, J)

    valid_mask = gt_valid.unsqueeze(-1).expand_as(err)   # (T, P, J)

    print("\n--- Per-joint error vs GT (geodesic°, valid GT frames only) ---")
    print(f"{'Joint':>12}  {'mean_deg':>10}")
    print("-" * 28)

    per_joint = []
    for j in range(J):
        e = err[:, :, j][valid_mask[:, :, j]]
        mean_e = e.mean().item() if e.numel() > 0 else float("nan")
        per_joint.append(mean_e)
        name = SMPLX_JOINT_NAMES[j] if j < len(SMPLX_JOINT_NAMES) else f"j{j:02d}"
        print(f"{name:>12}  {mean_e:>10.1f}")

    print("-" * 28)
    print(f"{'Overall mean':>12}  {np.nanmean(per_joint):>10.1f}")

    groups = [
        ("body  (0-21)", range(0, 22)),
        ("jaw+eyes(22-24)", range(22, 25)),
        ("hands (25-54)", range(25, 55)),
    ]
    print()
    for label, idx_range in groups:
        vals = [per_joint[i] for i in idx_range]
        print(f"  {label}: {np.nanmean(vals):.1f}°")

    # ── MetricCollection ─────────────────────────────────────────────────────
    print("\nComputing SMPL-X joint positions for MetricCollection metrics...")
    mc = MetricCollection([
        WMPJPE(), GAMPJPE(), PAMPJPE(),
        WMPJRE(), GAMPJRE(), PAMPJRE(),
        TranslationError(), ScaledTranslationError(),
        AngleError(),
        RRA(threshold=15.0), CCA(threshold=15.0), ScaledCCA(threshold=15.0),
    ])

    t_idx = torch.arange(0, T, METRIC_STRIDE)
    with torch.no_grad():
        pose_sub  = agg_pose[t_idx].unsqueeze(0)             # (1, T_sub, P, J, 6)
        shape_sub = gt_shape[t_idx].unsqueeze(0)              # (1, T_sub, P, 10)
        pred_joints_rel = get_smplx_joints(pose_sub, shape_sub).cpu().numpy()[..., :55, :]

        gt_pose_sub  = gt_pose[t_idx].unsqueeze(0)
        gt_shape_sub = gt_shape[t_idx].unsqueeze(0)
        gt_joints_rel = get_smplx_joints(gt_pose_sub, gt_shape_sub).cpu().numpy()[..., :55, :]

        pred_transl = agg_trans[t_idx].unsqueeze(0).cpu().numpy()    # (1, T_sub, P, 3)
        gt_transl   = gt_trans[t_idx].unsqueeze(0).cpu().numpy()

        pred_joints = pred_joints_rel + pred_transl[:, :, :, None, :]
        gt_joints   = gt_joints_rel   + gt_transl[:, :, :, None, :]

        pred_rotmats = rotation_6d_to_matrix(agg_pose[t_idx].unsqueeze(0)).cpu().numpy()
        gt_rotmats   = rotation_6d_to_matrix(gt_pose[t_idx].unsqueeze(0)).cpu().numpy()

        T_sub = len(t_idx)
        cam_valid = (gt_cam[t_idx, :, :4].norm(dim=-1) > 0.5).cpu().numpy()  # (T_sub, K)

        gt_R_w2c    = quaternion_to_matrix(gt_cam[t_idx, :, :4].reshape(-1, 4)).reshape(T_sub, K, 3, 3).cpu().numpy()
        gt_t_w2c    = gt_cam[t_idx, :, 4:7].cpu().numpy()
        gt_cam_centres = -np.einsum("...ji,...j->...i", gt_R_w2c, gt_t_w2c)  # (T_sub, K, 3)

        # Naive aggregator has no camera prediction → use GT cameras
        pred_cam_centres = gt_cam_centres.copy()

        gt_valid_sub = gt_valid[t_idx].cpu().numpy()  # (T_sub, P)

    t_mid = T_sub // 2
    valid_k = cam_valid[t_mid]
    Cp = pred_cam_centres[t_mid][valid_k]
    Cg = gt_cam_centres[t_mid][valid_k]
    Rp = gt_R_w2c[t_mid][valid_k]   # same as GT (no camera prediction)
    Rg = gt_R_w2c[t_mid][valid_k]

    pred_spread = float(np.linalg.norm(Cp - Cp.mean(0), axis=-1).max()) if valid_k.sum() >= 3 else 0.0
    if valid_k.sum() >= 3 and pred_spread > 1e-3:
        mc["TE"].update(Cp, Cg)
        mc["s-TE"].update(Cp, Cg)
        mc["CCA@15"].update(Cp, Cg)
        mc["s-CCA@15"].update(Cp, Cg)
    if valid_k.sum() >= 1:
        mc["AE"].update(Rp, Rg)
        mc["RRA@15"].update(Rp, Rg)

    for t in range(T_sub):
        if not gt_valid_sub[t].any():
            continue
        pj = pred_joints[0, t]   # (P, 55, 3)
        gj = gt_joints[0, t]
        pr = pred_rotmats[0, t]
        gr = gt_rotmats[0, t]
        mc["W-MPJPE"].update(pj, gj, pred_cam_centres[t][cam_valid[t]], gt_cam_centres[t][cam_valid[t]])
        mc["GA-MPJPE"].update(pj, gj)
        mc["PA-MPJPE"].update(pj, gj)
        mc["W-MPJRE"].update(pr, gr, pred_cam_centres[t][cam_valid[t]], gt_cam_centres[t][cam_valid[t]])
        mc["GA-MPJRE"].update(pr, gr)
        mc["PA-MPJRE"].update(pr, gr)

    print("\n--- MetricCollection results ---")
    results = mc.compute()
    for name, val_dict in results.items():
        vals = "  ".join(f"{k}={v:.4f}" for k, v in val_dict.items())
        print(f"  {name:<12}: {vals}")


if __name__ == "__main__":
    main()
