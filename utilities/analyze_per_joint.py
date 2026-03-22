"""
Per-joint analysis of a predictions.npz file from the overfitting test.

Usage:
    pixi run python scripts/analyze_per_joint.py 61039183
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from pytorch3d.transforms import rotation_6d_to_matrix

from utilities.smplx_utilities import get_smplx_joints


# ── SMPL-X joint names (first 55) ──────────────────────────────────────────
JOINT_NAMES = [
    "pelvis",           # 0
    "left_hip",         # 1
    "right_hip",        # 2
    "spine1",           # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine2",           # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine3",           # 9
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_collar",      # 13
    "right_collar",     # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
    "jaw",              # 22
    "left_eye_smplhf",  # 23
    "right_eye_smplhf", # 24
    # fingers left hand
    "left_index1", "left_index2", "left_index3",   # 25-27
    "left_middle1", "left_middle2", "left_middle3", # 28-30
    "left_pinky1", "left_pinky2", "left_pinky3",   # 31-33
    "left_ring1", "left_ring2", "left_ring3",       # 34-36
    "left_thumb1", "left_thumb2", "left_thumb3",   # 37-39
    # fingers right hand
    "right_index1", "right_index2", "right_index3",   # 40-42
    "right_middle1", "right_middle2", "right_middle3", # 43-45
    "right_pinky1", "right_pinky2", "right_pinky3",   # 46-48
    "right_ring1", "right_ring2", "right_ring3",       # 49-51
    "right_thumb1", "right_thumb2", "right_thumb3",   # 52-54
]


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "61039183"
    pred_path = Path("fusion_outputs") / f"{job_id}_predictions.npz"

    d = np.load(str(pred_path))
    print("Keys:", list(d.keys()))

    pose    = torch.from_numpy(d["pose"]).float()    # (T, P, J, 6)
    gt_pose = torch.from_numpy(d["gt_pose"]).float() # (T, P, J, 6)

    T, P, J, _ = pose.shape

    trans    = torch.from_numpy(d["trans"]).float()    if "trans"    in d else torch.zeros(T, P, 3)
    gt_trans = torch.from_numpy(d["gt_trans"]).float() if "gt_trans" in d else torch.zeros(T, P, 3)
    print(f"T={T}, P={P}, J={J}")

    # Add batch dim: (1, T, P, J, 6) and shape (1, T, P, 10)
    # shape_aggr is saved as (P, 10) (no T dim); gt_shape is (T, P, 10).
    pose_b    = pose.unsqueeze(0)      # (1, T, P, J, 6)
    gt_pose_b = gt_pose.unsqueeze(0)

    def _to_shape_b(arr):
        t = torch.from_numpy(arr).float()
        if t.ndim == 2:            # (P, 10) → expand over T
            t = t.unsqueeze(0).expand(T, -1, -1)   # (T, P, 10)
        return t.unsqueeze(0)      # (1, T, P, 10)

    shape_b    = _to_shape_b(d["shape"])    if "shape"    in d else torch.zeros(1, T, P, 10)
    gt_shape_b = _to_shape_b(d["gt_shape"]) if "gt_shape" in d else torch.zeros(1, T, P, 10)

    def fk_chunked(pose_b, shape_b, chunk=32):
        """Run get_smplx_joints in time chunks to avoid large CPU batches."""
        T = pose_b.shape[1]
        parts = []
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            out = get_smplx_joints(pose_b[:, t0:t1], shape_b[:, t0:t1])
            parts.append(out)
        return torch.cat(parts, dim=1)

    print("Running SMPL-X forward pass for predictions (chunked)...")
    pred_joints = fk_chunked(pose_b, shape_b)      # (1, T, P, Jout, 3)
    print("Running SMPL-X forward pass for GT (chunked)...")
    gt_joints   = fk_chunked(gt_pose_b, gt_shape_b) # (1, T, P, Jout, 3)

    # Take first 55 joints, squeeze batch dim
    pred_j = pred_joints[0, :, 0, :55, :].detach().numpy()  # (T, 55, 3)
    gt_j   = gt_joints[0, :, 0, :55, :].detach().numpy()    # (T, 55, 3)

    # World-frame: add translation (root position)
    # trans: (T, P, 3) → (T, 1, 3)
    pred_trans = trans[:, 0:1, :].numpy()  # (T, 1, 3)
    gt_trans_  = gt_trans[:, 0:1, :].numpy()

    pred_world = pred_j + pred_trans  # (T, 55, 3)
    gt_world   = gt_j + gt_trans_     # (T, 55, 3)

    # ── Root-aligned MPJPE (GA) ──────────────────────────────────────────
    pred_align = pred_j - pred_j[:, 0:1, :]  # pelvis-aligned
    gt_align   = gt_j   - gt_j[:, 0:1, :]

    per_joint_ga_mpjpe = np.sqrt(((pred_align - gt_align) ** 2).sum(-1)).mean(0)  # (55,)

    # ── World-frame MPJPE (W) ─────────────────────────────────────────────
    per_joint_w_mpjpe = np.sqrt(((pred_world - gt_world) ** 2).sum(-1)).mean(0)   # (55,)

    # ── Per-joint rotation error (geodesic) ──────────────────────────────
    pose_b_flat    = pose.view(T * P, J, 6)
    gt_pose_b_flat = gt_pose.view(T * P, J, 6)
    pred_rot = rotation_6d_to_matrix(pose_b_flat)    # (T, J, 3, 3)
    gt_rot   = rotation_6d_to_matrix(gt_pose_b_flat) # (T, J, 3, 3)
    # Geodesic: R_err = R_pred^T @ R_gt, angle = arccos((trace-1)/2)
    R_err = pred_rot.transpose(-1, -2) @ gt_rot  # (T, J, 3, 3)
    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
    geo_err_deg = torch.acos(cos_angle) * (180 / np.pi)   # (T, J)
    per_joint_geo = geo_err_deg.mean(0).numpy()            # (J=55,)

    # ── Print table ──────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"  Per-Joint Analysis — Job {job_id}  (T={T} frames, P={P} persons)")
    print("=" * 75)
    print(f"  {'Joint':<22}  {'GA-MPJPE (m)':>14}  {'W-MPJPE (m)':>13}  {'RotErr (°)':>12}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*13}  {'-'*12}")

    for i, name in enumerate(JOINT_NAMES[:J]):
        ga   = per_joint_ga_mpjpe[i]
        w    = per_joint_w_mpjpe[i]
        rot  = per_joint_geo[i]
        print(f"  {i:2d} {name:<20}  {ga:>14.4f}  {w:>13.4f}  {rot:>12.2f}")

    print(f"  {'-'*22}  {'-'*14}  {'-'*13}  {'-'*12}")
    print(f"  {'MEAN':<22}  {per_joint_ga_mpjpe.mean():>14.4f}  {per_joint_w_mpjpe.mean():>13.4f}  {per_joint_geo.mean():>12.2f}")
    print()

    # ── Body-part groupings ───────────────────────────────────────────────
    groups = {
        "Torso (0-9,12-14)": list(range(0, 10)) + [12, 13, 14],
        "Head/Neck (15)":    [15],
        "Left arm (16,18,20)":  [16, 18, 20],
        "Right arm (17,19,21)": [17, 19, 21],
        "Left leg (1,4,7,10)":  [1, 4, 7, 10],
        "Right leg (2,5,8,11)": [2, 5, 8, 11],
        "Fingers L (25-39)": list(range(25, 40)),
        "Fingers R (40-54)": list(range(40, 55)),
    }
    print("── Body-part summary ────────────────────────────────────────────────────")
    print(f"  {'Group':<30}  {'GA-MPJPE':>10}  {'W-MPJPE':>10}  {'RotErr':>10}")
    for grp, idxs in groups.items():
        valid = [i for i in idxs if i < J]
        ga  = per_joint_ga_mpjpe[valid].mean()
        w   = per_joint_w_mpjpe[valid].mean()
        rot = per_joint_geo[valid].mean()
        print(f"  {grp:<30}  {ga:>10.4f}  {w:>10.4f}  {rot:>10.2f}")


if __name__ == "__main__":
    main()
