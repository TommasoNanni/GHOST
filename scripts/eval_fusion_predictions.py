"""
Evaluate saved fusion predictions against GT.

Loads a .npz file produced by Trainer.save_predictions and prints:
  - Per-joint rotation error (degrees), averaged over annotated frames only
  - Root orientation error separately
  - Body joints (1-21) and hand joints (22-54) separately
  - Translation error (mean / median / max, cm), annotated frames only
  - Shape (betas) RMSE
  - Camera rotation error (degrees) and translation error (m): mean / min / max

Only frames flagged as annotated by gt_valid are used for body pose / translation
metrics.  Camera metrics are filtered separately by non-zero GT quaternion.

Usage:
    pixi run python scripts/eval_fusion_predictions.py \
        --pred fusion_outputs/Pavallion_003_phonesiteat.npz
"""

from __future__ import annotations

import argparse
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle in degrees between two (..., 3, 3) rotation tensors."""
    R_rel = R_pred @ R_gt.transpose(-1, -2)
    cos_a = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_a) * (180.0 / math.pi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to prediction .npz file")
    args = parser.parse_args()

    data = np.load(args.pred)
    print(f"Loaded: {args.pred}")
    print(f"Keys: {list(data.keys())}\n")

    pose_pred   = torch.tensor(data["pose"]).float()               # (T, P, 55, 6)
    pose_gt     = torch.tensor(data["gt_body_pose"]).float()       # (T, P, 55, 6)
    transl_pred = torch.tensor(data["body_transl_world"]).float()  # (T, P, 3)
    transl_gt   = torch.tensor(data["gt_body_transl_world"]).float()
    shape_pred  = torch.tensor(data["shape"]).float()              # (P, 10)
    shape_gt    = torch.tensor(data["gt_body_shape"]).float()      # (T, P, 10)
    cam_pred    = torch.tensor(data["camera"]).float()             # (T, K, 8)
    cam_gt      = torch.tensor(data["gt_camera"]).float()          # (T, K, 8)

    T, P, J, _ = pose_pred.shape

    # ── gt_valid: annotated frames ─────────────────────────────────────────────
    # Prefer the saved mask; fall back to inferring from non-zero translation.
    if "gt_valid" in data:
        gt_valid = torch.tensor(data["gt_valid"]).bool()  # (T, P)
    else:
        gt_valid = (transl_gt.norm(dim=-1) > 1e-6)        # (T, P) — zero = unannotated

    n_annotated = int(gt_valid.any(dim=-1).sum().item())
    print(f"Annotated frames : {n_annotated} / {T}  "
          f"({100*n_annotated/max(T,1):.1f}%)")
    print()

    # ── Rotation errors (annotated frames only) ───────────────────────────────
    # rotation_6d_to_matrix of an all-zero 6D vector → all-zero rotation matrix
    # (eps normalisation gives zero, not identity).  Geodesic distance to any
    # rotation then equals acos(-0.5) = 120°, badly contaminating unannotated frames.
    # Filtering by gt_valid avoids this entirely.
    R_pred = rotation_6d_to_matrix(pose_pred.reshape(-1, 6)).reshape(T, P, J, 3, 3)
    R_gt   = rotation_6d_to_matrix(pose_gt.reshape(-1, 6)).reshape(T, P, J, 3, 3)
    angle  = geodesic_deg(R_pred, R_gt)  # (T, P, J)

    # Mask: (T, P, J) — valid where the frame is annotated
    valid_mask = gt_valid.unsqueeze(-1).expand_as(angle)  # (T, P, J)

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
        """Mean over masked elements; returns nan if nothing is valid."""
        return float(x[mask].mean().item()) if mask.any() else float("nan")

    print("=" * 50)
    print("ROTATION ERROR (degrees, annotated frames only)")
    print("=" * 50)
    print(f"  Root orient (joint 0)  : {masked_mean(angle[:,:,0:1], valid_mask[:,:,0:1]):.2f}°")
    print(f"  Body joints (1-21)     : {masked_mean(angle[:,:,1:22], valid_mask[:,:,1:22]):.2f}°")
    print(f"  Hand joints (22-54)    : {masked_mean(angle[:,:,22:], valid_mask[:,:,22:]):.2f}°")
    print(f"  All 55 joints          : {masked_mean(angle, valid_mask):.2f}°")
    print()

    # Per-joint breakdown (mean over annotated T, P)
    print("  Per-joint breakdown (mean over annotated frames):")
    for j in range(J):
        tag = "root" if j == 0 else ("body" if j <= 21 else "hand")
        val = masked_mean(angle[:, :, j:j+1], valid_mask[:, :, j:j+1])
        print(f"    joint {j:2d} [{tag}]: {val:.2f}°")
    print()

    # ── Translation error (annotated frames only) ─────────────────────────────
    transl_err = (transl_pred - transl_gt).norm(dim=-1)  # (T, P)
    valid_transl = gt_valid  # (T, P)

    print("=" * 50)
    print("TRANSLATION ERROR (annotated frames only)")
    print("=" * 50)
    if valid_transl.any():
        err_valid = transl_err[valid_transl] * 100  # cm
        print(f"  Mean   : {err_valid.mean():.1f} cm")
        print(f"  Median : {err_valid.median():.1f} cm")
        print(f"  Max    : {err_valid.max():.1f} cm")
    else:
        print("  No annotated frames found.")
    print()

    # ── Shape error ───────────────────────────────────────────────────────────
    shape_gt_mean = shape_gt.mean(dim=0)  # (P, 10)
    shape_rmse = (shape_pred - shape_gt_mean).pow(2).mean(dim=-1).sqrt()  # (P,)
    print("=" * 50)
    print("SHAPE (betas) RMSE per person")
    print("=" * 50)
    for p in range(P):
        print(f"  Person {p}: {shape_rmse[p].item():.4f}")
    print()

    # ── Camera errors ─────────────────────────────────────────────────────────
    # Filter by non-zero GT quaternion (norm > 0.5 is the sentinel used elsewhere).
    cam_valid = cam_gt[..., :4].norm(dim=-1) > 0.5  # (T, K)
    K = cam_pred.shape[1]

    print("=" * 50)
    print("CAMERA ERROR")
    print("=" * 50)

    rot_errors_per_cam   = []
    trans_errors_per_cam = []

    for k in range(K):
        valid = cam_valid[:, k]  # (T,)
        if not valid.any():
            print(f"  Camera {k}: no valid GT frames, skipped")
            continue

        q_pred = F.normalize(cam_pred[valid, k, :4], dim=-1)
        q_gt   = F.normalize(cam_gt[valid, k, :4],   dim=-1)
        dot    = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1 - 1e-7)
        rot_err_deg = torch.acos(dot) * (2 * 180.0 / math.pi)  # (N,)

        t_err_m = (cam_pred[valid, k, 4:7] - cam_gt[valid, k, 4:7]).norm(dim=-1)  # (N,)

        rot_errors_per_cam.append(rot_err_deg)
        trans_errors_per_cam.append(t_err_m)

        print(f"  Camera {k}  ({int(valid.sum())} frames)")
        print(f"    Rotation  — mean: {rot_err_deg.mean():.2f}°  "
              f"min: {rot_err_deg.min():.2f}°  max: {rot_err_deg.max():.2f}°")
        print(f"    Transl(m) — mean: {t_err_m.mean():.3f}  "
              f"min: {t_err_m.min():.3f}  max: {t_err_m.max():.3f}")

    if rot_errors_per_cam:
        all_rot   = torch.cat(rot_errors_per_cam)
        all_trans = torch.cat(trans_errors_per_cam)
        print(f"\n  All cameras combined")
        print(f"    Rotation  — mean: {all_rot.mean():.2f}°  "
              f"min: {all_rot.min():.2f}°  max: {all_rot.max():.2f}°")
        print(f"    Transl(m) — mean: {all_trans.mean():.3f}  "
              f"min: {all_trans.min():.3f}  max: {all_trans.max():.3f}")


if __name__ == "__main__":
    main()
