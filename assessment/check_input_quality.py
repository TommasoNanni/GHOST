"""
Compare raw SAM3D input pose estimates against RICH GT pose annotations,
per joint. This tells us whether the input to the fusion model is already
far from GT (in which case the model can't overfit without learning something),
or close (in which case the task is easy and poor convergence is a model issue).

Run:
    pixi run python assessment/check_input_quality.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset

RICH_SCENE_DIR = Path(
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs"
    "/rich11_segmentation_test/Pavallion_003_phonesiteat"
)

SMPLX_JOINT_NAMES = (
    [f"body_{i:02d}" for i in range(22)]          # 0-21: pelvis + body + wrists
    + ["jaw", "eye_L", "eye_R"]                    # 22-24
    + [f"hand_{i:02d}" for i in range(30)]         # 25-54: hands
)


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle in degrees. Inputs: (..., 3, 3)."""
    trace = torch.einsum("...ij,...ij->...", R_pred, R_gt).clamp(-1 + 1e-6, 3 - 1e-6)
    cos_angle = (trace - 1.0) / 2.0
    return torch.acos(cos_angle.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / torch.pi)


def main():
    dp = RICHFusionDatapoint(scene_dir=RICH_SCENE_DIR, rich_data_root=CONFIG.data.rich_data_root)
    ds = RICHFusionDataset([dp])
    inputs, targets = ds[0]

    # inputs["pose"]  : (T, K, P, J, 6)
    # targets["pose"] : (T, P, J, 6)  — GT aggregated (same for all cams)
    inp_pose = inputs["pose"].float()    # (T, K, P, J, 6)
    gt_pose  = targets["pose"].float()  # (T, P, J, 6)

    T, K, P, J, _ = inp_pose.shape
    print(f"T={T}, K={K}, P={P}, J={J}")

    # Expand GT to match per-camera inputs
    gt_expanded = gt_pose.unsqueeze(1).expand(T, K, P, J, 6)  # (T, K, P, J, 6)

    R_inp = rotation_6d_to_matrix(inp_pose.reshape(-1, 6)).reshape(T, K, P, J, 3, 3)
    R_gt  = rotation_6d_to_matrix(gt_expanded.reshape(-1, 6)).reshape(T, K, P, J, 3, 3)

    err = geodesic_deg(R_inp, R_gt)   # (T, K, P, J)

    # joint_mask: (T, K, P, J) — only consider frames where joint was observed
    jm = inputs["joint_mask"]  # (T, K, P, J)

    print("\n--- Per-joint input error vs GT (geodesic degrees) ---")
    print(f"{'Joint':>12}  {'mean_all':>10}  {'mean_obs':>10}  {'obs_rate':>10}")
    print("-" * 50)

    per_joint_all = []
    per_joint_obs = []
    for j in range(J):
        e_all = err[:, :, :, j]                # (T, K, P)
        m     = jm[:, :, :, j].bool()         # (T, K, P)
        mean_all = e_all.mean().item()
        mean_obs = e_all[m].mean().item() if m.any() else float("nan")
        obs_rate = m.float().mean().item()
        per_joint_all.append(mean_all)
        per_joint_obs.append(mean_obs)

        name = SMPLX_JOINT_NAMES[j] if j < len(SMPLX_JOINT_NAMES) else f"j{j:02d}"
        print(f"{name:>12}  {mean_all:>10.1f}  {mean_obs:>10.1f}  {obs_rate:>10.1%}")

    print("-" * 50)
    print(f"\nOverall mean error (all frames):      {np.mean(per_joint_all):.1f}°")
    print(f"Overall mean error (observed frames): {np.nanmean(per_joint_obs):.1f}°")

    # Group summary
    groups = [
        ("pelvis+body (0-21)", range(0, 22)),
        ("jaw+eyes (22-24)",   range(22, 25)),
        ("hands (25-54)",      range(25, 55)),
    ]
    print("\n--- Group summary ---")
    for label, idx_range in groups:
        idxs = list(idx_range)
        all_e = np.array(per_joint_all)[idxs]
        obs_e = np.array(per_joint_obs)[idxs]
        print(f"{label}: mean_all={all_e.mean():.1f}°  mean_obs={np.nanmean(obs_e):.1f}°")


if __name__ == "__main__":
    main()
