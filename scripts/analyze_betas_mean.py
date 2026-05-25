"""
Analyze the quality of cross-view averaged betas vs. ground-truth SMPL-X betas.

For every (frame, person) in a RICH datapoint:
  - Average predicted smplx_betas across cameras where the person is detected.
  - Compare against GT betas with L2 error per beta coefficient and overall MSE.

Run:
    pixi run python scripts/analyze_betas_mean.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset

RICH_OUTPUT_ROOT = Path(
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test"
)

METRIC_STRIDE = 8


SCENE_DIRS = sorted(
    Path(RICH_OUTPUT_ROOT).iterdir()
) if RICH_OUTPUT_ROOT.exists() else []


def main() -> None:
    # Per-scene MSE accumulators (static estimate vs mean GT betas per person)
    scene_mse_mean:   list[float] = []
    scene_mse_median: list[float] = []

    for scene_dir in SCENE_DIRS:
        if not scene_dir.is_dir():
            continue
        try:
            dp = RICHFusionDatapoint(
                scene_dir=scene_dir,
                rich_data_root="/capstor/scratch/cscs/tnanni/datasets/rich",
            )
            ds = RICHFusionDataset([dp])
            inputs, targets = ds[0]
        except Exception as e:
            print(f"  SKIP {scene_dir.name}: {e}")
            continue

        # (T, K, P, 10), (T, K, P) bool
        pred_shape  = inputs["shape"].float()   # (T, K, P, 10)
        person_mask = inputs["person_mask"]     # (T, K, P) bool
        gt_shape    = targets["shape"].float()  # (T, P, 10)
        gt_valid    = targets["gt_valid"]       # (T, P) bool

        T, K, P, n_betas = pred_shape.shape

        scene_mean_mse   = []
        scene_median_mse = []

        for p in range(P):
            obs_mask = person_mask[:, :, p].bool()  # (T, K)
            gt_mask  = gt_valid[:, p]               # (T,)

            if not obs_mask.any() or not gt_mask.any():
                continue

            # All valid observations: (N_valid, 10)
            obs = pred_shape[:, :, p, :][obs_mask]  # (N_valid, 10)

            static_mean   = obs.mean(dim=0)    # (10,)
            static_median = obs.median(dim=0).values  # (10,)

            # GT target: mean of per-frame GT betas over valid frames
            gt_target = gt_shape[:, p, :][gt_mask].mean(dim=0)  # (10,)

            scene_mean_mse.append(((static_mean   - gt_target) ** 2).mean().item())
            scene_median_mse.append(((static_median - gt_target) ** 2).mean().item())

        if not scene_mean_mse:
            print(f"  SKIP {scene_dir.name}: no valid persons")
            continue

        m_mean   = np.mean(scene_mean_mse)
        m_median = np.mean(scene_median_mse)
        scene_mse_mean.append(m_mean)
        scene_mse_median.append(m_median)
        print(f"[{scene_dir.name}]  MSE(mean)={m_mean:.4f}  MSE(median)={m_median:.4f}")

    if not scene_mse_mean:
        print("No valid samples found.")
        return

    print(f"\n=== Static per-scene estimate vs mean GT betas ({len(scene_mse_mean)} scenes) ===")
    print(f"  Mean-pool  MSE : {np.mean(scene_mse_mean):.4f}")
    print(f"  Median-pool MSE: {np.mean(scene_mse_median):.4f}")


if __name__ == "__main__":
    main()
