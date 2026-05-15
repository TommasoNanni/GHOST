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
    all_errors: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for scene_dir in SCENE_DIRS:
        if not scene_dir.is_dir():
            continue
        try:
            dp = RICHFusionDatapoint(
                scene_dir=scene_dir,
                rich_data_root=CONFIG.data.rich_data_root,
            )
            ds = RICHFusionDataset([dp])
            inputs, targets = ds[0]
        except Exception as e:
            print(f"  SKIP {scene_dir.name}: {e}")
            continue

        # (T, K, P, 10), (T, K, P) bool
        pred_shape  = inputs["shape"].float()         # (T, K, P, 10)
        person_mask = inputs["person_mask"].float()   # (T, K, P)
        gt_shape    = targets["shape"].float()        # (T, P, 10)
        gt_valid    = targets["gt_valid"]             # (T, P) bool

        # Average predicted betas across cameras where person is detected
        view_mask  = person_mask.unsqueeze(-1)                             # (T, K, P, 1)
        mean_shape = (pred_shape * view_mask).sum(1) / view_mask.sum(1).clamp(min=1)
        # mean_shape: (T, P, 10)

        detected = person_mask.sum(1) > 0  # (T, P) bool
        valid    = gt_valid & detected

        if not valid.any():
            print(f"  SKIP {scene_dir.name}: no valid frames")
            continue

        err = (mean_shape - gt_shape).abs()  # (T, P, 10)
        err_np      = err[valid].numpy()
        gt_np       = gt_shape[valid].numpy()
        pred_np     = mean_shape[valid].numpy()

        all_errors.append(err_np)
        all_gt.append(gt_np)
        all_pred.append(pred_np)

        scene_mae = err_np.mean()
        scene_mse = (err_np ** 2).mean()
        print(f"[{scene_dir.name}]  MAE={scene_mae:.4f}  MSE={scene_mse:.4f}"
              f"  samples={valid.sum()}")

    if not all_errors:
        print("No valid samples found.")
        return

    errors = np.concatenate(all_errors, axis=0)   # (N_total, 10)
    gt_all = np.concatenate(all_gt, axis=0)
    pred_all = np.concatenate(all_pred, axis=0)

    print("\n=== Overall betas error (cross-view mean vs GT) ===")
    print(f"Total samples: {errors.shape[0]}")
    print(f"Overall  MAE : {errors.mean():.4f}")
    print(f"Overall  MSE : {(errors**2).mean():.4f}")
    print(f"Overall RMSE : {np.sqrt((errors**2).mean()):.4f}")

    print("\nPer-coefficient MAE:")
    for i, mae_i in enumerate(errors.mean(axis=0)):
        print(f"  beta[{i:02d}]: {mae_i:.4f}")

    # GT vs pred variance check: how much does GT vary across the dataset?
    gt_std = gt_all.std(axis=0)
    pred_std = pred_all.std(axis=0)
    print("\nPer-coefficient std  (GT / Pred):")
    for i, (gs, ps) in enumerate(zip(gt_std, pred_std)):
        print(f"  beta[{i:02d}]: GT={gs:.4f}  Pred={ps:.4f}")


if __name__ == "__main__":
    main()
