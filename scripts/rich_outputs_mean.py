"""
Naive multi-view mean aggregator baseline.

Reports RR-MPJPE / RR-MPJRE / Betas-MSE — the same metrics as train_rich_v2.py,
evaluated on the val split only (same round-robin logic as train_rich_v2.py).

For every (frame, person):
  - Joints 1-54 (parent-relative, camera-invariant): visibility-weighted mean
    of 6D representations across cameras, using joint_mask as weight.
  - Joint 0 (global orient): taken directly from GT, same as the training
    metric_fn.  This isolates the quality of relative body joints only,
    which is what the fusion model predicts.
  - Betas: visibility-weighted mean over cameras, then mean over valid frames.

No camera calibration required.

Run:
    pixi run python scripts/rich_outputs_mean.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from pytorch3d.transforms import rotation_6d_to_matrix

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.metrics_v2 import BetasMSE, MetricCollection, RootRelativeMPJPE, RootRelativeMPJRE
from utilities.smplx_utilities import get_smplx_joints

RICH_OUTPUT_ROOT = Path(
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test"
)

SKIP_SCENES: list[str] = ["Pavallion_013_plankjack"]
SKIP_CAMERAS: dict[str, list[str]] = {
    "Pavallion_003_018_tossball": ["cam_06"],
    "ParkingLot2_008_pushup2":   ["cam_03"],
    "ParkingLot2_014_takingphotos2": ["cam_01"],
}
NUM_VAL_SCENES = 10

METRIC_STRIDE = 8


def _val_scenes(output_root: Path) -> list[Path]:
    """Round-robin val split — mirrors _split_by_location in train_rich_v2.py."""
    all_dirs = sorted(d for d in output_root.iterdir() if d.is_dir())
    scenes = [d for d in all_dirs if d.name not in SKIP_SCENES]

    by_location: dict[str, list[Path]] = defaultdict(list)
    for s in scenes:
        by_location[s.name.split("_")[0]].append(s)

    available: dict[str, list[Path]] = {}
    for loc in sorted(by_location):
        loc_scenes = sorted(by_location[loc], key=lambda s: s.name)
        available[loc] = loc_scenes[1:]  # first scene per location → mandatory train

    locs = sorted(available)
    val: list[Path] = []
    while len(val) < NUM_VAL_SCENES:
        picked_any = False
        for loc in locs:
            if len(val) >= NUM_VAL_SCENES:
                break
            if available[loc]:
                val.append(available[loc].pop())
                picked_any = True
        if not picked_any:
            break

    return val


def _process_scene(scene_dir: Path, mc: MetricCollection) -> bool:
    try:
        dp = RICHFusionDatapoint(
            scene_dir=scene_dir,
            rich_data_root=CONFIG.data.rich_data_root,
            rich_gt_dir=CONFIG.data.rich_gt_dir,
            exclude_cameras=SKIP_CAMERAS.get(scene_dir.name, []),
        )
        ds = RICHFusionDataset([dp])
        inputs, targets = ds[0]
    except Exception as e:
        print(f"  SKIP {scene_dir.name} (load): {e}")
        return False

    try:
        pose_cam    = inputs["pose"].float()        # (T, K, P, J, 6)
        person_mask = inputs["person_mask"].bool()  # (T, K, P)
        joint_mask  = inputs["joint_mask"].float()  # (T, K, P, J)

        gt_pose  = targets["pose"].float()          # (T, P, J, 6)
        gt_shape = targets["shape"].float()         # (T, P, 10)
        gt_valid = targets["gt_valid"]              # (T, P) bool

        T, K, P, J, _ = pose_cam.shape

        if P == 0:
            print(f"  SKIP {scene_dir.name}: no persons")
            return False

        # Joints 1+: visibility-weighted mean across cameras (camera-invariant).
        jvis = joint_mask * person_mask.float().unsqueeze(-1)  # (T, K, P, J)
        jvis_1plus = jvis[..., 1:, None]                       # (T, K, P, J-1, 1)
        jvis_sum   = jvis[..., 1:].sum(dim=1).clamp(min=1e-8) # (T, P, J-1)
        agg_joints = (pose_cam[..., 1:, :] * jvis_1plus).sum(dim=1) / jvis_sum.unsqueeze(-1)
        # (T, P, J-1, 6)

        # Joint 0: use GT global orient, same as training metric_fn.
        pose_full = torch.cat([gt_pose[..., :1, :], agg_joints], dim=-2)  # (T, P, J, 6)

        # Betas: visibility-weighted mean over cameras, then mean over valid frames.
        pred_shape_raw = inputs["shape"].float()                              # (T, K, P, 10)
        vis = person_mask.float().unsqueeze(-1)                               # (T, K, P, 1)
        agg_shape_frame = (pred_shape_raw * vis).sum(1) / vis.sum(1).clamp(min=1)  # (T, P, 10)

        valid_f = gt_valid.float().unsqueeze(-1)                              # (T, P, 1)
        agg_shape_static = (agg_shape_frame * valid_f).sum(0) / valid_f.sum(0).clamp(min=1)
        gt_shape_static  = (gt_shape * valid_f).sum(0) / valid_f.sum(0).clamp(min=1)

        # FK on strided frames.
        t_idx = torch.arange(0, T, METRIC_STRIDE)

        with torch.no_grad():
            pred_joints = get_smplx_joints(
                pose_full[t_idx].unsqueeze(0),
                agg_shape_frame[t_idx].unsqueeze(0),
            ).cpu().numpy()[0, :, :, :55, :]    # (T', P, 55, 3)

            gt_joints = get_smplx_joints(
                gt_pose[t_idx].unsqueeze(0),
                gt_shape[t_idx].unsqueeze(0),
            ).cpu().numpy()[0, :, :, :55, :]    # (T', P, 55, 3)

            rot_pred = rotation_6d_to_matrix(
                pose_full[t_idx].reshape(-1, 6)
            ).reshape(len(t_idx), P, J, 3, 3).cpu().numpy()

            rot_gt = rotation_6d_to_matrix(
                gt_pose[t_idx].reshape(-1, 6)
            ).reshape(len(t_idx), P, J, 3, 3).cpu().numpy()

        gt_valid_sub = gt_valid[t_idx].cpu().numpy()  # (T', P)

        for t in range(len(t_idx)):
            for p in range(P):
                if not gt_valid_sub[t, p]:
                    continue
                mc["RR-MPJPE"].update(pred_joints[t, p], gt_joints[t, p])
                mc["RR-MPJRE"].update(rot_pred[t, p], rot_gt[t, p])

        for p in range(P):
            mc["Betas-MSE"].update(
                agg_shape_static[p].cpu().numpy(),
                gt_shape_static[p].cpu().numpy(),
            )

    except Exception as e:
        import traceback
        print(f"  SKIP {scene_dir.name} (processing): {e}")
        traceback.print_exc()
        return False

    return True


def main():
    scene_dirs = _val_scenes(RICH_OUTPUT_ROOT)
    print(f"Val scenes ({len(scene_dirs)}) under {RICH_OUTPUT_ROOT.name}:")
    for d in sorted(scene_dirs, key=lambda s: s.name):
        print(f"  {d.name}")

    mc = MetricCollection([RootRelativeMPJPE(), RootRelativeMPJRE(), BetasMSE()])

    n_ok = 0
    for i, scene_dir in enumerate(scene_dirs):
        print(f"[{i+1}/{len(scene_dirs)}] {scene_dir.name}")
        if _process_scene(scene_dir, mc):
            n_ok += 1

    print(f"\nProcessed {n_ok}/{len(scene_dirs)} scenes successfully.")

    results = mc.compute()
    print("\n" + "=" * 50)
    print("MEAN-POOL BASELINE  (same metrics as train_rich_v2)")
    print("=" * 50)
    for name, d in results.items():
        unit = " m" if "MPJPE" in name else ("°" if "MPJRE" in name else "")
        print(f"  {name:<12}: mean={d['mean']:.4f}{unit}  median={d['median']:.4f}{unit}  count={d['count']:.0f}")


if __name__ == "__main__":
    main()
