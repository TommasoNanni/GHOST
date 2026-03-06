"""
Overfitting test for SSTNetwork using a real RICH pipeline output.

Uses EgoExoFusionDataset from data.fusion_dataset to load one temporal window
from the ghost pipeline output, then trains SSTNetwork to memorise that
single sample.

Since this RICH run has no SMPL-X conversion active, pose/shape inputs are
zeros; camera comes from pred_cam_t. Targets are self-supervised (mirror
inputs). The test verifies the full forward/backward loop works and the
loss converges to zero.

Needs a GPU (flex_attention requires CUDA).
Run via SLURM or on an interactive GPU node:
    pixi run python -m test.test_fusion_overfit
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Subset

from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_axis_angle

from data.fusion_dataset import EgoExoFusionDataset
from fusion.fusion_module import SSTNetwork
from fusion.loss import (
    MSELoss,
    EpipolarLoss,
    TemporalSmoothnessLoss,
    VPoserLoss,
    BoneLengthconsistencyLoss,
    BetaConsistencyLoss,
    CameraMSELoss,
)
from fusion.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────

RICH_SCENE_DIR  = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs"
    "/rich1_sam3_segmentation_test/BBQ_001_guitar"
)
WINDOW_SIZE     = 128
TEMPORAL_WINDOW = 32   # each frame attends to 2*32+1 = 65 neighbours


# ─── Loss instances ────────────────────────────────────────────────────────────

_mse      = MSELoss()
_epipolar = EpipolarLoss()
_temporal = TemporalSmoothnessLoss()
_bone     = BoneLengthconsistencyLoss()
_beta     = BetaConsistencyLoss()
_cam_mse  = CameraMSELoss()

try:
    _vposer     = VPoserLoss()
    _has_vposer = True
except Exception as _e:
    logger.warning(f"VPoserLoss unavailable ({_e}); skipping.")
    _vposer     = None
    _has_vposer = False


# ─── Loss wrappers ─────────────────────────────────────────────────────────────
# preds  : (pose_aggr, shape_aggr, camera, pose_per_cam, shape_per_cam)
#   pose_aggr    : (B, T, P, J, 6)
#   shape_aggr   : (B, T, P, 10)
#   camera       : (B, T, K, 7)
#   pose_per_cam : (B, T, K, P, J, 6)
#   shape_per_cam: (B, T, K, P, 10)
#
# targets: dict — pose (B,T,K,P,J,6), shape (B,T,K,P,10),
#                 camera (B,T,K,7), keypoints_3d (B,T,K,P,70,3)

def pose_loss(preds, targets):
    pose_aggr, _, _, _, _ = preds
    return _mse(pose_aggr, targets["pose"].mean(dim=2))


def shape_loss(preds, targets):
    _, shape_aggr, _, _, _ = preds
    return _mse(shape_aggr, targets["shape"].mean(dim=2))


def camera_loss(preds, targets):
    _, _, camera_pred, _, _ = preds
    return _mse(camera_pred, targets["camera"])


def epipolar_loss(preds, targets):
    """EpipolarLoss over per-camera pose/shape/camera — only meaningful if K >= 2."""
    _, _, camera_pred, pose_per_cam, shape_per_cam = preds
    if pose_per_cam.shape[2] < 2:
        return pose_per_cam.new_zeros([])
    return _epipolar(pose_per_cam, shape_per_cam, camera_pred)


def temporal_loss(preds, targets):
    """Acceleration-smoothness on the aggregated pose stream."""
    pose_aggr, _, _, _, _ = preds
    if pose_aggr.shape[1] < 3:
        return pose_aggr.new_zeros([])
    return _temporal(
        current=pose_aggr[:, 2:],
        previous1=pose_aggr[:, 1:-1],
        previous2=pose_aggr[:, :-2],
    )


def bone_loss(preds, targets):
    """BoneLengthconsistencyLoss — uses 3D keypoints from targets."""
    return _bone(targets["keypoints_3d"])


def beta_loss(preds, targets):
    """BetaConsistencyLoss — consistent shape across cameras."""
    _, _, _, _, shape_per_cam = preds
    return _beta(shape_per_cam)


def camera_mse_loss(preds, targets):
    """CameraMSELoss — geodesic + translation between predicted and target cameras."""
    _, _, camera_pred, _, _ = preds
    cam_gt = targets["camera"]
    # camera: (B, T, K, 7) = [quat(4), trans(3)]
    R_pred = quaternion_to_matrix(camera_pred[..., :4])
    t_pred = camera_pred[..., 4:]
    R_gt   = quaternion_to_matrix(cam_gt[..., :4])
    t_gt   = cam_gt[..., 4:]
    return _cam_mse(R_pred, t_pred, R_gt, t_gt)


def vposer_loss(preds, targets):
    """VPoserLoss — KL prior on body pose; skipped if checkpoint unavailable."""
    if not _has_vposer:
        return preds[0].new_zeros([])
    pose_aggr, _, _, _, _ = preds
    # pose_aggr: (B, T, P, J, 6) — convert 6D rotation → axis-angle for VPoser
    rot_mat    = rotation_6d_to_matrix(pose_aggr)              # (..., J, 3, 3)
    axis_angle = matrix_to_axis_angle(rot_mat.reshape(-1, 3, 3))
    axis_angle = axis_angle.reshape(*pose_aggr.shape[:-1], 3)  # (..., J, 3)
    return _vposer(axis_angle)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build dataset — EgoExoFusionDataset works on any ghost pipeline output
    # (no GT needed; targets mirror inputs in self-supervised mode).
    ds = EgoExoFusionDataset(
        scene_dir=RICH_SCENE_DIR,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_SIZE,
    )

    # Overfit on the first window only.
    loader = DataLoader(Subset(ds, [0]), batch_size=1, shuffle=False)

    logger.info(
        f"WindowedTemporalAttention — flex_attention + block_mask  "
        f"(temporal_window={TEMPORAL_WINDOW}, T={WINDOW_SIZE}  "
        f"-> each frame attends to {min(2 * TEMPORAL_WINDOW + 1, WINDOW_SIZE)} frames)"
    )

    model = SSTNetwork(
        embedding_dim=64,
        num_heads=4,
        num_layers=4,
        max_temporal_len=WINDOW_SIZE + 16,
        dropout=0.0,
        temporal_window=TEMPORAL_WINDOW,
    )

    logger.info("\n" + model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = {
        "pose":       (pose_loss,       1.0),
        "shape":      (shape_loss,      1.0),
        "camera":     (camera_loss,     1.0),
        "epipolar":   (epipolar_loss,   0.1),
        "temporal":   (temporal_loss,   0.1),
        "bone":       (bone_loss,       0.1),
        "beta":       (beta_loss,       0.1),
        "camera_mse": (camera_mse_loss, 0.1),
        "vposer":     (vposer_loss,     0.01),
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        losses=losses,
        max_epochs=500,
        use_wandb=False,  # set True + call wandb.init() to log to W&B
    )

    trainer.train()

    # ── Final check ────────────────────────────────────────────────────────
    trainer.model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs, targets = trainer._unpack_batch(batch)
        preds = trainer._forward(inputs)
        final_loss = sum(fn(preds, targets).item() for fn, _ in losses.values())

    logger.info(f"\nFinal combined loss: {final_loss:.6f}")
    if final_loss < 0.01:
        logger.info("PASS — model successfully overfit the sample.")
    else:
        logger.warning(
            f"FAIL — loss {final_loss:.4f} is still high. "
            "Try more epochs or a larger model."
        )


if __name__ == "__main__":
    main()
