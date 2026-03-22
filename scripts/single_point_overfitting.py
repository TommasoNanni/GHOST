"""
Overfitting test for SSTNetwork using a real RICH pipeline output.

Uses RICHFusionDataset from data.fusion_dataset to load one temporal window
from the ghost pipeline output, then trains SSTNetwork to memorise that
single sample.

Needs a GPU (flex_attention requires CUDA).
Run on an interactive GPU node:
    pixi run python scripts/single_point_overfitting.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix
from torch.utils.data import DataLoader

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.fusion_module import SSTNetwork
from fusion.loss import (
    PoseMSELoss,
    ShapeMSELoss,
    EpipolarLoss,
    TemporalSmoothnessLoss,
    VPoserLoss,
    BoneLengthconsistencyLoss,
    CameraMSELoss,
    TriangulationLoss,
)
from fusion.metric import (
    MetricCollection,
    WMPJPE, GAMPJPE, PAMPJPE,
    WMPJRE, GAMPJRE, PAMPJRE,
    TranslationError, ScaledTranslationError,
    AngleError,
    RRA, CCA, ScaledCCA,
)
from fusion.trainer import Trainer
from utilities.smplx_utilities import get_smplx_joints

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


RICH_SCENE_DIR  = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs"
    "/rich10_segmentation_test/BBQ_001_guitar"
)


def main():
    # Architecture
    embedding_dim           = CONFIG.fusion.architecture.embedding_dimension
    temporal_window         = CONFIG.fusion.architecture.temporal_window
    num_heads               = CONFIG.fusion.architecture.num_heads
    num_layers              = CONFIG.fusion.architecture.num_layers
    max_cameras             = CONFIG.fusion.architecture.max_cameras
    dropout                 = CONFIG.fusion.architecture.dropout
    # Loss weights
    pose_mse_weight         = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight        = CONFIG.fusion.loss.shape_mse_weight
    epipolar_weight         = CONFIG.fusion.loss.epipolar_weight
    temporal_weight         = CONFIG.fusion.loss.temporal_weight
    bone_length_weight      = CONFIG.fusion.loss.bone_length_weight
    camera_mse_weight          = CONFIG.fusion.loss.camera_mse_weight
    triangulation_weight       = CONFIG.fusion.loss.triangulation_weight
    vposer_weight           = CONFIG.fusion.loss.vposer_weight
    # Training params
    lr                      = CONFIG.fusion.training.lr
    max_epochs              = CONFIG.fusion.training.max_epochs
    batch_size              = CONFIG.fusion.training.batch_size
    grad_clip               = CONFIG.fusion.training.grad_clip
    scheduler_name          = getattr(CONFIG.fusion.training, "scheduler", None)
    patience                = getattr(CONFIG.fusion.training, "patience", None)

    # create the dataset
    dp = RICHFusionDatapoint(scene_dir=RICH_SCENE_DIR, rich_data_root = CONFIG.data.rich_data_root)
    img_size = dp.img_size
    ds = RICHFusionDataset([dp])
    inputs, targets = ds[0]
    # inputs:  dict ['pose', 'shape', 'camera', 'joint_mask', 'person_mask']
    # targets: dict ['pose', 'shape', 'camera', 'keypoints_3d']
    T = inputs["pose"].shape[0]
    print({k: tuple(v.shape) for k, v in inputs.items()})
    # 'pose': (382, 8, 1, 55, 6), 'shape': (382, 8, 1, 10), 'camera': (382, 8, 8), 'joint_mask': (382, 8, 1, 55), 'person_mask': (382, 8, 1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logger.info(
        f"WindowedTemporalAttention — flex_attention + block_mask  "
        f"(temporal_window={temporal_window}, T={T}  "
        f"-> each frame attends to {min(2 * temporal_window + 1, T)} frames)"
    )

    model = SSTNetwork(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_temporal_len=T,
        dropout=dropout,
        temporal_window=temporal_window,
        max_cameras=max_cameras,
    )

    logger.info("\n" + model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.1
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    else:
        scheduler = None

    try:
        vposer_loss = VPoserLoss()
        logging.info("VPoser successfully loaded")
    except Exception as e:
        logger.warning(f"VPoserLoss unavailable ({e}); skipping.")
        vposer_loss = None
    losses = {
        "pose":       (PoseMSELoss(),                    pose_mse_weight              ),
        "shape":      (ShapeMSELoss(),                   shape_mse_weight             ),
        "epipolar":   (EpipolarLoss(img_size=img_size),  epipolar_weight              ),
        "temporal":   (TemporalSmoothnessLoss(),         temporal_weight              ),
        "bone":       (BoneLengthconsistencyLoss(),      bone_length_weight           ),
        "camera_mse":      (CameraMSELoss(img_size=img_size),      camera_mse_weight      ),
        "triangulation":   (TriangulationLoss(),                  triangulation_weight   ),
        **({"vposer": (vposer_loss, vposer_weight)} if vposer_loss is not None else {}),
    }

    # ── Metrics ──────────────────────────────────────────────────────────────
    # Evaluated on the training batch at the end of every epoch.
    # We pick the middle frame of the window as the representative frame so
    # we avoid averaging rotation matrices over T (which breaks SO(3)).
    metrics = MetricCollection([
        WMPJPE(), GAMPJPE(), PAMPJPE(),
        WMPJRE(), GAMPJRE(), PAMPJRE(),
        TranslationError(), ScaledTranslationError(),
        AngleError(),
        RRA(threshold=15.0), CCA(threshold=15.0), ScaledCCA(threshold=15.0),
    ])

    def metric_fn(preds, targets, mc):
        pose_aggr, shape_aggr, camera_pred, trans_aggr = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        K = camera_pred.shape[2]
        t_mid = T // 2   # representative frame

        with torch.no_grad():
            # shape_aggr is (B, P, 10) — expand to (B, T, P, 10) for get_smplx_joints
            shape_exp = shape_aggr.unsqueeze(1).expand(B, T, P, 10)
            # 3D joints via SMPL-X: (B, T, P, Jout, 3)
            pred_joints = get_smplx_joints(
                pose_aggr.float(), shape_exp.float()
            ).cpu().numpy()[..., :55, :]
            gt_joints = get_smplx_joints(
                targets["pose"].float(), targets["shape"].float()
            ).cpu().numpy()[..., :55, :]

            # Per-joint rotation matrices from 6D pose: (B, T, P, J, 3, 3)
            pred_rotmats = rotation_6d_to_matrix(
                pose_aggr.float()
            ).cpu().numpy()
            gt_rotmats = rotation_6d_to_matrix(
                targets["pose"].float()
            ).cpu().numpy()

            # Camera rotations (B, T, K, 3, 3), translations (B, T, K, 3)
            R_pred = quaternion_to_matrix(
                camera_pred[..., :4].float().reshape(-1, 4)
            ).reshape(B, T, K, 3, 3).cpu().numpy()
            t_pred = camera_pred[..., 4:7].float().cpu().numpy()
            f_pred = camera_pred[..., 7].float().cpu().numpy()   # (B, T, K)

            R_gt = quaternion_to_matrix(
                targets["camera"][..., :4].float().reshape(-1, 4)
            ).reshape(B, T, K, 3, 3).cpu().numpy()
            t_gt   = targets["camera"][..., 4:7].float().cpu().numpy()
            f_gt   = targets["camera"][..., 7].float().cpu().numpy()

        # Camera centres: C = -R^T t
        cam_centres_pred = -np.einsum("...ji,...j->...i", R_pred, t_pred)  # (B, T, K, 3)
        cam_centres_gt   = -np.einsum("...ji,...j->...i", R_gt,   t_gt)

        gt_valid_np = targets["gt_valid"].cpu().numpy() if "gt_valid" in targets else None  # (B, T, P)

        for b in range(B):
            # Camera metrics: GT cameras are static in RICH — use middle frame
            Cp = cam_centres_pred[b, t_mid]      # (K, 3)
            Cg = cam_centres_gt[b, t_mid]
            Rp = R_pred[b, t_mid]                # (K, 3, 3)
            Rg = R_gt[b, t_mid]
            mc["TE"].update(Cp, Cg)
            mc["s-TE"].update(Cp, Cg)
            mc["CCA@15"].update(Cp, Cg)
            mc["s-CCA@15"].update(Cp, Cg)
            mc["AE"].update(Rp, Rg)
            mc["RRA@15"].update(Rp, Rg)

            # Body metrics: average over all annotated frames
            for t in range(T):
                if gt_valid_np is not None and not gt_valid_np[b, t].any():
                    continue
                pj = pred_joints[b, t]           # (P, J, 3)
                gj = gt_joints[b, t]
                pr = pred_rotmats[b, t]          # (P, J, 3, 3)
                gr = gt_rotmats[b, t]
                mc["W-MPJPE"].update(pj, gj, Cp, Cg)
                mc["GA-MPJPE"].update(pj, gj)
                mc["PA-MPJPE"].update(pj, gj)
                mc["W-MPJRE"].update(pr, gr, Cp, Cg)
                mc["GA-MPJRE"].update(pr, gr)
                mc["PA-MPJRE"].update(pr, gr)

    if CONFIG.fusion.use_wandb:
        import wandb
        wandb.init(
            project="ghost-fusion",
            name="single_point_overfitting",
            config={
                **vars(CONFIG.fusion.architecture),
                **vars(CONFIG.fusion.loss),
                **vars(CONFIG.fusion.training),
            },
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        losses=losses,
        max_epochs=max_epochs,
        use_wandb=CONFIG.fusion.use_wandb,
        dtype=torch.bfloat16,
        grad_clip=grad_clip,
        scheduler=scheduler,
        early_stopping_patience=patience,
        metrics=metrics,
        metric_fn=metric_fn,
        prediction_save_path=CONFIG.data.fusion_output_dir,
    )

    trainer.train()

    trainer.model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs, targets = trainer._unpack_batch(batch)
        preds = trainer._forward(inputs)
        final_loss = sum(w * fn(preds, targets).item() for fn, w in losses.values())

    logger.info(f"\nFinal combined loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
