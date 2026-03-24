"""Train SSTNetwork on multiple RICH scenes with a train/val split.

Scenes with broken ReID or other issues can be added to SKIP_SCENES below.
The remaining scenes are split into train and val sets.

Run:
    pixi run python scripts/train_rich.py

Via SLURM:
    sbatch bash_jobs/train_rich.sh
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
    BoneLengthconsistencyLoss,
    CameraMSELoss,
    EpipolarLoss,
    PoseMSELoss,
    ShapeMSELoss,
    TemporalSmoothnessLoss,
    TranslationMSELoss,
    TriangulationLoss,
    VPoserLoss,
)
from fusion.metric import (
    GAMPJPE, GAMPJRE, PAMPJPE, PAMPJRE, WMPJPE, WMPJRE,
    AngleError, CCA, MetricCollection, RRA, ScaledCCA,
    ScaledTranslationError, TranslationError,
)
from fusion.trainer import Trainer
from utilities.smplx_utilities import get_smplx_joints

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCENES_ROOT = Path("test_outputs/rich10_segmentation_test")
# Scenes to exclude (broken ReID or other issues).
# Add scene directory names here to skip them.
SKIP_SCENES: list[str] = [
    "ParkingLot2_008_pushup2",
    "Pavallion_003_018_tossball",
]
# Last N scenes (alphabetically) are used for validation, rest for training.
NUM_VAL_SCENES = 2


def load_datapoints(scenes: list[Path]) -> list[RICHFusionDatapoint]:
    datapoints = []
    for scene_dir in scenes:
        try:
            dp = RICHFusionDatapoint(
                scene_dir=scene_dir,
                rich_data_root=CONFIG.data.rich_data_root,
            )
            datapoints.append(dp)
            logger.info(f"  loaded {scene_dir.name}")
        except Exception as e:
            logger.warning(f"  skipping {scene_dir.name}: {e}")
    return datapoints


def main():
    # ── Discover scenes ───────────────────────────────────────────────────────
    all_scenes = sorted(SCENES_ROOT.iterdir())
    scenes = [
        s for s in all_scenes
        if s.is_dir() and s.name not in SKIP_SCENES
    ]
    if not scenes:
        raise RuntimeError(f"No scenes found in {SCENES_ROOT}")

    train_scenes = scenes[:-NUM_VAL_SCENES]
    val_scenes   = scenes[-NUM_VAL_SCENES:]

    logger.info(f"Train scenes ({len(train_scenes)}):")
    for s in train_scenes:
        logger.info(f"  {s.name}")
    logger.info(f"Val scenes ({len(val_scenes)}):")
    for s in val_scenes:
        logger.info(f"  {s.name}")

    # ── Load datapoints ───────────────────────────────────────────────────────
    logger.info("Loading train datapoints...")
    train_dps = load_datapoints(train_scenes)
    logger.info("Loading val datapoints...")
    val_dps   = load_datapoints(val_scenes)

    if not train_dps:
        raise RuntimeError("No valid training scenes could be loaded.")

    train_ds = RICHFusionDataset(train_dps)  # type: ignore[arg-type]
    val_ds   = RICHFusionDataset(val_dps) if val_dps else None  # type: ignore[arg-type]

    # ── Architecture ──────────────────────────────────────────────────────────
    embedding_dim    = CONFIG.fusion.architecture.embedding_dimension
    temporal_window  = CONFIG.fusion.architecture.temporal_window
    max_T            = CONFIG.fusion.architecture.max_temporal_len
    num_heads        = CONFIG.fusion.architecture.num_heads
    num_layers       = CONFIG.fusion.architecture.num_layers
    max_cameras      = CONFIG.fusion.architecture.max_cameras
    dropout          = CONFIG.fusion.architecture.dropout

    # Warn if any sequence is longer than the PE table
    for dp in train_dps + val_dps:
        T = dp._frame_end - dp._frame_start
        if T > max_T:
            logger.warning(
                f"{dp.scene_dir.name} has {T} frames but max_temporal_len={max_T} "
                f"— positional encoding will be out of range. Increase max_temporal_len in config."
            )

    # ── Loss weights ──────────────────────────────────────────────────────────
    pose_mse_weight        = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight       = CONFIG.fusion.loss.shape_mse_weight
    epipolar_weight        = CONFIG.fusion.loss.epipolar_weight
    temporal_weight        = CONFIG.fusion.loss.temporal_weight
    bone_length_weight     = CONFIG.fusion.loss.bone_length_weight
    camera_mse_weight      = CONFIG.fusion.loss.camera_mse_weight
    triangulation_weight   = CONFIG.fusion.loss.triangulation_weight
    translation_mse_weight = CONFIG.fusion.loss.translation_mse_weight
    vposer_weight          = CONFIG.fusion.loss.vposer_weight

    # ── Training params ───────────────────────────────────────────────────────
    lr             = CONFIG.fusion.training.lr
    max_epochs     = CONFIG.fusion.training.max_epochs
    batch_size     = CONFIG.fusion.training.batch_size
    grad_clip      = CONFIG.fusion.training.grad_clip
    scheduler_name = getattr(CONFIG.fusion.training, "scheduler", None)
    patience       = getattr(CONFIG.fusion.training, "patience", None)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False) if val_ds else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SSTNetwork(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_temporal_len=max_T,
        dropout=dropout,
        temporal_window=temporal_window,
        max_cameras=max_cameras,
    )
    logger.info("\n" + model.summary())

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.1
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    else:
        scheduler = None

    # ── Losses ────────────────────────────────────────────────────────────────
    # img_size: use first train scene as reference
    img_size = train_dps[0].img_size

    try:
        vposer_loss = VPoserLoss()
        logger.info("VPoser successfully loaded")
    except Exception as e:
        logger.warning(f"VPoserLoss unavailable ({e}); skipping.")
        vposer_loss = None

    losses = {
        "pose":            (PoseMSELoss(),                       pose_mse_weight),
        "shape":           (ShapeMSELoss(),                      shape_mse_weight),
        "epipolar":        (EpipolarLoss(img_size=img_size),     epipolar_weight),
        "temporal":        (TemporalSmoothnessLoss(),            temporal_weight),
        "bone":            (BoneLengthconsistencyLoss(),         bone_length_weight),
        "camera_mse":      (CameraMSELoss(img_size=img_size),    camera_mse_weight),
        "triangulation":   (TriangulationLoss(),                 triangulation_weight),
        "translation_mse": (TranslationMSELoss(),                translation_mse_weight),
        **({"vposer": (vposer_loss, vposer_weight)} if vposer_loss is not None else {}),
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = MetricCollection([
        WMPJPE(), GAMPJPE(), PAMPJPE(),
        WMPJRE(), GAMPJRE(), PAMPJRE(),
        TranslationError(), ScaledTranslationError(),
        AngleError(),
        RRA(threshold=15.0), CCA(threshold=15.0), ScaledCCA(threshold=15.0),
    ])

    def metric_fn(preds, targets, mc):
        pose_aggr, shape_aggr, camera_pred, _ = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        K = camera_pred.shape[2]
        t_mid = T // 2

        with torch.no_grad():
            shape_exp  = shape_aggr.unsqueeze(1).expand(B, T, P, 10)
            pred_joints = get_smplx_joints(
                pose_aggr.float(), shape_exp.float()
            ).cpu().numpy()[..., :55, :]
            gt_joints = get_smplx_joints(
                targets["pose"].float(), targets["shape"].float()
            ).cpu().numpy()[..., :55, :]

            pred_rotmats = rotation_6d_to_matrix(pose_aggr.float()).cpu().numpy()
            gt_rotmats   = rotation_6d_to_matrix(targets["pose"].float()).cpu().numpy()

            cam_rot_w2c    = quaternion_to_matrix(
                camera_pred[..., :4].float().reshape(-1, 4)
            ).reshape(B, T, K, 3, 3).cpu().numpy()
            cam_transl_w2c = camera_pred[..., 4:7].float().cpu().numpy()

            gt_cam_rot_w2c    = quaternion_to_matrix(
                targets["camera"][..., :4].float().reshape(-1, 4)
            ).reshape(B, T, K, 3, 3).cpu().numpy()
            gt_cam_transl_w2c = targets["camera"][..., 4:7].float().cpu().numpy()

        cam_centres    = -np.einsum("...ji,...j->...i", cam_rot_w2c,    cam_transl_w2c)
        gt_cam_centres = -np.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)

        gt_valid_np = targets["gt_valid"].cpu().numpy() if "gt_valid" in targets else None

        for b in range(B):
            Cp = cam_centres[b, t_mid]
            Cg = gt_cam_centres[b, t_mid]
            Rp = cam_rot_w2c[b, t_mid]
            Rg = gt_cam_rot_w2c[b, t_mid]
            mc["TE"].update(Cp, Cg)
            mc["s-TE"].update(Cp, Cg)
            mc["CCA@15"].update(Cp, Cg)
            mc["s-CCA@15"].update(Cp, Cg)
            mc["AE"].update(Rp, Rg)
            mc["RRA@15"].update(Rp, Rg)

            for t in range(T):
                if gt_valid_np is not None and not gt_valid_np[b, t].any():
                    continue
                pj = pred_joints[b, t]
                gj = gt_joints[b, t]
                pr = pred_rotmats[b, t]
                gr = gt_rotmats[b, t]
                mc["W-MPJPE"].update(pj, gj, Cp, Cg)
                mc["GA-MPJPE"].update(pj, gj)
                mc["PA-MPJPE"].update(pj, gj)
                mc["W-MPJRE"].update(pr, gr, Cp, Cg)
                mc["GA-MPJRE"].update(pr, gr)
                mc["PA-MPJRE"].update(pr, gr)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if CONFIG.fusion.use_wandb:
        import wandb
        wandb.init(
            project="ghost-fusion",
            name="train_rich",
            config={
                "train_scenes": [s.name for s in train_scenes],
                "val_scenes":   [s.name for s in val_scenes],
                **vars(CONFIG.fusion.architecture),
                **vars(CONFIG.fusion.loss),
                **vars(CONFIG.fusion.training),
            },
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
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


if __name__ == "__main__":
    main()
