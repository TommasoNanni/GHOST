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
from fusion.fusion_module import SSTNetwork, WindowedTemporalAttention
from fusion.loss import (
    BoneLengthconsistencyLoss,
    CameraMSELoss,
    EpipolarLoss,
    PoseMSELoss,
    ShapeMSELoss,
    ShapeRegularizationLoss,
    TemporalSmoothnessLoss,
    TranslationMSELoss,
    TranslationSmoothnessLoss,
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

SCENES_ROOT = Path(CONFIG.data.output_directory)
# Scenes to exclude (broken ReID or other issues).
# Add scene directory names here to skip them.
SKIP_SCENES: list[str] = [
    "ParkingLot2_008_pushup2",
    "Pavallion_003_018_tossball",
    "ParkingLot1_002_burpee3",
]
NUM_VAL_SCENES = 2
DISABLED_LOSSES: list[str] = []


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


def _split_by_location(scenes: list[Path], num_val: int) -> tuple[list[Path], list[Path]]:
    """Split scenes into train/val ensuring every location appears in train.

    Algorithm:
    1. Group scenes by location prefix (first '_'-separated token, e.g. 'BBQ').
    2. From each location take the first scene (alphabetically) as a mandatory
       train scene — guarantees full location coverage in training.
    3. All remaining scenes form a pool sorted alphabetically.
    4. The last ``num_val`` scenes in the pool go to val (preferring diversity
       since the pool already has one scene per location removed); the rest go
       to train.  If the pool has fewer than ``num_val`` scenes, all pool scenes
       go to val and a warning is logged.
    """
    from collections import defaultdict

    by_location: dict[str, list[Path]] = defaultdict(list)
    for s in scenes:
        loc = s.name.split('_')[0]
        by_location[loc].append(s)

    mandatory_train: list[Path] = []
    pool: list[Path] = []
    for loc in sorted(by_location):
        loc_scenes = sorted(by_location[loc], key=lambda s: s.name)
        mandatory_train.append(loc_scenes[0])   # one per location, always in train
        pool.extend(loc_scenes[1:])

    pool = sorted(pool, key=lambda s: s.name)

    if num_val > len(pool):
        logger.warning(
            f"NUM_VAL_SCENES={num_val} but only {len(pool)} scenes available for val "
            f"after reserving one per location for train. Using all {len(pool)} as val."
        )
        val_scenes = pool
        extra_train: list[Path] = []
    else:
        val_scenes   = pool[-num_val:]
        extra_train  = pool[:-num_val]

    train_scenes = sorted(mandatory_train + extra_train, key=lambda s: s.name)
    return train_scenes, val_scenes


def main():
    # ── Discover scenes ───────────────────────────────────────────────────────
    all_scenes = sorted(SCENES_ROOT.iterdir())
    scenes = [
        s for s in all_scenes
        if s.is_dir() and s.name not in SKIP_SCENES
    ]
    if not scenes:
        raise RuntimeError(f"No scenes found in {SCENES_ROOT}")

    train_scenes, val_scenes = _split_by_location(scenes, NUM_VAL_SCENES)

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
    translation_mse_weight      = CONFIG.fusion.loss.translation_mse_weight
    shape_reg_weight            = CONFIG.fusion.loss.shape_reg_weight
    translation_temporal_weight = CONFIG.fusion.loss.translation_temporal_weight
    vposer_weight               = CONFIG.fusion.loss.vposer_weight

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

    # flex_attention requires torch.compile to use its sparse backward pass.
    # Without it, the backward falls back to a dense O(T²) implementation that
    # OOMs on long sequences (T=1157 tried to allocate 12.95 GiB).
    # We compile only the LocalWindowTemporalAttention layers (not the whole model)
    # to avoid breaking the cross-attention modules which fail with full compile.
    for module in model.modules():
        if isinstance(module, WindowedTemporalAttention):
            module.forward = torch.compile(module.forward, dynamic=True)

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

    _all_losses = {
        "pose":            (PoseMSELoss(),                       pose_mse_weight),
        "shape":           (ShapeMSELoss(),                      shape_mse_weight),
        "epipolar":        (EpipolarLoss(img_size=img_size),     epipolar_weight),
        "temporal":        (TemporalSmoothnessLoss(),            temporal_weight),
        "bone":            (BoneLengthconsistencyLoss(),         bone_length_weight),
        "camera_mse":      (CameraMSELoss(img_size=img_size),    camera_mse_weight),
        "triangulation":   (TriangulationLoss(),                 triangulation_weight),
        "translation_mse":      (TranslationMSELoss(),                translation_mse_weight),
        "shape_reg":            (ShapeRegularizationLoss(),           shape_reg_weight),
        "translation_temporal": (TranslationSmoothnessLoss(),         translation_temporal_weight),
        **({"vposer": (vposer_loss, vposer_weight)} if vposer_loss is not None else {}),
    }
    losses = {k: v for k, v in _all_losses.items() if k not in DISABLED_LOSSES}
    if DISABLED_LOSSES:
        logger.info(f"Disabled losses: {DISABLED_LOSSES}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = MetricCollection([
        WMPJPE(), GAMPJPE(), PAMPJPE(),
        WMPJRE(), GAMPJRE(), PAMPJRE(),
        TranslationError(), ScaledTranslationError(),
        AngleError(),
        RRA(threshold=15.0), CCA(threshold=15.0), ScaledCCA(threshold=15.0),
    ])

    METRIC_STRIDE = 8  # evaluate every Nth frame to keep SMPL-X memory bounded

    def metric_fn(preds, targets, mc):
        pose_aggr, shape_aggr, camera_pred, body_transl_world = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        K = camera_pred.shape[2]

        # Subsample time axis to bound the two SMPL-X forward passes below
        t_idx = torch.arange(0, T, METRIC_STRIDE, device=pose_aggr.device)

        with torch.no_grad():
            pose_sub   = pose_aggr[:, t_idx].float()
            shape_sub  = shape_aggr.unsqueeze(1).expand(B, len(t_idx), P, 10).float()
            # Root-relative joints from SMPL-X forward kinematics
            pred_joints_rel = get_smplx_joints(pose_sub, shape_sub).cpu().numpy()[..., :55, :]

            gt_pose_sub  = targets["pose"][:, t_idx].float()
            gt_shape_sub = targets["shape"][:, t_idx].float()
            gt_joints_rel = get_smplx_joints(gt_pose_sub, gt_shape_sub).cpu().numpy()[..., :55, :]

            # World-frame joints: add root translation so cameras and joints
            # live in the same coordinate system (fixes W-MPJPE alignment).
            # transl shape: (B, T_sub, P, 3) → broadcast over J → (B, T_sub, P, 1, 3)
            pred_transl = body_transl_world[:, t_idx].float().cpu().numpy()   # (B, T_sub, P, 3)
            pred_joints  = pred_joints_rel + pred_transl[:, :, :, None, :]   # (B, T_sub, P, J, 3)

            if "trans" in targets:
                gt_transl   = targets["trans"][:, t_idx].float().cpu().numpy()  # (B, T_sub, P, 3)
                gt_joints   = gt_joints_rel + gt_transl[:, :, :, None, :]
            else:
                gt_joints   = gt_joints_rel

            pred_rotmats = rotation_6d_to_matrix(pose_aggr[:, t_idx].float()).cpu().numpy()
            gt_rotmats   = rotation_6d_to_matrix(targets["pose"][:, t_idx].float()).cpu().numpy()

            T_sub = len(t_idx)
            cam_rot_w2c    = quaternion_to_matrix(
                camera_pred[:, t_idx, :, :4].float().reshape(-1, 4)
            ).reshape(B, T_sub, K, 3, 3).cpu().numpy()
            cam_transl_w2c = camera_pred[:, t_idx, :, 4:7].float().cpu().numpy()

            gt_cam_rot_w2c    = quaternion_to_matrix(
                targets["camera"][:, t_idx, :, :4].float().reshape(-1, 4)
            ).reshape(B, T_sub, K, 3, 3).cpu().numpy()
            gt_cam_transl_w2c = targets["camera"][:, t_idx, :, 4:7].float().cpu().numpy()

        cam_centres    = -np.einsum("...ji,...j->...i", cam_rot_w2c,    cam_transl_w2c)
        gt_cam_centres = -np.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)

        gt_valid_np = targets["gt_valid"][:, t_idx].cpu().numpy() if "gt_valid" in targets else None

        # cam_valid: cameras where GT quaternion is a real rotation (norm > 0.5).
        # Cameras with no person detection stay at zero quaternion → excluded.
        cam_valid_np = (
            targets["camera"][:, t_idx, :, :4].float().norm(dim=-1) > 0.5
        ).cpu().numpy()  # (B, T_sub, K)

        t_mid_sub = T_sub // 2
        for b in range(B):
            valid = cam_valid_np[b, t_mid_sub]  # (K,) bool
            Cp = cam_centres[b, t_mid_sub][valid]
            Cg = gt_cam_centres[b, t_mid_sub][valid]
            Rp = cam_rot_w2c[b, t_mid_sub][valid]
            Rg = gt_cam_rot_w2c[b, t_mid_sub][valid]
            # umeyama requires ≥3 cameras and non-degenerate predicted positions.
            # At early training the camera head outputs near-identity for all cameras
            # → all Cp collapse to ~same point → covariance ≈ 0 → SVD fails.
            pred_spread = float(np.linalg.norm(Cp - Cp.mean(0), axis=-1).max()) if valid.sum() >= 3 else 0.0
            if valid.sum() >= 3 and pred_spread > 1e-3:
                mc["TE"].update(Cp, Cg)
                mc["s-TE"].update(Cp, Cg)
                mc["CCA@15"].update(Cp, Cg)
                mc["s-CCA@15"].update(Cp, Cg)
            if valid.sum() >= 1:
                mc["AE"].update(Rp, Rg)
                mc["RRA@15"].update(Rp, Rg)

            for t in range(T_sub):
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

    # ── Curriculum schedule ───────────────────────────────────────────────────
    # Epoch 0:   MSE losses only (safe, no geometric instability)
    # Epoch 50:  add regularisation (temporal smoothness, bone length, VPoser)
    # Epoch 100: add geometric losses (epipolar, triangulation) — only once
    #            poses and cameras are already roughly aligned
    curriculum_schedule = {
        0:   ["pose", "shape", "camera_mse", "shape_reg", "translation_temporal"],
        25:  ["translation_mse", "temporal", "bone", "vposer"],
        100: ["epipolar", "triangulation"],
    }

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        losses=losses,
        max_epochs=max_epochs,
        use_wandb=CONFIG.fusion.use_wandb,
        dtype=None,
        use_amp=True,
        grad_clip=grad_clip,
        scheduler=scheduler,
        early_stopping_patience=patience,
        checkpoint_dir=CONFIG.fusion.checkpoint_dir,
        metrics=metrics,
        metric_fn=metric_fn,
        prediction_save_path=CONFIG.data.fusion_output_dir,
        curriculum_schedule=curriculum_schedule,
    )

    trainer.train()


if __name__ == "__main__":
    main()
