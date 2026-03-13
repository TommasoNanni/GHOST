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

import torch
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
    BetaConsistencyLoss,
    CameraMSELoss,
)
from fusion.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


RICH_SCENE_DIR  = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs"
    "/rich5_segmentation_test/BBQ_001_guitar"
)


def main():
    # Architecture
    embedding_dim           = CONFIG.fusion.architecture.embedding_dimension
    temporal_window         = CONFIG.fusion.architecture.temporal_window
    num_heads               = CONFIG.fusion.architecture.num_heads
    num_layers              = CONFIG.fusion.architecture.num_layers
    dropout                 = CONFIG.fusion.architecture.dropout
    # Loss weights
    pose_mse_weight         = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight        = CONFIG.fusion.loss.shape_mse_weight
    epipolar_weight         = CONFIG.fusion.loss.epipolar_weight
    temporal_weight         = CONFIG.fusion.loss.temporal_weight
    bone_length_weight      = CONFIG.fusion.loss.bone_length_weight
    beta_consistency_weight = CONFIG.fusion.loss.beta_consistency_weight
    camera_mse_weight       = CONFIG.fusion.loss.camera_mse_weight
    vposer_weight           = CONFIG.fusion.loss.vposer_weight
    # Training params
    lr                      = CONFIG.fusion.training.lr
    max_epochs              = CONFIG.fusion.training.max_epochs
    batch_size              = CONFIG.fusion.training.batch_size

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
    )

    logger.info("\n" + model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        "beta":       (BetaConsistencyLoss(),            beta_consistency_weight      ),
        "camera_mse": (CameraMSELoss(img_size=img_size), camera_mse_weight            ),
        **({"vposer": (vposer_loss, vposer_weight)} if vposer_loss is not None else {}),
    }

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
    )

    trainer.train()

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
