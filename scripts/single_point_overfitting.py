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
TEMPORAL_WINDOW = 32   # each frame attends to 2*32+1 = 65 neighbours


def main():
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
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    logger.info(
        f"WindowedTemporalAttention — flex_attention + block_mask  "
        f"(temporal_window={TEMPORAL_WINDOW}, T={T}  "
        f"-> each frame attends to {min(2 * TEMPORAL_WINDOW + 1, T)} frames)"
    )

    model = SSTNetwork(
        embedding_dim=64,
        num_heads=8,
        num_layers=8,
        max_temporal_len=T,
        dropout=0.1,
        temporal_window=TEMPORAL_WINDOW,
    )

    logger.info("\n" + model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        vposer = VPoserLoss()
    except Exception as e:
        logger.warning(f"VPoserLoss unavailable ({e}); skipping.")
        vposer = None

    losses = {
        "pose":       (PoseMSELoss(),             1.0),
        "shape":      (ShapeMSELoss(),            1.0),
        "epipolar":   (EpipolarLoss(img_size=img_size),            0.1),
        "temporal":   (TemporalSmoothnessLoss(),  0.1),
        "bone":       (BoneLengthconsistencyLoss(), 0.0),
        "beta":       (BetaConsistencyLoss(),     0.1),
        "camera_mse": (CameraMSELoss(img_size=img_size), 0.1),
        **({"vposer": (vposer, 0.01)} if vposer is not None else {}),
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        losses=losses,
        max_epochs=500,
        use_wandb=False,
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
