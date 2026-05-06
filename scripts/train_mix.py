"""Train SSTNetwork on a mixed RICH + DNA-Rendering dataset.

RICH scenes are split by location (same strategy as train_rich.py).
DNA-Rendering scenes are split by a simple last-N-as-val strategy, since
their naming convention doesn't follow the RICH ``Location_XXX_name`` pattern.

Both pools are combined into a :class:`MixedFusionDataset`.  Use
``DNA_WEIGHT`` to oversample DNA scenes and compensate for a smaller pool.

Run:
    pixi run python scripts/train_mix.py

Via SLURM:
    sbatch bash_jobs/train_mix.sh
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
from data.fusion_dataset import (
    DNARenderingFusionDatapoint,
    MixedFusionDataset,
    RICHFusionDatapoint,
    RICHFusionDataset,
)
from fusion.fusion_module import SSTNetwork, WindowedTemporalAttention
from fusion.loss import (
    BoneLengthconsistencyLoss,
    CameraMSELoss,
    EpipolarLoss,
    JointPositionLoss,
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

# ── Scene roots ───────────────────────────────────────────────────────────────
# RICH: ghost pipeline output (same as train_rich.py)
RICH_SCENES_ROOT = Path(CONFIG.data.output_directory)
# DNA-Rendering: ghost pipeline output for DNA scenes.
# Set this to the directory that holds the processed DNA-Rendering scenes.
DNA_SCENES_ROOT  = Path(CONFIG.data.output_directory)   # ← change when DNA is ready

# ── Scene filters ─────────────────────────────────────────────────────────────
RICH_SKIP_SCENES: list[str] = [
    "ParkingLot2_008_pushup2",
    "Pavallion_003_018_tossball",
    "ParkingLot1_002_burpee3",
]
DNA_SKIP_SCENES: list[str] = []

# ── Split sizes ───────────────────────────────────────────────────────────────
RICH_NUM_VAL_SCENES = 2   # held out from the pool after mandatory-train selection
DNA_NUM_VAL_SCENES  = 1   # last N DNA scenes (alphabetical) go to val

# ── Mixing weight ─────────────────────────────────────────────────────────────
# Each DNA scene will appear DNA_WEIGHT times per epoch relative to each RICH
# scene.  Set to 1.0 for no oversampling.
DNA_WEIGHT: float = 1.0

DISABLED_LOSSES: list[str] = []


# ── Scene discovery helpers ───────────────────────────────────────────────────

def _discover_rich_scenes() -> list[Path]:
    all_scenes = sorted(RICH_SCENES_ROOT.iterdir())
    return [
        s for s in all_scenes
        if s.is_dir()
        and s.name not in RICH_SKIP_SCENES
        and s.name not in DNA_SKIP_SCENES   # avoid double-counting if roots overlap
        and _looks_like_rich(s.name)
    ]


def _discover_dna_scenes() -> list[Path]:
    if DNA_SCENES_ROOT == RICH_SCENES_ROOT:
        # Roots overlap: pick only directories that are NOT RICH-style names.
        all_scenes = sorted(DNA_SCENES_ROOT.iterdir())
        return [
            s for s in all_scenes
            if s.is_dir()
            and s.name not in DNA_SKIP_SCENES
            and not _looks_like_rich(s.name)
        ]
    all_scenes = sorted(DNA_SCENES_ROOT.iterdir())
    return [
        s for s in all_scenes
        if s.is_dir() and s.name not in DNA_SKIP_SCENES
    ]


def _looks_like_rich(name: str) -> bool:
    """Heuristic: RICH scene names follow ``Location_NNN_activity`` pattern."""
    parts = name.split("_")
    return len(parts) >= 3 and parts[1].isdigit() and len(parts[1]) == 3


# ── Train/val splits ──────────────────────────────────────────────────────────

def _split_rich_by_location(
    scenes: list[Path], num_val: int
) -> tuple[list[Path], list[Path]]:
    """Split RICH scenes ensuring every location appears in train.

    Same algorithm as train_rich.py:
    1. Group by location prefix (first '_'-separated token).
    2. One mandatory-train scene per location (alphabetically first).
    3. Remaining pool: last ``num_val`` go to val, rest to train.
    """
    from collections import defaultdict

    by_location: dict[str, list[Path]] = defaultdict(list)
    for s in scenes:
        loc = s.name.split("_")[0]
        by_location[loc].append(s)

    mandatory_train: list[Path] = []
    pool: list[Path] = []
    for loc in sorted(by_location):
        loc_scenes = sorted(by_location[loc], key=lambda s: s.name)
        mandatory_train.append(loc_scenes[0])
        pool.extend(loc_scenes[1:])

    pool = sorted(pool, key=lambda s: s.name)

    if num_val > len(pool):
        logger.warning(
            f"RICH_NUM_VAL_SCENES={num_val} but only {len(pool)} scenes in pool. "
            f"Using all {len(pool)} as val."
        )
        val_scenes = pool
        extra_train: list[Path] = []
    else:
        val_scenes  = pool[-num_val:]
        extra_train = pool[:-num_val]

    train_scenes = sorted(mandatory_train + extra_train, key=lambda s: s.name)
    return train_scenes, val_scenes


def _split_dna_last_n(
    scenes: list[Path], num_val: int
) -> tuple[list[Path], list[Path]]:
    """Split DNA scenes alphabetically: last ``num_val`` go to val."""
    scenes = sorted(scenes, key=lambda s: s.name)
    if num_val >= len(scenes):
        logger.warning(
            f"DNA_NUM_VAL_SCENES={num_val} ≥ total DNA scenes ({len(scenes)}). "
            "All DNA scenes will be val — no DNA in train."
        )
        return [], scenes
    return scenes[:-num_val], scenes[-num_val:]


# ── Datapoint loaders ─────────────────────────────────────────────────────────

def load_rich_datapoints(scenes: list[Path]) -> list[RICHFusionDatapoint]:
    datapoints = []
    for scene_dir in scenes:
        try:
            dp = RICHFusionDatapoint(
                scene_dir=scene_dir,
                rich_data_root=CONFIG.data.rich_data_root,
            )
            datapoints.append(dp)
            logger.info(f"  [RICH] loaded {scene_dir.name}")
        except Exception as e:
            logger.warning(f"  [RICH] skipping {scene_dir.name}: {e}")
    return datapoints


def load_dna_datapoints(scenes: list[Path]) -> list[DNARenderingFusionDatapoint]:
    datapoints = []
    for scene_dir in scenes:
        # The annots file is expected at <DNA_SCENES_ROOT>/<scene>/<scene>_annots.smc
        # or wherever you place it.  Pass None and let the class warn if absent.
        annots_path = DNA_SCENES_ROOT / scene_dir.name / f"{scene_dir.name}_annots.smc"
        if not annots_path.exists():
            annots_path = None  # type: ignore[assignment]
        try:
            dp = DNARenderingFusionDatapoint(
                scene_dir=scene_dir,
                annots_path=annots_path,
            )
            datapoints.append(dp)
            logger.info(f"  [DNA] loaded {scene_dir.name}")
        except Exception as e:
            logger.warning(f"  [DNA] skipping {scene_dir.name}: {e}")
    return datapoints


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Discover and split RICH scenes ───────────────────────────────────────
    rich_scenes = _discover_rich_scenes()
    if not rich_scenes:
        logger.warning(f"No RICH scenes found under {RICH_SCENES_ROOT}")
    rich_train_scenes, rich_val_scenes = _split_rich_by_location(rich_scenes, RICH_NUM_VAL_SCENES)

    logger.info(f"RICH train scenes ({len(rich_train_scenes)}):")
    for s in rich_train_scenes:
        logger.info(f"  {s.name}")
    logger.info(f"RICH val scenes ({len(rich_val_scenes)}):")
    for s in rich_val_scenes:
        logger.info(f"  {s.name}")

    # ── Discover and split DNA scenes ─────────────────────────────────────────
    dna_scenes = _discover_dna_scenes()
    if not dna_scenes:
        logger.warning(f"No DNA scenes found under {DNA_SCENES_ROOT}")
    dna_train_scenes, dna_val_scenes = _split_dna_last_n(dna_scenes, DNA_NUM_VAL_SCENES)

    logger.info(f"DNA train scenes ({len(dna_train_scenes)}):")
    for s in dna_train_scenes:
        logger.info(f"  {s.name}")
    logger.info(f"DNA val scenes ({len(dna_val_scenes)}):")
    for s in dna_val_scenes:
        logger.info(f"  {s.name}")

    # ── Load datapoints ───────────────────────────────────────────────────────
    logger.info("Loading RICH train datapoints...")
    rich_train_dps = load_rich_datapoints(rich_train_scenes)
    logger.info("Loading RICH val datapoints...")
    rich_val_dps   = load_rich_datapoints(rich_val_scenes)

    logger.info("Loading DNA train datapoints...")
    dna_train_dps  = load_dna_datapoints(dna_train_scenes)
    logger.info("Loading DNA val datapoints...")
    dna_val_dps    = load_dna_datapoints(dna_val_scenes)

    all_train_dps = rich_train_dps + dna_train_dps
    all_val_dps   = rich_val_dps   + dna_val_dps

    if not all_train_dps:
        raise RuntimeError("No valid training scenes could be loaded from either dataset.")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = MixedFusionDataset(
        sources=[
            ("rich", rich_train_dps),
            ("dna",  dna_train_dps),
        ],
        weights=[1.0, DNA_WEIGHT],
    )
    # Val dataset: plain list wrapped in RICHFusionDataset (no oversampling needed).
    val_ds = RICHFusionDataset(all_val_dps) if all_val_dps else None  # type: ignore[arg-type]

    # ── Architecture ──────────────────────────────────────────────────────────
    embedding_dim   = CONFIG.fusion.architecture.embedding_dimension
    temporal_window = CONFIG.fusion.architecture.temporal_window
    max_T           = CONFIG.fusion.architecture.max_temporal_len
    num_heads       = CONFIG.fusion.architecture.num_heads
    num_layers      = CONFIG.fusion.architecture.num_layers
    max_cameras     = CONFIG.fusion.architecture.max_cameras
    dropout         = CONFIG.fusion.architecture.dropout

    for dp in all_train_dps + all_val_dps:
        T = dp._frame_end - dp._frame_start
        if T > max_T:
            logger.warning(
                f"{dp.scene_dir.name} has {T} frames but max_temporal_len={max_T} "
                f"— positional encoding will be out of range. Increase max_temporal_len in config."
            )

    # ── Loss weights ──────────────────────────────────────────────────────────
    pose_mse_weight             = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight            = CONFIG.fusion.loss.shape_mse_weight
    epipolar_weight             = CONFIG.fusion.loss.epipolar_weight
    temporal_weight             = CONFIG.fusion.loss.temporal_weight
    bone_length_weight          = CONFIG.fusion.loss.bone_length_weight
    camera_mse_weight           = CONFIG.fusion.loss.camera_mse_weight
    triangulation_weight        = CONFIG.fusion.loss.triangulation_weight
    translation_mse_weight      = CONFIG.fusion.loss.translation_mse_weight
    shape_reg_weight            = CONFIG.fusion.loss.shape_reg_weight
    translation_temporal_weight = CONFIG.fusion.loss.translation_temporal_weight
    vposer_weight               = CONFIG.fusion.loss.vposer_weight
    joint_position_weight       = CONFIG.fusion.loss.joint_position_weight

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
    img_size = all_train_dps[0].img_size

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
        "joint_position": (JointPositionLoss(),                  joint_position_weight),
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

    METRIC_STRIDE = 8

    def metric_fn(preds, targets, mc):
        pose_aggr, shape_aggr, camera_pred, body_transl_world = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        K = camera_pred.shape[2]

        t_idx = torch.arange(0, T, METRIC_STRIDE, device=pose_aggr.device)

        with torch.no_grad():
            pose_sub  = pose_aggr[:, t_idx].float()
            shape_sub = shape_aggr.unsqueeze(1).expand(B, len(t_idx), P, 10).float()
            pred_joints_rel = get_smplx_joints(pose_sub, shape_sub).cpu().numpy()[..., :55, :]

            gt_pose_sub  = targets["pose"][:, t_idx].float()
            gt_shape_sub = targets["shape"][:, t_idx].float()
            gt_joints_rel = get_smplx_joints(gt_pose_sub, gt_shape_sub).cpu().numpy()[..., :55, :]

            pred_transl = body_transl_world[:, t_idx].float().cpu().numpy()
            pred_joints = pred_joints_rel + pred_transl[:, :, :, None, :]

            if "trans" in targets:
                gt_transl = targets["trans"][:, t_idx].float().cpu().numpy()
                gt_joints = gt_joints_rel + gt_transl[:, :, :, None, :]
            else:
                gt_joints = gt_joints_rel

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
        cam_valid_np = (
            targets["camera"][:, t_idx, :, :4].float().norm(dim=-1) > 0.5
        ).cpu().numpy()

        t_mid_sub = T_sub // 2
        for b in range(B):
            valid = cam_valid_np[b, t_mid_sub]
            Cp = cam_centres[b, t_mid_sub][valid]
            Cg = gt_cam_centres[b, t_mid_sub][valid]
            Rp = cam_rot_w2c[b, t_mid_sub][valid]
            Rg = gt_cam_rot_w2c[b, t_mid_sub][valid]
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
            name="train_mix",
            config={
                "rich_train_scenes": [s.name for s in rich_train_scenes],
                "rich_val_scenes":   [s.name for s in rich_val_scenes],
                "dna_train_scenes":  [s.name for s in dna_train_scenes],
                "dna_val_scenes":    [s.name for s in dna_val_scenes],
                "dna_weight":        DNA_WEIGHT,
                **vars(CONFIG.fusion.architecture),
                **vars(CONFIG.fusion.loss),
                **vars(CONFIG.fusion.training),
            },
        )

    # ── Curriculum schedule ───────────────────────────────────────────────────
    curriculum_schedule = {
        0:   ["pose", "shape", "camera_mse", "shape_reg", "translation_temporal", "joint_position"],
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
