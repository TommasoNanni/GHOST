"""Train FusionWithBetas on multiple RICH scenes with a train/val split.

Uses:
  - FusionWithBetas  (PoseFusionModule + BetasAggregator) from fusion_module_v2.py
  - TrainerV2        from trainer_v2.py
  - loss_v2.py       losses adapted for the 2-output preds format

Scenes with broken ReID can be added to SKIP_SCENES below.

Run:
    pixi run python scripts/train_rich_v2.py

Via SLURM:
    sbatch bash_jobs/train_rich.sh   # (update the script path inside)
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
from pathlib import Path

faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.fusion_module_v2 import BetasAggregator, FusionWithBetas, PoseFusionModule
from fusion.loss_v2 import (
    JointPositionLoss,
    PoseMSELoss,
    ShapeMSELoss,
    TemporalSmoothnessLoss,
    VPoserLoss,
)
from fusion.metrics_v2 import BetasMSE, MetricCollection, RootRelativeMPJPE, RootRelativeMPJRE
from fusion.trainer_v2 import TrainerV2

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _setup_ddp() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


SCENES_ROOT = Path(CONFIG.data.output_directory)


SKIP_SCENES: list[str] = [
    "Pavallion_013_plankjack",
]
SKIP_CAMERAS: dict[str, list[str]] = {
    "Pavallion_003_018_tossball": ["cam_06"],
    "ParkingLot2_008_pushup2":   ["cam_03"],
    "ParkingLot2_014_takingphotos2": ["cam_01"],
}

NUM_VAL_SCENES = 10
DISABLED_LOSSES: list[str] = []


def load_datapoints(scenes: list[Path]) -> list[RICHFusionDatapoint]:
    datapoints = []
    for scene_dir in scenes:
        try:
            dp = RICHFusionDatapoint(
                scene_dir=scene_dir,
                rich_data_root=CONFIG.data.rich_data_root,
                rich_gt_dir=CONFIG.data.rich_gt_dir,
                exclude_cameras=SKIP_CAMERAS.get(scene_dir.name, []),
            )
            if dp.num_frames == 0:
                logger.warning(f"  skipping {scene_dir.name}: 0 prediction frames")
                continue
            if not dp.has_gt:
                logger.warning(f"  skipping {scene_dir.name}: no GT data")
                continue
            datapoints.append(dp)
            logger.info(f"  loaded {scene_dir.name}")
        except Exception as e:
            logger.warning(f"  skipping {scene_dir.name}: {e}")
    return datapoints


def _split_by_location(scenes: list[Path], num_val: int) -> tuple[list[Path], list[Path]]:
    from collections import defaultdict

    by_location: dict[str, list[Path]] = defaultdict(list)
    for s in scenes:
        loc = s.name.split("_")[0]
        by_location[loc].append(s)

    # One mandatory scene per location goes to train (alphabetically first).
    # The rest are eligible for val via round-robin across locations.
    mandatory_train: list[Path] = []
    available: dict[str, list[Path]] = {}
    for loc in sorted(by_location):
        loc_scenes = sorted(by_location[loc], key=lambda s: s.name)
        mandatory_train.append(loc_scenes[0])
        available[loc] = loc_scenes[1:]  # sorted ascending; we pop from the end each round

    locs = sorted(available)
    val_scenes: list[Path] = []
    while len(val_scenes) < num_val:
        picked_any = False
        for loc in locs:
            if len(val_scenes) >= num_val:
                break
            if available[loc]:
                val_scenes.append(available[loc].pop())
                picked_any = True
        if not picked_any:
            logger.warning(
                f"All location pools exhausted after {len(val_scenes)} val scenes "
                f"(requested {num_val})."
            )
            break

    leftover = [s for loc in locs for s in available[loc]]
    train_scenes = sorted(mandatory_train + leftover, key=lambda s: s.name)
    return train_scenes, val_scenes


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last checkpoint in CONFIG.fusion.checkpoint_dir.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    use_ddp    = "LOCAL_RANK" in os.environ
    local_rank = 0
    rank       = 0
    world_size = 1

    if use_ddp:
        local_rank, rank, world_size = _setup_ddp()

    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Discover scenes ───────────────────────────────────────────────────────
    all_scenes = sorted(SCENES_ROOT.iterdir())
    scenes = [s for s in all_scenes if s.is_dir() and s.name not in SKIP_SCENES]
    if not scenes:
        raise RuntimeError(f"No scenes found in {SCENES_ROOT}")

    train_scenes, val_scenes = _split_by_location(scenes, NUM_VAL_SCENES)

    if is_main:
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

    train_ds = RICHFusionDataset(train_dps, augment=False)
    val_ds   = RICHFusionDataset(val_dps) if val_dps else None

    # ── Architecture ──────────────────────────────────────────────────────────
    embedding_dim   = CONFIG.fusion.architecture.embedding_dimension
    temporal_window = CONFIG.fusion.architecture.temporal_window
    max_T           = CONFIG.fusion.architecture.max_temporal_len
    num_heads       = CONFIG.fusion.architecture.num_heads
    num_layers      = CONFIG.fusion.architecture.num_layers
    dropout         = CONFIG.fusion.architecture.dropout

    for dp in train_dps + val_dps:
        T = dp._frame_end - dp._frame_start
        if T > max_T:
            logger.warning(
                f"{dp.scene_dir.name} has {T} frames but max_temporal_len={max_T} "
                f"— positional encoding will be out of range."
            )

    # ── Training params ───────────────────────────────────────────────────────
    lr             = CONFIG.fusion.training.lr
    max_epochs     = CONFIG.fusion.training.max_epochs
    batch_size     = CONFIG.fusion.training.batch_size
    grad_clip      = CONFIG.fusion.training.grad_clip
    scheduler_name = getattr(CONFIG.fusion.training, "scheduler", None)
    patience       = getattr(CONFIG.fusion.training, "patience", None)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    # ── Model ─────────────────────────────────────────────────────────────────
    pose_module = PoseFusionModule(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_temporal_len=max_T,
        dropout=dropout,
        temporal_window=temporal_window,
    )
    betas_aggregator = BetasAggregator(n_betas=10, embedding_dim=64, num_inducing=4, num_heads=4, dropout=0.3, input_noise_std=0.5)
    model = FusionWithBetas(pose_module, betas_aggregator).to(device)

    if is_main:
        n_pose  = sum(p.numel() for p in pose_module.parameters())
        n_betas = sum(p.numel() for p in betas_aggregator.parameters())
        logger.info(f"PoseFusionModule : {n_pose:,} params")
        logger.info(f"BetasAggregator  : {n_betas:,} params")
        logger.info(f"Total            : {n_pose + n_betas:,} params")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    betas_params = set(betas_aggregator.parameters())
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if p not in betas_params]},
        {"params": list(betas_params), "weight_decay": 1e-3},
    ], lr=lr)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.1
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    else:
        scheduler = None

    # ── Loss weights ──────────────────────────────────────────────────────────
    pose_mse_weight        = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight       = CONFIG.fusion.loss.shape_mse_weight
    temporal_weight        = CONFIG.fusion.loss.temporal_weight
    vposer_weight          = CONFIG.fusion.loss.vposer_weight
    joint_position_weight  = CONFIG.fusion.loss.joint_position_weight

    # ── Losses ────────────────────────────────────────────────────────────────
    # JointPositionLoss uses GT betas for FK to keep pose gradients clean.
    # Betas supervision comes from ShapeMSELoss (direct MSE on preds[1]).
    losses = {
        "pose":           (PoseMSELoss(),                        pose_mse_weight),
        "shape":          (ShapeMSELoss(),                       shape_mse_weight),
        "joint_position": (JointPositionLoss(use_gt_betas=False), joint_position_weight),
    }

    # ── Curriculum ────────────────────────────────────────────────────────────
    curriculum_schedule = {
        0: ["pose", "shape", "joint_position"],
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    from pytorch3d.transforms import rotation_6d_to_matrix
    from utilities.smplx_utilities import get_smplx_joints

    metrics = MetricCollection([RootRelativeMPJPE(), RootRelativeMPJRE(), BetasMSE()])
    METRIC_STRIDE = 8  # evaluate every Nth frame to keep FK memory bounded

    def metric_fn(preds, targets, mc):
        pose_aggr = preds[0]              # (B, T, P, J, 6)  — J=54 (no root)
        betas_out = preds[1]              # (B, P, 10)
        gt_joints = targets["gt_joints"]  # (B, T, P, 55, 3)

        B, T, P = pose_aggr.shape[:3]
        t_idx = torch.arange(0, T, METRIC_STRIDE, device=pose_aggr.device)

        with torch.no_grad():
            pose_sub  = pose_aggr[:, t_idx].float()                        # (B, T', P, 54, 6)

            # Prepend GT global_orient so pred and GT FK share the same root frame.
            # After root-relative subtraction, global rotation cancels — pure body pose comparison.
            gt_root   = targets["pose"][:, t_idx, :, :1, :].float()       # (B, T', P, 1, 6)
            pose_full = torch.cat([gt_root, pose_sub], dim=3)              # (B, T', P, 55, 6)

            betas_exp   = betas_out.unsqueeze(1).expand(B, len(t_idx), P, 10).float()
            pred_joints = get_smplx_joints(pose_full, betas_exp).cpu().numpy()[..., :55, :]  # (B,T',P,55,3)
            gt_joints_sub = gt_joints[:, t_idx].float().cpu().numpy()      # (B, T', P, 55, 3)

            rot_pred = rotation_6d_to_matrix(pose_full.cpu().reshape(-1, 6)).reshape(B, len(t_idx), P, 55, 3, 3).numpy()

            gt_pose_sub = targets["pose"][:, t_idx].float().cpu()          # (B, T', P, 55, 6)
            rot_gt = rotation_6d_to_matrix(gt_pose_sub.reshape(-1, 6)).reshape(B, len(t_idx), P, 55, 3, 3).numpy()

            betas_np = betas_out.float().cpu().numpy()                     # (B, P, 10)

            # Compute masked time-mean of GT betas — same reference as ShapeMSELoss.
            gt_valid = targets.get("gt_valid")
            shape_gt = targets["shape"].float()                            # (B, T, P, 10)
            if gt_valid is not None:
                mask_f = gt_valid.float().unsqueeze(-1)                    # (B, T, P, 1)
                count  = mask_f.sum(dim=1).clamp(min=1)                    # (B, P, 1)
                gt_betas_mean = ((shape_gt * mask_f).sum(dim=1) / count).cpu().numpy()  # (B, P, 10)
            else:
                gt_betas_mean = shape_gt.mean(dim=1).cpu().numpy()        # (B, P, 10)

        for b in range(B):
            for t in range(len(t_idx)):
                if gt_valid is not None and not gt_valid[b, t_idx[t]].any():
                    continue
                for p in range(P):
                    if gt_valid is not None and not gt_valid[b, t_idx[t], p]:
                        continue
                    mc["RR-MPJPE"].update(pred_joints[b, t, p], gt_joints_sub[b, t, p])
                    mc["RR-MPJRE"].update(rot_pred[b, t, p], rot_gt[b, t, p])

            for p in range(P):
                # Only update if person has at least one valid GT frame.
                if gt_valid is not None and not gt_valid[b, :, p].any():
                    continue
                mc["Betas-MSE"].update(betas_np[b, p], gt_betas_mean[b, p])

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    resume_checkpoint = None
    if args.resume:
        last_pt = Path(CONFIG.fusion.checkpoint_dir) / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(
                f"--resume specified but no checkpoint at {last_pt}"
            )
        resume_checkpoint = last_pt
        if is_main:
            logger.info(f"Resuming from {last_pt}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if CONFIG.fusion.use_wandb and is_main:
        import wandb
        _wandb_run_id = None
        if resume_checkpoint is not None:
            _ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            _wandb_run_id = _ckpt.get("wandb_run_id")
            del _ckpt
        wandb.init(
            project="ghost-fusion",
            name="train_rich_v2",
            id=_wandb_run_id,
            resume="allow",
            config={
                "train_scenes": [s.name for s in train_scenes],
                "val_scenes":   [s.name for s in val_scenes],
                "world_size":   world_size,
                **vars(CONFIG.fusion.architecture),
                **vars(CONFIG.fusion.loss),
                **vars(CONFIG.fusion.training),
            },
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = TrainerV2(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        losses=losses,
        max_epochs=max_epochs,
        use_wandb=CONFIG.fusion.use_wandb,
        use_amp=True,
        grad_clip=grad_clip,
        scheduler=scheduler,
        early_stopping_patience=patience,
        checkpoint_dir=CONFIG.fusion.checkpoint_dir,
        metrics=metrics,
        metric_fn=metric_fn,
        curriculum_schedule=curriculum_schedule,
        is_main_process=is_main,
        train_sampler=train_sampler,
        device=device,
        resume_checkpoint=resume_checkpoint,
    )

    try:
        trainer.train()
    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
