"""Train PoseFusionModuleV3 (residual fusion) on RICH with a train/val split.

Copy of scripts/train_rich_v2.py. Differences, and ONLY these:

  * imports PoseFusionModuleV3 (fusion/fusion_module_v3.py) — predicts a
    CORRECTION to the chordal mean of the per-camera estimates, from tokens that
    carry each camera's DEVIATION from that mean. Parameter shapes are identical
    to v2, so nothing else in the pipeline changes.
  * the zero-initialised output layer gets its own optimizer param group with
    `head_lr_mult` x the base LR (removes the warm-up ramp; 1.0 disables).
  * model_config comes from `module.arch_config()` rather than a hand-written
    dict, so a flag can never be added to the module and forgotten here.
  * `--max_epochs` override, so the cosine schedule can be set to the length the
    run will actually reach (previous runs stopped near 270-296 of a 400-epoch
    schedule and never entered the low-LR refinement phase).

Losses, split, batching, metrics and curriculum are untouched.

Uses:
  - PoseFusionModuleV3 from fusion_module_v3.py
  - TrainerV2        from trainer_v2.py
  - loss_v2.py

Scenes with broken ReID can be added to SKIP_SCENES below.

Run:
    pixi run python scripts/train_rich_v3.py

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
# human_body_prior submodule: the actual Python package is one level deep.
# Must be inserted before any other import touches human_body_prior.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "human_body_prior"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.fusion_module_v3 import PoseFusionModuleV3
from fusion.loss_v2 import (
    JointPositionLoss,
    PoseMSELoss,
    TemporalSmoothnessLoss,
    VPoserLoss,
)
from fusion.metrics_v2 import MetricCollection, RootRelativeMPJPE, RootRelativeMPJRE
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
    "Pavallion_013_phonesiteat",
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
    # R0/R1/R2 overrides. Each defaults to None = "take the config value", so the
    # config carries the R2 setting and the earlier runs switch pieces off.
    parser.add_argument(
        "--kintree_k", type=int, default=None,
        help="CHANGE 2 hop radius for the joint-attention kintree mask. "
             "Negative disables the mask (R0/R1). Omit to use the config value.",
    )
    parser.add_argument(
        "--joint_body_only", type=int, choices=(0, 1), default=None,
        help="CHANGE 1: restrict JointPositionLoss to root + 21 body joints. "
             "0 for the R0 baseline. Omit to use the config value.",
    )
    parser.add_argument(
        "--pose_hand_weight", type=float, default=None,
        help="CHANGE 1: per-joint hand weight inside PoseMSELoss. "
             "1.0 for the R0 baseline. Omit to use the config value.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Override CONFIG.fusion.training.max_epochs. The cosine schedule is "
             "built from this, so set it to the length the run will actually reach.",
    )
    parser.add_argument(
        "--run_tag", default=None,
        help="Label printed at the top of the log (e.g. R0/R1/R2).",
    )
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Override CONFIG.fusion.checkpoint_dir so R0/R1/R2 never share a "
             "directory. Omit to use the config value.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Applied before anything reads it (resume path, wandb, trainer).
    if args.checkpoint_dir:
        CONFIG.fusion.checkpoint_dir = args.checkpoint_dir

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
    max_epochs     = args.max_epochs or CONFIG.fusion.training.max_epochs
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
    # ── R0 / R1 / R2 flags ────────────────────────────────────────────────────
    # Config holds the R2 values; the CLI overrides them so all three runs come
    # from one codebase. Resolved here and logged below so every run's log states
    # exactly which changes were active.
    kintree_mask_k = getattr(CONFIG.fusion.architecture, "kintree_mask_k", None)
    if args.kintree_k is not None:
        kintree_mask_k = None if args.kintree_k < 0 else args.kintree_k
    joint_body_only = getattr(CONFIG.fusion.loss, "joint_position_body_only", False)
    if args.joint_body_only is not None:
        joint_body_only = bool(args.joint_body_only)
    pose_hand_weight = getattr(CONFIG.fusion.loss, "pose_hand_weight", 1.0)
    if args.pose_hand_weight is not None:
        pose_hand_weight = args.pose_hand_weight
    pose_body_weight = getattr(CONFIG.fusion.loss, "pose_body_weight", 1.0)
    pose_face_weight = getattr(CONFIG.fusion.loss, "pose_face_weight", 1.0)

    if is_main:
        logger.info("=" * 72)
        logger.info(f"RUN CONFIG          : {args.run_tag or '(untagged)'}")
        logger.info(f"  CHANGE 1 L_joint body_only : {joint_body_only}")
        logger.info(f"  CHANGE 1 L_pose weights    : body={pose_body_weight} "
                    f"hands={pose_hand_weight} face={pose_face_weight}")
        logger.info(f"  CHANGE 2 kintree_mask_k    : {kintree_mask_k}")
        logger.info(f"  joint confidence bias      : "
                    f"{getattr(CONFIG.fusion, 'use_joint_confidence', False)}")
        logger.info(f"  checkpoint_dir             : {CONFIG.fusion.checkpoint_dir}")
        logger.info("=" * 72)

    residual_head  = getattr(CONFIG.fusion.architecture, "residual_head", True)
    centered_input = getattr(CONFIG.fusion.architecture, "centered_input", True)
    head_lr_mult   = float(getattr(CONFIG.fusion.architecture, "head_lr_mult", 1.0))

    pose_module = PoseFusionModuleV3(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_temporal_len=max_T,
        dropout=dropout,
        temporal_window=temporal_window,
        kintree_mask_k=kintree_mask_k,
        residual_head=residual_head,
        centered_input=centered_input,
    )

    if is_main:
        logger.info(f"  V3 residual_head={residual_head}  centered_input={centered_input}  "
                    f"head_lr_mult={head_lr_mult}  max_epochs={max_epochs}")
    model = pose_module.to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"PoseFusionModuleV3 : {n_params:,} params")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    # The residual head starts at zero, so gradients reaching the transformer are
    # scaled by ||W_head||: it must escape zero quickly or the body spends the
    # first epochs learning at ~lr/step * ||W||. A larger LR on that ONE layer
    # removes the ramp without touching the rest of the schedule.
    head_params = list(pose_module.decoder[-1].parameters())
    head_ids    = {id(p) for p in head_params}
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    if residual_head and head_lr_mult != 1.0:
        optimizer = torch.optim.Adam(
            [{"params": base_params, "lr": lr},
             {"params": head_params, "lr": lr * head_lr_mult}],
            lr=lr,
        )
        if is_main:
            logger.info(f"  optimizer: head LR {lr * head_lr_mult:.2e} "
                        f"({head_lr_mult}x), body LR {lr:.2e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    temporal_weight        = CONFIG.fusion.loss.temporal_weight
    vposer_weight          = CONFIG.fusion.loss.vposer_weight
    joint_position_weight  = CONFIG.fusion.loss.joint_position_weight

    # ── Losses ────────────────────────────────────────────────────────────────
    # JointPositionLoss uses GT betas for FK to keep pose gradients clean.
    losses = {
        "pose":           (PoseMSELoss(body_weight=pose_body_weight,
                                        hand_weight=pose_hand_weight,
                                        face_weight=pose_face_weight),
                                                   pose_mse_weight),
        "joint_position": (JointPositionLoss(body_only=joint_body_only),
                                                   joint_position_weight),
        "vposer":         (VPoserLoss(),           vposer_weight),
        "temporal":       (TemporalSmoothnessLoss(), temporal_weight),
    }

    # ── Curriculum ────────────────────────────────────────────────────────────
    # vposer/temporal deliberately absent: the fusion module that produced our good
    # numbers trained without either. VPoser also dominated the objective here —
    # its raw value (~28) x 0.001 was ~75% of the total loss, vs 14% for
    # joint_position — so leaving it in makes any regression unattributable
    # between the data fixes and the changed objective.
    curriculum_schedule = {
        0: ["pose", "joint_position"],
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    from pytorch3d.transforms import rotation_6d_to_matrix
    from utilities.smplx_utilities import get_smplx_joints

    metrics = MetricCollection([RootRelativeMPJPE(), RootRelativeMPJRE()])
    METRIC_STRIDE = 8  # evaluate every Nth frame to keep FK memory bounded

    def metric_fn(preds, targets, mc):
        pose_aggr = preds[0]              # (B, T, P, J, 6)  — J=54 (no root)
        gt_joints = targets["gt_joints"]  # (B, T, P, 55, 3)

        B, T, P = pose_aggr.shape[:3]
        t_idx = torch.arange(0, T, METRIC_STRIDE, device=pose_aggr.device)

        gt_valid = targets.get("gt_valid")

        with torch.no_grad():
            pose_sub  = pose_aggr[:, t_idx].float()                        # (B, T', P, 54, 6)

            gt_root   = targets["pose"][:, t_idx, :, :1, :].float()       # (B, T', P, 1, 6)
            pose_full = torch.cat([gt_root, pose_sub], dim=3)              # (B, T', P, 55, 6)

            betas_exp   = targets["shape"][:, t_idx].float()              # (B, T', P, 10)
            pred_joints = get_smplx_joints(pose_full, betas_exp).cpu().numpy()[..., :55, :]  # (B,T',P,55,3)
            gt_joints_sub = gt_joints[:, t_idx].float().cpu().numpy()      # (B, T', P, 55, 3)

            rot_pred = rotation_6d_to_matrix(pose_full.cpu().reshape(-1, 6)).reshape(B, len(t_idx), P, 55, 3, 3).numpy()

            gt_pose_sub = targets["pose"][:, t_idx].float().cpu()          # (B, T', P, 55, 6)
            rot_gt = rotation_6d_to_matrix(gt_pose_sub.reshape(-1, 6)).reshape(B, len(t_idx), P, 55, 3, 3).numpy()

        for b in range(B):
            for t in range(len(t_idx)):
                if gt_valid is not None and not gt_valid[b, t_idx[t]].any():
                    continue
                for p in range(P):
                    if gt_valid is not None and not gt_valid[b, t_idx[t], p]:
                        continue
                    mc["RR-MPJPE"].update(pred_joints[b, t, p], gt_joints_sub[b, t, p])
                    mc["RR-MPJRE"].update(rot_pred[b, t, p], rot_gt[b, t, p])

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
            name="train_rich_v3",
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
        # Straight from the module: v3's flags change what the SAME weights mean
        # and cannot be inferred from the state dict, so the module reports them
        # itself rather than relying on this script to stay in sync.
        model_config=pose_module.arch_config(),
    )

    try:
        trainer.train()
    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
