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
from fusion.fusion_module import SSTNetwork, WindowedTemporalAttention
from fusion.loss import (
    BoneLengthconsistencyLoss,
    CameraMSELoss,
    CameraMSELossVGGT,
    CameraRotationMSELoss,
    CameraTranslationMSELoss,
    EpipolarLoss,
    JointPositionLoss,
    PoseMSELoss,
    ShapeMSELoss,
    ShapeRegularizationLoss,
    TemporalSmoothnessLoss,
    GTCameraTranslationMSELoss,
    TranslationMSELoss,
    TranslationSmoothnessLoss,
    TriangulationLoss,
    VPoserLoss,
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


RICH_SCENE_DIR = Path(
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs"
    "/rich11_segmentation_test/Pavallion_003_phonesiteat"
)

DISABLED_LOSSES: list[str] = [
    "temporal", "bone", "shape_reg", "translation_temporal", "vposer",
]

# Set to True to bypass the camera head and use GT cameras for root back-projection.
# This isolates whether root-orientation error comes from the camera head or the root head.
FORCE_GT_CAMERAS: bool = False

# Slice the sequence to the first MAX_T frames before training.
# Set to None to use the full sequence.
MAX_T: int = 30


def main():
    # Architecture
    embedding_dim           = CONFIG.fusion.architecture.embedding_dimension
    temporal_window         = CONFIG.fusion.architecture.temporal_window
    max_temporal_len        = CONFIG.fusion.architecture.max_temporal_len
    num_heads               = CONFIG.fusion.architecture.num_heads
    num_layers              = CONFIG.fusion.architecture.num_layers
    max_cameras             = CONFIG.fusion.architecture.max_cameras
    dropout                 = CONFIG.fusion.architecture.dropout
    # Loss weights
    pose_mse_weight              = CONFIG.fusion.loss.pose_mse_weight
    shape_mse_weight             = CONFIG.fusion.loss.shape_mse_weight
    epipolar_weight              = CONFIG.fusion.loss.epipolar_weight
    temporal_weight              = CONFIG.fusion.loss.temporal_weight
    bone_length_weight           = CONFIG.fusion.loss.bone_length_weight
    camera_rotation_mse_weight    = CONFIG.fusion.loss.camera_rotation_mse_weight
    camera_translation_mse_weight = CONFIG.fusion.loss.camera_translation_mse_weight
    gt_translation_mse_weight     = CONFIG.fusion.loss.gt_translation_mse_weight
    triangulation_weight          = CONFIG.fusion.loss.triangulation_weight
    translation_mse_weight        = CONFIG.fusion.loss.translation_mse_weight
    shape_reg_weight             = CONFIG.fusion.loss.shape_reg_weight
    translation_temporal_weight  = CONFIG.fusion.loss.translation_temporal_weight
    vposer_weight                = CONFIG.fusion.loss.vposer_weight
    joint_position_weight        = CONFIG.fusion.loss.joint_position_weight
    # Training params
    lr                      = CONFIG.fusion.training.lr
    camera_lr               = CONFIG.fusion.training.camera_lr
    max_epochs              = CONFIG.fusion.training.max_epochs
    batch_size              = CONFIG.fusion.training.batch_size
    grad_clip               = CONFIG.fusion.training.grad_clip
    scheduler_name          = getattr(CONFIG.fusion.training, "scheduler", None)
    patience                = getattr(CONFIG.fusion.training, "patience", None)

    # create the dataset
    dp = RICHFusionDatapoint(scene_dir=RICH_SCENE_DIR, rich_data_root = CONFIG.data.rich_data_root)
    img_size = dp.img_size
    ds = RICHFusionDataset([dp])

    if MAX_T is not None:
        class _TemporalSliceDataset(torch.utils.data.Dataset):
            def __init__(self, inner, t): self.inner = inner; self.t = t
            def __len__(self): return len(self.inner)
            def __getitem__(self, idx):
                inp, tgt = self.inner[idx]
                inp = {k: v[:self.t] if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > self.t else v for k, v in inp.items()}
                tgt = {k: v[:self.t] if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > self.t else v for k, v in tgt.items()}
                return inp, tgt
        ds = _TemporalSliceDataset(ds, MAX_T)
        logger.info(f"Slicing sequence to first {MAX_T} frames")

    inputs, targets = ds[0]
    # inputs:  dict ['pose', 'shape', 'camera', 'joint_mask', 'person_mask']
    # targets: dict ['pose', 'shape', 'camera', 'keypoints_3d']
    T = inputs["pose"].shape[0]
    print({k: tuple(v.shape) for k, v in inputs.items()})

    # ── Confidence distribution across cameras ────────────────────────────────
    jm = inputs["joint_mask"]   # (T, K, P, J)
    pm = inputs["person_mask"]  # (T, K, P) bool
    _, K, P, J = jm.shape
    logger.info("=== Confidence distribution per camera ===")
    for k in range(K):
        presence = pm[:, k, :].float().mean().item()          # fraction of frames where person detected
        conf_k   = jm[:, k, :, :]                             # (T, P, J)
        # only over frames where person is present
        mask_k   = pm[:, k, :].unsqueeze(-1).expand_as(conf_k)
        if mask_k.any():
            vals = conf_k[mask_k]
            logger.info(
                f"  cam {k:2d}  presence={presence*100:.1f}%  "
                f"joint_conf  mean={vals.mean():.3f}  "
                f"median={vals.median():.3f}  "
                f"min={vals.min():.3f}  "
                f"p10={vals.float().quantile(0.10):.3f}  "
                f"p25={vals.float().quantile(0.25):.3f}"
            )
        else:
            logger.info(f"  cam {k:2d}  presence=0% — no detections")
    logger.info("==========================================")
    # ─────────────────────────────────────────────────────────────────────────

    if FORCE_GT_CAMERAS:
        # Replace predicted cameras in inputs with GT cameras from targets so that
        # the model back-projects root with perfect extrinsics. The delta in
        # SSTOutputHeads is also zeroed (force_gt_cameras flag) so camera_out == GT.
        class _GTCameraDataset(torch.utils.data.Dataset):
            def __init__(self, inner): self.inner = inner
            def __len__(self): return len(self.inner)
            def __getitem__(self, idx):
                inp, tgt = self.inner[idx]
                inp = dict(inp)
                inp["camera"] = tgt["camera"]
                return inp, tgt
        ds = _GTCameraDataset(ds)

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
        max_temporal_len=max_temporal_len,
        dropout=dropout,
        temporal_window=temporal_window,
        max_cameras=max_cameras,
    )

    model.output_heads.force_gt_cameras = FORCE_GT_CAMERAS
    if FORCE_GT_CAMERAS:
        logger.info("force_gt_cameras=True: camera head delta zeroed, GT cameras used for back-projection")

    logger.info("\n" + model.summary())

    for module in model.modules():
        if isinstance(module, WindowedTemporalAttention):
            module.forward = torch.compile(module.forward, dynamic=True)

    camera_modules = {"camera_layers", "camera_pose_encoding", "output_heads.camera_rot_trans_head"}
    camera_params = set()
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in camera_modules):
            camera_params.add(param)
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if p not in camera_params], "lr": lr},
        {"params": list(camera_params), "lr": camera_lr},
    ])

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
    _all_losses = {
        "pose":                  (PoseMSELoss(body_only=False),        pose_mse_weight),
        "joint_position":        (JointPositionLoss(),                 joint_position_weight),
        "shape":                 (ShapeMSELoss(),                      shape_mse_weight),
        "epipolar":              (EpipolarLoss(img_size=img_size),     epipolar_weight),
        "temporal":              (TemporalSmoothnessLoss(),            temporal_weight),
        "bone":                  (BoneLengthconsistencyLoss(),         bone_length_weight),
        "camera_rotation_mse":   (CameraRotationMSELoss(),            camera_rotation_mse_weight),
        "camera_translation_mse":(CameraTranslationMSELoss(),         camera_translation_mse_weight),

        "triangulation":         (TriangulationLoss(),                 triangulation_weight),
        "translation_mse":       (TranslationMSELoss(),               translation_mse_weight),
        "gt_translation_mse":    (GTCameraTranslationMSELoss(),       gt_translation_mse_weight),
        "shape_reg":             (ShapeRegularizationLoss(),           shape_reg_weight),
        "translation_temporal":  (TranslationSmoothnessLoss(),         translation_temporal_weight),
        **({"vposer": (vposer_loss, vposer_weight)} if vposer_loss is not None else {}),
    }
    losses = {k: v for k, v in _all_losses.items() if k not in DISABLED_LOSSES}
    if DISABLED_LOSSES:
        logger.info(f"Disabled losses: {DISABLED_LOSSES}")

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

    METRIC_STRIDE = 8  # evaluate every Nth frame to keep SMPL-X memory bounded

    def metric_fn(preds, targets, mc):
        pose_aggr, shape_aggr, camera_pred, body_transl_world = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        K = camera_pred.shape[1]  # camera_pred is (B, K, 8) — static

        t_idx = torch.arange(0, T, METRIC_STRIDE, device=pose_aggr.device)

        with torch.no_grad():
            pose_sub  = pose_aggr[:, t_idx].float()
            shape_sub = shape_aggr.unsqueeze(1).expand(B, len(t_idx), P, 10).float()
            pred_joints_rel = get_smplx_joints(pose_sub, shape_sub).cpu().numpy()[..., :55, :]

            gt_pose_sub  = targets["pose"][:, t_idx].float()
            gt_shape_sub = targets["shape"][:, t_idx].float()
            gt_joints_rel = get_smplx_joints(gt_pose_sub, gt_shape_sub).cpu().numpy()[..., :55, :]

            pred_transl = body_transl_world[:, t_idx].float().cpu().numpy()   # (B, T_sub, P, 3)
            pred_joints = pred_joints_rel + pred_transl[:, :, :, None, :]

            if "trans" in targets:
                gt_transl = targets["trans"][:, t_idx].float().cpu().numpy()
                gt_joints = gt_joints_rel + gt_transl[:, :, :, None, :]
            else:
                gt_joints = gt_joints_rel

            pred_rotmats = rotation_6d_to_matrix(pose_aggr[:, t_idx].float()).cpu().numpy()
            gt_rotmats   = rotation_6d_to_matrix(targets["pose"][:, t_idx].float()).cpu().numpy()

            T_sub = len(t_idx)
            # Cameras are static — (B, K, 8), no T dimension.
            cam_rot_w2c    = quaternion_to_matrix(
                camera_pred[..., :4].float().reshape(B * K, 4)
            ).reshape(B, K, 3, 3).cpu().numpy()
            cam_transl_w2c = camera_pred[..., 4:7].float().cpu().numpy()  # (B, K, 3)

            gt_cam_rot_w2c    = quaternion_to_matrix(
                targets["camera"][..., :4].float().reshape(B * K, 4)
            ).reshape(B, K, 3, 3).cpu().numpy()
            gt_cam_transl_w2c = targets["camera"][..., 4:7].float().cpu().numpy()  # (B, K, 3)

        cam_centres    = -np.einsum("...ji,...j->...i", cam_rot_w2c,    cam_transl_w2c)   # (B, K, 3)
        gt_cam_centres = -np.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)

        gt_valid_np  = targets["gt_valid"][:, t_idx].cpu().numpy() if "gt_valid" in targets else None
        cam_valid_np = (
            targets["camera"][..., :4].float().norm(dim=-1) > 0.5
        ).cpu().numpy()  # (B, K)

        for b in range(B):
            valid = cam_valid_np[b]        # (K,)
            Cp = cam_centres[b][valid]
            Cg = gt_cam_centres[b][valid]
            Rp = cam_rot_w2c[b][valid]
            Rg = gt_cam_rot_w2c[b][valid]
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

    curriculum_schedule = {
        0:   ["camera_rotation_mse", "pose", "shape", "gt_translation_mse"],
        50:  ["camera_translation_mse"],
        200: ["epipolar"],
        400: ["triangulation"],
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
        dtype=None,
        use_amp=True,
        grad_clip=grad_clip,
        scheduler=scheduler,
        early_stopping_patience=patience,
        metrics=metrics,
        metric_fn=metric_fn,
        prediction_save_path=CONFIG.data.fusion_output_dir,
        curriculum_schedule=curriculum_schedule,
    )

    trainer.train()

    trainer.model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs, targets = trainer._unpack_batch(batch)
        preds = trainer._forward(inputs)
        preds = trainer._append_smplx_joints(preds)
        final_loss = sum(w * fn(preds, targets).item() for fn, w in losses.values())

    logger.info(f"\nFinal combined loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
