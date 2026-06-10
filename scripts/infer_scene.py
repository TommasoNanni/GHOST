"""Run fusion inference on a single RICH scene and (optionally) launch the visualizer.

Usage
-----
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar --no_visualize
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar \\
        --scenes_root test_outputs/rich10_segmentation_test \\
        --checkpoint checkpoints/fusion_module/best.pt

The script:
  1. Loads the RICHFusionDatapoint for the requested scene.
  2. Loads the SSTNetwork from a checkpoint.
  3. Runs a forward pass over the full sequence.
  4. Saves predictions to fusion_outputs/<scene_name>.npz  (visualizer-ready format).
  5. Launches visualize/visualize_fusion.py unless --no_visualize is set.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tyro

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.fusion_module_v2 import BetasAggregator, FusionWithBetas, PoseFusionModule
from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

# Import PnP placement helpers from inference.py (same scripts/ dir)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from inference import _run_placer, _build_vggt_cameras

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _R_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to 6D (first two rows)."""
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def _build_model() -> FusionWithBetas:
    arch = CONFIG.fusion.architecture
    pose_module = PoseFusionModule(
        embedding_dim    = arch.embedding_dimension,
        num_heads        = arch.num_heads,
        num_layers       = arch.num_layers,
        max_temporal_len = arch.max_temporal_len,
        dropout          = arch.dropout,
        temporal_window  = arch.temporal_window,
    )
    betas_aggregator = BetasAggregator(n_betas=10, embedding_dim=64, num_inducing=4, num_heads=4, dropout=0.3, input_noise_std=0.5)
    return FusionWithBetas(pose_module, betas_aggregator)


def _load_checkpoint(model: FusionWithBetas, checkpoint: Path) -> None:
    logger.info(f"Loading checkpoint: {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state["model"])
    epoch = state.get("epoch", "?")
    logger.info(f"  Loaded (epoch {epoch})")


def _run_forward(
    model:   FusionWithBetas,
    dp:      RICHFusionDatapoint,
    device:  torch.device,
) -> dict[str, np.ndarray]:
    """Run a full-sequence forward pass and return model predictions + GT arrays.

    Returns pred_pose_54 (root excluded) so that main() can prepend the
    PnP-estimated global_orient to build the full 55-joint pose.
    """
    ds     = RICHFusionDataset([dp])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    model.to(device)

    def _s(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).float().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch

            inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pose_aggr, betas_out = model(
                    pose        = inp["pose"],          # (B, T, K, P, 55, 6)
                    person_mask = inp["person_mask"],   # (B, T, K, P)
                    joint_mask  = inp["joint_mask"],    # (B, T, K, P, 55)
                    shape       = inp["shape"],         # (B, T, K, P, 10)
                )

            # pose_aggr: (B, T, P, 54, 6) — root excluded by model
            # betas_out: (B, P, 10)
            pred_pose_54 = _s(pose_aggr)   # (T, P, 54, 6)
            pred_shape   = _s(betas_out)   # (P, 10)

            # GT arrays (cam-0 = world frame)
            gt_pose  = _s(targets["pose"])   # (T, P, 55, 6)
            gt_shape = _s(targets["shape"])  # (T, P, 10)
            gt_trans = _s(targets["trans"])  # (T, P, 3)
            gt_valid = _s(targets["gt_valid"])  # (T, P)

            # GT camera is static (K, 8); tile to (T, K, 8) for the visualizer.
            gt_cam_static = _s(targets["camera"])               # (K, 8)
            T_local = pred_pose_54.shape[0]
            gt_cam_tiled = np.broadcast_to(
                gt_cam_static[None], (T_local,) + gt_cam_static.shape
            ).copy()  # (T, K, 8)

            break  # single scene, single batch

    return {
        "pred_pose_54":      pred_pose_54,    # (T, P, 54, 6) — root excluded
        "pred_shape":        pred_shape,      # (P, 10)
        "gt_body_pose":      gt_pose,         # (T, P, 55, 6)
        "gt_body_shape":     gt_shape[0],     # (P, 10) — first frame
        "gt_camera":         gt_cam_tiled,    # (T, K, 8)
        "gt_body_transl_world": gt_trans,     # (T, P, 3)
        "gt_valid":          gt_valid,        # (T, P)
    }


def main(
    scene:        str,
    scenes_root:  Path = Path(CONFIG.data.output_directory),
    checkpoint:   Path | None = None,
    out_dir:      Path = Path(CONFIG.data.fusion_output_dir),
    port:         int  = 9090,
    no_visualize: bool = False,
    frame_start:  int  = 0,
    show_gt:      bool = True,
    body_split:   str  = "train_body",
    device:       str  = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Parameters
    ----------
    scene        : scene directory name, e.g. "BBQ_001_guitar"
    scenes_root  : root that contains the scene directory (default: config output_directory)
    checkpoint   : path to a .pt checkpoint; defaults to best.pt in config checkpoint_dir
    out_dir      : directory where the predictions .npz is written
    port         : viser WebSocket port
    no_visualize : save predictions only — do not launch the viewer
    frame_start  : first RICH frame index (for image loading in the viewer)
    show_gt      : show GT body in the viewer (if available)
    device       : "cuda" or "cpu"
    """
    scene_dir = Path(scenes_root) / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # ── resolve checkpoint ────────────────────────────────────────────────────
    if checkpoint is None:
        checkpoint = Path(CONFIG.fusion.checkpoint_dir) / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Train first with:  pixi run python scripts/train_rich.py"
        )

    # ── load scene ────────────────────────────────────────────────────────────
    logger.info(f"Loading scene: {scene_dir}")
    dp = RICHFusionDatapoint(
        scene_dir      = scene_dir,
        rich_data_root = CONFIG.data.rich_data_root,
        rich_gt_dir    = CONFIG.data.rich_gt_dir,
        body_split     = body_split,
    )
    T = dp._frame_end - dp._frame_start
    logger.info(f"  {T} frames, {dp.num_cameras} cameras, {dp.max_persons} persons")

    # ── build and load model ──────────────────────────────────────────────────
    model = _build_model()
    _load_checkpoint(model, checkpoint)

    # ── forward pass ──────────────────────────────────────────────────────────
    dev = torch.device(device)
    logger.info(f"Running forward pass on {dev} …")
    raw_arrays = _run_forward(model, dp, dev)

    pred_pose_54 = raw_arrays["pred_pose_54"]   # (T, P, 54, 6)
    pred_shape   = raw_arrays["pred_shape"]      # (P, 10)
    T_scene = pred_pose_54.shape[0]
    P       = pred_pose_54.shape[1]

    # ── BodyPlacer: PnP translation + orientation + VGGT cameras ─────────────
    cam_dirs = sorted(
        d for d in scene_dir.iterdir()
        if d.is_dir() and (d / "body_data").is_dir()
    )
    # pid ordering must match dataset slot ordering (same sorted ghost pids)
    all_pids_ordered = sorted(
        set(pid for pids_k in dp._pid_order for pid in pids_k)
    )
    # raw dict for _run_placer (it needs frame_indices from body data)
    raw_body: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_persons: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid_num = int(npz_path.stem.split("_")[1])
            if pid_num not in all_pids_ordered:
                continue
            d = np.load(npz_path, allow_pickle=False)
            cam_persons[pid_num] = {k: d[k] for k in d.files}
        raw_body.append(cam_persons)

    _gender_json = Path(__file__).resolve().parent.parent / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(CONFIG.data.smplx_model_path).parent, _gender_json)
        if _gender_json.exists() else Path(CONFIG.data.smplx_model_path)
    )

    logger.info("Running BodyPlacer (Procrustes DLT) …")
    root_translation, orient_R, vggt_cameras = _run_placer(
        scene_dir      = scene_dir,
        cam_dirs       = cam_dirs,
        raw            = raw_body,
        all_pids       = all_pids_ordered,
        frame_start    = dp._frame_start,
        T              = T_scene,
        smplx_model_path = _smplx_arg,
        fused_betas    = pred_shape,     # (P, 10) fused betas for FK
        fused_pose     = pred_pose_54,   # (T, P, 54, 6) fused pose for Procrustes FK
    )
    # orient_R: (T, P, 3, 3) — NaN where Procrustes DLT failed

    # ── Build full 55-joint pose: Procrustes root + fusion non-root ──────────
    # Convert Procrustes rotation matrices to 6D; fall back to GT root where NaN.
    gt_root_6d = raw_arrays["gt_body_pose"][:, :, 0, :]   # (T, P, 6)
    proc_root_6d = np.where(
        np.isnan(orient_R[:, :, 0, 0])[..., np.newaxis],   # (T, P, 1) mask
        gt_root_6d,
        _R_to_6d(orient_R),                                 # (T, P, 6)
    )  # (T, P, 6)
    pred_pose_55 = np.concatenate(
        [proc_root_6d[:, :, np.newaxis, :], pred_pose_54], axis=2
    )  # (T, P, 55, 6)

    # ── Assemble visualizer-ready dict ────────────────────────────────────────
    arrays: dict[str, np.ndarray] = {
        # Predicted
        "pose":              pred_pose_55,             # (T, P, 55, 6)
        "shape":             pred_shape,               # (P, 10)
        "camera":            vggt_cameras,             # (T, K, 8) — VGGT cameras
        "body_transl_world": root_translation,         # (T, P, 3) — PnP translation
        # GT reference (for visualizer comparison)
        "gt_body_pose":          raw_arrays["gt_body_pose"],
        "gt_body_shape":         raw_arrays["gt_body_shape"],
        "gt_camera":             raw_arrays["gt_camera"],
        "gt_body_transl_world":  raw_arrays["gt_body_transl_world"],
        "gt_valid":              raw_arrays["gt_valid"],
    }

    # ── save predictions ──────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scene}.npz"
    np.savez_compressed(str(out_file), **arrays)
    logger.info(f"Predictions saved → {out_file}  (keys: {list(arrays)})")

    # ── launch visualizer ─────────────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent
    show_gt_flag = "--show-gt" if show_gt else "--no-show-gt"
    vis_cmd = (
        f"pixi run python visualize/visualize_fusion.py"
        f" --predictions {out_file}"
        f" --scene-dir {scene_dir}"
        f" --smplx-model-dir {CONFIG.data.smplx_model_path}"
        f" --frame-start {frame_start}"
        f" {show_gt_flag}"
        f" --port {port}"
    )

    if no_visualize:
        logger.info("\nTo visualize, run:")
        logger.info(f"  {vis_cmd}")
    else:
        logger.info("Launching visualizer …")
        subprocess.run(vis_cmd, shell=True, cwd=str(repo_root))


if __name__ == "__main__":
    tyro.cli(main)
