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
from fusion.fusion_module import SSTNetwork, WindowedTemporalAttention

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _build_model() -> SSTNetwork:
    arch = CONFIG.fusion.architecture
    model = SSTNetwork(
        embedding_dim    = arch.embedding_dimension,
        num_heads        = arch.num_heads,
        num_layers       = arch.num_layers,
        max_temporal_len = arch.max_temporal_len,
        dropout          = arch.dropout,
        temporal_window  = arch.temporal_window,
        max_cameras      = arch.max_cameras,
    )
    # Match the compile applied in train_rich.py so checkpoint weights load cleanly.
    for module in model.modules():
        if isinstance(module, WindowedTemporalAttention):
            module.forward = torch.compile(module.forward, dynamic=True)
    return model


def _load_checkpoint(model: SSTNetwork, checkpoint: Path) -> None:
    logger.info(f"Loading checkpoint: {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state["model"])
    epoch = state.get("epoch", "?")
    logger.info(f"  Loaded (epoch {epoch})")


def _run_forward(
    model:   SSTNetwork,
    dp:      RICHFusionDatapoint,
    device:  torch.device,
) -> dict[str, np.ndarray]:
    """Run a full-sequence forward pass and return a visualizer-ready dict."""
    ds     = RICHFusionDataset([dp])
    # Dataset returns the full sequence as a single item (no windowing for full-pass).
    # Use batch_size=1 and no shuffling.
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    model.to(device)

    all_pose, all_shape, all_camera, all_transl = [], [], [], []
    gt_pose_list, gt_shape_list, gt_camera_list, gt_transl_list = [], [], [], []
    has_gt = False

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch

            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(
                    pose               = inputs["pose"],
                    shape              = inputs["shape"],
                    camera             = inputs["camera"],
                    joint_mask         = inputs["joint_mask"],
                    person_mask        = inputs["person_mask"],
                    body_transl_cam_in = inputs["body_transl_cam_in"],
                )

            pose_aggr, shape_aggr, camera_pred, body_transl_world = preds[:4]

            # squeeze batch dim, move to CPU float32
            def _s(t):
                return t.squeeze(0).float().cpu().numpy().astype(np.float32)

            all_pose.append(_s(pose_aggr))          # (T, P, J, 6)
            all_shape.append(_s(shape_aggr))         # (P, 10)
            all_camera.append(_s(camera_pred))       # (T, K, 8)
            all_transl.append(_s(body_transl_world)) # (T, P, 3)

            # GT — targets["pose"] is (B, T, K, P, J, 6); cam-0 slice is world frame
            if isinstance(targets, dict):
                if "pose" in targets:
                    has_gt = True
                    gt_pose   = _s(targets["pose"])    # (T, K, P, J, 6)
                    gt_pose_list.append(gt_pose[:, 0])  # (T, P, J, 6) — cam-0 = world
                if "shape" in targets:
                    gt_shape  = _s(targets["shape"])   # (T, K, P, 10)
                    gt_shape_list.append(gt_shape[:, 0])  # (T, P, 10)
                if "camera" in targets:
                    gt_camera_list.append(_s(targets["camera"]))  # (T, K, 8)
                if "trans" in targets:
                    gt_transl_list.append(_s(targets["trans"]))   # (T, P, 3)

    # Concatenate along time axis (there is only one batch here, but keep it general)
    out: dict[str, np.ndarray] = {
        "pose":             np.concatenate(all_pose,   axis=0),  # (T, P, J, 6)
        "shape":            all_shape[0],                         # (P, 10) — time-invariant
        "camera":           np.concatenate(all_camera, axis=0),  # (T, K, 8)
        "body_transl_world": np.concatenate(all_transl, axis=0), # (T, P, 3)
    }

    if has_gt:
        if gt_pose_list:
            out["gt_body_pose"]         = np.concatenate(gt_pose_list,   axis=0)  # (T, P, J, 6)
        if gt_shape_list:
            out["gt_body_shape"]        = gt_shape_list[0][0]  # (P, 10) — first frame, cam-0
        if gt_camera_list:
            out["gt_camera"]            = np.concatenate(gt_camera_list, axis=0)  # (T, K, 8)
        if gt_transl_list:
            out["gt_body_transl_world"] = np.concatenate(gt_transl_list, axis=0)  # (T, P, 3)

    return out


def main(
    scene:        str,
    scenes_root:  Path = Path(CONFIG.data.output_directory),
    checkpoint:   Path | None = None,
    out_dir:      Path = Path(CONFIG.data.fusion_output_dir),
    port:         int  = 9090,
    no_visualize: bool = False,
    frame_start:  int  = 0,
    show_gt:      bool = True,
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
    )
    T = dp._frame_end - dp._frame_start
    logger.info(f"  {T} frames, {dp.num_cameras} cameras, {dp.max_persons} persons")

    # ── build and load model ──────────────────────────────────────────────────
    model = _build_model()
    _load_checkpoint(model, checkpoint)

    # ── forward pass ──────────────────────────────────────────────────────────
    dev = torch.device(device)
    logger.info(f"Running forward pass on {dev} …")
    arrays = _run_forward(model, dp, dev)

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
