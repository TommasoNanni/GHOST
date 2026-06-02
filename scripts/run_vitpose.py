"""Run easy_ViTPose on all processed scenes and save per-person COCO keypoints.

Uses bounding boxes already stored in body_data/person_<pid>.npz to crop each
person directly — no YOLO needed. For each scene → camera → person:

    <cam_dir>/vitpose_kps_person_<pid>.npz
        keypoints  (T_local, 17, 3)  float32 — [x, y, conf] in original image pixels
                                               zeros for frames where crop fails

Usage:
    pixi run python scripts/run_vitpose.py \\
        --vitpose-weights checkpoints/vitpose/vitpose_b_coco.pth \\
        --model-name b --device cuda:0 \\
        [--scenes SCENE1 SCENE2 ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configuration import CONFIG
from data.video_dataset import RichDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_vitpose_model(weights: str, model_name: str, device: str):
    """Load ViTPose model directly — no YOLO needed."""
    from easy_ViTPose.easy_ViTPose.vit_models.model import ViTPose
    from easy_ViTPose.easy_ViTPose.vit_utils.util import dyn_model_import
    from easy_ViTPose.easy_ViTPose.configs.ViTPose_common import data_cfg

    model_cfg  = dyn_model_import("coco", model_name)
    model      = ViTPose(model_cfg)
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(torch.device(device)).eval()
    target_size = data_cfg["image_size"]   # [W, H] e.g. [192, 176]
    return model, target_size


def _preprocess(img: np.ndarray, target_size: list[int]):
    """Resize + normalise crop for ViTPose. Returns (1,3,H,W) float32."""
    org_h, org_w = img.shape[:2]
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    resized = (resized - _MEAN) / _STD
    return resized.transpose(2, 0, 1)[None].astype(np.float32), org_w, org_h


def _run_vitpose_on_bbox(
    model,
    target_size: list[int],
    device: str,
    img: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray | None:
    """Run ViTPose on a single person crop. Returns (17, 3) [x, y, conf] or None."""
    from easy_ViTPose.easy_ViTPose.vit_utils.inference import pad_image
    from easy_ViTPose.easy_ViTPose.inference import VitInference

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    padded, (left_pad, top_pad) = pad_image(crop, 3 / 4)

    inp, org_w, org_h = _preprocess(padded, target_size)

    with torch.no_grad():
        heatmaps = model(torch.from_numpy(inp).to(device))

    # postprocess(heatmaps, org_w, org_h) — note: width before height
    kps = VitInference.postprocess(heatmaps.cpu().numpy(), org_w, org_h)[0]  # (17,3) [y,x,conf]

    # Transform from padded-crop space → original image space, then swap [y,x] → [x,y]
    kps[:, 0] += y1 - top_pad
    kps[:, 1] += x1 - left_pad
    kps = kps[:, [1, 0, 2]]   # [y, x, conf] → [x, y, conf]
    return kps.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes",           nargs="+", default=None)
    parser.add_argument("--vitpose-weights",  required=True, type=str)
    parser.add_argument("--model-name",       default="b", choices=["s", "b", "l", "h"])
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--data-root",        default=None,
                        help="Override config rich_data_root")
    parser.add_argument("--frames-base-dir",  default=None,
                        help="Override config rich_frames_base_dir")
    args = parser.parse_args()

    model, target_size = _load_vitpose_model(
        args.vitpose_weights, args.model_name, args.device)
    logger.info(f"ViTPose-{args.model_name.upper()} loaded, target_size={target_size}.")

    output_dir = Path(CONFIG.data.output_directory)
    data_root  = args.data_root or CONFIG.data.rich_data_root
    # Dataset is already resized inside the squashfs — no resize needed, no frames_base_dir.
    ds = RichDataset(
        data_root=data_root,
        slice=getattr(CONFIG.data, "slice", None),
        max_side=None,
        frames_base_dir=None,
    )

    scenes = [s for s in ds.scenes
              if args.scenes is None or s.scene_id in args.scenes]
    logger.info(f"Processing {len(scenes)} scene(s).")

    for scene in scenes:
        scene_out = output_dir / scene.scene_id
        cam_dirs  = sorted(d for d in scene_out.iterdir()
                           if d.is_dir() and (d / "body_data").is_dir())
        if not cam_dirs:
            logger.warning(f"[{scene.scene_id}] No cam_dirs — skipping.")
            continue

        logger.info(f"\n=== {scene.scene_id} ===")

        videos_sorted = sorted(scene.videos, key=lambda v: v.video_id)
        for cam_dir, video in zip(cam_dirs, videos_sorted):

            pid_data: dict[int, dict] = {}
            for bf in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(bf.stem.split("_")[1])
                d   = np.load(bf, allow_pickle=False)
                if "frame_indices" not in d.files or "bbox" not in d.files:
                    continue
                pid_data[pid] = {
                    "frame_indices": d["frame_indices"].astype(int),
                    "bbox":          d["bbox"].astype(np.float32),
                }

            if not pid_data:
                continue

            if all((cam_dir / f"vitpose_kps_person_{pid}.npz").exists()
                   for pid in pid_data):
                logger.info(f"  {cam_dir.name}: already done — skipping.")
                continue

            fdir = video.frames_home
            if fdir is None or not fdir.is_dir():
                logger.warning(f"  {cam_dir.name}: no frames dir — skipping.")
                continue
            frame_files = sorted(p for p in fdir.iterdir()
                                  if p.suffix.lower() in _IMAGE_EXTS)
            frame_map: dict[int, Path] = {i: p for i, p in enumerate(frame_files)}
            if not frame_map:
                raise RuntimeError(f"{cam_dir.name}: frames_home={fdir} has no images")

            kp_buffers: dict[int, np.ndarray] = {
                pid: np.zeros((len(pd["frame_indices"]), 17, 3), dtype=np.float32)
                for pid, pd in pid_data.items()
            }

            n_total = sum(len(pd["frame_indices"]) for pd in pid_data.values())
            logger.info(f"  {cam_dir.name}: {len(pid_data)} PIDs, {n_total} crops")

            # Cache loaded images to avoid re-reading the same frame for multiple PIDs
            img_cache: dict[int, np.ndarray | None] = {}

            for pid, pd in pid_data.items():
                for local_t, global_t in enumerate(pd["frame_indices"]):
                    gt = int(global_t)
                    if gt not in img_cache:
                        img_path = frame_map.get(gt)
                        if img_path is not None:
                            raw = cv2.imread(str(img_path))
                            img_cache[gt] = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw is not None else None
                        else:
                            img_cache[gt] = None

                    img = img_cache[gt]
                    if img is None:
                        continue

                    kps = _run_vitpose_on_bbox(
                        model, target_size, args.device, img, pd["bbox"][local_t])

                    if kps is not None:
                        kp_buffers[pid][local_t] = kps

                # Clear cache periodically to limit memory use
                if len(img_cache) > 500:
                    img_cache.clear()

            for pid, buf in kp_buffers.items():
                out_path = cam_dir / f"vitpose_kps_person_{pid}.npz"
                np.savez_compressed(out_path, keypoints=buf)
            logger.info(f"  {cam_dir.name}: saved.")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
