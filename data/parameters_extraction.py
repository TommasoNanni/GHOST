"""SAM3D Body estimation for tracked persons in multi-view scenes.

Reads pre-existing segmentation output (frames, masks, JSON metadata) and
runs per-frame 3D body model estimation via SAM3D Body, with optional
temporal smoothing.

Expected input layout (produced by :class:`PersonSegmenter`)::

    output_dir/<scene_id>/<video_id>/
        frames/          extracted JPEGs
        mask_data/       .npy uint16 masks
        json_data/       .json per-frame instance metadata

Output (added to existing directories)::

    output_dir/<scene_id>/<video_id>/
        body_data/
            person_<id>.npz                 <- per-person body params across frames
            body_params_summary.json        <- metadata
"""

from __future__ import annotations

import gc
import json
import math
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
import smplx

from data.video_dataset import Scene, Video
from mhr.mhr import MHR
from conversion import Conversion


# ======================================================================
# One-Euro Filter for temporal smoothing
# ======================================================================

class OneEuroFilter:
    """Attempt jitter reduction on noisy signals (standard algorithm).

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency.  Lower = more smoothing.
    beta : float
        Speed coefficient.  Higher = less lag when signal changes fast.
    d_cutoff : float
        Cutoff frequency for the derivative filter.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev.copy()

        # Derivative estimation.
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class BodyParameterEstimator:
    """Estimate 3D body parameters for tracked persons.

    Reads pre-existing segmentation output and runs SAM3D Body on each
    detected person crop, with optional temporal smoothing via one-euro
    filtering.  Does **not** perform segmentation itself.

    Parameters
    ----------
    sam3d_hf_repo : str
        HuggingFace repo ID for SAM3D Body model.
    sam3d_step : int
        Run SAM3D every *sam3d_step* frames (1 = every frame).
    smooth : bool
        Whether to apply temporal smoothing to body parameters.
    smooth_params : dict | None
        One-euro filter parameters: ``{min_cutoff, beta, d_cutoff}``.
    bbox_padding : float
        Fractional padding around bounding boxes for person crops.
    """

    # Keys from SAM3D output that we store as arrays.
    _PARAM_KEYS = (
        "pred_keypoints_3d",
        "pred_keypoints_2d",
        "pred_vertices",
        "pred_cam_t",
        "body_pose_params",
        "hand_pose_params",
        "shape_params",
        "global_rot",
        "scale_params",
        "focal_length",
    )

    # Keys to smooth (shape is averaged, not smoothed).
    _SMOOTH_KEYS = (
        "pred_keypoints_3d",
        "pred_keypoints_2d",
        "pred_vertices",
        "pred_cam_t",
        "body_pose_params",
        "hand_pose_params",
        "global_rot",
        "scale_params",
    )

    def __init__(
        self,
        sam3d_hf_repo: str = "facebook/sam-3d-body-dinov3",
        sam3d_step: int = 1,
        smooth: bool = False,
        smooth_params: dict | None = None,
        bbox_padding: float = 0.2,
        smplx_model_path: str | None = None,
        mhr_model_path: str | None = None,
    ):
        self.sam3d_hf_repo = sam3d_hf_repo
        self.sam3d_step = sam3d_step
        self.smooth = smooth
        self.smooth_params = smooth_params or {
            "min_cutoff": 1.0,
            "beta": 0.007,
            "d_cutoff": 1.0,
        }
        self.bbox_padding = bbox_padding
        self.smplx_model_path = smplx_model_path
        self.mhr_model_path = mhr_model_path

        self._estimator = None
        self._converter = None

    def _init_sam3d(self) -> None:
        """Lazy-load the SAM3D Body estimator."""
        if self._estimator is not None:
            return
        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(
                "sam-3d-body package not installed. "
                "Please install it and ensure notebook.utils is available."
            ) from e
        self._estimator = setup_sam_3d_body(hf_repo_id=self.sam3d_hf_repo)
        print(f"SAM3D Body loaded from {self.sam3d_hf_repo}")

    def estimate_scene(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
    ) -> None:
        """Run SAM3D Body estimation on pre-segmented scene data.

        Parameters
        ----------
        scene : Scene
            The scene whose videos to process.
        video_dirs : dict[str, Path]
            Mapping of ``video_id`` -> output directory (as returned by
            :meth:`PersonSegmenter.segment_scene`).  Each directory must
            contain ``frames/``, ``json_data/``, and ``mask_data/`` subdirs.
        """
        self._init_sam3d()
        for video in tqdm(scene.videos, desc="SAM3D Body estimation"):
            video_dir = video_dirs[video.video_id]
            self._estimate_bodies(video, video_dir)
            gc.collect()
            torch.cuda.empty_cache()

    def _estimate_bodies(self, video: Video, video_dir: Path) -> None:
        """Run SAM3D Body on person crops for a single video."""
        json_dir = video_dir / "json_data"
        frame_dir = video_dir / "frames"
        body_dir = video_dir / "body_data"
        body_dir.mkdir(exist_ok=True)

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            print(f"  {video.video_id}: no JSON data, skipping body estimation")
            return

        # Collect per-person tracks: {person_id: {frame_idx: body_params}}.
        tracks: dict[int, dict[int, dict]] = {}

        for json_path in tqdm(
            json_files, desc=f"  SAM3D {video.video_id}", leave=False
        ):
            # Parse frame index from filename like "mask_000042.json".
            frame_idx_str = json_path.stem.replace("mask_", "")
            frame_idx = int(frame_idx_str)

            # Step filtering.
            if self.sam3d_step > 1 and frame_idx % self.sam3d_step != 0:
                continue

            with open(json_path) as f:
                meta = json.load(f)

            labels = meta.get("labels", {})
            if not labels:
                logging.warning(f"No people labels available for frame {frame_idx} in video {video.video_id}")
                continue

            # Load the frame image.
            frame_path = frame_dir / f"{frame_idx_str}.jpg"
            if not frame_path.exists():
                logging.warning(f"Frame image not found for frame {frame_idx} in video {video.video_id} at path {frame_path}")
                continue
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                logging.warning(f"Unable to read image {frame_idx} for video {video.video_id}")
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame_rgb.shape[:2]

            for str_id, info in labels.items():
                person_id = int(str_id)
                x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

                # Skip tiny boxes.
                bw, bh = x2 - x1, y2 - y1
                if bw < 10 or bh < 10:
                    logging.warning(f"Person {str_id} in video {video.video_id} had too small bounding box, skipping")
                    continue

                # Pad and clamp the crop.
                pad_w = int(bw * self.bbox_padding)
                pad_h = int(bh * self.bbox_padding)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(img_w, x2 + pad_w)
                cy2 = min(img_h, y2 + pad_h)

                crop = frame_rgb[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # Run SAM3D on the crop.
                try:
                    outputs = self._estimator.process_one_image(crop)
                except Exception as e:
                    logging.warning(f"    SAM3D failed on frame {frame_idx} for person {person_id}: {e}")
                    continue

                if not outputs:
                    logging.warning(f"No outputs found for frame {frame_idx} for video {video.video_id}")
                    continue

                # Match to target person if multiple detections.
                body = self._match_detection(
                    outputs, target_bbox=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
                )
                if body is None:
                    continue

                # Store the parameters.
                params = {"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)}
                for key in self._PARAM_KEYS:
                    if key in body:
                        val = body[key]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        params[key] = np.asarray(val, dtype=np.float32)

                tracks.setdefault(person_id, {})[frame_idx] = params

        if not tracks:
            logging.warning(f"  {video.video_id}: no body detections")
            return

        # Smooth and save.
        if self.smooth:
            tracks = self._smooth_tracks(tracks)

        self._save_body_data(tracks, body_dir, video.video_id)

    @staticmethod
    def _match_detection(
        outputs: list[dict],
        target_bbox: tuple[int, int, int, int],
    ) -> dict | None:
        """Pick the SAM3D detection best matching target_bbox (IoU)."""
        if len(outputs) == 1:
            return outputs[0]

        tx1, ty1, tx2, ty2 = target_bbox
        t_area = max((tx2 - tx1) * (ty2 - ty1), 1)
        best, best_iou = None, -1.0

        for det in outputs:
            # Try common bbox keys from SAM3D output.
            bbox = None
            for bkey in ("bbox", "pred_bbox", "bboxes"):
                if bkey in det:
                    bbox = det[bkey]
                    break
            if bbox is None:
                # If no bbox, fall back to first detection.
                if best is None:
                    best = det
                continue

            if isinstance(bbox, torch.Tensor):
                bbox = bbox.detach().cpu().numpy()
            bbox = np.asarray(bbox).flatten()[:4]
            dx1, dy1, dx2, dy2 = bbox

            inter_x1 = max(tx1, dx1)
            inter_y1 = max(ty1, dy1)
            inter_x2 = min(tx2, dx2)
            inter_y2 = min(ty2, dy2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            d_area = max((dx2 - dx1) * (dy2 - dy1), 1)
            iou = inter / (t_area + d_area - inter)

            if iou > best_iou:
                best_iou = iou
                best = det

        return best

    def _smooth_tracks(
        self, tracks: dict[int, dict[int, dict]]
    ) -> dict[int, dict[int, dict]]:
        """Apply one-euro filtering to body parameters per person."""
        for person_id, frames in tracks.items():
            if len(frames) < 3:
                continue

            sorted_idxs = sorted(frames.keys())

            # Average shape params (should be constant per person).
            shape_vals = [
                frames[fi]["shape_params"]
                for fi in sorted_idxs
                if "shape_params" in frames[fi]
            ]
            if shape_vals:
                avg_shape = np.mean(shape_vals, axis=0)
                for fi in sorted_idxs:
                    if "shape_params" in frames[fi]:
                        frames[fi]["shape_params"] = avg_shape.copy()

            # One-euro filter for other parameters.
            for key in self._SMOOTH_KEYS:
                if key == "shape_params":
                    continue

                # Check that all frames have this key.
                vals = [frames[fi].get(key) for fi in sorted_idxs]
                if vals[0] is None:
                    continue

                orig_shape = vals[0].shape
                flat_vals = [v.flatten() for v in vals if v is not None]
                if not flat_vals:
                    continue

                filt = OneEuroFilter(**self.smooth_params)
                for i, fi in enumerate(sorted_idxs):
                    if key not in frames[fi]:
                        continue
                    smoothed = filt(float(fi), flat_vals[i])
                    frames[fi][key] = smoothed.reshape(orig_shape)

        return tracks

    def _save_body_data(
        self,
        tracks: dict[int, dict[int, dict]],
        body_dir: Path,
        video_id: str,
    ) -> None:
        """Save per-person .npz files and a summary JSON."""
        summary = {"video_id": video_id, "persons": {}}

        for person_id, frames in tracks.items():
            sorted_idxs = sorted(frames.keys())
            n = len(sorted_idxs)
            if n == 0:
                continue

            arrays: dict[str, np.ndarray] = {
                "frame_indices": np.array(sorted_idxs, dtype=np.int32),
            }

            # Collect all param keys present in any frame.
            all_keys = set()
            for fi in sorted_idxs:
                all_keys.update(frames[fi].keys())

            for key in list(self._PARAM_KEYS) + ["bbox"]:
                if key not in all_keys:
                    continue
                vals = []
                for fi in sorted_idxs:
                    v = frames[fi].get(key)
                    if v is not None:
                        vals.append(v)
                    else:
                        # Use zeros with same shape as first available.
                        ref = next(
                            (frames[fj][key] for fj in sorted_idxs if key in frames[fj]),
                            None,
                        )
                        vals.append(np.zeros_like(ref) if ref is not None else None)
                # Filter out any remaining None.
                if any(v is None for v in vals):
                    continue
                arrays[key] = np.stack(vals, axis=0)

            npz_path = body_dir / f"person_{person_id}.npz"
            np.savez(str(npz_path), **arrays)

            # Summary entry.
            param_shapes = {
                k: list(v.shape) for k, v in arrays.items()
            }
            summary["persons"][str(person_id)] = {
                "num_frames": n,
                "frame_range": [int(sorted_idxs[0]), int(sorted_idxs[-1])],
                "param_shapes": param_shapes,
            }

        summary_path = body_dir / "body_params_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        total_frames = sum(
            info["num_frames"] for info in summary["persons"].values()
        )
        print(
            f"  {video_id}: saved body data for {len(summary['persons'])} "
            f"persons ({total_frames} total frame estimates) -> {body_dir}"
        )

    def convert_hmr_to_smplx(self, sam3d_outputs):
        """Convert HMR parameters to SMPLX parameters"""
        mhr_model = MHR.from_files(
            model_path = self.mhr_model_path,
            lod = 1, device = self.device,
        )
        smplx_model = smplx.create(
            model_path = self.smplx_model_path,
            model_type = 'smplx',
            gender = 'neutral',
            use_pca = False,
            batch_size = 1,
        ).to(self_device)

        # Launch the converter
        converter = Conversion(
            mhr_model = mhr_model,
            smpl_model = smplx_model,
            method = "pytorch",
        )

        results = converter.convert_sam3d_output_to_smpl(
            sam3d_outputs = sam3d_outputs,
            return_smpl_meshes=True,
            return_smpl_parameters=True,
            return_smpl_vertices=True,
            return_fitting_errors=True,
        )

        raise ValueError("Find a way to save these based on what it outputs")



