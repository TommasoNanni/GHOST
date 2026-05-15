"""
VGGT camera + depth preprocessing for multi-camera synchronized scenes.

Runs VGGT frame-by-frame on all available cameras and saves:
  - vggt_cameras.npz  : extrinsics, intrinsics, coordinate mapping info
  - vggt_depth.npz    : depth maps and confidence at VGGT native resolution (518×518)

Coordinate conventions
----------------------
Extrinsics are in OpenCV convention: x_cam = R @ x_world + t  (camera-from-world).
The world origin is defined as camera 0 (cam0) of each frame.
  R_i' = R_i @ R_0^T
  t_i' = t_i − R_i @ R_0^T @ t_0
Depth maps live in each camera's own frame (z-axis = depth), so re-rooting the
world frame does not affect depth values.

Querying depth at a body keypoint
----------------------------------
VGGT runs on square-padded 518×518 images. The original image is first padded to
square then resized. `original_coords[t, k] = [x1, y1, x2, y2]` records where the
top-left (x1,y1) and bottom-right (x2,y2) corners of the original image land in
518-space. To back-project a 2D keypoint (u_orig, v_orig) from original resolution:

    scale_x = (x2 - x1) / W_orig   # = scale_y since square padding
    u_518   = x1 + u_orig * scale_x
    v_518   = y1 + v_orig * scale_x

Then look up depth[t, k] at (v_518, u_518) with bilinear interpolation and
back-project with intrinsics[t, k] (calibrated for the 518×518 padded image):
    X = (u_518 - cx) * depth / fx
    Y = (v_518 - cy) * depth / fy
    Z = depth

Depth is stored as uint16 millimetres. To recover metres:
    depth_m = depth_mm.astype(np.float32) / 1000

Multi-GPU
---------
When multiple devices are given to process_scene(), T frames are distributed
round-robin across devices. Each device worker loads a separate VGGT model instance,
processes its chunk, and writes a temporary .npz. The main process merges them.

Usage example
-------------
    from preprocessing.run_vggt import VGGTPreprocessor

    preprocessor = VGGTPreprocessor(weights="facebook/VGGT-1B")

    # frame_paths[t][k] = Path to image of camera k at frame t, or None if absent.
    preprocessor.process_scene(
        frame_paths  = frame_paths,
        camera_names = camera_names,
        output_dir   = Path("outputs/scene_01"),
        devices      = ["cuda:0", "cuda:1"],
    )
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms as TF

# Add VGGT submodule to the import path
_VGGT_ROOT = Path(__file__).resolve().parent.parent / "vggt"
if str(_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_ROOT))

logger = logging.getLogger(__name__)

VGGT_RESOLUTION = 518   # hard-coded inference resolution for VGGT


class VGGTPreprocessor:
    """Wraps VGGT for frame-by-frame camera + depth estimation.

    Parameters
    ----------
    weights    : HuggingFace repo ID (e.g. "facebook/VGGT-1B") or local directory.
    device     : torch device string, e.g. "cuda:0".
    resolution : VGGT inference resolution (default 518, do not change).
    """

    def __init__(
        self,
        weights:    str = "facebook/VGGT-1B",
        device:     str = "cuda:0",
        resolution: int = VGGT_RESOLUTION,
    ):
        self.weights    = weights
        self.device     = torch.device(device)
        self.resolution = resolution
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
            else torch.float16
        )

        from vggt.models.vggt import VGGT
        logger.info(f"Loading VGGT on {device} from '{weights}' …")
        self.model = VGGT.from_pretrained(weights, enable_track=False)
        self.model.eval().to(self.device)
        logger.info("VGGT ready.")

    # ── Image loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(
        path: Path,
        resolution: int,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Load one image, pad to square, resize to resolution.

        Returns
        -------
        img_t          : (3, resolution, resolution) float32 in [0, 1]
        original_coords: (4,) float32 [x1, y1, x2, y2] in resolution-space
        original_size  : (2,) int32   [W, H] of the original image
        """
        img = Image.open(path).convert("RGB")
        W, H = img.size

        max_dim = max(W, H)
        left    = (max_dim - W) // 2
        top     = (max_dim - H) // 2
        scale   = resolution / max_dim

        coords = np.array(
            [left * scale, top * scale, (left + W) * scale, (top + H) * scale],
            dtype=np.float32,
        )
        square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square.paste(img, (left, top))
        square = square.resize((resolution, resolution), Image.Resampling.BICUBIC)

        return TF.ToTensor()(square), coords, np.array([W, H], dtype=np.int32)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    @staticmethod
    def _reroot_to_cam0(extrinsics: np.ndarray) -> np.ndarray:
        """Re-express K extrinsics so that camera 0 is the world origin.

        Parameters
        ----------
        extrinsics : (K, 3, 4) camera-from-world in OpenCV convention.

        Returns
        -------
        (K, 3, 4) with cam0 extrinsic = [I | 0].
        """
        R0  = extrinsics[0, :3, :3]
        t0  = extrinsics[0, :3,  3]
        R0T = R0.T
        out = np.empty_like(extrinsics)
        for k in range(len(extrinsics)):
            Rk = extrinsics[k, :3, :3]
            tk = extrinsics[k, :3,  3]
            out[k, :3, :3] = Rk @ R0T
            out[k, :3,  3] = tk - Rk @ R0T @ t0
        return out

    # ── Single-frame inference ────────────────────────────────────────────────

    @torch.no_grad()
    def run_frame(
        self,
        image_paths: list[Path | None],
    ) -> dict[str, np.ndarray]:
        """Run VGGT on the K cameras of a single timestep.

        Parameters
        ----------
        image_paths : length-K list; entry k is a Path or None if camera k is absent.

        Returns
        -------
        dict with keys:
          extrinsics      (K_present, 3, 4) float32 — cam-from-world, cam0 is world origin
          intrinsics      (K_present, 3, 3) float32 — calibrated for 518×518 padded image
          original_coords (K_present, 4)   float32 — [x1,y1,x2,y2] in 518-space
          original_size   (K_present, 2)   int32   — [W, H] of original image
          depth           (K_present, 518, 518) float32 — metric depth in metres
          depth_conf      (K_present, 518, 518) float32
          present_indices (K_present,)     int32   — which k indices had images
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        present_indices = [k for k, p in enumerate(image_paths) if p is not None]
        present_paths   = [image_paths[k] for k in present_indices]

        if not present_paths:
            raise ValueError("run_frame: all image paths are None.")

        imgs, coords_list, sizes_list = [], [], []
        for path in present_paths:
            img_t, oc, os_ = self._load_image(path, self.resolution)
            imgs.append(img_t)
            coords_list.append(oc)
            sizes_list.append(os_)

        images   = torch.stack(imgs).to(self.device)    # (K_present, 3, H, W)
        images_b = images.unsqueeze(0)                  # (1, K_present, 3, H, W)

        with torch.amp.autocast("cuda", dtype=self.dtype):
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(images_b)

        # Camera head enforces float32 internally
        pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]  # (1, K_present, 9)
        extrinsics_t, intrinsics_t = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(self.resolution, self.resolution)
        )  # (1, K_present, 3, 4) and (1, K_present, 3, 3)

        # Depth head also enforces float32
        with torch.amp.autocast("cuda", enabled=False):
            depth, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images_b, patch_start_idx
            )
        # depth: (1, K_present, H, W, 1),  depth_conf: (1, K_present, H, W)

        extrinsics_np = extrinsics_t.squeeze(0).float().cpu().numpy()       # (K_present, 3, 4)
        intrinsics_np = intrinsics_t.squeeze(0).float().cpu().numpy()       # (K_present, 3, 3)
        depth_np      = depth.squeeze(0).squeeze(-1).float().cpu().numpy()  # (K_present, H, W)
        depth_conf_np = depth_conf.squeeze(0).float().cpu().numpy()         # (K_present, H, W)

        extrinsics_np = self._reroot_to_cam0(extrinsics_np)

        return {
            "extrinsics":      extrinsics_np,
            "intrinsics":      intrinsics_np,
            "original_coords": np.stack(coords_list),
            "original_size":   np.stack(sizes_list),
            "depth":           depth_np,
            "depth_conf":      depth_conf_np,
            "present_indices": np.array(present_indices, dtype=np.int32),
        }

    # ── Frame-loop (single GPU) ───────────────────────────────────────────────

    def _process_frames(
        self,
        frame_paths:  list[list[Path | None]],
        camera_names: list[str],
        output_dir:   Path,
        tmp_path:     Path | None = None,
    ) -> None:
        """Process a list of frames on this instance's device and save results.

        Parameters
        ----------
        frame_paths  : frame_paths[t][k] = Path or None.
        camera_names : K camera name strings.
        output_dir   : destination for the final .npz files (used when tmp_path is None).
        tmp_path     : if given, write a temporary chunk file here instead; the main
                       process will merge all chunks into the final output.
        """
        T = len(frame_paths)
        K = len(camera_names)
        R = self.resolution

        all_extrinsics      = np.full((T, K, 3, 4), np.nan, dtype=np.float32)
        all_intrinsics      = np.full((T, K, 3, 3), np.nan, dtype=np.float32)
        all_original_coords = np.full((T, K, 4),    np.nan, dtype=np.float32)
        all_original_size   = np.zeros((T, K, 2),           dtype=np.int32)
        all_depth           = np.full((T, K, R, R), np.nan, dtype=np.float32)
        all_depth_conf      = np.full((T, K, R, R), np.nan, dtype=np.float32)
        valid               = np.zeros((T, K),               dtype=bool)

        for t, paths in enumerate(frame_paths):
            if not any(p is not None for p in paths):
                logger.warning(f"Frame {t:05d}: no cameras present — skipping.")
                continue
            try:
                res = self.run_frame(paths)
                ki  = res["present_indices"]
                all_extrinsics[t,      ki] = res["extrinsics"]
                all_intrinsics[t,      ki] = res["intrinsics"]
                all_original_coords[t, ki] = res["original_coords"]
                all_original_size[t,   ki] = res["original_size"]
                all_depth[t,           ki] = res["depth"]
                all_depth_conf[t,      ki] = res["depth_conf"]
                valid[t, ki] = True
            except Exception:
                logger.exception(f"Frame {t:05d} failed.")

            if (t + 1) % 20 == 0 or (t + 1) == T:
                logger.info(f"  [{self.device}] {t + 1}/{T} frames done.")

        if tmp_path is not None:
            self._save_chunk(
                tmp_path,
                all_extrinsics, all_intrinsics,
                all_original_coords, all_original_size,
                all_depth, all_depth_conf, valid,
            )
        else:
            self._save_outputs(
                output_dir, camera_names,
                all_extrinsics, all_intrinsics,
                all_original_coords, all_original_size,
                all_depth, all_depth_conf, valid,
            )

    # ── Multi-GPU entry point ─────────────────────────────────────────────────

    def process_scene(
        self,
        frame_paths:  list[list[Path | None]],
        camera_names: list[str],
        output_dir:   Path,
        devices:      list[str] | None = None,
    ) -> None:
        """Process all frames of a scene and save vggt_cameras.npz / vggt_depth.npz.

        Parameters
        ----------
        frame_paths  : frame_paths[t][k] = Path or None. len = T, inner len = K.
        camera_names : K camera name strings, same ordering as the K dimension.
        output_dir   : directory where outputs are written.
        devices      : list of CUDA device strings to use. If None or length 1,
                       runs on self.device without spawning subprocesses.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not devices or len(devices) <= 1:
            self._process_frames(frame_paths, camera_names, output_dir)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            mp.spawn(
                VGGTPreprocessor._worker_fn,
                args=(frame_paths, camera_names, output_dir, devices, self.weights, tmp_dir),
                nprocs=len(devices),
                join=True,
            )
            self._merge_chunks(
                Path(tmp_dir), output_dir, camera_names,
                len(frame_paths), len(camera_names),
            )

    # ── Multiprocessing worker (staticmethod so mp.spawn can pickle it) ───────

    @staticmethod
    def _worker_fn(
        rank:         int,
        frame_paths:  list[list[Path | None]],
        camera_names: list[str],
        output_dir:   Path,
        devices:      list[str],
        weights:      str,
        tmp_dir:      str,
    ) -> None:
        """One worker process: handles a round-robin slice of frames."""
        device = devices[rank]
        T      = len(frame_paths)

        my_indices = list(range(rank, T, len(devices)))
        my_paths   = [frame_paths[t] for t in my_indices]
        tmp_path   = Path(tmp_dir) / f"chunk_{rank:02d}.npz"

        worker = VGGTPreprocessor(weights=weights, device=device)
        worker._process_frames(my_paths, camera_names, output_dir, tmp_path=tmp_path)

        # Append the original frame indices to the chunk so the merge step can
        # reconstruct the correct temporal ordering.
        data = dict(np.load(tmp_path, allow_pickle=False))
        data["frame_indices"] = np.array(my_indices, dtype=np.int32)
        np.savez_compressed(str(tmp_path), **data)
        logger.info(f"[rank {rank}] chunk saved → {tmp_path}")

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_chunk(
        path:            Path,
        extrinsics:      np.ndarray,
        intrinsics:      np.ndarray,
        original_coords: np.ndarray,
        original_size:   np.ndarray,
        depth:           np.ndarray,
        depth_conf:      np.ndarray,
        valid:           np.ndarray,
    ) -> None:
        """Save one worker's results to a temporary compressed .npz."""
        np.savez_compressed(
            str(path),
            extrinsics      = extrinsics,
            intrinsics      = intrinsics,
            original_coords = original_coords,
            original_size   = original_size,
            depth           = depth.astype(np.float16),
            depth_conf      = depth_conf.astype(np.float16),
            valid           = valid,
        )

    @staticmethod
    def _merge_chunks(
        tmp_dir:      Path,
        output_dir:   Path,
        camera_names: list[str],
        T:            int,
        K:            int,
    ) -> None:
        """Merge per-GPU chunk files into the final output .npz files."""
        R = VGGT_RESOLUTION
        all_extrinsics      = np.full((T, K, 3, 4), np.nan, dtype=np.float32)
        all_intrinsics      = np.full((T, K, 3, 3), np.nan, dtype=np.float32)
        all_original_coords = np.full((T, K, 4),    np.nan, dtype=np.float32)
        all_original_size   = np.zeros((T, K, 2),           dtype=np.int32)
        all_depth           = np.full((T, K, R, R), np.nan, dtype=np.float16)
        all_depth_conf      = np.full((T, K, R, R), np.nan, dtype=np.float16)
        valid               = np.zeros((T, K),               dtype=bool)

        for chunk_path in sorted(tmp_dir.glob("chunk_*.npz")):
            data = np.load(chunk_path, allow_pickle=False)
            idxs = data["frame_indices"]
            all_extrinsics[idxs]      = data["extrinsics"]
            all_intrinsics[idxs]      = data["intrinsics"]
            all_original_coords[idxs] = data["original_coords"]
            all_original_size[idxs]   = data["original_size"]
            all_depth[idxs]           = data["depth"]
            all_depth_conf[idxs]      = data["depth_conf"]
            valid[idxs]               = data["valid"]

        VGGTPreprocessor._save_outputs(
            output_dir, camera_names,
            all_extrinsics, all_intrinsics,
            all_original_coords, all_original_size,
            all_depth.astype(np.float32), all_depth_conf.astype(np.float32), valid,
        )

    @staticmethod
    def _save_outputs(
        output_dir:      Path,
        camera_names:    list[str],
        extrinsics:      np.ndarray,
        intrinsics:      np.ndarray,
        original_coords: np.ndarray,
        original_size:   np.ndarray,
        depth:           np.ndarray,
        depth_conf:      np.ndarray,
        valid:           np.ndarray,
    ) -> None:
        """Write vggt_cameras.npz and vggt_depth.npz to output_dir."""
        cam_path   = output_dir / "vggt_cameras.npz"
        depth_path = output_dir / "vggt_depth.npz"

        np.savez_compressed(
            str(cam_path),
            extrinsics      = extrinsics,       # (T, K, 3, 4) float32 — cam-from-world in cam0 frame
            intrinsics      = intrinsics,       # (T, K, 3, 3) float32 — for 518×518 padded image
            original_coords = original_coords, # (T, K, 4)   float32 — [x1,y1,x2,y2] in 518-space
            original_size   = original_size,   # (T, K, 2)   int32   — [W, H] original image
            valid           = valid,            # (T, K)      bool
            camera_names    = np.array(camera_names, dtype="S64"),  # (K,) byte strings
        )
        logger.info(f"Saved cameras → {cam_path}  extrinsics shape={extrinsics.shape}")

        # Depth quantised to uint16 millimetres (range 0–65.5 m, resolution 1 mm).
        # Integer values compress ~2× better than float16 under gzip for smooth depth.
        # To recover metres: depth_m = depth_mm.astype(np.float32) / 1000
        depth_mm = np.clip(
            np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535
        ).astype(np.uint16)

        np.savez_compressed(
            str(depth_path),
            depth       = depth_mm,                         # (T, K, 518, 518) uint16 mm
            depth_conf  = depth_conf.astype(np.float16),   # (T, K, 518, 518) float16
            depth_valid = valid,                            # (T, K) bool
        )
        logger.info(f"Saved depth   → {depth_path}  depth shape={depth_mm.shape} dtype=uint16")
