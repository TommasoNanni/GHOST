"""
VGGT camera + depth preprocessing for multi-camera synchronized scenes.

Runs VGGT frame-by-frame on all available cameras and saves:
  - vggt_cameras.npz  : extrinsics, intrinsics, coordinate mapping info
  - vggt_depth.npz    : depth maps and confidence at VGGT-Omega output resolution

Coordinate conventions
----------------------
Extrinsics are in OpenCV convention: x_cam = R @ x_world + t  (camera-from-world).
The world origin is defined as camera 0 (cam0) of each frame.
  R_i' = R_i @ R_0^T
  t_i' = t_i − R_i @ R_0^T @ t_0
Depth maps live in each camera's own frame (z-axis = depth), so re-rooting the
world frame does not affect depth values.

Frames where camera 0 has no image are anchored on the first camera that does;
_regauge_to_cam0 converts them back into the camera-0 frame before saving, so
every frame of a scene shares one world origin.

Querying depth at a body keypoint
----------------------------------
VGGT-Omega resizes images with aspect ratio preserved (no square padding).
`original_coords[t, k] = [0, 0, W_vggt, H_vggt]`. To back-project a 2D keypoint
(u_orig, v_orig) from original resolution:

    scale_x = W_vggt / W_orig
    scale_y = H_vggt / H_orig
    u_vggt  = u_orig * scale_x
    v_vggt  = v_orig * scale_y

Then look up depth[t, k] at (v_vggt, u_vggt) with bilinear interpolation and
back-project with intrinsics[t, k] (calibrated for the VGGT output image):
    X = (u_vggt - cx) * depth / fx
    Y = (v_vggt - cy) * depth / fy
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

    preprocessor = VGGTPreprocessor(weights="/path/to/vggt_omega.pt")

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
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from vggt_omega import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = logging.getLogger(__name__)

# Base resolution passed to load_and_preprocess_images.  The actual output
# dimensions depend on the input aspect ratio; e.g. for RICH 1440×1053 images
# this yields 448×592 depth/camera maps (not a square).
VGGT_RESOLUTION = 512


class VGGTPreprocessor:
    """Wraps VGGT for frame-by-frame camera + depth estimation.

    Parameters
    ----------
    weights    : HuggingFace repo ID (e.g. "facebook/VGGT-1B") or local directory.
    device     : torch device string, e.g. "cuda:0".
    resolution : Base resolution passed to load_and_preprocess_images (default 512).
                 Actual output H×W depends on input aspect ratio.
    """

    def __init__(
        self,
        weights:    str,
        device:     str = "cuda:0",
        resolution: int = VGGT_RESOLUTION,
    ):
        self.weights    = weights
        self.device     = torch.device(device)
        self.resolution = resolution

        logger.info(f"Loading VGGT-Omega on {device} from '{weights}' …")
        self.model = VGGTOmega().eval()
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        logger.info("VGGT-Omega ready.")

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

    @staticmethod
    def _regauge_to_cam0(
        extrinsics: np.ndarray,
        valid:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Put every frame in the camera-0 world frame, including those without camera 0.

        _reroot_to_cam0 anchors on the first *present* camera, which is camera 0
        only when camera 0 has an image at that timestep. If it is absent — a late
        start, a dropped frame — the frame is silently expressed relative to a
        different camera, so its extrinsics live in a different world frame than
        every other frame's.

        Anchoring is only a gauge choice, so such a frame can be converted rather
        than discarded. With E_k the world-to-camera-k extrinsic and E^a_k the
        gauge anchored at camera a::

            E^0_k = E_k E_0^-1 = (E_k E_a^-1)(E_a E_0^-1) = E^a_k E^0_a

        E^0_a is camera a's pose in the camera-0 gauge, which is constant for a
        static rig and can be read off any frame where camera 0 and camera a are
        both present. In [R|t] form::

            R^0_k = R^a_k R^0_a          t^0_k = R^a_k t^0_a + t^a_k

        This is exact for static rigs only. A moving camera 0 makes E^0_a
        time-varying and the composition wrong — but a moving camera 0 already
        makes the per-frame world origin follow it, which is a deeper problem.

        Frames whose anchor camera never co-occurs with camera 0 have no such
        transform and are marked invalid: nothing observed both, so no relative
        pose exists.

        Parameters
        ----------
        extrinsics : (T, K, 3, 4) camera-from-world, as produced by _reroot_to_cam0.
        valid      : (T, K) bool; valid[t, k] is True when camera k was solved at t.

        Returns
        -------
        (extrinsics, valid) — the same arrays when no frame needs re-gauging.
        """
        if valid.ndim != 2 or not valid.any():
            return extrinsics, valid

        anchor = np.argmax(valid, axis=1)          # first present camera per frame
        solved = valid.any(axis=1)                 # frames VGGT actually solved
        off    = solved & (anchor != 0)            # ... but not anchored on camera 0
        if not off.any():
            return extrinsics, valid               # nothing to do: leave input untouched

        extrinsics = extrinsics.copy()
        valid      = valid.copy()

        for a in np.unique(anchor[off]):
            sel = off & (anchor == a)
            # Frames already in the camera-0 gauge that also saw camera a.
            ref = solved & (anchor == 0) & valid[:, a]
            if not ref.any():
                logger.warning(
                    f"Camera {a} never co-occurs with camera 0 — cannot re-gauge "
                    f"{int(sel.sum())} frame(s) anchored on it; marking them invalid."
                )
                extrinsics[sel] = np.nan
                valid[sel]      = False
                continue

            # Chordal mean of camera a's rotation over the reference frames,
            # projected back onto SO(3); averaging also damps VGGT's per-frame jitter.
            M       = extrinsics[ref, a, :3, :3].mean(axis=0)
            U, _, Vh = np.linalg.svd(M)
            D       = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vh)))])
            R0a     = (U @ D @ Vh).astype(np.float32)
            t0a     = extrinsics[ref, a, :3, 3].mean(axis=0).astype(np.float32)

            block = extrinsics[sel]                       # (n, K, 3, 4)
            # Copies, not views: writing into block below would otherwise feed the
            # already-updated rotation into the translation.
            Rka   = block[:, :, :3, :3].copy()            # (n, K, 3, 3)
            tka   = block[:, :, :3,  3].copy()            # (n, K, 3)
            block[:, :, :3, :3] = Rka @ R0a               # NaN entries stay NaN
            block[:, :, :3,  3] = Rka @ t0a + tka
            extrinsics[sel] = block

            logger.info(
                f"Re-gauged {int(sel.sum())} frame(s) anchored on camera {a} "
                f"into the camera-0 frame ({int(ref.sum())} reference frame(s))."
            )

        return extrinsics, valid

    # ── Single-frame inference ────────────────────────────────────────────────

    @torch.no_grad()
    def run_frame(
        self,
        image_paths: list[Path | None],
    ) -> dict[str, np.ndarray]:
        """Run VGGT-Omega on the K cameras of a single timestep.

        Parameters
        ----------
        image_paths : length-K list; entry k is a Path or None if camera k is absent.

        Returns
        -------
        dict with keys:
          extrinsics      (K_present, 3, 4) float32 — cam-from-world, cam0 is world origin
          intrinsics      (K_present, 3, 3) float32 — calibrated for the VGGT output resolution
          original_coords (K_present, 4)   float32 — [0, 0, W_vggt, H_vggt] per camera
          original_size   (K_present, 2)   int32   — [W, H] of original image
          depth           (K_present, H_vggt, W_vggt) float32 — metric depth in metres
          depth_conf      (K_present, H_vggt, W_vggt) float32
          present_indices (K_present,)     int32   — which k indices had images
        """
        present_indices = [k for k, p in enumerate(image_paths) if p is not None]
        present_paths   = [image_paths[k] for k in present_indices]

        if not present_paths:
            raise ValueError("run_frame: all image paths are None.")

        # Read original image sizes before preprocessing.
        sizes_list = []
        for path in present_paths:
            with Image.open(path) as img:
                sizes_list.append(np.array([img.width, img.height], dtype=np.int32))

        # Load and preprocess: aspect-ratio-preserving resize, common padding if needed.
        # Output: (K_present, 3, H_vggt, W_vggt)
        images = load_and_preprocess_images(
            [str(p) for p in present_paths], image_resolution=self.resolution
        ).to(self.device)

        # Forward pass — VGGTOmega adds batch dim internally for 4-D input.
        # predictions["images"] stores the (1, K_present, 3, H_vggt, W_vggt) tensor.
        predictions = self.model(images)

        H_vggt, W_vggt = predictions["images"].shape[-2:]

        # Decode cameras from 9-D pose encoding.
        extrinsics_t, intrinsics_t = encoding_to_camera(
            predictions["pose_enc"], (H_vggt, W_vggt)
        )  # (1, K_present, 3, 4) and (1, K_present, 3, 3)

        extrinsics_np = extrinsics_t.squeeze(0).float().cpu().numpy()  # (K_present, 3, 4)
        intrinsics_np = intrinsics_t.squeeze(0).float().cpu().numpy()  # (K_present, 3, 3)

        # depth: (1, K_present, H_vggt, W_vggt, 1);  depth_conf: (1, K_present, H_vggt, W_vggt)
        depth_np      = predictions["depth"][0, :, :, :, 0].float().cpu().numpy()
        depth_conf_np = predictions["depth_conf"][0].float().cpu().numpy()

        extrinsics_np = self._reroot_to_cam0(extrinsics_np)

        # original_coords: [0, 0, W_vggt, H_vggt] — no padding in VGGT-Omega, image fills
        # the full output.  Downstream _orig_to_vggt uses (x2-x1)/W_orig and (y2-y1)/H_orig
        # as x and y scale factors, which are W_vggt/W_orig and H_vggt/H_orig respectively.
        coords = np.array([0.0, 0.0, float(W_vggt), float(H_vggt)], dtype=np.float32)

        return {
            "extrinsics":      extrinsics_np,
            "intrinsics":      intrinsics_np,
            "original_coords": np.stack([coords] * len(present_paths)),
            "original_size":   np.stack(sizes_list),
            "depth":           depth_np,
            "depth_conf":      depth_conf_np,
            "present_indices": np.array(present_indices, dtype=np.int32),
        }

    # ── Resolution helper ─────────────────────────────────────────────────────

    def _get_vggt_hw(self, frame_paths: list[list[Path | None]]) -> tuple[int, int]:
        """Determine actual VGGT-Omega output (H, W) by preprocessing all cameras of one frame.

        Must use all cameras together (not just one) because load_and_preprocess_images
        pads to a common size when cameras have different aspect ratios. Using a single
        image underestimates the padded output dimensions, causing a shape mismatch in
        the depth assignment loop and silently leaving depth all-zero.
        """
        for paths in frame_paths:
            present = [str(p) for p in paths if p is not None]
            if present:
                sample = load_and_preprocess_images(present, image_resolution=self.resolution)
                return int(sample.shape[-2]), int(sample.shape[-1])
        raise ValueError("No valid images found in frame_paths — cannot determine VGGT output size.")

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

        # Compute actual VGGT output dimensions (aspect-ratio-aware, not square).
        H_vggt, W_vggt = self._get_vggt_hw(frame_paths)
        logger.info(f"  VGGT-Omega output resolution: {H_vggt}×{W_vggt}")

        all_extrinsics      = np.full((T, K, 3, 4),              np.nan, dtype=np.float32)
        all_intrinsics      = np.full((T, K, 3, 3),              np.nan, dtype=np.float32)
        all_original_coords = np.full((T, K, 4),                 np.nan, dtype=np.float32)
        all_original_size   = np.zeros((T, K, 2),                        dtype=np.int32)
        all_depth           = np.full((T, K, H_vggt, W_vggt),   np.nan, dtype=np.float32)
        all_depth_conf      = np.full((T, K, H_vggt, W_vggt),   np.nan, dtype=np.float32)
        valid               = np.zeros((T, K),                           dtype=bool)

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
            nprocs = min(len(devices), len(frame_paths))
            mp.spawn(
                VGGTPreprocessor._worker_fn,
                args=(frame_paths, camera_names, output_dir, devices, self.weights, tmp_dir),
                nprocs=nprocs,
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
        chunk_paths = sorted(tmp_dir.glob("chunk_*.npz"))
        first       = np.load(chunk_paths[0], allow_pickle=False)
        H_vggt, W_vggt = first["depth"].shape[-2:]   # read actual dims from first chunk

        all_extrinsics      = np.full((T, K, 3, 4),             np.nan, dtype=np.float32)
        all_intrinsics      = np.full((T, K, 3, 3),             np.nan, dtype=np.float32)
        all_original_coords = np.full((T, K, 4),                np.nan, dtype=np.float32)
        all_original_size   = np.zeros((T, K, 2),                       dtype=np.int32)
        all_depth           = np.full((T, K, H_vggt, W_vggt),  np.nan, dtype=np.float16)
        all_depth_conf      = np.full((T, K, H_vggt, W_vggt),  np.nan, dtype=np.float16)
        valid               = np.zeros((T, K),                          dtype=bool)

        for chunk_path in chunk_paths:
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
        cam_path   = output_dir / "vggt_cameras_centered.npz"
        depth_path = output_dir / "vggt_depth_centered.npz"

        # Frames where camera 0 was absent were anchored on another camera; move
        # them into the camera-0 world frame. No-op when camera 0 is always present.
        extrinsics, valid = VGGTPreprocessor._regauge_to_cam0(extrinsics, valid)

        np.savez_compressed(
            str(cam_path),
            extrinsics      = extrinsics,       # (T, K, 3, 4) float32 — cam-from-world in cam0 frame
            intrinsics      = intrinsics,       # (T, K, 3, 3) float32 — for VGGT output image
            original_coords = original_coords, # (T, K, 4)   float32 — [0,0,W_vggt,H_vggt]
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

        valid_frames = int(valid.any(axis=1).sum())
        nonzero_px   = int(np.count_nonzero(depth_mm))
        depth_finite = depth[np.isfinite(depth)]
        depth_stats  = (f"raw depth: min={depth_finite.min():.4f} max={depth_finite.max():.4f} "
                        f"mean={depth_finite.mean():.4f}") if depth_finite.size else "raw depth: all NaN"
        logger.info(
            f"Depth stats → valid_frames={valid_frames}/{depth_mm.shape[0]}  "
            f"nonzero_pixels={nonzero_px}  {depth_stats}"
        )

        np.savez_compressed(
            str(depth_path),
            depth       = depth_mm,                         # (T, K, H_vggt, W_vggt) uint16 mm
            depth_conf  = depth_conf.astype(np.float16),   # (T, K, H_vggt, W_vggt) float16
            depth_valid = valid,                            # (T, K) bool
        )
        logger.info(f"Saved depth   → {depth_path}  depth shape={depth_mm.shape} dtype=uint16")
