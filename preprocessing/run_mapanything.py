"""
preprocessing/run_mapanything.py

Estimates a per-frame metric scale for VGGT depth maps using MapAnything.

For each frame t in a scene, passes all valid cameras' images + VGGT depth maps
+ VGGT camera poses to MapAnything (is_metric_scale=False) and computes:

    scale[t] = median over valid cameras of  median(pred_depth / vggt_depth)

The result is a float32 (T,) array saved as {scene_dir}/mapanything_scale.npy.

Standalone usage (all scenes):
    pixi run python preprocessing/run_mapanything.py \
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
        --rich_root         /tmp/rich_train

Standalone usage (single scene):
    pixi run python preprocessing/run_mapanything.py \
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
        --rich_root         /tmp/rich_train \
        --scenes            BBQ_001_guitar
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH       = 14           # ViT-G/14 patch size — H and W must be multiples
DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD  = [0.229, 0.224, 0.225]
HF_REPO     = "facebook/map-anything"


# ---------------------------------------------------------------------------
# MapAnythingScaleEstimator
# ---------------------------------------------------------------------------

class MapAnythingScaleEstimator:
    """Estimates per-frame metric scale for VGGT depth maps using MapAnything.

    Parameters
    ----------
    device     : torch device string, e.g. "cuda:0".
    batch_size : Number of consecutive frames processed per MapAnything call.
    force      : If True, recompute even when mapanything_scale.npy exists.
    """

    def __init__(
        self,
        device:     str = "cuda:0",
        batch_size: int = 8,
        force:      bool = False,
    ):
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.force      = force
        self._model     = None   # lazy-loaded

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        from mapanything.models import MapAnything
        logger.info(f"Loading MapAnything from '{HF_REPO}' on {self.device} …")
        self._model = (
            MapAnything.from_pretrained(HF_REPO)
            .to(self.device)
            .eval()
        )
        logger.info("MapAnything ready.")

    # ── Image / geometry helpers ───────────────────────────────────────────────

    @staticmethod
    def _load_img(path: Path, H: int, W: int) -> torch.Tensor:
        """Load JPEG → (1, 3, H, W) DINOv2-normalised float32."""
        img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
        t = TF.to_tensor(img)
        return TF.normalize(t, DINOV2_MEAN, DINOV2_STD).unsqueeze(0)

    @staticmethod
    def _w2c_to_c2w(ext: np.ndarray) -> np.ndarray:
        """[R|t] world-to-cam (3,4) → (4,4) cam-to-world."""
        R, t = ext[:3, :3], ext[:3, 3]
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = R.T
        m[:3, 3]  = -R.T @ t
        return m

    @staticmethod
    def _scale_intrinsics(K: np.ndarray,
                          W_src: int, H_src: int,
                          W_dst: int, H_dst: int) -> np.ndarray:
        K = K.copy()
        K[0, 0] *= W_dst / W_src;  K[0, 2] *= W_dst / W_src
        K[1, 1] *= H_dst / H_src;  K[1, 2] *= H_dst / H_src
        return K

    # ── Core batch inference ───────────────────────────────────────────────────

    def _run_batch(
        self,
        batch_frames:   list[int],
        valid_ks:       list[int],
        cam_file_lists: dict[int, list[Path]],
        vggt_exts:      np.ndarray,
        vggt_intrs:     np.ndarray,
        depth_mm:       np.ndarray,
        H_vggt: int, W_vggt: int,
        H_ma:   int, W_ma:   int,
    ) -> dict[int, float]:
        """Run MapAnything on one batch of frames.

        Returns {frame_idx: scale} for each successfully processed frame.
        """
        B = len(batch_frames)
        views        = []
        depth_nps    = []   # (B, H, W) per camera, for scale extraction

        for k in valid_ks:
            files = cam_file_lists[k]
            imgs, intrs, depths, poses = [], [], [], []

            ok = True
            for fi in batch_frames:
                if fi >= len(files):
                    ok = False; break
                jp = files[fi]

                imgs.append(self._load_img(jp, H_ma, W_ma))
                intrs.append(torch.from_numpy(
                    self._scale_intrinsics(vggt_intrs[fi, k],
                                           W_vggt, H_vggt, W_ma, H_ma)
                ).float())

                d_m = depth_mm[fi, k].astype(np.float32) / 1000.0
                d_t = F.interpolate(
                    torch.from_numpy(d_m).unsqueeze(0).unsqueeze(0),
                    size=(H_ma, W_ma), mode="nearest",
                ).squeeze()
                depths.append(d_t)

                poses.append(torch.from_numpy(
                    self._w2c_to_c2w(vggt_exts[fi, k])).float())

            if not ok:
                continue

            depth_nps.append(torch.stack(depths).numpy())   # (B, H, W)
            views.append({
                "img":             torch.cat(imgs).to(self.device),     # (B,3,H,W)
                "data_norm_type":  ["dinov2"] * B,
                "intrinsics":      torch.stack(intrs).to(self.device),  # (B,3,3)
                "depth_z":         torch.stack(depths).to(self.device), # (B,H,W)
                "camera_poses":    torch.stack(poses).to(self.device),  # (B,4,4)
                "is_metric_scale": torch.zeros(B, dtype=torch.bool,
                                               device=self.device),
            })

        if len(views) < 2:
            return {}

        # Normalise poses per batch element so cam_0 → [I|0]
        ref_poses = views[0]["camera_poses"].cpu().double().numpy()  # (B,4,4)
        for v in views:
            pi = v["camera_poses"].cpu().double().numpy()
            norm = np.stack([np.linalg.inv(ref_poses[b]) @ pi[b]
                             for b in range(B)])
            v["camera_poses"] = torch.from_numpy(norm).float().to(self.device)

        with torch.no_grad():
            preds = self._model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=False,
                mask_edges=False,
                ignore_depth_scale_inputs=True,
                ignore_pose_scale_inputs=True,
            )

        results = {}
        for b, fi in enumerate(batch_frames):
            cam_scales = []
            for pred, vggt_d in zip(preds, depth_nps):
                pd   = pred["depth_z"][b].squeeze(-1).cpu().numpy()  # (H, W)
                vd   = vggt_d[b]
                mask = (pd > 0) & (vd > 0)
                if mask.sum() > 100:
                    cam_scales.append(float(np.median(pd[mask] / vd[mask])))
            if cam_scales:
                results[fi] = float(np.median(cam_scales))

        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_scene(
        self,
        scene_dir: Path,
        rich_root: Path,
    ) -> np.ndarray | None:
        """Estimate per-frame scale for one scene.

        Parameters
        ----------
        scene_dir : ghost output directory for the scene
                    (contains vggt_cameras.npz, vggt_depth.npz).
        rich_root : root of the mounted RICH squash
                    (contains {scene}/{cam}/{frame:05d}_{cam_num:02d}.jpeg).

        Returns
        -------
        float32 (T,) scale array, or None if prerequisites are missing.
        Also saves the array to {scene_dir}/mapanything_scale_centered.npy.
        """
        out_path = scene_dir / "mapanything_scale_centered.npy"
        if out_path.exists() and not self.force:
            logger.info(f"{scene_dir.name}: already done, loading from disk")
            return np.load(out_path)

        cam_path   = scene_dir / "vggt_cameras_centered.npz"
        depth_path = scene_dir / "vggt_depth_centered.npz"
        if not cam_path.exists() or not depth_path.exists():
            logger.warning(f"{scene_dir.name}: missing vggt_cameras or vggt_depth — skip")
            return None

        cam_npz   = np.load(cam_path,   mmap_mode="r")
        depth_npz = np.load(depth_path, mmap_mode="r")

        cam_names  = [n.decode() if isinstance(n, bytes) else n
                      for n in cam_npz["camera_names"]]
        vggt_exts  = cam_npz["extrinsics"].astype(np.float64)   # (T, K, 3, 4)
        vggt_intrs = cam_npz["intrinsics"].astype(np.float64)   # (T, K, 3, 3)
        vggt_valid = cam_npz["valid"]                            # (T, K) bool
        depth_mm   = depth_npz["depth"]                         # (T, K, H, W) uint16

        T, K, H_vggt, W_vggt = depth_mm.shape
        H_ma = (H_vggt // PATCH) * PATCH
        W_ma = (W_vggt // PATCH) * PATCH

        scene_name = scene_dir.name
        img_root   = rich_root / scene_name

        # Build sorted file lists per camera — handles any starting frame number.
        cam_file_lists: dict[int, list[Path]] = {}
        for k, cn in enumerate(cam_names):
            cam_dir = img_root / cn
            if cam_dir.is_dir():
                cam_file_lists[k] = sorted(
                    p for p in cam_dir.iterdir()
                    if p.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp"}
                )

        valid_cam_file_ks = [k for k, fl in cam_file_lists.items() if len(fl) >= T]
        if len(valid_cam_file_ks) < 2:
            logger.warning(f"{scene_name}: images not found under {img_root} — skip")
            return None

        self._load_model()
        logger.info(f"{scene_name}: T={T}  K={K}  MA={H_ma}×{W_ma}")

        scale_arr = np.full(T, np.nan, dtype=np.float32)
        t0 = time.perf_counter()

        for batch_start in range(0, T, self.batch_size):
            batch_end    = min(batch_start + self.batch_size, T)
            batch_frames = list(range(batch_start, batch_end))
            B            = len(batch_frames)

            valid_ks = [k for k in valid_cam_file_ks
                        if vggt_valid[batch_frames, k].all()]
            if len(valid_ks) < 2:
                continue

            results = self._run_batch(
                batch_frames, valid_ks,
                cam_file_lists,
                vggt_exts, vggt_intrs, depth_mm,
                H_vggt, W_vggt, H_ma, W_ma,
            )
            for fi, s in results.items():
                scale_arr[fi] = s

            torch.cuda.empty_cache()

        # Fill any NaN frames with the overall median
        valid_scales = scale_arr[np.isfinite(scale_arr)]
        if len(valid_scales) == 0:
            logger.warning(f"{scene_name}: no valid scale estimates")
            return None

        global_median = float(np.median(valid_scales))
        nan_mask = ~np.isfinite(scale_arr)
        scale_arr[nan_mask] = global_median

        elapsed = time.perf_counter() - t0
        logger.info(
            f"{scene_name}: scale mean={scale_arr.mean():.4f}  "
            f"std={scale_arr.std():.4f}  nan_filled={nan_mask.sum()}  "
            f"time={elapsed:.1f}s"
        )

        np.save(out_path, scale_arr)
        return scale_arr


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--ghost_output_root", required=True)
    p.add_argument("--rich_root",         required=True)
    p.add_argument("--scenes",            default="",
                   help="Comma-separated scene names; empty = all scenes")
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--force",             action="store_true")
    args = p.parse_args()

    output_root = Path(args.ghost_output_root)
    rich_root   = Path(args.rich_root)

    if args.scenes:
        scene_dirs = [output_root / s.strip() for s in args.scenes.split(",")]
    else:
        scene_dirs = sorted(
            d for d in output_root.iterdir()
            if d.is_dir() and (d / "vggt_cameras_centered.npz").exists()
        )

    estimator = MapAnythingScaleEstimator(
        device=args.device,
        batch_size=args.batch_size,
        force=args.force,
    )

    t_total = time.perf_counter()
    logger.info(f"Processing {len(scene_dirs)} scene(s) with batch_size={args.batch_size}")

    for scene_dir in scene_dirs:
        estimator.process_scene(scene_dir, rich_root)

    logger.info(f"All done in {(time.perf_counter()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
