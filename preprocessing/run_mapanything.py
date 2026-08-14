"""
preprocessing/run_mapanything.py

Estimates a per-frame metric scale for VGGT reconstructions using MapAnything.

For each frame t in a scene, MapAnything runs IMAGES-ONLY (its in-distribution
mode) over all valid cameras and reconstructs the rig metrically; VGGT gives the
same rig up to scale.  The two differ by one similarity, whose scale is:

    scale[t] = median over camera pairs of
               (MapAnything baseline / VGGT baseline)

No focal length, depth map or pose is fed to MapAnything, so the estimate is
convention-free.  The result is a float32 (T,) array saved as
{scene_dir}/mapanything_scale_baseline.npy.

Standalone usage (all scenes):
    pixi run python preprocessing/run_mapanything.py \
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
        --img_root          /tmp/rich_train

Standalone usage (single scene):
    pixi run python preprocessing/run_mapanything.py \
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
        --img_root          /tmp/rich_train \
        --scenes            BBQ_001_guitar
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
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
    """Estimates per-frame metric scale for VGGT reconstructions using MapAnything.

    The scale comes from CAMERA BASELINE RATIOS with MapAnything run images-only
    — see :meth:`_run_batch_baselines`.  A conditioned variant (feed MapAnything
    the vggt intrinsics + depth + poses, take median(MA_depth / vggt_depth)) used
    to live here and was removed 2026-08-14: it is BIASED on wide-FOV cameras,
    because MA's metric scale head is conditioned on the input rays and
    geometry-in/scale-out is a rare (~5%) training configuration (arXiv
    2509.13414; no wide-angle training data).  On EgoHumans undistorted fisheye
    (~101 deg FOV) it came out ~1.56x too small; on 031_badminton it gave 13.2
    where the baseline ratio gives 21.50 against a GT of 21.54 (0.2%).  Nothing
    downstream read its output — ``fusion/placer.py`` loads the baseline file
    only — so the legacy ``mapanything_scale_centered.npy`` files still on disk
    are inert.

    Parameters
    ----------
    device     : torch device string, e.g. "cuda:0".
    batch_size : Number of consecutive frames processed per MapAnything call.
    force      : If True, recompute even when mapanything_scale_baseline.npy exists.
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

    # ── Core batch inference ───────────────────────────────────────────────────

    def _run_batch_baselines(
        self,
        batch_frames:   list[int],
        valid_ks:       list[int],
        cam_file_lists: dict[int, list[Path]],
        vggt_exts:      np.ndarray,
        H_ma: int, W_ma: int,
    ) -> dict[int, float]:
        """Images-only MapAnything; scale from camera-baseline ratios.

        For each frame: MA reconstructs the rig metrically from images alone,
        vggt gives the same rig up-to-scale; the two differ by one similarity,
        whose scale is the ratio of any camera-pair distance. Median over all
        pairs. Returns {frame_idx: scale}.
        """
        B = len(batch_frames)
        views = []
        kept_ks: list[int] = []   # camera index behind each view, in view order
        for k in valid_ks:
            files = cam_file_lists[k]
            imgs = []
            ok = True
            for fi in batch_frames:
                if fi >= len(files):
                    ok = False
                    break
                imgs.append(self._load_img(files[fi], H_ma, W_ma))
            if not ok:
                continue
            kept_ks.append(k)
            views.append({
                "img":            torch.cat(imgs).to(self.device),   # (B,3,H,W)
                "data_norm_type": ["dinov2"] * B,
            })
        if len(views) < 2:
            return {}

        with torch.no_grad():
            preds = self._model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=False,
                mask_edges=False,
            )

        results: dict[int, float] = {}
        n_cams = len(views)
        for b, fi in enumerate(batch_frames):
            # MA camera centres (metric, view-0 frame): c2w translation
            ma_c = [preds[v]["camera_poses"][b].cpu().double().numpy()[:3, 3]
                    for v in range(n_cams)]
            # vggt camera centres (up-to-scale): -R^T t from w2c extrinsics.
            # Indexed by kept_ks, NOT valid_ks — a camera whose file list came up
            # short is skipped above, and slicing valid_ks positionally would
            # then pair every later MA view with the wrong vggt camera.
            vg_c = []
            for k in kept_ks:
                E = vggt_exts[fi, k]
                vg_c.append(-E[:3, :3].T @ E[:3, 3])
            ratios = []
            for i in range(n_cams):
                for j in range(i + 1, n_cams):
                    vb = float(np.linalg.norm(vg_c[i] - vg_c[j]))
                    if vb < 1e-6:
                        continue
                    ratios.append(float(np.linalg.norm(ma_c[i] - ma_c[j])) / vb)
            if ratios:
                results[fi] = float(np.median(ratios))
        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_scene(
        self,
        scene_dir: Path,
        img_root: Path,
    ) -> np.ndarray | None:
        """Estimate per-frame scale for one scene.

        Parameters
        ----------
        scene_dir : ghost output directory for the scene
                    (contains vggt_cameras_centered.npz, vggt_depth_centered.npz).
        img_root  : directory containing one sub-folder per camera (named after
                    the camera, e.g. ``cam_01/`` or ``cam01/``).  Images may
                    live directly in that sub-folder or one level deeper (e.g.
                    inside an ``images_undistorted/`` sub-directory).

        Returns
        -------
        float32 (T,) scale array, or None if prerequisites are missing.
        Also saves the array to {scene_dir}/mapanything_scale_baseline.npy.
        """
        _IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}
        out_path = scene_dir / "mapanything_scale_baseline.npy"
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
        vggt_valid = cam_npz["valid"]                            # (T, K) bool
        # MapAnything is fed images only; the depth array survives purely as the
        # source of T/K and of the grid that sizes MA's input below. TODO: read
        # the header instead of decompressing the whole (T,K,H,W) uint16 array.
        depth_mm   = depth_npz["depth"]                         # (T, K, H, W) uint16

        T, K, H_vggt, W_vggt = depth_mm.shape
        H_ma = (H_vggt // PATCH) * PATCH
        W_ma = (W_vggt // PATCH) * PATCH

        scene_name = scene_dir.name

        # Build sorted file lists per camera.
        # Searches cam_dir directly, then one level deeper (e.g. images_undistorted/).
        cam_file_lists: dict[int, list[Path]] = {}
        for k, cn in enumerate(cam_names):
            cam_dir = img_root / cn
            if not cam_dir.is_dir():
                continue
            files = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
            if not files:
                for subdir in sorted(cam_dir.iterdir()):
                    if subdir.is_dir():
                        files = sorted(p for p in subdir.iterdir()
                                       if p.suffix.lower() in _IMG_EXTS)
                        if not files:
                            # one more level (e.g. images_undistorted/frames/)
                            for subdir2 in sorted(subdir.iterdir()):
                                if subdir2.is_dir():
                                    files = sorted(p for p in subdir2.iterdir()
                                                   if p.suffix.lower() in _IMG_EXTS)
                                    if files:
                                        break
                        if files:
                            break
            if files:
                cam_file_lists[k] = files

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

            results = self._run_batch_baselines(
                batch_frames, valid_ks, cam_file_lists,
                vggt_exts, H_ma, W_ma,
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
    p.add_argument("--img_root",          required=True,
                   help="Directory with one sub-folder per camera (cam_01/ etc.). "
                        "For RICH: rich_root/scene_name. For EgoHumans: seq_dir/exo.")
    p.add_argument("--scenes",            default="",
                   help="Comma-separated scene names; empty = all scenes")
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--force",             action="store_true")
    args = p.parse_args()

    output_root = Path(args.ghost_output_root)
    img_root    = Path(args.img_root)

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
        estimator.process_scene(scene_dir, img_root)

    logger.info(f"All done in {(time.perf_counter()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
