"""SAM3D Body estimation for tracked persons in multi-view scenes.

Reads pre-existing segmentation output (frames, masks, JSON metadata) and
runs per-frame 3D body model estimation via SAM3D Body.

Expected input layout (produced by :class:`PersonSegmenter`)::

    output_dir/<scene_id>/<video_id>/
        mask_data.npz    compressed mask archive
        json_data/       .json per-frame instance metadata

Frames live next to the source data, not inside the segmentation output::

    data_root/<scene_id>/<video_id>/
        frames/          extracted JPEGs

Output (added to existing directories)::

    output_dir/<scene_id>/<video_id>/
        body_data/
            person_<id>.npz                 <- per-person body params across frames
            body_params_summary.json        <- metadata
"""

from __future__ import annotations

import gc
import io
import json
import logging
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import smplx

# sam-3d-body ships a `tools/` package that is not part of any installed
# distribution — it must be on sys.path so that notebook/utils.py can do
# `from tools.build_detector import HumanDetector`.  Previously this was
# handled by a PYTHONPATH entry in ~/.bashrc; we inject it here instead so
# the code is self-contained regardless of shell environment.
_SAM3D_ROOT = Path(__file__).resolve().parents[1] / "sam-3d-body"
# Always insert at position 0 so sam-3d-body/tools/ takes priority over
# site-packages/tools/ (detectron2 installs a tools/ package there that
# shadows ours when the .pth-file appends sam-3d-body/ after site-packages).
_sam3d_root_str = str(_SAM3D_ROOT)
if _sam3d_root_str in sys.path:
    sys.path.remove(_sam3d_root_str)
sys.path.insert(0, _sam3d_root_str)

from data.video_dataset import Scene
from mhr.mhr import MHR
from conversion import Conversion
from preprocessing.confidence import ConfidenceEstimator
from preprocessing.within_video_reidentifier import InVideoReidentifier
from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2 as CrossVideoReidentifier


class ParametersExtractor:
    """Estimate 3D body parameters for tracked persons.

    Reads pre-existing segmentation output and runs SAM3D Body on each
    detected person crop.  Does **not** perform segmentation itself.

    Parameters
    ----------
    sam3d_hf_repo : str
        HuggingFace repo ID for SAM3D Body model.
    sam3d_step : int
        Run SAM3D every *sam3d_step* frames (1 = every frame).
    bbox_padding : float
        Fractional padding around bounding boxes before passing to SAM3D.
    """

    # Re-identification defaults — overridden by constructor args (from config).
    _REID_THRESHOLD: float = 0.65
    _GALLERY_MAX_SIZE: int = 30
    _REID_MATCH_WINDOW: int = 5

    def __init__(
        self,
        sam3d_hf_repo: str = "facebook/sam-3d-body-dinov3",
        sam3d_step: int = 1,
        bbox_padding: float = 0.2,
        smplx_model_path: str | None = None,
        mhr_model_path: str | None = None,
        reid_threshold: float | None = None,
        reid_match_window: int | None = None,
        rich_data_root: str | None = None,
    ):
        self.sam3d_hf_repo = sam3d_hf_repo
        self.sam3d_step = sam3d_step
        self.bbox_padding = bbox_padding
        self.smplx_model_path = smplx_model_path
        self.mhr_model_path = mhr_model_path
        self.reid_threshold = reid_threshold if reid_threshold is not None else self._REID_THRESHOLD
        self.reid_match_window = reid_match_window if reid_match_window is not None else self._REID_MATCH_WINDOW
        self.rich_data_root = rich_data_root

        self._estimator: object | None = None

    @staticmethod
    def _load_rich_cam_intrinsics(rich_data_root: str, video_dir: str, img_h: int, img_w: int) -> np.ndarray | None:
        """Load and resize RICH camera intrinsics from scan_calibration XML."""
        import re, xml.etree.ElementTree as ET
        video_path = Path(video_dir)
        scene_name = video_path.parent.name          # e.g. "BBQ_001_guitar"
        cam_id = video_path.name                     # e.g. "cam_00"
        location = re.sub(r'_\d{3}.*', '', scene_name)  # "BBQ", "LectureHall", …
        cam_idx = int(cam_id.split('_')[-1])
        xml_path = Path(rich_data_root) / "scan_calibration" / location / "calibration" / f"{cam_idx:03d}.xml"
        if not xml_path.exists():
            return None
        tree = ET.parse(xml_path)
        for child in tree.getroot():
            if child.tag == "Intrinsics":
                vals = [float(x) for x in child.find("data").text.split()]
                K = np.array(vals, dtype=np.float32).reshape(3, 3)
                scale = img_w / (K[0, 2] * 2)   # cx * 2 ≈ original image width
                K[0] *= scale
                K[1] *= scale
                return K
        return None

    @staticmethod
    def _is_estimated(video_dir: Path) -> bool:
        """Return True if body estimation has already been run for this video.

        Uses the existence of any person_*.npz file as the completion marker,
        which is robust to runs that were interrupted before appearance_gallery.npz
        was written.
        """
        return any((Path(video_dir) / "body_data").glob("person_*.npz"))

    def _init_sam3d(self) -> None:
        """Lazy-load the SAM3D Body estimator."""
        if self._estimator is not None:
            logging.warning("The estimator was already loaded, skipping the loading")
            return
        # Safety: if a bf16 autocast context leaked from the SAM3 segmentation step
        # (Sam3TrackerPredictor.__init__ enters one globally), operations in SAM3D Body
        # will silently run in bfloat16 and fail with "Got unsupported ScalarType BFloat16".
        if torch.is_autocast_enabled():
            raise RuntimeError(
                "bf16 autocast context is active before SAM3D loading — "
                "this will cause 'Got unsupported ScalarType BFloat16' errors. "
                "Check that _free_models() properly exited the SAM3 tracker's bf16_context."
            )
        # Force deterministic cuBLAS / cuDNN so that the DINOv3 backbone produces
        # bit-identical features regardless of how many forward passes have run
        # before this scene.  Without this, the cuBLAS workspace carries floating-
        # point residuals from prior scenes, shifting backbone outputs by ~1e-3.
        # Those shifts are enough to flip borderline within-video ReID decisions,
        # cascade into wrong gallery descriptors, and ultimately cause cross-view
        # mismatches (e.g. a cameraman stealing a foreground person's global ID).
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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

        Videos are processed in parallel across all available GPUs using a
        dynamic task queue for better load balancing.

        Parameters
        ----------
        scene : Scene
            The scene whose videos to process.
        video_dirs : dict[str, Path]
            Mapping of ``video_id`` -> output directory (as returned by
            `PersonSegmenter.segment_scene`).  Each directory must
            contain ``frames/``, ``json_data/`` directories and a mask_data.npz file.
        """
        num_gpus = torch.cuda.device_count()
        num_videos = len(scene.videos)

        param_keys = self._AGNOSTIC_KEYS + self._SMPLX_PARAM_KEYS

        if num_gpus <= 1:
            # Fallback: sequential on single GPU
            converter = None
            for video in tqdm(scene.videos, desc="SAM3D Body estimation"):
                video_dir = video_dirs[video.video_id]
                if ParametersExtractor._is_estimated(video_dir):
                    logging.info(f"  {video.video_id}: body data already exists, skipping")
                    continue
                # Lazy-load model only when at least one video needs processing
                if self._estimator is None:
                    self._init_sam3d()
                    converter = self._create_converter(self.mhr_model_path, self.smplx_model_path)
                cam_int = None
                if self.rich_data_root:
                    h, w = video.frame_resolution
                    cam_int = ParametersExtractor._load_rich_cam_intrinsics(
                        self.rich_data_root, str(video_dir), h, w
                    )
                ParametersExtractor._process_video_core(
                    self._estimator,
                    video.video_id,
                    str(video_dir),
                    self.sam3d_step,
                    self.bbox_padding,
                    param_keys,
                    frames_dir=str(video.frames_home) if video.frames_home else None,
                    reid_threshold=self.reid_threshold,
                    reid_match_window=self.reid_match_window,
                    fps=video.fps,
                    converter=converter,
                    cam_int=cam_int,
                )
                gc.collect()
                torch.cuda.empty_cache()
            return

        logging.info(f"Parallel body estimation: {num_videos} videos across {num_gpus} GPUs")

        # Free main-process estimator before spawning
        self._estimator = None
        gc.collect()
        torch.cuda.empty_cache()

        # Pre-warm the torch.hub cache for facebookresearch/dinov3 in the main
        # process. Without this, all worker processes race to clone the GitHub
        # repo and write hubconf.py simultaneously, which corrupts the cache and
        # raises "Cannot find callable dinov2_vitb14 in hubconf".
        logging.info("Pre-warming torch.hub cache for facebookresearch/dinov3 ...")
        torch.hub.list("facebookresearch/dinov3")
        logging.info("torch.hub cache ready.")

        # Dynamic task queue — workers pull tasks until they receive a None sentinel
        mp.set_start_method("spawn", force=True)
        task_queue: mp.Queue = mp.Queue()
        for video in scene.videos:
            if ParametersExtractor._is_estimated(video_dirs[video.video_id]):
                logging.info(f"  {video.video_id}: body data already exists, skipping")
                continue
            cam_int = None
            if self.rich_data_root:
                h, w = video.frame_resolution
                cam_int = ParametersExtractor._load_rich_cam_intrinsics(
                    self.rich_data_root, str(video_dirs[video.video_id]), h, w
                )
            task_queue.put((
                video.video_id,
                str(video_dirs[video.video_id]),
                str(video.frames_home) if video.frames_home else None,
                video.fps,
                cam_int,
            ))
        # If every video was already estimated, nothing to do
        if task_queue.empty():
            logging.info("All videos already estimated — skipping body estimation workers")
            return

        # One sentinel per worker signals end of work
        num_workers = min(num_gpus, num_videos)
        for _ in range(num_workers):
            task_queue.put(None)

        processes: list[tuple[int, mp.Process]] = []
        for gpu_id in range(num_workers):
            # Launch processes in parallel on the available GPUs
            p = mp.Process(
                target=ParametersExtractor._gpu_worker,
                args=(
                    gpu_id,
                    task_queue,
                    self.sam3d_hf_repo,
                    self.sam3d_step,
                    self.bbox_padding,
                    param_keys,
                    self.reid_threshold,
                    self.reid_match_window,
                    self.mhr_model_path,
                    self.smplx_model_path,
                ),
            )
            p.start()
            processes.append((gpu_id, p))

        for gpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                logging.error(
                    f"[GPU {gpu_id}] Worker process (pid={p.pid}) terminated with "
                    f"exit code {p.exitcode} — videos assigned to this GPU may be "
                    f"incomplete or missing"
                )

    @staticmethod
    def _gpu_worker(
        gpu_id: int,
        task_queue: mp.Queue,
        sam3d_hf_repo: str,
        sam3d_step: int,
        bbox_padding: float,
        param_keys: tuple[str, ...],
        reid_threshold: float = 0.65,
        reid_match_window: int = 5,
        mhr_model_path: str | None = None,
        smplx_model_path: str | None = None,
    ) -> None:
        """Worker process: load SAM3D once, then consume videos from the queue.

        Receives a None sentinel to stop.
        """
        torch.cuda.set_device(gpu_id)
        gpu_label = f"[GPU {gpu_id}] "

        # With mp.set_start_method("spawn") the child process starts fresh.
        # The editable-install .pth file appends sam-3d-body/ AFTER
        # site-packages/, where detectron2 installs its own tools/ package.
        # We must insert sam-3d-body/ at position 0 so our tools/ wins.
        import sys as _sys
        _sam3d_root = str(Path(__file__).resolve().parents[1] / "sam-3d-body")
        if _sam3d_root in _sys.path:
            _sys.path.remove(_sam3d_root)
        _sys.path.insert(0, _sam3d_root)

        # Mirror the determinism settings from _init_sam3d() so that cuBLAS
        # workspace residuals do not accumulate across videos processed
        # sequentially by the same worker.  Without this, DINOv3 features
        # for the N-th video in a worker differ by ~1e-3 from a fresh run,
        # which can flip borderline within-video ReID merge decisions.
        import os as _os
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            logging.info(f"{gpu_label}Loading SAM3D...")
            try:
                from notebook.utils import setup_sam_3d_body
            except ImportError as e:
                raise ImportError(
                    "sam-3d-body package not installed. "
                    "Please install it and ensure notebook.utils is available."
                ) from e
            estimator = setup_sam_3d_body(hf_repo_id=sam3d_hf_repo)
            logging.info(f"{gpu_label}SAM3D loaded.")

            converter = ParametersExtractor._create_converter(mhr_model_path, smplx_model_path)
            logging.info(f"{gpu_label}SMPLX converter initialised.")

            # finish the queue, until a None trigger is hit
            while True:
                task = task_queue.get()  # blocks until a task is available
                if task is None:
                    break

                video_id, video_dir, frames_dir, video_fps, cam_int = task
                logging.info(f"{gpu_label}Processing {video_id}")
                try:
                    ParametersExtractor._process_video_core(
                        estimator,
                        video_id,
                        video_dir,
                        sam3d_step,
                        bbox_padding,
                        param_keys,
                        frames_dir=frames_dir,
                        gpu_label=gpu_label,
                        reid_threshold=reid_threshold,
                        reid_match_window=reid_match_window,
                        fps=video_fps,
                        converter=converter,
                        cam_int=cam_int,
                    )
                except Exception as e:
                    logging.error(
                        f"{gpu_label}Error processing {video_id}: {e}", exc_info=True
                    )

                gc.collect()
                torch.cuda.empty_cache()

            del estimator
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"{gpu_label}Worker done.")
        except Exception:
            logging.error(
                f"{gpu_label}Worker crashed during initialisation or processing",
                exc_info=True,
            )
            raise

    # Model-agnostic SAM3D output keys saved regardless of conversion.
    _AGNOSTIC_KEYS: tuple[str, ...] = (
        "pred_keypoints_3d",
        "pred_keypoints_2d",
        "pred_cam_t",
        "focal_length",
        "pred_joint_confidence",
    )

    # SMPLX parameter keys produced by the Conversion class.
    _SMPLX_PARAM_KEYS: tuple[str, ...] = (
        "smplx_betas",
        "smplx_body_pose",
        "smplx_global_orient",
        "smplx_transl",
        "smplx_left_hand_pose",
        "smplx_right_hand_pose",
        "smplx_expression",
    )

    @staticmethod
    def _process_video_core(
        estimator,
        video_id: str,
        video_dir: str,
        sam3d_step: int,
        bbox_padding: float,
        param_keys: tuple[str, ...],
        frames_dir: str | None = None,
        gpu_label: str = "",
        reid_threshold: float = 0.65,
        reid_match_window: int = 5,
        fps: float = 15.0,
        converter,
        cam_int: np.ndarray | None = None,
    ) -> None:
        """Process all frames of one video with batched per-frame inference.

        All persons detected in a single frame are forwarded through SAM3D in
        one call.
        """

        # create the output directories
        video_path = Path(video_dir)
        json_dir = video_path / "json_data"
        # Frames live co-located with the source data, not inside video_dir.
        # frames_dir is the canonical data/<scene>/<video_id>/frames/ path.
        frame_dir = Path(frames_dir) if frames_dir else video_path / "frames"
        body_dir = video_path / "body_data"
        body_dir.mkdir(exist_ok=True)

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            logging.warning(f"{gpu_label}{video_id}: no JSON data, skipping")
            return

        # Open the SAM3 mask archive once for the whole video.
        # The archive holds one uint16 frame per key (pixel value = person ID).
        mask_npz_path = video_path / "mask_data.npz"
        _mask_zip: zipfile.ZipFile | None = None
        if mask_npz_path.exists():
            _mask_zip = zipfile.ZipFile(str(mask_npz_path), "r")
        else:
            logging.warning(f"{gpu_label}{video_id}: mask_data.npz not found — confidence will be all-ones")

        confidence_estimator = ConfidenceEstimator()

        tracks: dict[int, dict[int, dict]] = {}

        reidentifier = InVideoReidentifier(
            reid_threshold=reid_threshold,
            reid_match_window=reid_match_window,
            fps=fps,
            gpu_label=gpu_label,
            video_id=video_id,
        )
        reidentifier.build_covisibility(json_files)

        # Pass 2: batched MHR→SMPL-X conversion after the frame loop.
        # Accumulate per-person SAM3D outputs during the frame loop, then
        # convert the entire track at once (one optimizer run per person).
        # canonical_id → list of (frame_idx, body_output, orig_person_id, mask_key, img_h, img_w)
        pending_conversion: dict[int, list] = {}

        for json_path in tqdm(
            json_files, desc=f"{gpu_label}SAM3D {video_id}", leave=False
        ):
            # Load frame idx and bounding boxes

            frame_idx_str = json_path.stem.replace("mask_", "")
            frame_idx = int(frame_idx_str.split("_")[0])

            if sam3d_step > 1 and frame_idx % sam3d_step != 0:
                continue

            with open(json_path) as f:
                meta = json.load(f)

            labels = meta.get("labels", {})
            if not labels:
                continue

            frame_path = next(
                (frame_dir / f"{frame_idx_str}{ext}"
                 for ext in (".jpg", ".jpeg", ".png", ".bmp")
                 if (frame_dir / f"{frame_idx_str}{ext}").exists()),
                None,
            )
            if frame_path is None:
                continue
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame_rgb.shape[:2]

            # Load the full segmentation mask for this frame (uint16, pixel = person_id).
            mask_key = json_path.stem + ".npy"
            frame_mask: np.ndarray | None = None
            if _mask_zip is not None:
                if mask_key in _mask_zip.namelist():
                    with _mask_zip.open(mask_key) as _mf:
                        frame_mask = np.load(io.BytesIO(_mf.read()))

            # Collect all valid persons for this frame.
            # Each entry: (person_id, padded_x1, padded_y1, padded_x2, padded_y2,
            #               orig_x1, orig_y1, orig_x2, orig_y2)
            valid_persons = []
            for str_id, info in labels.items():
                person_id = int(str_id)
                x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

                # skip bboxes that are too small
                bw, bh = x2 - x1, y2 - y1
                if bw < 10 or bh < 10:
                    continue

                # Skip degenerate detections whose bbox covers most of
                # the frame — these are remnants of flood-fill masks and
                # would produce scene-level features that corrupt re-ID.
                if bw * bh > 0.80 * img_w * img_h:
                    logging.warning(
                        f"{gpu_label}Skipping degenerate bbox for person "
                        f"{person_id} in frame {frame_idx}: "
                        f"{bw}x{bh} covers "
                        f"{100*bw*bh/(img_w*img_h):.1f}% "
                        f"of {img_w}x{img_h}"
                    )
                    continue
                
                # pad the bboxes
                pad_w = int(bw * bbox_padding)
                pad_h = int(bh * bbox_padding)
                px1 = max(0, x1 - pad_w)
                py1 = max(0, y1 - pad_h)
                px2 = min(img_w, x2 + pad_w)
                py2 = min(img_h, y2 + pad_h)
                valid_persons.append(
                    (person_id, px1, py1, px2, py2, x1, y1, x2, y2)
                )

            if not valid_persons:
                continue

            # Single batched forward pass for all persons in this frame.
            bboxes_arr = np.array(
                [[p[1], p[2], p[3], p[4]] for p in valid_persons], # pass the bbox coordinates
                dtype=np.float32,
            )

            # Hook into the DINOv3 ViT norm layer to capture the CLS token for re-ID.
            # Dinov3Backbone.forward() calls encoder.get_intermediate_layers(..., norm=True)
            # which applies encoder.norm to the full token sequence (B, N_tokens, D) before
            # slicing off patch tokens.  Token 0 is always the CLS token in DINOv2/DINOv3 ViTs.
            # We hook here (rather than on backbone itself) because the backbone's forward()
            # discards the CLS token before returning, exposing only the patch tokens.
            # We guard on _hook_feats to ignore subsequent calls from hand passes.
            _hook_feats: list[np.ndarray] = []

            def _backbone_hook(_module, _input, output):
                if _hook_feats:   # ignore subsequent hand-branch calls
                    return
                # output: (N_persons, N_tokens, D) — token 0 is CLS
                cls = output[:, 0, :].float()  # (N_persons, D)
                cls = cls / cls.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                _hook_feats.append(cls.detach().cpu().numpy())

            try:
                _hook_handle = estimator.model.backbone.encoder.norm.register_forward_hook(
                    _backbone_hook
                )
            except AttributeError:
                logging.error("estimator doesn't have a model.backbone.encoder.norm module")
                _hook_handle = None

            try:
                cam_int_t = (
                    torch.tensor(cam_int[None], dtype=torch.float32)
                    if cam_int is not None else None
                )
                outputs = estimator.process_one_image(
                    frame_rgb, bboxes=bboxes_arr, cam_int=cam_int_t
                )
            except Exception as e:
                logging.error(
                    f"{gpu_label}SAM3D failed frame {frame_idx} in {video_id}: {e}, returning None"
                )
                outputs = None
            finally:
                # Finally free the hook
                if _hook_handle is not None:
                    _hook_handle.remove()

            if not outputs:
                if outputs is not None:
                    logging.error(
                        f"{gpu_label}No outputs for frame {frame_idx} in {video_id}"
                    )
                continue

            # vis_feats[i] is the L2-normalised backbone descriptor for valid_persons[i].
            vis_feats: np.ndarray | None = _hook_feats[0] if _hook_feats else None

            # outputs[i] corresponds to valid_persons[i] (same order as bboxes_arr).
            # SMPL-X conversion is deferred to Pass 2 (batched per person track).
            for i, (person_id, _, _, _, _, x1, y1, x2, y2) in enumerate(
                valid_persons
            ):
                if i >= len(outputs):
                    break
                body = outputs[i]
                if body is None:
                    continue

                # Visual re-identification
                feat_i = (
                    vis_feats[i]
                    if vis_feats is not None and i < len(vis_feats)
                    else None
                )

                params = {"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)}

                # Always keep model-agnostic SAM3D outputs.
                for key in ParametersExtractor._AGNOSTIC_KEYS:
                    if key in body:
                        val = body[key]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        params[key] = np.asarray(val, dtype=np.float32)

                # --- Per-joint confidence (MHR-based; overwritten by Pass 2 when converter present) ---
                # Computed before ReID so the scalar mean can weight gallery entries.
                kp3d = params.get("pred_keypoints_3d")
                kp2d = params.get("pred_keypoints_2d")
                cam_t = params.get("pred_cam_t")
                fl = params.get("focal_length")
                fl_val = float(fl) if fl is not None and np.ndim(fl) == 0 else (float(fl[0]) if fl is not None else None)

                verts_raw = body.get("pred_vertices")
                if isinstance(verts_raw, torch.Tensor):
                    verts_raw = verts_raw.detach().cpu().numpy()
                if (
                    kp3d is not None
                    and kp2d is not None
                    and cam_t is not None
                    and fl_val is not None
                    and verts_raw is not None
                    and frame_mask is not None
                ):
                    person_mask = (frame_mask == person_id).astype(np.uint8)
                    try:
                        params["pred_joint_confidence"] = confidence_estimator.estimate(
                            pred_vertices=np.asarray(verts_raw, dtype=np.float32),
                            pred_keypoints_3d=kp3d,
                            pred_keypoints_2d=kp2d,
                            pred_cam_t=cam_t,
                            focal_length=fl_val,
                            person_mask=person_mask,
                            img_h=img_h,
                            img_w=img_w,
                        )
                    except Exception as _ce:
                        logging.warning(
                            f"{gpu_label}Confidence estimation failed frame "
                            f"{frame_idx} person {person_id}: {_ce}"
                        )
                elif kp3d is not None:
                    params["pred_joint_confidence"] = np.ones(
                        kp3d.shape[0], dtype=np.float32
                    )

                reid_confidence: float | None = None
                _conf_arr = params.get("pred_joint_confidence")
                if _conf_arr is not None:
                    reid_confidence = float(_conf_arr.mean())

                # Extract raw MHR shape params for betas-based reid signal.
                betas_i: np.ndarray | None = None
                _sp = body.get("shape_params")
                if _sp is not None:
                    if isinstance(_sp, torch.Tensor):
                        _sp = _sp.detach().cpu().numpy()
                    _sp = np.asarray(_sp, dtype=np.float32).ravel()
                    _norm = np.linalg.norm(_sp)
                    if _norm > 0:
                        betas_i = _sp / _norm

                canonical_id: int = reidentifier.process_detection(
                    person_id, feat_i, frame_idx, valid_persons,
                    confidence=reid_confidence, betas=betas_i,
                )

                tracks.setdefault(canonical_id, {})[frame_idx] = params

                # Accumulate raw SAM3D output for batched SMPL-X conversion in Pass 2.
                # Move tensors to CPU now to avoid holding GPU memory for the whole loop.
                body_cpu = {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in body.items()
                }
                pending_conversion.setdefault(canonical_id, []).append(
                    (frame_idx, body_cpu, person_id, mask_key, img_h, img_w)
                )

        # Pass 2: batched SMPL-X conversion + SMPL-X-based confidence override.
        # One optimizer run per person track (all frames batched together).
        if pending_conversion:
            _smplx_device = next(converter._smpl_model.parameters()).device
            for canonical_id, frame_list in pending_conversion.items():
                frame_list.sort(key=lambda x: x[0])
                sam3d_outputs = [entry[1] for entry in frame_list]
                try:
                    conv_result = converter.convert_sam3d_output_to_smpl(
                        sam3d_outputs=sam3d_outputs,
                        return_smpl_meshes=False,
                        return_smpl_parameters=True,
                        return_smpl_vertices=True,
                        return_fitting_errors=False,
                        batch_size=64,
                    )
                except Exception as _e:
                    logging.warning(
                        f"{gpu_label}Batched SMPLX conversion failed person "
                        f"{canonical_id} in {video_id}: {_e}",
                        exc_info=True,
                    )
                    continue

                # Collect smplx params as numpy, indexed by position in frame_list.
                smplx_np: dict[str, np.ndarray] = {}
                if conv_result.result_parameters is not None:
                    for _pk, _pv in conv_result.result_parameters.items():
                        if isinstance(_pv, torch.Tensor):
                            smplx_np[f"smplx_{_pk}"] = _pv.detach().cpu().numpy()
                        else:
                            smplx_np[f"smplx_{_pk}"] = np.asarray(_pv, dtype=np.float32)

                # Copy vertices to numpy now, then free the GPU tensor immediately.
                # result_vertices is (N, 10475, 3) on GPU — ~147 MB for 1166 frames.
                # Holding it while running the joints forward below causes OOM.
                smplx_verts_np: np.ndarray | None = None
                if conv_result.result_vertices is not None:
                    _rv = conv_result.result_vertices
                    if isinstance(_rv, torch.Tensor):
                        smplx_verts_np = _rv.detach().cpu().numpy().astype(np.float32)
                    else:
                        smplx_verts_np = np.asarray(_rv, dtype=np.float32)
                del conv_result
                torch.cuda.empty_cache()

                # Batch SMPL-X forward for joints, in chunks to avoid OOM.
                _CHUNK = 128
                smplx_joints_batch: np.ndarray | None = None
                if "smplx_betas" in smplx_np:
                    try:
                        _betas_np = smplx_np["smplx_betas"]
                        _body_pose_np = smplx_np["smplx_body_pose"]
                        N = _body_pose_np.shape[0]
                        # betas may be (10,) or (1, 10) — broadcast to (N, 10) in numpy.
                        if _betas_np.ndim == 1:
                            _betas_np = _betas_np[None]
                        if _betas_np.shape[0] == 1 and N > 1:
                            _betas_np = np.broadcast_to(_betas_np, (N, _betas_np.shape[1])).copy()

                        _lhp_np = smplx_np.get("smplx_left_hand_pose")
                        _rhp_np = smplx_np.get("smplx_right_hand_pose")
                        _expr_np = smplx_np.get("smplx_expression")

                        chunks = []
                        with torch.no_grad():
                            for _s in range(0, N, _CHUNK):
                                _e = min(_s + _CHUNK, N)
                                _c = _e - _s
                                _b = torch.from_numpy(_betas_np[_s:_e]).float().to(_smplx_device)
                                _bp = torch.from_numpy(_body_pose_np[_s:_e]).float().to(_smplx_device)
                                _go = torch.from_numpy(smplx_np["smplx_global_orient"][_s:_e]).float().to(_smplx_device)
                                _tr = torch.from_numpy(smplx_np["smplx_transl"][_s:_e]).float().to(_smplx_device)
                                _lh = torch.from_numpy(_lhp_np[_s:_e]).float().to(_smplx_device) if _lhp_np is not None else torch.zeros(_c, converter._hand_pose_dim, device=_smplx_device)
                                _rh = torch.from_numpy(_rhp_np[_s:_e]).float().to(_smplx_device) if _rhp_np is not None else torch.zeros(_c, converter._hand_pose_dim, device=_smplx_device)
                                _ex = torch.from_numpy(_expr_np[_s:_e]).float().to(_smplx_device) if _expr_np is not None else torch.zeros(_c, converter._smpl_model.num_expression_coeffs, device=_smplx_device)
                                _z1 = torch.zeros(_c, 1, 3, device=_smplx_device)
                                _out = converter._smpl_model(
                                    betas=_b,
                                    body_pose=_bp,
                                    global_orient=_go,
                                    transl=_tr,
                                    jaw_pose=_z1,
                                    leye_pose=_z1,
                                    reye_pose=_z1,
                                    left_hand_pose=_lh,
                                    right_hand_pose=_rh,
                                    expression=_ex,
                                )
                                chunks.append(_out.joints[:, :55].cpu().numpy().astype(np.float32))
                        smplx_joints_batch = np.concatenate(chunks, axis=0)
                    except Exception as _je:
                        logging.warning(
                            f"{gpu_label}Batched SMPLX joints failed person "
                            f"{canonical_id} in {video_id}: {_je}",
                            exc_info=True,
                        )

                for j, (frame_idx, _body, orig_pid, f_mask_key, img_h, img_w) in enumerate(frame_list):
                    frame_params = tracks.get(canonical_id, {}).get(frame_idx)
                    if frame_params is None:
                        continue

                    # Store per-frame smplx params.
                    for _pk, _pv_all in smplx_np.items():
                        frame_params[_pk] = np.asarray(_pv_all[j], dtype=np.float32)

                    # Override confidence with SMPL-X vertices + joints.
                    if (
                        smplx_verts_np is not None
                        and smplx_joints_batch is not None
                        and _mask_zip is not None
                        and f_mask_key in _mask_zip.namelist()
                    ):
                        smplx_verts_cam = smplx_verts_np[j]
                        smplx_joints_cam = smplx_joints_batch[j]  # (55, 3) camera space

                        kp3d = frame_params.get("pred_keypoints_3d")
                        kp2d = frame_params.get("pred_keypoints_2d")
                        cam_t = frame_params.get("pred_cam_t")
                        fl = frame_params.get("focal_length")
                        fl_val = (
                            float(fl) if fl is not None and np.ndim(fl) == 0
                            else (float(fl[0]) if fl is not None else None)
                        )

                        if kp3d is not None and cam_t is not None and kp2d is not None and fl_val is not None:
                            mhr_kpts_cam = kp3d + cam_t
                            cx, cy = ConfidenceEstimator._recover_principal_point(
                                mhr_kpts_cam, kp2d, fl_val
                            )
                            try:
                                with _mask_zip.open(f_mask_key) as _mf2:
                                    f_frame_mask = np.load(io.BytesIO(_mf2.read()))
                                person_mask = (f_frame_mask == orig_pid).astype(np.uint8)
                                frame_params["pred_joint_confidence"] = confidence_estimator.estimate(
                                    pred_vertices=smplx_verts_cam,
                                    pred_keypoints_3d=smplx_joints_cam,
                                    pred_keypoints_2d=None,
                                    pred_cam_t=np.zeros(3, dtype=np.float32),
                                    focal_length=fl_val,
                                    person_mask=person_mask,
                                    img_h=img_h,
                                    img_w=img_w,
                                    cx_cy=(cx, cy),
                                )
                            except Exception as _ce:
                                logging.warning(
                                    f"{gpu_label}SMPLX confidence failed person "
                                    f"{canonical_id} frame {frame_idx}: {_ce}"
                                )

        if _mask_zip is not None:
            _mask_zip.close()

        # Expose reidentifier state for post-loop gallery saving and remap.
        person_feat_buffer = reidentifier.feature_buffer
        id_remap = reidentifier.id_remap

        if not tracks:
            logging.warning(f"{gpu_label}{video_id}: no body detections")
            return

        # Prune tracks where SAM3D failed to produce body params in most
        # frames.  A hand, phone, or other non-person object may survive the
        # segmentation fill-ratio filter (e.g. the propagated bbox grows but
        # the mask stays tiny) yet SAM3D will still fail to fit a real body
        # model to it.  Require ≥30% of a track's frames to have valid 3D
        # keypoints; tracks below this are almost certainly not people.
        _MIN_BODY_HIT_RATIO = 0.30
        pruned_ids: list[int] = []
        for pid, frames in list(tracks.items()):
            n_total = len(frames)
            n_valid = sum(
                1 for fdata in frames.values()
                if "pred_keypoints_3d" in fdata
            )
            if n_total > 0 and (n_valid / n_total) < _MIN_BODY_HIT_RATIO:
                pruned_ids.append(pid)
                del tracks[pid]
        if pruned_ids:
            logging.info(
                f"{gpu_label}{video_id}: pruned {len(pruned_ids)} non-person "
                f"track(s) with low body-fit rate: {pruned_ids}"
            )

        if not tracks:
            logging.warning(
                f"{gpu_label}{video_id}: no body detections remain after pruning"
            )
            return

        ParametersExtractor._save_body_data_static(
            tracks, body_dir, video_id, param_keys
        )

        # Persist per-person appearance feature matrices for cross-view re-ID.
        # Each person gets a (N, D) matrix of L2-normalised DINOv3 features,
        # one row per frame that passes an adaptive per-track confidence threshold.
        # Keeping individual frames (instead of collapsing to a mean vector)
        # lets the matcher use Chamfer similarity, which is robust to occlusion:
        # even if most frames of a tracklet are partially occluded, the best
        # matching pair across two cameras still produces a reliable score.
        if person_feat_buffer:
            gallery_arrays: dict[str, np.ndarray] = {}
            for pid, feat_list in person_feat_buffer.items():
                pid_frames = tracks.get(pid, {})

                # Collect (feat, scalar_conf) pairs for all frames.
                all_feats: list[np.ndarray] = []
                all_confs: list[float] = []
                for fi, feat in feat_list:
                    conf = pid_frames.get(fi, {}).get("pred_joint_confidence")
                    scalar_conf = float(np.mean(conf)) if conf is not None else 1.0
                    all_feats.append(feat)
                    all_confs.append(scalar_conf)

                # Adaptive threshold: anchor each track to its own peak confidence
                # so that background people (whose Z-buffer confidence is capped low
                # due to vertex crowding at small image scales) are not starved of
                # gallery frames by a global absolute threshold.
                # The 0.15 floor prevents degenerate near-zero peaks from setting a
                # negative threshold.
                peak_conf = max(all_confs)
                dynamic_threshold = max(0.15, peak_conf - 0.20)

                filtered_feats = [f for f, c in zip(all_feats, all_confs) if c >= dynamic_threshold]
                filtered_confs = [c for c in all_confs if c >= dynamic_threshold]

                # If the gallery has fewer than 3 frames (e.g. one lucky high-conf
                # outlier sets a strict threshold), relax by 0.05 steps until we
                # collect at least 3 frames or exhaust the buffer.
                _MIN_GALLERY = 3
                relaxed_threshold = dynamic_threshold
                while len(filtered_feats) < _MIN_GALLERY and relaxed_threshold > 0.05:
                    relaxed_threshold -= 0.05
                    filtered_feats = [f for f, c in zip(all_feats, all_confs) if c >= relaxed_threshold]
                    filtered_confs = [c for c in all_confs if c >= relaxed_threshold]

                # Last resort: use all frames (can happen for very short tracks).
                if not filtered_feats:
                    filtered_feats = all_feats
                    filtered_confs = all_confs

                # Store feature matrix (N, D) and per-frame confidence weights (N,).
                # The matcher uses confidence-weighted Chamfer similarity so that
                # clearly-visible frames contribute more than occluded ones.
                gallery_arrays[str(pid)] = np.stack(filtered_feats)                    # (N, D)
                gallery_arrays[f"{pid}_conf"] = np.array(filtered_confs, dtype=np.float32)  # (N,)

            gallery_path = body_dir / "appearance_gallery.npz"
            np.savez(str(gallery_path), **gallery_arrays)

        # Persist and apply the within-video re-ID mapping so that
        # mask_data.npz and json_data/ stay consistent with body_data/.
        if id_remap:
            reid_path = body_dir / "reid_id_mapping.json"
            with open(reid_path, "w") as f:
                json.dump({str(k): v for k, v in id_remap.items()}, f, indent=2)
            print(
                f"  {video_id}: re-ID merged {len(id_remap)} SAM2 track(s) "
                f"→ updating masks and JSON metadata"
            )
            CrossVideoReidentifier.apply_reid_remap(
                video_path, id_remap, gpu_label
            )

    @staticmethod
    def _save_body_data_static(
        tracks: dict[int, dict[int, dict]],
        body_dir: Path,
        video_id: str,
        param_keys: tuple[str, ...],
    ) -> None:
        """Save per-person .npz files and a summary JSON (static)."""
        summary = {"video_id": video_id, "persons": {}}

        for person_id, frames in tracks.items():
            sorted_idxs = sorted(frames.keys())
            n = len(sorted_idxs)
            if n == 0:
                continue

            arrays: dict[str, np.ndarray] = {
                "frame_indices": np.array(sorted_idxs, dtype=np.int32),
            }

            all_keys = set()
            for fi in sorted_idxs:
                all_keys.update(frames[fi].keys())

            for key in list(param_keys) + ["bbox"]:
                if key not in all_keys:
                    continue
                vals = []
                for fi in sorted_idxs:
                    v = frames[fi].get(key)
                    if v is not None:
                        vals.append(v)
                    else:
                        ref = next(
                            (frames[fj][key] for fj in sorted_idxs if key in frames[fj]),
                            None,
                        )
                        vals.append(np.zeros_like(ref) if ref is not None else None)
                if any(v is None for v in vals):
                    continue
                arrays[key] = np.stack(vals, axis=0)

            npz_path = body_dir / f"person_{person_id}.npz"
            np.savez(str(npz_path), **arrays)

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

    @staticmethod
    def _create_converter(mhr_model_path: str | None, smplx_model_path: str | None):
        """Create and return a Conversion object."""
        if not mhr_model_path or not smplx_model_path:
            raise RuntimeError(
                "mhr_model_path and smplx_model_path must both be set — "
                "check configuration/config.yaml"
            )
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mhr_model = MHR.from_files(folder=Path(mhr_model_path), lod=1, device=_device)
        _sp = Path(smplx_model_path)
        _smplx_kwargs = (
            {"model_path": str(_sp), "ext": _sp.suffix.lstrip(".")}
            if _sp.is_file() else
            {"model_path": smplx_model_path, "ext": "pkl"}
        )
        smplx_model = smplx.create(
            **_smplx_kwargs,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            batch_size=1,
        )
        return Conversion(
            mhr_model=mhr_model,
            smpl_model=smplx_model,
            method="pytorch",
        )


