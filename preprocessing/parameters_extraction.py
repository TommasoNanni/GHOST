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

class BodyParameterEstimator:
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

    # Re-identification defaults — overridden by constructor args (from config).
    _REID_THRESHOLD: float = 0.65
    _GALLERY_EMA_ALPHA: float = 0.9
    _REID_MATCH_WINDOW: int = 5

    def __init__(
        self,
        sam3d_hf_repo: str = "facebook/sam-3d-body-dinov3",
        sam3d_step: int = 1,
        bbox_padding: float = 0.2,
        smplx_model_path: str | None = None,
        mhr_model_path: str | None = None,
        reid_threshold: float | None = None,
        gallery_ema_alpha: float | None = None,
        reid_match_window: int | None = None,
    ):
        self.sam3d_hf_repo = sam3d_hf_repo
        self.sam3d_step = sam3d_step
        self.bbox_padding = bbox_padding
        self.smplx_model_path = smplx_model_path
        self.mhr_model_path = mhr_model_path
        self.reid_threshold = reid_threshold if reid_threshold is not None else self._REID_THRESHOLD
        self.gallery_ema_alpha = gallery_ema_alpha if gallery_ema_alpha is not None else self._GALLERY_EMA_ALPHA
        self.reid_match_window = reid_match_window if reid_match_window is not None else self._REID_MATCH_WINDOW

        self._estimator: object | None = None
        self._converter: object | None = None

    @staticmethod
    def _is_estimated(video_dir: Path) -> bool:
        """Return True if body estimation has already been run for this video.

        Uses ``appearance_gallery.npz`` as the completion marker because it is
        written last in ``_process_video_core``.
        """
        return (Path(video_dir) / "body_data" / "appearance_gallery.npz").exists()

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

        # When an MHR->SMPLX converter is configured, save only SMPL-X keys
        # plus the model-agnostic geometric outputs.  MHR-native params
        # (body_pose_params, shape_params, global_rot, etc.) are discarded.
        param_keys = (
            self._AGNOSTIC_KEYS + self._SMPLX_PARAM_KEYS
            if (self.mhr_model_path is not None and self.smplx_model_path is not None)
            else self._PARAM_KEYS
         )

        if num_gpus <= 1:
            # Fallback: sequential on single GPU
            self._init_sam3d()
            converter = self._create_converter(self.mhr_model_path, self.smplx_model_path)
            for video in tqdm(scene.videos, desc="SAM3D Body estimation"):
                video_dir = video_dirs[video.video_id]
                if BodyParameterEstimator._is_estimated(video_dir):
                    logging.info(f"  {video.video_id}: body data already exists, skipping")
                    continue
                BodyParameterEstimator._process_video_core(
                    self._estimator,
                    video.video_id,
                    str(video_dir),
                    self.sam3d_step,
                    self.bbox_padding,
                    param_keys,
                    frames_dir=str(video.frames_home) if video.frames_home else None,
                    reid_threshold=self.reid_threshold,
                    gallery_ema_alpha=self.gallery_ema_alpha,
                    reid_match_window=self.reid_match_window,
                    converter=converter,
                )
                gc.collect()
                torch.cuda.empty_cache()
            return

        logging.info(f"Parallel body estimation: {num_videos} videos across {num_gpus} GPUs")

        # Free main-process estimator before spawning
        self._estimator = None
        gc.collect()
        torch.cuda.empty_cache()

        # Dynamic task queue — workers pull tasks until they receive a None sentinel
        mp.set_start_method("spawn", force=True)
        task_queue: mp.Queue = mp.Queue()
        for video in scene.videos:
            if BodyParameterEstimator._is_estimated(video_dirs[video.video_id]):
                logging.info(f"  {video.video_id}: body data already exists, skipping")
                continue
            task_queue.put((
                video.video_id,
                str(video_dirs[video.video_id]),
                str(video.frames_home) if video.frames_home else None,
            ))
        # If every video was already estimated, nothing to do
        if task_queue.empty():
            logging.info("All videos already estimated — skipping body estimation workers")
            return

        # One sentinel per worker signals end of work
        num_workers = min(num_gpus, num_videos)
        for _ in range(num_workers):
            task_queue.put(None)

        processes = []
        for gpu_id in range(num_workers):
            # Launch processes in parallel on the available GPUs
            p = mp.Process(
                target=BodyParameterEstimator._gpu_worker,
                args=(
                    gpu_id,
                    task_queue,
                    self.sam3d_hf_repo,
                    self.sam3d_step,
                    self.bbox_padding,
                    param_keys,
                    self.reid_threshold,
                    self.gallery_ema_alpha,
                    self.reid_match_window,
                    self.mhr_model_path,
                    self.smplx_model_path,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    @staticmethod
    def _gpu_worker(
        gpu_id: int,
        task_queue: mp.Queue,
        sam3d_hf_repo: str,
        sam3d_step: int,
        bbox_padding: float,
        param_keys: tuple[str, ...],
        reid_threshold: float = 0.65,
        gallery_ema_alpha: float = 0.9,
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

        converter = BodyParameterEstimator._create_converter(mhr_model_path, smplx_model_path)
        if converter is not None:
            logging.info(f"{gpu_label}SMPLX converter initialised.")

        # finish the queue, until a None trigger is hit
        while True:
            task = task_queue.get()  # blocks until a task is available
            if task is None:
                break

            video_id, video_dir, frames_dir = task
            logging.info(f"{gpu_label}Processing {video_id}")
            try:
                BodyParameterEstimator._process_video_core(
                    estimator,
                    video_id,
                    video_dir,
                    sam3d_step,
                    bbox_padding,
                    param_keys,
                    frames_dir=frames_dir,
                    gpu_label=gpu_label,
                    reid_threshold=reid_threshold,
                    gallery_ema_alpha=gallery_ema_alpha,
                    reid_match_window=reid_match_window,
                    converter=converter,
                )
            except Exception as e:
                logging.error(f"{gpu_label}Error processing {video_id}: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        del estimator
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"{gpu_label}Worker done.")

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
        gallery_ema_alpha: float = 0.9,
        reid_match_window: int = 5,
        converter=None,
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

        # Phase 0 — pre-scan every JSON frame to find which SAM3 track IDs are
        # ever SIMULTANEOUSLY visible.  ID pairs co-visible in >= N frames must
        # belong to different people and can never be merged by re-ID.
        # Using a threshold > 1 avoids blocking re-ID across detection-transition
        # frames where SAM3 briefly emits both the old and the new track ID
        # (e.g. the frame where one person's track fades while a new one
        # initialises), which would otherwise permanently block valid merges.
        _COVISIBILITY_THRESHOLD = 3
        _covisibility_counter: Counter = Counter()
        for jp in json_files:
            with open(jp) as _f:
                _meta = json.load(_f)
            _ids = sorted(int(s) for s in _meta.get("labels", {}).keys())
            for _ai, _a in enumerate(_ids):
                for _b in _ids[_ai + 1:]:
                    _covisibility_counter[frozenset({_a, _b})] += 1
        covisible_ids: set[frozenset] = {
            pair for pair, count in _covisibility_counter.items()
            if count >= _COVISIBILITY_THRESHOLD
        }

        # Gallery for visual re-identification across SAM2 track interruptions.
        # person_gallery: canonical_id → L2-normalised appearance descriptor (EMA).
        #   Used live during within-video track merging.
        # person_feat_buffer: canonical_id → list of (frame_idx, feat) pairs.
        #   Collects all confirmed appearance features; at save time the top-K
        #   highest-confidence frames are used to build the cross-view gallery
        #   (confidence-weighted mean), which is more robust than a flat EMA.
        # id_remap: raw SAM3 id → canonical_id for ids that were re-identified.
        # pending_reid: new SAM3 ids that weren't matched on first sight —
        #   maps person_id → frames_remaining for retry attempts.
        # We employ DINOv3 backbone (given by SAM 3D) in order to match people across
        # frames using cosine similarity between visual features
        person_gallery: dict[int, np.ndarray] = {}
        person_feat_buffer: dict[int, list[tuple[int, np.ndarray]]] = {}
        id_remap: dict[int, int] = {}
        pending_reid: dict[int, int] = {}  # person_id → frames remaining

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

            frame_path = frame_dir / f"{frame_idx_str}.jpg"
            if not frame_path.exists():
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
                outputs = estimator.process_one_image(
                    frame_rgb, bboxes=bboxes_arr
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
                canonical_id: int = id_remap.get(person_id, person_id)

                if feat_i is not None:
                    # Determine whether to attempt ReID matching for this track.
                    # Brand-new tracks get matched immediately. If the first
                    # match fails, the track enters a retry window of
                    # `reid_match_window` frames before being finalised as a
                    # genuinely new person.
                    should_try_reid = False
                    if person_id not in person_gallery and person_id not in id_remap:
                        # First time seeing this SAM2 track
                        should_try_reid = True
                        if person_id not in pending_reid:
                            pending_reid[person_id] = reid_match_window
                    elif person_id in pending_reid:
                        # Still within the retry window
                        should_try_reid = True

                    if should_try_reid and person_gallery:
                        # Canonical IDs already present in THIS frame — we must
                        # never merge into one of them (they are a different
                        # person who is simultaneously visible).
                        frame_canonical_ids = {
                            id_remap.get(p[0], p[0]) for p in valid_persons
                        }

                        sims = {
                            pid: float(np.dot(feat_i, gfeat))
                            for pid, gfeat in person_gallery.items()
                            if pid not in frame_canonical_ids
                        }
                        if sims:
                            best_id = max(sims, key=lambda pid: sims[pid])
                        else:
                            best_id = None
                        # Match only if similarity is high enough AND the two SAM2
                        # track IDs were never seen in the same frame (co-visible
                        # IDs must belong to different people).
                        if (
                            best_id is not None
                            and sims[best_id] >= reid_threshold
                            and frozenset({person_id, best_id}) not in covisible_ids
                        ):
                            id_remap[person_id] = best_id
                            canonical_id = best_id
                            pending_reid.pop(person_id, None)
                            # Merge any buffered frames from the old track into
                            # the canonical person's buffer.
                            if person_id in person_feat_buffer:
                                person_feat_buffer.setdefault(best_id, []).extend(
                                    person_feat_buffer.pop(person_id)
                                )
                            logging.info(
                                f"{gpu_label}Re-ID: SAM3 id {person_id} → "
                                f"person {best_id} (sim={sims[best_id]:.3f}) "
                                f"in {video_id} frame {frame_idx}"
                            )
                        else:
                            if best_id is not None and sims:
                                covis_blocked = frozenset({person_id, best_id}) in covisible_ids
                                logging.info(
                                    f"[within-video reid] {gpu_label}{video_id} "
                                    f"frame {frame_idx}  SAM3 id {person_id} → "
                                    f"best match person {best_id}  "
                                    f"sim={sims[best_id]:.3f}  "
                                    f"({'covis-blocked' if covis_blocked else f'rejected (thr={reid_threshold:.2f})'})"
                                )
                            # Decrement retry counter; finalise as new person
                            # when the window expires.
                            if person_id in pending_reid:
                                pending_reid[person_id] -= 1
                                if pending_reid[person_id] <= 0:
                                    # Window expired — genuinely new person
                                    person_gallery[person_id] = feat_i.copy()
                                    person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat_i.copy()))
                                    del pending_reid[person_id]
                            else:
                                person_gallery[person_id] = feat_i.copy()
                                person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat_i.copy()))
                    elif should_try_reid and not person_gallery:
                        # Very first person ever — no gallery to match against
                        person_gallery[person_id] = feat_i.copy()
                        person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat_i.copy()))
                        pending_reid.pop(person_id, None)

                    # Update gallery descriptor with EMA for the canonical person.
                    if canonical_id in person_gallery:
                        alpha = gallery_ema_alpha
                        g = alpha * person_gallery[canonical_id] + (1 - alpha) * feat_i
                        norm = np.linalg.norm(g)
                        person_gallery[canonical_id] = g / norm if norm > 0 else g
                        # Also buffer this frame for the confidence-weighted cross-view gallery.
                        person_feat_buffer.setdefault(canonical_id, []).append((frame_idx, feat_i.copy()))

                params = {"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)}

                # Always keep model-agnostic SAM3D outputs.
                for key in BodyParameterEstimator._AGNOSTIC_KEYS:
                    if key in body:
                        val = body[key]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        params[key] = np.asarray(val, dtype=np.float32)

                # When no converter, fall back to raw MHR params immediately.
                if converter is None:
                    for key in param_keys:
                        if key in body and key not in BodyParameterEstimator._AGNOSTIC_KEYS:
                            val = body[key]
                            if isinstance(val, torch.Tensor):
                                val = val.detach().cpu().numpy()
                            params[key] = np.asarray(val, dtype=np.float32)

                # --- Per-joint confidence (MHR-based; overwritten by Pass 2 when converter present) ---
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

                tracks.setdefault(canonical_id, {})[frame_idx] = params

                # Accumulate raw SAM3D output for batched SMPL-X conversion in Pass 2.
                # Move tensors to CPU now to avoid holding GPU memory for the whole loop.
                if converter is not None:
                    body_cpu = {
                        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in body.items()
                    }
                    pending_conversion.setdefault(canonical_id, []).append(
                        (frame_idx, body_cpu, person_id, mask_key, img_h, img_w)
                    )

        # Pass 2: batched SMPL-X conversion + SMPL-X-based confidence override.
        # One optimizer run per person track (all frames batched together).
        if converter is not None and pending_conversion:
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
                    )
                except Exception as _e:
                    logging.warning(
                        f"{gpu_label}Batched SMPLX conversion failed person "
                        f"{canonical_id} in {video_id}: {_e}"
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

                # Batch SMPL-X forward for joints (one pass for all frames).
                smplx_joints_batch: np.ndarray | None = None
                if conv_result.result_parameters is not None and "smplx_betas" in smplx_np:
                    try:
                        with torch.no_grad():
                            _betas = torch.from_numpy(smplx_np["smplx_betas"]).float().to(_smplx_device)
                            _body_pose = torch.from_numpy(smplx_np["smplx_body_pose"]).float().to(_smplx_device)
                            _global_orient = torch.from_numpy(smplx_np["smplx_global_orient"]).float().to(_smplx_device)
                            _transl = torch.from_numpy(smplx_np["smplx_transl"]).float().to(_smplx_device)
                            # betas may be (10,) or (1, 10) (one set per sequence)
                            # while pose params are (N_frames, *) — expand to match.
                            if _betas.dim() == 1:
                                _betas = _betas.unsqueeze(0)
                            if _betas.shape[0] == 1 and _body_pose.shape[0] > 1:
                                _betas = _betas.expand(_body_pose.shape[0], -1)
                            _smplx_out = converter._smpl_model(
                                betas=_betas,
                                body_pose=_body_pose,
                                global_orient=_global_orient,
                                transl=_transl,
                            )
                        # (N_frames, J_total, 3) → keep first 55
                        smplx_joints_batch = _smplx_out.joints[:, :55].cpu().numpy().astype(np.float32)
                    except Exception as _je:
                        logging.warning(
                            f"{gpu_label}Batched SMPLX joints failed person "
                            f"{canonical_id} in {video_id}: {_je}"
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
                        conv_result.result_vertices is not None
                        and smplx_joints_batch is not None
                        and _mask_zip is not None
                        and f_mask_key in _mask_zip.namelist()
                    ):
                        smplx_verts_cam = np.asarray(conv_result.result_vertices[j], dtype=np.float32)
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

        BodyParameterEstimator._save_body_data_static(
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
            BodyParameterEstimator._apply_reid_remap(
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

    def _save_body_data(
        self,
        tracks: dict[int, dict[int, dict]],
        body_dir: Path,
        video_id: str,
    ) -> None:
        """Save per-person .npz files and a summary JSON."""
        self._save_body_data_static(tracks, body_dir, video_id, self._PARAM_KEYS)

    @staticmethod
    def _match_persons_across_views_static(
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        cross_view_reid_threshold: float = 0.4,
        appearance_weight: float = 0.7,
        shape_weight: float = 0.3,
    ) -> None:
        """Core logic for cross-view person re-identification.

        Parameters
        ----------
        video_dirs : dict[str, Path]
            Mapping of ``video_id`` → output directory.
        scene_id : str
            Used only for logging.
        cross_view_reid_threshold : float
            Minimum hybrid cosine similarity to accept a cross-view match.
        appearance_weight : float
            Weight for the DINOv3 appearance component of the hybrid descriptor.
        shape_weight : float
            Weight for the SMPL-X shape (beta) component.
        """
        video_ids = list(video_dirs.keys())
        if len(video_ids) < 2:
            logging.info(
                f"Scene {scene_id}: only one video, skipping cross-view re-ID"
            )
            return

        # ── 1. Load per-person descriptors ────────────────────────────────
        # person_descs[video_id][person_id] = (app_feat, shape_feat)
        #   app_feat  : L2-normalised DINOv3 float32 vector, or None
        #   shape_feat: L2-normalised median SMPL-X beta vector, or None
        # Keeping modalities separate lets us compute cosine similarities
        # independently and combine them with properly normalised weights,
        # avoiding the dimension-imbalance distortion of concatenation.
        person_descs: dict[str, dict[int, tuple[np.ndarray | None, np.ndarray | None]]] = {}
        person_pids: dict[str, list[int]] = {}

        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / "body_data"
            gallery_path = body_dir / "appearance_gallery.npz"
            summary_path = body_dir / "body_params_summary.json"

            if not summary_path.exists():
                logging.warning(
                    f"{scene_id}/{vid_id}: no body_params_summary.json, "
                    f"skipping cross-view re-ID for this video"
                )
                continue

            with open(summary_path) as _f:
                summary = json.load(_f)
            pids = [int(k) for k in summary.get("persons", {}).keys()]
            if not pids:
                continue

            # Load appearance gallery: pid → (feat_matrix (N,D), conf_weights (N,))
            app_gallery: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                feat_keys = [k for k in gdata.files if not k.endswith("_conf")]
                for k in feat_keys:
                    pid_key = int(k)
                    conf_key = f"{k}_conf"
                    feats = gdata[k]                                          # (N, D)
                    confs = gdata[conf_key] if conf_key in gdata.files else np.ones(len(feats), dtype=np.float32)
                    app_gallery[pid_key] = (feats, confs)

            descs: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue

                # Shape descriptor: confidence-weighted mean of beta vectors,
                # L2-normalised.  Using per-frame mean joint confidence as weights
                # down-weights frames with poor body fits (occlusion, blur, bad
                # pose estimate) that would otherwise corrupt the descriptor.
                # Falls back to the plain median when confidence is unavailable.
                shape_feat: np.ndarray | None = None
                with np.load(str(npz_path)) as pdata:
                    if "smplx_betas" in pdata:
                        shape_vecs = pdata["smplx_betas"]  # (N, 10)
                        if len(shape_vecs) > 0:
                            conf = pdata.get("pred_joint_confidence")  # (N, J) or None
                            if conf is not None and len(conf) == len(shape_vecs):
                                # Mean confidence per frame → confidence-weighted mean of betas
                                frame_conf = np.mean(conf, axis=-1).astype(np.float32)
                                total_conf = frame_conf.sum()
                                if total_conf > 0:
                                    shape_med = (frame_conf[:, None] * shape_vecs).sum(0) / total_conf
                                else:
                                    shape_med = np.median(shape_vecs, axis=0).astype(np.float32)
                            else:
                                shape_med = np.median(shape_vecs, axis=0).astype(np.float32)
                            norm = np.linalg.norm(shape_med)
                            shape_feat = (
                                shape_med / norm if norm > 0 else shape_med
                            )

                # app_feat is (feat_matrix (N,D), conf_weights (N,)) or None
                app_feat: tuple[np.ndarray, np.ndarray] | None = app_gallery.get(pid)

                if app_feat is None and shape_feat is None:
                    continue

                descs[pid] = (app_feat, shape_feat)

            if descs:
                person_descs[vid_id] = descs
                person_pids[vid_id] = sorted(descs.keys())

        active_vids = [v for v in video_ids if v in person_descs]
        if len(active_vids) < 2:
            logging.info(
                f"Scene {scene_id}: fewer than 2 videos with descriptors, "
                f"skipping cross-view re-ID"
            )
            return

        # ── 2. Union-Find with edge tracking ──────────────────────────────
        # Nodes: (video_id, person_id) tuples.
        # We track accepted edges (similarity, node_a, node_b) so we can
        # remove the weakest one when resolving same-video conflicts.
        parent: dict[tuple, tuple] = {}
        rank_uf: dict[tuple, int] = {}
        edges: list[tuple[float, tuple, tuple]] = []

        def _find(x: tuple) -> tuple:
            if parent.setdefault(x, x) != x:
                parent[x] = _find(parent[x])
            return parent[x]

        def _union(x: tuple, y: tuple) -> None:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            if rank_uf.get(rx, 0) < rank_uf.get(ry, 0):
                rx, ry = ry, rx
            parent[ry] = rx
            if rank_uf.get(rx, 0) == rank_uf.get(ry, 0):
                rank_uf[rx] = rank_uf.get(rx, 0) + 1

        # Register all nodes so isolated persons are not forgotten.
        for _vid in active_vids:
            for _pid in person_pids[_vid]:
                _find((_vid, _pid))

        # ── 3. All-pairs Hungarian matching ───────────────────────────────
        def _chamfer_sim(
            feats_a: np.ndarray,   # (N, D) L2-normalised
            confs_a: np.ndarray,   # (N,)  unused
            feats_b: np.ndarray,   # (M, D) L2-normalised
            confs_b: np.ndarray,   # (M,)  unused
        ) -> float:
            """Symmetric Chamfer similarity.

            For each frame in A find its nearest neighbour in B (max cosine
            similarity), take the plain mean, and vice-versa.
            """
            S = feats_a @ feats_b.T                 # (N, M) cosine similarities
            score_a2b = float(S.max(axis=1).mean())
            score_b2a = float(S.max(axis=0).mean())
            return 0.5 * (score_a2b + score_b2a)

        def _weighted_sim_mat(
            pids_a: list[int],
            pids_b: list[int],
            descs_a: dict[int, tuple],
            descs_b: dict[int, tuple],
            w_app: float,
            w_shape: float,
        ) -> np.ndarray:
            """Compute (Na, Nb) similarity matrix as a weighted sum of per-modality
            similarities.  Weights are re-normalised per cell so that a missing
            modality on either side just falls back to the other one.

            Appearance uses confidence-weighted Chamfer similarity between the
            full per-tracklet feature matrices.  Shape uses a plain cosine
            similarity between confidence-weighted mean beta vectors (shape is
            already a stable summary statistic, no multi-shot needed).
            """
            Na, Nb = len(pids_a), len(pids_b)
            sim_mat = np.zeros((Na, Nb), dtype=np.float32)
            weight_mat = np.zeros((Na, Nb), dtype=np.float32)

            # --- appearance (confidence-weighted Chamfer) ---
            for i, pa in enumerate(pids_a):
                app_a = descs_a[pa][0]   # (N, D), (N,) or None
                if app_a is None:
                    continue
                feats_a, confs_a = app_a
                for j, pb in enumerate(pids_b):
                    app_b = descs_b[pb][0]
                    if app_b is None:
                        continue
                    feats_b, confs_b = app_b
                    sim_mat[i, j] += w_app * _chamfer_sim(feats_a, confs_a, feats_b, confs_b)
                    weight_mat[i, j] += w_app

            # --- shape (cosine similarity between mean beta vectors) ---
            shape_a = [descs_a[p][1] for p in pids_a]
            shape_b = [descs_b[p][1] for p in pids_b]
            mask_a = np.array([f is not None for f in shape_a], dtype=np.float32)
            mask_b = np.array([f is not None for f in shape_b], dtype=np.float32)
            if mask_a.any() and mask_b.any():
                dim = next(f for f in shape_a if f is not None).shape[0]
                zero = np.zeros(dim, dtype=np.float32)
                mat_a = np.stack([f if f is not None else zero for f in shape_a])
                mat_b = np.stack([f if f is not None else zero for f in shape_b])
                shape_sim = mat_a @ mat_b.T   # (Na, Nb)
                shape_w = np.outer(mask_a, mask_b) * w_shape
                sim_mat += shape_w * shape_sim
                weight_mat += shape_w

            # Normalise by the actual weight used per cell
            sim_mat = np.where(weight_mat > 0, sim_mat / weight_mat, 0.0)
            return sim_mat

        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                pids_a = person_pids[vid_a]
                pids_b = person_pids[vid_b]

                # Weighted sum of per-modality cosine similarities
                sim_mat = _weighted_sim_mat(
                    pids_a, pids_b,
                    person_descs[vid_a], person_descs[vid_b],
                    appearance_weight, shape_weight,
                )
                cost_mat = 1.0 - sim_mat

                row_ind, col_ind = linear_sum_assignment(cost_mat)
                for r, c in zip(row_ind, col_ind):
                    sim = float(sim_mat[r, c])
                    accepted = sim >= cross_view_reid_threshold
                    logging.info(
                        f"[cross-view reid] {scene_id}  "
                        f"{vid_a}/P{pids_a[r]} ↔ {vid_b}/P{pids_b[c]}  "
                        f"sim={sim:.3f}  "
                        f"({'MATCH' if accepted else f'rejected (thr={cross_view_reid_threshold:.2f})'})"
                    )
                    if accepted:
                        node_a = (vid_a, pids_a[r])
                        node_b = (vid_b, pids_b[c])
                        edges.append((sim, node_a, node_b))
                        _union(node_a, node_b)

        # ── 4. Conflict detection and resolution ──────────────────────────
        # A conflict occurs when two persons from the same video end up in
        # the same component (a single camera cannot show the same person
        # twice under different SAM2 IDs after within-video re-ID).  Resolve
        # by iteratively removing the weakest edge that bridges conflicting
        # nodes until no component contains a duplicate video ID.

        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _vid in active_vids:
                for _pid in person_pids[_vid]:
                    _node = (_vid, _pid)
                    _root = _find(_node)
                    comps.setdefault(_root, []).append(_node)
            return comps

        for _ in range(len(edges) + 1):
            comps = _get_components()
            conflict_found = False

            for members in comps.values():
                vid_to_nodes: dict[str, list[tuple]] = {}
                for node in members:
                    vid_to_nodes.setdefault(node[0], []).append(node)

                for vid_id, nodes in vid_to_nodes.items():
                    if len(nodes) < 2:
                        continue

                    # Conflict — find the weakest accepted edge whose both
                    # endpoints currently live in this component.
                    conflict_found = True
                    conflict_nodes = set(nodes)
                    worst_sim, worst_idx = float("inf"), -1
                    for ei, (sim, na, nb) in enumerate(edges):
                        if _find(na) == _find(nb):  # same component
                            if na in conflict_nodes or nb in conflict_nodes:
                                if sim < worst_sim:
                                    worst_sim, worst_idx = sim, ei

                    if worst_idx >= 0:
                        edges.pop(worst_idx)
                        # Rebuild union-find without the removed edge.
                        parent.clear()
                        rank_uf.clear()
                        for _vid in active_vids:
                            for _pid in person_pids[_vid]:
                                _find((_vid, _pid))
                        for _sim, _na, _nb in edges:
                            _union(_na, _nb)
                        logging.info(
                            f"Scene {scene_id}: removed conflicting cross-view "
                            f"edge (sim={worst_sim:.3f}) to resolve "
                            f"same-video duplicate in {vid_id}"
                        )
                    break  # restart outer conflict loop after rebuild

            if not conflict_found:
                break

        # ── 4.5. Consolidation pass with component centroids ──────────────
        # After the strict pairwise matching, some persons may remain
        # unlinked because their appearance differed enough (e.g. the same
        # person goes from a solo shot to a crowded scene) that no single
        # pairwise similarity exceeded the threshold.
        #
        # Fix: compute the L2-normalised *centroid* of each confirmed
        # multi-view component (averaging out per-view noise) and try to
        # attach isolated single-video persons to the best-matching
        # component at a slightly relaxed threshold.
        consolidation_threshold = max(0.15, cross_view_reid_threshold - 0.15)
        comps = _get_components()

        multi_view_comps: dict[tuple, list[tuple]] = {
            root: members
            for root, members in comps.items()
            if len({v for v, _ in members}) >= 2
        }
        isolated: list[tuple] = [
            (vid, pid)
            for root, members in comps.items()
            if root not in multi_view_comps
            for vid, pid in members
        ]

        # Compute per-modality L2-normalised centroids for each multi-view component.
        # Storing (app_centroid, shape_centroid) mirrors the per-person descriptor
        # structure so we can reuse the same weighted-similarity logic.
        comp_centroids: dict[tuple, tuple[np.ndarray | None, np.ndarray | None]] = {}
        for root, members in multi_view_comps.items():
            app_vecs = [
                person_descs[vid][pid][0]
                for vid, pid in members
                if vid in person_descs and pid in person_descs.get(vid, {})
                and person_descs[vid][pid][0] is not None
            ]
            shape_vecs = [
                person_descs[vid][pid][1]
                for vid, pid in members
                if vid in person_descs and pid in person_descs.get(vid, {})
                and person_descs[vid][pid][1] is not None
            ]
            app_c: np.ndarray | None = None
            shape_c: np.ndarray | None = None
            if app_vecs:
                # Each entry is (feats (N,D), confs (N,)) — concatenate across members
                all_feats = np.concatenate([f for f, _ in app_vecs], axis=0)
                all_confs = np.concatenate([c for _, c in app_vecs], axis=0)
                total_w = all_confs.sum()
                m = ((all_confs[:, None] * all_feats).sum(0) / total_w
                     if total_w > 0 else all_feats.mean(0))
                n = np.linalg.norm(m)
                app_c = (m / n if n > 0 else m).astype(np.float32)
            if shape_vecs:
                m = np.mean(np.stack(shape_vecs), axis=0).astype(np.float32)
                n = np.linalg.norm(m)
                shape_c = m / n if n > 0 else m
            if app_c is not None or shape_c is not None:
                comp_centroids[root] = (app_c, shape_c)

        def _weighted_sim_scalar(
            feat: tuple[np.ndarray | None, np.ndarray | None],
            centroid: tuple[np.ndarray | None, np.ndarray | None],
            w_app: float,
            w_shape: float,
        ) -> float:
            sim, weight = 0.0, 0.0
            if feat[0] is not None and centroid[0] is not None:
                feats, confs = feat[0]
                total_w = confs.sum()
                mean_feat = ((confs[:, None] * feats).sum(0) / total_w
                             if total_w > 0 else feats.mean(0))
                sim += w_app * float(np.dot(mean_feat, centroid[0]))
                weight += w_app
            if feat[1] is not None and centroid[1] is not None:
                sim += w_shape * float(np.dot(feat[1], centroid[1]))
                weight += w_shape
            return sim / weight if weight > 0 else 0.0

        # Try to link each isolated person to its best-matching multi-view
        # component, skipping components that already contain a member from
        # the same video (one camera cannot show the same person twice).
        consolidation_edges_added = False
        for vid, pid in isolated:
            if vid not in person_descs or pid not in person_descs.get(vid, {}):
                continue
            feat = person_descs[vid][pid]

            best_root, best_sim = None, -1.0
            for root, centroid in comp_centroids.items():
                if any(v == vid for v, _ in multi_view_comps[root]):
                    continue
                sim = _weighted_sim_scalar(feat, centroid, appearance_weight, shape_weight)
                if sim > best_sim:
                    best_sim, best_root = sim, root

            if best_root is not None and best_sim >= consolidation_threshold:
                comp_member = multi_view_comps[best_root][0]
                edges.append((best_sim, (vid, pid), comp_member))
                _union((vid, pid), comp_member)
                consolidation_edges_added = True
                logging.info(
                    f"Scene {scene_id}: consolidation linked "
                    f"{vid}/P{pid} → component {best_root} "
                    f"(centroid_sim={best_sim:.3f})"
                )

        # Re-run conflict resolution if the consolidation added new edges.
        if consolidation_edges_added:
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False

                for members in comps.values():
                    vid_to_nodes: dict[str, list[tuple]] = {}
                    for node in members:
                        vid_to_nodes.setdefault(node[0], []).append(node)

                    for vid_id, nodes in vid_to_nodes.items():
                        if len(nodes) < 2:
                            continue

                        conflict_found = True
                        conflict_nodes = set(nodes)
                        worst_sim, worst_idx = float("inf"), -1
                        for ei, (sim, na, nb) in enumerate(edges):
                            if _find(na) == _find(nb):
                                if na in conflict_nodes or nb in conflict_nodes:
                                    if sim < worst_sim:
                                        worst_sim, worst_idx = sim, ei

                        if worst_idx >= 0:
                            edges.pop(worst_idx)
                            parent.clear()
                            rank_uf.clear()
                            for _vid in active_vids:
                                for _pid in person_pids[_vid]:
                                    _find((_vid, _pid))
                            for _sim, _na, _nb in edges:
                                _union(_na, _nb)
                            logging.info(
                                f"Scene {scene_id}: consolidation conflict "
                                f"resolved — removed edge (sim={worst_sim:.3f})"
                            )
                        break

                if not conflict_found:
                    break

        # ── 5. Assign global IDs ──────────────────────────────────────────
        # Multi-view components (confirmed cross-camera persons) claim their
        # preferred global ID (min local ID across the group) first.
        # Single-video components (unmatched or new persons) get their
        # preferred ID only if it doesn't collide; otherwise they receive a
        # new ID beyond the current maximum.  This prevents a new person in
        # a later video who happened to receive a low local SAM2 ID (e.g. 1)
        # from silently overwriting an established cross-camera person with
        # the same global ID during the file-rename step.
        comps = _get_components()
        used_global_ids: set[int] = set()
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}

        # First pass: multi-view components get strictly sequential global IDs.
        # Using min(local_pid) was buggy: multiple distinct components can all
        # have min=1 (local IDs restart at 1 in every camera), causing collisions.
        pending_single: list[tuple[int, list[tuple]]] = []
        global_counter = 1
        for members in comps.values():
            if len({v for v, _ in members}) >= 2:
                global_id = global_counter
                global_counter += 1
                used_global_ids.add(global_id)
                for vid_id, pid in members:
                    if pid != global_id:
                        global_remap[vid_id][pid] = global_id
            else:
                proposed_id = min(pid for (_, pid) in members)
                pending_single.append((proposed_id, list(members)))

        # Second pass: single-video components also get sequential IDs.
        next_new_id = global_counter
        for proposed_id, members in sorted(pending_single):
            global_id = next_new_id
            next_new_id += 1
            used_global_ids.add(global_id)
            for vid_id, pid in members:
                if pid != global_id:
                    global_remap[vid_id][pid] = global_id

        total_remaps = sum(len(m) for m in global_remap.values())
        logging.info(
            f"Scene {scene_id}: cross-view re-ID → {len(comps)} global "
            f"person(s), {total_remaps} local-ID remap(s) across "
            f"{len(active_vids)} view(s)"
        )
        if total_remaps == 0:
            logging.info(
                f"Scene {scene_id}: all local IDs are already globally "
                f"consistent, no remap needed"
            )

        # ── 6. Apply remaps to body_data, masks, and JSON ─────────────────
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"

            # Rename person_<old>.npz → person_<new>.npz via a temporary name
            # to avoid clobbering when two IDs effectively swap (e.g. 1↔2).
            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.xviewtmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    # Another source file already claimed this global ID
                    # (same-camera tracks merged via cross-view transitivity).
                    # Keep the existing file (the original tracked person) and
                    # discard the duplicate rather than overwriting good data.
                    logging.warning(
                        f"{vid_id}: cross-view remap — {dst.name} already exists, "
                        f"discarding duplicate track from {tmp.name}"
                    )
                    tmp.unlink()
                else:
                    tmp.rename(dst)

            # Update body_params_summary.json with new IDs.
            summary_path = body_dir / "body_params_summary.json"
            if summary_path.exists():
                with open(summary_path) as _f:
                    summary = json.load(_f)
                new_persons: dict[str, object] = {}
                for str_id, info in summary.get("persons", {}).items():
                    new_persons[str(remap.get(int(str_id), int(str_id)))] = info
                summary["persons"] = new_persons
                with open(summary_path, "w") as _f:
                    json.dump(summary, _f, indent=2)

            # Update appearance_gallery.npz with new IDs.
            # Keys are either "{pid}" (feature matrix) or "{pid}_conf" (weights).
            gallery_path = body_dir / "appearance_gallery.npz"
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                new_gallery = {}
                for k in gdata.files:
                    if k.endswith("_conf"):
                        old_pid = int(k[:-5])
                        new_key = f"{remap.get(old_pid, old_pid)}_conf"
                    else:
                        old_pid = int(k)
                        new_key = str(remap.get(old_pid, old_pid))
                    new_gallery[new_key] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            # Save cross-view mapping for provenance / downstream consumers.
            mapping_path = body_dir / "cross_view_id_mapping.json"
            with open(mapping_path, "w") as _f:
                json.dump({str(k): v for k, v in remap.items()}, _f, indent=2)

            # Rewrite mask_data.npz and json_data/*.json with the global IDs.
            BodyParameterEstimator._apply_reid_remap(vid_dir, remap)
            print(
                f"  {vid_id}: cross-view re-ID — "
                f"{len(remap)} ID remap(s): {remap}"
            )

        # Write scene-level summary. This file also serves as the completion
        # marker checked by match_across_views to skip already-processed scenes.
        reid_summary_path = scene_dir / "cross_view_reid.json"
        reid_summary = {
            vid_id: {str(k): v for k, v in remap.items()}
            for vid_id, remap in global_remap.items()
        }
        with open(reid_summary_path, "w") as _f:
            json.dump(reid_summary, _f, indent=2)

    @staticmethod
    def _apply_reid_remap(
        video_dir: Path,
        id_remap: dict[int, int],
        gpu_label: str = "",
    ) -> None:
        """Rewrite mask_data.npz and json_data/*.json with re-identified IDs.

        Uses the same stream-remap logic as PersonSegmenter._apply_id_mapping
        so that the segmentation files stay consistent with body_data/ after
        within-video re-identification.

        Parameters
        ----------
        video_dir : Path
            Root output directory for one video (contains mask_data.npz and
            json_data/).
        id_remap : dict[int, int]
            Mapping of raw SAM2 ID → canonical person ID discovered during
            body parameter estimation.
        """
        if not id_remap or all(k == v for k, v in id_remap.items()):
            return

        npz_path = video_dir / "mask_data.npz"
        json_dir = video_dir / "json_data"

        # Remap pixel values in the compressed mask archive
        if npz_path.exists():
            tmp_path = npz_path.with_suffix(".tmp.npz")
            with (
                zipfile.ZipFile(str(npz_path), "r") as zf_in,
                zipfile.ZipFile(
                    str(tmp_path), "w",
                    compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                ) as zf_out,
            ):
                for name in sorted(zf_in.namelist()):
                    with zf_in.open(name) as f:
                        mask_img = np.load(io.BytesIO(f.read()))
                    total_pixels = mask_img.shape[0] * mask_img.shape[1]
                    new_mask = np.zeros_like(mask_img)
                    for old_id, new_id in id_remap.items():
                        old_region = mask_img == old_id
                        # Do not remap degenerate masks that cover >80% of
                        # the frame — they are flood-fill artefacts that
                        # should not be attributed to any real person.
                        if int(old_region.sum()) > 0.80 * total_pixels:
                            logging.warning(
                                f"{gpu_label}Skipping remap {old_id}→{new_id} "
                                f"in {name}: mask covers >80% of frame"
                            )
                            continue
                        new_mask[old_region] = new_id
                    # IDs not in the remap keep their original value.
                    for uid in set(np.unique(mask_img)) - {0} - set(id_remap.keys()):
                        new_mask[mask_img == uid] = uid
                    buf = io.BytesIO()
                    np.save(buf, new_mask)
                    zf_out.writestr(name, buf.getvalue())
            tmp_path.replace(npz_path)

        # Remap instance IDs in the per-frame JSON metadata.
        # Guard against collisions: if two different old IDs remap to the same
        # new ID in the same frame (which would mean the remap is wrong for that
        # frame), keep the original ID for the later entry rather than silently
        # overwriting the earlier one and losing a person's bounding box.
        for json_path in sorted(json_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            if "labels" in data:
                new_labels = {}
                # Process non-remapped entries first so they always claim their
                # canonical slots before any remapped entry can collide with them.
                sorted_items = sorted(
                    data["labels"].items(),
                    key=lambda kv: 0 if id_remap.get(int(kv[0]), int(kv[0])) == int(kv[0]) else 1,
                )
                for str_id, info in sorted_items:
                    old_id = int(str_id)
                    new_id = id_remap.get(old_id, old_id)
                    info["instance_id"] = new_id
                    if str(new_id) in new_labels:
                        # Collision: two persons would share the same ID in this
                        # frame — this indicates an incorrect merge.  Preserve
                        # both entries by keeping the original ID for this one.
                        logging.warning(
                            f"{gpu_label}Re-ID collision for id {new_id} in "
                            f"{json_path.name}: keeping original id {old_id}"
                        )
                        info["instance_id"] = old_id
                        new_labels[str(old_id)] = info
                    else:
                        new_labels[str(new_id)] = info
                data["labels"] = new_labels
            with open(json_path, "w") as f:
                json.dump(data, f)

        logging.info(
            f"{gpu_label}Re-ID segmentation remap applied in {video_dir.name}: "
            f"{id_remap}"
        )

    @staticmethod
    def _create_converter(mhr_model_path: str | None, smplx_model_path: str | None):
        """Create and return a Conversion object, or None if paths are not provided."""
        if not mhr_model_path or not smplx_model_path:
            return None
        try:
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
        except Exception as e:
            import traceback as _tb
            logging.warning(f"Could not initialise SMPLX converter: {e}\n{_tb.format_exc()}")
            return None

    def convert_hmr_to_smplx(self, sam3d_outputs):
        """Convert HMR/MHR parameters to SMPLX parameters.

        Args:
            sam3d_outputs: List of SAM3D output dicts (one per person) from
                process_one_image().

        Returns:
            ConversionResult with result_parameters containing SMPLX params.
        """
        if self._converter is None:
            self._converter = self._create_converter(
                self.mhr_model_path, self.smplx_model_path
            )
        if self._converter is None:
            raise RuntimeError(
                "SMPLX converter could not be initialised. "
                "Check mhr_model_path and smplx_model_path."
            )
        return self._converter.convert_sam3d_output_to_smpl(
            sam3d_outputs=sam3d_outputs,
            return_smpl_meshes=False,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            return_fitting_errors=False,
        )


class CrossViewReidentifier:
    """Assign consistent global person IDs across camera views in a scene.

    Parameters
    ----------
    threshold : float
        Minimum hybrid cosine similarity to accept a cross-view match.
    appearance_weight : float
        Weight for the DINOv3 appearance component of the hybrid descriptor.
    shape_weight : float
        Weight for the SMPL-X shape (beta) component.
    """

    _THRESHOLD: float = 0.4
    _APPEARANCE_WEIGHT: float = 0.7
    _SHAPE_WEIGHT: float = 0.3

    def __init__(
        self,
        threshold: float | None = None,
        appearance_weight: float | None = None,
        shape_weight: float | None = None,
    ):
        self.threshold = threshold if threshold is not None else self._THRESHOLD
        self.appearance_weight = appearance_weight if appearance_weight is not None else self._APPEARANCE_WEIGHT
        self.shape_weight = shape_weight if shape_weight is not None else self._SHAPE_WEIGHT

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
    ) -> None:
        """Assign consistent global person IDs across all camera views in a scene."""
        scene_dir = Path(next(iter(video_dirs.values()))).parent

        if (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene.scene_id}: cross-view ReID already done, skipping")
            return

        BodyParameterEstimator._match_persons_across_views_static(
            video_dirs=video_dirs,
            scene_id=scene.scene_id,
            scene_dir=scene_dir,
            cross_view_reid_threshold=self.threshold,
            appearance_weight=self.appearance_weight,
            shape_weight=self.shape_weight,
        )
