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

from data.video_dataset import Scene
from mhr.mhr import MHR
from conversion import Conversion

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

    # Cross-view re-identification defaults.
    _CROSS_VIEW_REID_THRESHOLD: float = 0.4
    _CROSS_VIEW_APPEARANCE_WEIGHT: float = 0.7
    _CROSS_VIEW_SHAPE_WEIGHT: float = 0.3

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
        cross_view_reid_threshold: float | None = None,
        cross_view_appearance_weight: float | None = None,
        cross_view_shape_weight: float | None = None,
    ):
        self.sam3d_hf_repo = sam3d_hf_repo
        self.sam3d_step = sam3d_step
        self.bbox_padding = bbox_padding
        self.smplx_model_path = smplx_model_path
        self.mhr_model_path = mhr_model_path
        self.reid_threshold = reid_threshold if reid_threshold is not None else self._REID_THRESHOLD
        self.gallery_ema_alpha = gallery_ema_alpha if gallery_ema_alpha is not None else self._GALLERY_EMA_ALPHA
        self.reid_match_window = reid_match_window if reid_match_window is not None else self._REID_MATCH_WINDOW
        self.cross_view_reid_threshold = (
            cross_view_reid_threshold
            if cross_view_reid_threshold is not None
            else self._CROSS_VIEW_REID_THRESHOLD
        )
        self.cross_view_appearance_weight = (
            cross_view_appearance_weight
            if cross_view_appearance_weight is not None
            else self._CROSS_VIEW_APPEARANCE_WEIGHT
        )
        self.cross_view_shape_weight = (
            cross_view_shape_weight
            if cross_view_shape_weight is not None
            else self._CROSS_VIEW_SHAPE_WEIGHT
        )

        self._estimator: object | None = None
        self._converter: object | None = None

    def _init_sam3d(self) -> None:
        """Lazy-load the SAM3D Body estimator."""
        if self._estimator is not None:
            logging.warning("The estimator was already loaded, skipping the loading")
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

        if num_gpus <= 1:
            # Fallback: sequential on single GPU
            self._init_sam3d()
            converter = self._create_converter(self.mhr_model_path, self.smplx_model_path)
            for video in tqdm(scene.videos, desc="SAM3D Body estimation"):
                video_dir = video_dirs[video.video_id]
                BodyParameterEstimator._process_video_core(
                    self._estimator,
                    video.video_id,
                    str(video_dir),
                    self.sam3d_step,
                    self.bbox_padding,
                    self._PARAM_KEYS,
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
            task_queue.put((
                video.video_id,
                str(video_dirs[video.video_id]),
                str(video.frames_home) if video.frames_home else None,
            ))
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
                    self._PARAM_KEYS,
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

        tracks: dict[int, dict[int, dict]] = {}

        # Phase 0 — pre-scan every JSON frame to find which SAM2 track IDs are
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
        # id_remap: raw SAM2 id → canonical_id for ids that were re-identified.
        # pending_reid: new SAM2 ids that weren't matched on first sight —
        #   maps person_id → frames_remaining for retry attempts.
        # We employ DINOv3 backbone (given by SAM 3D) in order to match people across
        # frames using cosine similarity between visual features
        person_gallery: dict[int, np.ndarray] = {}
        id_remap: dict[int, int] = {}
        pending_reid: dict[int, int] = {}  # person_id → frames remaining

        for json_path in tqdm(
            json_files, desc=f"{gpu_label}SAM3D {video_id}", leave=False
        ):
            # Load frame idx and bounding boxes

            frame_idx_str = json_path.stem.replace("mask_", "")
            frame_idx = int(frame_idx_str)

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

            # Collect all valid persons for this frame.
            # Each entry: (person_id, padded_x1, padded_y1, padded_x2, padded_y2,
            #               orig_x1, orig_y1, orig_x2, orig_y2)
            valid_persons = []
            for str_id, info in labels.items():
                person_id = int(str_id)
                x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

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

            # Hook into the backbone to capture visual features for re-ID.
            # The backbone runs first for the body pass (N_persons crops); we only
            # want that first call — hand passes use different crops.
            # This will allow us to keep track of the visual features
            _hook_feats: list[np.ndarray] = []

            def _backbone_hook(_module, _input, output):
                if _hook_feats:   # ignore subsequent hand-branch calls
                    return
                emb = output[-1] if isinstance(output, tuple) else output
                # Global-average-pool: (N, C, H, W) → (N, C), then L2-normalise.
                feat = emb.float().mean(dim=(-2, -1))
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                _hook_feats.append(feat.detach().cpu().numpy())

            try:
                _hook_handle = estimator.model.backbone.register_forward_hook(
                    _backbone_hook
                )
            except AttributeError:
                logging.error("estimator  doesn't have a model.backbone.register_forward_hook method")
                _hook_handle = None

            try:
                outputs = estimator.process_one_image(
                    frame_rgb, bboxes=bboxes_arr
                )
            except Exception as e:
                logging.warning(
                    f"{gpu_label}SAM3D failed frame {frame_idx} in {video_id}: {e}, returning None"
                )
                outputs = None
            finally:
                # FInally free the hook
                if _hook_handle is not None:
                    _hook_handle.remove()

            if not outputs:
                if outputs is not None:
                    logging.warning(
                        f"{gpu_label}No outputs for frame {frame_idx} in {video_id}"
                    )
                continue

            # vis_feats[i] is the L2-normalised backbone descriptor for valid_persons[i].
            vis_feats: np.ndarray | None = _hook_feats[0] if _hook_feats else None

            # Convert SAM3D (MHR) outputs to SMPLX parameters when a converter is
            # available.  The converter accepts all non-None person outputs for this
            # frame at once and returns per-person SMPLX parameters indexed by their
            # position in the non-None subset.
            smplx_params_by_out_idx: dict[int, dict[str, np.ndarray]] | None = None
            if converter is not None and outputs:
                # Map: position in `outputs` → position in the non-None subset.
                valid_idx_map: dict[int, int] = {}
                valid_outputs_list = []
                for _oi, _o in enumerate(outputs):
                    if _o is not None:
                        valid_idx_map[_oi] = len(valid_outputs_list)
                        valid_outputs_list.append(_o)

                if valid_outputs_list:
                    try:
                        conv_result = converter.convert_sam3d_output_to_smpl(
                            sam3d_outputs=valid_outputs_list,
                            return_smpl_meshes=False,
                            return_smpl_parameters=True,
                            return_smpl_vertices=False,
                            return_fitting_errors=False,
                        )
                        if conv_result.result_parameters is not None:
                            smplx_params_by_out_idx = {}
                            for _param_key, _param_val in conv_result.result_parameters.items():
                                if isinstance(_param_val, torch.Tensor):
                                    _param_np = _param_val.detach().cpu().numpy()
                                else:
                                    _param_np = np.asarray(_param_val, dtype=np.float32)
                                for _out_idx, _v_idx in valid_idx_map.items():
                                    smplx_params_by_out_idx.setdefault(_out_idx, {})[
                                        f"smplx_{_param_key}"
                                    ] = _param_np[_v_idx]
                    except Exception as _e:
                        logging.warning(
                            f"{gpu_label}SMPLX conversion failed for frame "
                            f"{frame_idx} in {video_id}: {_e}"
                        )

            # outputs[i] corresponds to valid_persons[i] (same order as bboxes_arr).
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
                            logging.info(
                                f"{gpu_label}Re-ID: SAM2 id {person_id} → "
                                f"person {best_id} (sim={sims[best_id]:.3f}) "
                                f"in {video_id} frame {frame_idx}"
                            )
                        else:
                            # Decrement retry counter; finalise as new person
                            # when the window expires.
                            if person_id in pending_reid:
                                pending_reid[person_id] -= 1
                                if pending_reid[person_id] <= 0:
                                    # Window expired — genuinely new person
                                    person_gallery[person_id] = feat_i.copy()
                                    del pending_reid[person_id]
                            else:
                                person_gallery[person_id] = feat_i.copy()
                    elif should_try_reid and not person_gallery:
                        # Very first person ever — no gallery to match against
                        person_gallery[person_id] = feat_i.copy()
                        pending_reid.pop(person_id, None)

                    # Update gallery descriptor with EMA for the canonical person.
                    if canonical_id in person_gallery:
                        alpha = gallery_ema_alpha
                        g = alpha * person_gallery[canonical_id] + (1 - alpha) * feat_i
                        norm = np.linalg.norm(g)
                        person_gallery[canonical_id] = g / norm if norm > 0 else g

                params = {"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)}

                # Always keep model-agnostic SAM3D outputs.
                for key in BodyParameterEstimator._AGNOSTIC_KEYS:
                    if key in body:
                        val = body[key]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        params[key] = np.asarray(val, dtype=np.float32)

                # Prefer SMPLX params from conversion; fall back to raw MHR params.
                if smplx_params_by_out_idx is not None and i in smplx_params_by_out_idx:
                    for k, v in smplx_params_by_out_idx[i].items():
                        params[k] = np.asarray(v, dtype=np.float32)
                else:
                    for key in param_keys:
                        if key in body and key not in BodyParameterEstimator._AGNOSTIC_KEYS:
                            val = body[key]
                            if isinstance(val, torch.Tensor):
                                val = val.detach().cpu().numpy()
                            params[key] = np.asarray(val, dtype=np.float32)

                tracks.setdefault(canonical_id, {})[frame_idx] = params

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

        # Persist per-person appearance descriptors for cross-view re-ID.
        # The gallery holds the final EMA-updated DINOv3 descriptor (1024-D,
        # L2-normalised) for each canonical person after within-video re-ID.
        if person_gallery:
            gallery_arrays = {str(pid): feat for pid, feat in person_gallery.items()}
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

    # ──────────────────────────────────────────────────────────────────────
    # Cross-view person re-identification
    # ──────────────────────────────────────────────────────────────────────

    def match_persons_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
    ) -> None:
        """Assign consistent global person IDs across all camera views in a scene.

        After within-video re-ID, each video has locally consistent person IDs
        that are independent across cameras.  This method builds hybrid
        appearance+shape descriptors for each person in each video, solves
        all-pairs optimal bipartite matching (Hungarian algorithm), closes the
        transitive chains via Union-Find, and remaps every output file
        (body_data/*.npz, mask_data.npz, json_data/*.json) so that the same
        physical person carries the same integer ID across all views.

        Persons visible in only a subset of views (e.g. only in views B and C
        but not A) are still correctly re-identified because all view *pairs*
        are matched, not just pairs that include a reference view.

        Parameters
        ----------
        scene : Scene
            The scene whose videos to process.
        video_dirs : dict[str, Path]
            Mapping of ``video_id`` → output directory as returned by
            ``segment_scene`` / ``estimate_scene``.
        """
        BodyParameterEstimator._match_persons_across_views_static(
            video_dirs=video_dirs,
            scene_id=scene.scene_id,
            cross_view_reid_threshold=self.cross_view_reid_threshold,
            appearance_weight=self.cross_view_appearance_weight,
            shape_weight=self.cross_view_shape_weight,
        )

    @staticmethod
    def _match_persons_across_views_static(
        video_dirs: dict[str, Path],
        scene_id: str,
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

        # ── 1. Load per-person hybrid descriptors ─────────────────────────
        # person_descs[video_id][person_id] = L2-normalised float32 vector
        # person_pids[video_id]             = sorted list of person_ids
        person_descs: dict[str, dict[int, np.ndarray]] = {}
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

            # Load appearance gallery (DINOv3, 1024-D, L2-normalised)
            app_gallery: dict[int, np.ndarray] = {}
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                for k in gdata.files:
                    app_gallery[int(k)] = gdata[k]

            descs: dict[int, np.ndarray] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue

                # Shape descriptor: median beta across frames, L2-normalised
                shape_feat: np.ndarray | None = None
                with np.load(str(npz_path)) as pdata:
                    if "shape_params" in pdata:
                        shape_vecs = pdata["shape_params"]  # (N, 10)
                        if len(shape_vecs) > 0:
                            shape_med = np.median(shape_vecs, axis=0).astype(
                                np.float32
                            )
                            norm = np.linalg.norm(shape_med)
                            shape_feat = (
                                shape_med / norm if norm > 0 else shape_med
                            )

                app_feat: np.ndarray | None = app_gallery.get(pid)

                # Build hybrid descriptor by concatenating weighted parts
                parts: list[np.ndarray] = []
                if app_feat is not None:
                    parts.append(appearance_weight * app_feat)
                if shape_feat is not None:
                    parts.append(shape_weight * shape_feat)
                if not parts:
                    continue

                hybrid = np.concatenate(parts)
                norm = np.linalg.norm(hybrid)
                descs[pid] = hybrid / norm if norm > 0 else hybrid

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
        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                pids_a = person_pids[vid_a]
                pids_b = person_pids[vid_b]

                mat_a = np.stack(
                    [person_descs[vid_a][p] for p in pids_a]
                )  # (Na, D)
                mat_b = np.stack(
                    [person_descs[vid_b][p] for p in pids_b]
                )  # (Nb, D)

                # Cosine similarity: dot product (vectors are L2-normalised)
                sim_mat = mat_a @ mat_b.T  # (Na, Nb)
                cost_mat = 1.0 - sim_mat

                row_ind, col_ind = linear_sum_assignment(cost_mat)
                for r, c in zip(row_ind, col_ind):
                    sim = float(sim_mat[r, c])
                    if sim >= cross_view_reid_threshold:
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

        # ── 5. Assign global IDs ──────────────────────────────────────────
        # Each connected component represents one physical person.  We use
        # the smallest local ID within the component as the global ID, which
        # keeps numbers stable across runs and preserves low IDs.
        comps = _get_components()
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}
        for members in comps.values():
            global_id = min(pid for (_, pid) in members)
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
            return

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
            gallery_path = body_dir / "appearance_gallery.npz"
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                new_gallery = {
                    str(remap.get(int(k), int(k))): gdata[k]
                    for k in gdata.files
                }
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
            mhr_model = MHR.from_files(folder=Path(mhr_model_path), lod=1)
            smplx_model = smplx.create(
                model_path=smplx_model_path,
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
            logging.warning(f"Could not initialise SMPLX converter: {e}")
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
