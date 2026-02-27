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
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
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

    # Re-identification: minimum cosine similarity to merge a new SAM2 track
    # into an existing person, and EMA weight for updating gallery features.
    _REID_THRESHOLD: float = 0.65
    _GALLERY_EMA_ALPHA: float = 0.9

    def __init__(
        self,
        sam3d_hf_repo: str = "facebook/sam-3d-body-dinov3",
        sam3d_step: int = 1,
        bbox_padding: float = 0.2,
        smplx_model_path: str | None = None,
        mhr_model_path: str | None = None,
    ):
        self.sam3d_hf_repo = sam3d_hf_repo
        self.sam3d_step = sam3d_step
        self.bbox_padding = bbox_padding
        self.smplx_model_path = smplx_model_path
        self.mhr_model_path = mhr_model_path

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
                )
            except Exception as e:
                logging.error(f"{gpu_label}Error processing {video_id}: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        del estimator
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"{gpu_label}Worker done.")

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
        # ever SIMULTANEOUSLY visible.  Two IDs that share a frame must belong to
        # two different people and can never be merged by re-ID, regardless of how
        # similar their visual features appear.  This prevents the common failure
        # where P2 enters the scene in the single frame where P1 is momentarily
        # missed by SAM3D, causing re-ID to incorrectly fold P2 into P1.
        covisible_ids: set[frozenset] = set()
        for jp in json_files:
            with open(jp) as _f:
                _meta = json.load(_f)
            _ids = {int(s) for s in _meta.get("labels", {}).keys()}
            for _a in _ids:
                for _b in _ids:
                    if _a != _b:
                        covisible_ids.add(frozenset({_a, _b}))

        # Gallery for visual re-identification across SAM2 track interruptions.
        # person_gallery: canonical_id → L2-normalised appearance descriptor (EMA).
        # id_remap: raw SAM2 id → canonical_id for ids that were re-identified.
        # We employ DINOv3 backbone (given by SAM 3D) in order to match people across
        # frames using cosine similarity between visual features
        person_gallery: dict[int, np.ndarray] = {}
        id_remap: dict[int, int] = {}

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
                    if person_id not in person_gallery and person_id not in id_remap:
                        # Brand-new SAM2 track — check whether this person was seen
                        # before under a different ID (e.g. after occlusion).
                        if person_gallery:
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
                                and sims[best_id] >= BodyParameterEstimator._REID_THRESHOLD
                                and frozenset({person_id, best_id}) not in covisible_ids
                            ):
                                id_remap[person_id] = best_id
                                canonical_id = best_id
                                logging.info(
                                    f"{gpu_label}Re-ID: SAM2 id {person_id} → "
                                    f"person {best_id} (sim={sims[best_id]:.3f}) "
                                    f"in {video_id} frame {frame_idx}"
                                )
                            else:
                                person_gallery[person_id] = feat_i.copy()
                        else:
                            person_gallery[person_id] = feat_i.copy()

                    # Update gallery descriptor with EMA for the canonical person.
                    if canonical_id in person_gallery:
                        alpha = BodyParameterEstimator._GALLERY_EMA_ALPHA
                        g = alpha * person_gallery[canonical_id] + (1 - alpha) * feat_i
                        norm = np.linalg.norm(g)
                        person_gallery[canonical_id] = g / norm if norm > 0 else g

                params = {"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)}
                for key in param_keys:
                    if key in body:
                        val = body[key]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        params[key] = np.asarray(val, dtype=np.float32)

                tracks.setdefault(canonical_id, {})[frame_idx] = params

        if not tracks:
            logging.warning(f"{gpu_label}{video_id}: no body detections")
            return

        BodyParameterEstimator._save_body_data_static(
            tracks, body_dir, video_id, param_keys
        )

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
                    new_mask = np.zeros_like(mask_img)
                    for old_id, new_id in id_remap.items():
                        new_mask[mask_img == old_id] = new_id
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
        ).to(self.device)

        # Launch the converter
        converter = Conversion(
            mhr_model = mhr_model,
            smpl_model = smplx_model,
            method = "pytorch", # The alternative is pymomentum
        )

        results = converter.convert_sam3d_output_to_smpl(
            sam3d_outputs = sam3d_outputs,
            return_smpl_meshes=True,
            return_smpl_parameters=True,
            return_smpl_vertices=True,
            return_fitting_errors=True,
        )

        raise ValueError("Find a way to save these based on what it outputs")
