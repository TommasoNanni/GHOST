"""Person segmentation and tracking for multi-view scenes.

Uses Grounding DINO for zero-shot person detection and SAM2 for
mask prediction and temporal tracking.  Provides consistent person IDs
within each video (SAM2 video propagation) and across videos
(appearance-based matching).

Output layout for each scene::

    output_dir/
        <scene_id>/
            <video_id>/
                frames/          extracted JPEGs (can be cleaned up)
                mask_data.npz    compressed mask archive (uint16, pixel value = person ID)
                json_data/       .json per-frame instance metadata
                result/          (optional) annotated visualisation frames
                segmentation.mp4 (optional) visualisation video
            cross_video_id_mapping.json   (if match_across_videos=True)
"""

from __future__ import annotations

import copy
import gc
import io
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.gdsam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from sam2.gdsam2_utils.common_utils import CommonUtils
from sam2.gdsam2_utils.video_utils import create_video_from_images
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from decord import VideoReader, cpu

from data.video_dataset import Scene, Video


class PersonSegmenter:
    """Segment and track people across all videos in a :class:`Scene`.

    Models are loaded lazily on the first call to :meth:`segment_scene`.

    Parameters
    ----------
    sam2_checkpoint : str
        Path to a SAM2 model checkpoint file.
    model_cfg : str
        Path to the SAM2 YAML config (relative to the SAM2 package).
    gdino_model_id : str
        HuggingFace model id for Grounding DINO.
    device : str
        ``"cuda"`` or ``"cpu"``.
    text_prompt : str
        Detection query for Grounding DINO (lowercase, ends with ``'.'``).
    box_threshold : float
        Minimum confidence for detected bounding boxes.
    text_threshold : float
        Minimum confidence for text-grounded detections.
    detection_step : int
        Run Grounding DINO every *detection_step* frames; SAM2 propagates
        masks for the frames in between.
    """

    def __init__(
        self,
        sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        text_prompt: str = "person.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        detection_step: int = 50,
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.gdino_model_id = gdino_model_id
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detection_step = detection_step

        # Populated by _init_models()
        self._video_predictor = None
        self._image_predictor = None
        self._gdino_processor = None
        self._gdino_model = None
        self._models_ready = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        """Load SAM2 and Grounding DINO into GPU memory (once)."""
        if self._models_ready:
            return

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._video_predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint
        )
        sam2_model = build_sam2(
            self.model_cfg, self.sam2_checkpoint, device=self.device
        )
        self._image_predictor = SAM2ImagePredictor(sam2_model)

        self._gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self._gdino_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.gdino_model_id)
            .to(self.device)
        )

        self._models_ready = True
        print(f"PersonSegmenter: models loaded on {self.device}")

    def _free_models(self) -> None:
        """Release GPU models so child processes can use the VRAM."""
        self._video_predictor = None
        self._image_predictor = None
        self._gdino_processor = None
        self._gdino_model = None
        self._models_ready = False
        gc.collect()
        torch.cuda.empty_cache()

    def segment_scene(
        self,
        scene: Scene,
        output_dir: str | Path,
        vis: bool = False,
        match_across_videos: bool = True,
        _objects_count_start: int = 0,
    ) -> dict[str, Path]:
        """Segment every video in *scene* and (optionally) unify IDs.

        Videos are processed **in parallel** across all available GPUs.
        Each video is assigned to a GPU and runs in its own process with
        its own model instances.

        Parameters
        ----------
        scene : Scenep
            Scene containing one or more :class:`Video` objects.
        output_dir : str | Path
            Root directory where results are written.
        vis : bool
            If ``True``, render annotated frames and an mp4 per video.
        match_across_videos : bool
            If ``True``, run appearance-based matching to assign consistent
            person IDs across the different camera views.

        Returns
        -------
        dict mapping ``video_id`` → ``Path`` to that video's output folder.
        """
        scene_dir = Path(output_dir) / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        num_videos = len(scene.videos)

        if num_gpus <= 1:
            return self._segment_scene_sequential(
                scene, output_dir, vis, match_across_videos, _objects_count_start
            )

        print(f"Parallel segmentation: {num_videos} videos across {num_gpus} GPUs")

        # Build worker arguments — each video gets a GPU (round-robin)
        # and a large ID offset so person IDs don't collide.
        video_results: dict[str, dict] = {}
        worker_args = []
        for vi, video in enumerate(scene.videos):
            vid_out = scene_dir / video.video_id
            if PersonSegmenter._is_segmented(vid_out):
                print(f"  {video.video_id}: already segmented, loading cached data")
                video_results[video.video_id] = PersonSegmenter._load_cached_result(
                    vid_out, video.video_id
                )
                continue
            gpu_id = vi % num_gpus
            worker_args.append((
                gpu_id,
                str(video.path),
                video.video_id,
                str(vid_out),
                _objects_count_start + vi * 10000,
                self.sam2_checkpoint,
                self.model_cfg,
                self.gdino_model_id,
                self.text_prompt,
                self.box_threshold,
                self.text_threshold,
                self.detection_step,
            ))

        newly_segmented = bool(worker_args)
        if newly_segmented:
            # Free the main-process models before spawning children
            self._free_models()

            mp.set_start_method("spawn", force=True)
            with mp.Pool(processes=min(num_gpus, len(worker_args))) as pool:
                results = pool.starmap(PersonSegmenter._segment_video_on_gpu, worker_args)

            for r in results:
                video_results[r["video_id"]] = r

            # --- compact per-frame .npy files into a single compressed .npz ---
            for video in scene.videos:
                PersonSegmenter._compact_mask_data(scene_dir / video.video_id)

        # --- cross-video ID matching (reads from .npz) ---
        if newly_segmented and match_across_videos and len(scene.videos) > 1:
            id_mapping = self._match_across_videos(
                scene, video_results, scene_dir
            )
            if id_mapping:
                self._apply_id_mapping(id_mapping, scene_dir, scene)

        # --- optional visualisation (unpacks .npz temporarily) ---
        if vis:
            for video in scene.videos:
                self._visualize(video, scene_dir / video.video_id)

        return {v.video_id: scene_dir / v.video_id for v in scene.videos}

    def cleanup_frames(self, output_dir: str | Path, scene: Scene) -> None:
        """Delete extracted JPEG directories to reclaim disk space."""
        scene_dir = Path(output_dir) / scene.scene_id
        for video in scene.videos:
            frame_dir = scene_dir / video.video_id / "frames"
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
                print(f"Removed {frame_dir}")

    # ------------------------------------------------------------------
    # Sequential fallback (single GPU)
    # ------------------------------------------------------------------

    def _segment_scene_sequential(
        self,
        scene: Scene,
        output_dir: str | Path,
        vis: bool = False,
        match_across_videos: bool = True,
        objects_count_start: int = 0,
    ) -> dict[str, Path]:
        """Original sequential fallback for single-GPU environments."""
        scene_dir = Path(output_dir) / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        video_results: dict[str, dict] = {}
        global_objects_count = objects_count_start
        newly_segmented = False

        for video in tqdm(scene.videos, desc=f"Segmenting {scene.scene_id}"):
            vid_out = scene_dir / video.video_id
            if PersonSegmenter._is_segmented(vid_out):
                print(f"  {video.video_id}: already segmented, loading cached data")
                result = PersonSegmenter._load_cached_result(vid_out, video.video_id)
                video_results[video.video_id] = result
                global_objects_count = result["objects_count"]
                continue

            if not self._models_ready:
                self._init_models()

            result = self._segment_video(video, vid_out, global_objects_count)
            video_results[video.video_id] = result
            global_objects_count = result["objects_count"]
            newly_segmented = True

            del result
            gc.collect()
            torch.cuda.empty_cache()

        if newly_segmented:
            # --- compact per-frame .npy files into a single compressed .npz ---
            for video in scene.videos:
                PersonSegmenter._compact_mask_data(scene_dir / video.video_id)

        if newly_segmented and match_across_videos and len(scene.videos) > 1:
            id_mapping = self._match_across_videos(
                scene, video_results, scene_dir
            )
            if id_mapping:
                self._apply_id_mapping(id_mapping, scene_dir, scene)

        if vis:
            for video in scene.videos:
                self._visualize(video, scene_dir / video.video_id)

        return {v.video_id: scene_dir / v.video_id for v in scene.videos}

    # ------------------------------------------------------------------
    # Parallel GPU worker
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_video_on_gpu(
        gpu_id: int,
        video_path: str,
        video_id: str,
        output_dir: str,
        objects_count_start: int,
        sam2_checkpoint: str,
        model_cfg: str,
        gdino_model_id: str,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        detection_step: int,
    ) -> dict:
        """Segment one video on a specific GPU (runs in a child process).

        Loads its own models and uses decord directly for frame extraction
        (no Video object re-creation needed).
        """
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        if torch.cuda.get_device_properties(gpu_id).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load models on this GPU
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        image_predictor = SAM2ImagePredictor(sam2_model)
        gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        gdino_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(gdino_model_id)
            .to(device)
        )

        print(f"  [GPU {gpu_id}] Models loaded for {video_id}")

        # Set up output directories
        vid_out = Path(output_dir)
        vid_out.mkdir(parents=True, exist_ok=True)
        mask_data_dir = vid_out / "mask_data"
        json_data_dir = vid_out / "json_data"
        mask_data_dir.mkdir(exist_ok=True)
        json_data_dir.mkdir(exist_ok=True)

        # Extract frames directly with decord (no Video object needed)
        frame_dir = vid_out / "frames"
        frame_names = PersonSegmenter._extract_frames_from_path(
            video_path, video_id, frame_dir
        )

        # Init SAM2 video predictor
        inference_state = video_predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        sam2_masks = MaskDictionaryModel()
        objects_count = objects_count_start
        step = detection_step

        print(f"  [GPU {gpu_id}] {video_id}: {len(frame_names)} frames, step={step}")

        for start_idx in range(0, len(frame_names), step):
            img_path = frame_dir / frame_names[start_idx]
            image = Image.open(img_path)
            base_name = frame_names[start_idx].split(".")[0]

            mask_dict = MaskDictionaryModel(
                promote_type="mask",
                mask_name=f"mask_{base_name}.npy",
            )

            # Grounding DINO detection
            inputs = gdino_processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = gdino_model(**inputs)

            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )

            input_boxes = results[0]["boxes"]
            labels = results[0]["labels"]

            if input_boxes.shape[0] != 0:
                image_predictor.set_image(np.array(image.convert("RGB")))
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(device),
                    box_list=torch.tensor(input_boxes),
                    label_list=labels,
                    scores_list=torch.tensor(scores).to(device),
                )

                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks,
                    iou_threshold=0.8,
                    objects_count=objects_count,
                )
            else:
                mask_dict = sam2_masks

            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(
                    str(mask_data_dir),
                    str(json_data_dir),
                    image_name_list=frame_names[start_idx : start_idx + step],
                )
                continue

            # SAM2 video propagation
            video_predictor.reset_state(inference_state)

            for obj_id, obj_info in mask_dict.labels.items():
                video_predictor.add_new_mask(
                    inference_state, start_idx, obj_id, obj_info.mask,
                )

            video_segments: dict[int, MaskDictionaryModel] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in (
                video_predictor.propagate_in_video(
                    inference_state,
                    max_frame_num_to_track=step,
                    start_frame_idx=start_idx,
                )
            ):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = out_mask_logits[i] > 0.0
                    obj_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id),
                    )
                    obj_info.update_box()
                    frame_masks.labels[out_obj_id] = obj_info
                    frame_masks.mask_name = (
                        f"mask_{frame_names[out_frame_idx].split('.')[0]}.npy"
                    )
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            # Save masks + metadata
            for frame_idx, fmasks in video_segments.items():
                mask_img = torch.zeros(fmasks.mask_height, fmasks.mask_width)
                for obj_id, obj_info in fmasks.labels.items():
                    mask_img[obj_info.mask == True] = obj_id

                np.save(
                    str(mask_data_dir / fmasks.mask_name),
                    mask_img.numpy().astype(np.uint16),
                )
                json_path = json_data_dir / fmasks.mask_name.replace(".npy", ".json")
                with open(json_path, "w") as f:
                    json.dump(fmasks.to_dict(), f)

        del inference_state, video_predictor, image_predictor, gdino_model, gdino_processor
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  [GPU {gpu_id}] {video_id}: done")
        return {
            "video_id": video_id,
            "objects_count": objects_count,
            "frame_dir": str(frame_dir),
            "mask_data_dir": str(mask_data_dir),
            "json_data_dir": str(json_data_dir),
        }

    @staticmethod
    def _extract_frames_from_path(
        video_path: str, video_id: str, frame_dir: Path
    ) -> list[str]:
        """Extract frames using decord directly from a video path."""
        frame_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            p.name for p in frame_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg")
        )
        if existing:
            print(f"  Reusing {len(existing)} existing frames in {frame_dir}")
            return existing

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        frame_names: list[str] = []

        for idx in tqdm(range(total), desc=f"  Extracting {video_id}", leave=False):
            frame_np = vr[idx].asnumpy()
            name = f"{idx:06d}.jpg"
            cv2.imwrite(
                str(frame_dir / name),
                cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR),
            )
            frame_names.append(name)
            del frame_np

        del vr
        return frame_names

    # ------------------------------------------------------------------
    # Single-video processing (used by sequential fallback)
    # ------------------------------------------------------------------

    def _segment_video(
        self,
        video: Video,
        output_dir: Path,
        objects_count: int = 0,
    ) -> dict:
        """Run the full GDINO + SAM2 pipeline on a single video."""

        # Create output folders

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_data_dir = output_dir / "mask_data"
        json_data_dir = output_dir / "json_data"
        mask_data_dir.mkdir(exist_ok=True)
        json_data_dir.mkdir(exist_ok=True)

        # SAM2 requires a directory of numbered JPEGs.
        frame_dir = output_dir / "frames"
        frame_names = self._extract_frames(video, frame_dir)

        # Initialize the video predictor
        # We don't predict again people at every frame, we predict every
        # step frames, and propagate to the remaining frames using
        # SAM2's video predictor

        inference_state = self._video_predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        sam2_masks = MaskDictionaryModel()
        step = self.detection_step

        print(f"  {video.video_id}: {len(frame_names)} frames, step={step}")

        for start_idx in range(0, len(frame_names), step):
            img_path = frame_dir / frame_names[start_idx]
            image = Image.open(img_path)
            base_name = frame_names[start_idx].split(".")[0]

            mask_dict = MaskDictionaryModel(
                promote_type="mask",
                mask_name=f"mask_{base_name}.npy",
            )

            # ---- Grounding DINO detection ----
            inputs = self._gdino_processor(
                images=image, text=self.text_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._gdino_model(**inputs)

            # Predict the boxes for people in the rooted frames

            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )

            input_boxes = results[0]["boxes"]
            labels = results[0]["labels"]

            if input_boxes.shape[0] != 0:
                # Now that we detected people, we just extract the masks for them
                self._image_predictor.set_image(np.array(image.convert("RGB")))
                masks, scores, logits = self._image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(self.device),
                    box_list=torch.tensor(input_boxes),
                    label_list=labels,
                    scores_list=torch.tensor(scores).to(self.device),
                )

                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks,
                    iou_threshold=0.8,
                    objects_count=objects_count,
                )
            else:
                mask_dict = sam2_masks

            # Nothing detected in this window — save empties.
            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(
                    str(mask_data_dir),
                    str(json_data_dir),
                    image_name_list=frame_names[start_idx : start_idx + step],
                )
                continue

            # ---- SAM2 video propagation ----
            self._video_predictor.reset_state(inference_state)

            for obj_id, obj_info in mask_dict.labels.items():
                self._video_predictor.add_new_mask(
                    inference_state, start_idx, obj_id, obj_info.mask,
                )

            video_segments: dict[int, MaskDictionaryModel] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in (
                self._video_predictor.propagate_in_video(
                    inference_state,
                    max_frame_num_to_track=step,
                    start_frame_idx=start_idx,
                )
            ):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = out_mask_logits[i] > 0.0
                    obj_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id),
                    )
                    obj_info.update_box()
                    frame_masks.labels[out_obj_id] = obj_info
                    frame_masks.mask_name = (
                        f"mask_{frame_names[out_frame_idx].split('.')[0]}.npy"
                    )
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            # ---- Save masks + metadata ----
            for frame_idx, fmasks in video_segments.items():
                mask_img = torch.zeros(fmasks.mask_height, fmasks.mask_width)
                for obj_id, obj_info in fmasks.labels.items():
                    mask_img[obj_info.mask == True] = obj_id

                np.save(
                    str(mask_data_dir / fmasks.mask_name),
                    mask_img.numpy().astype(np.uint16),
                )
                json_path = json_data_dir / fmasks.mask_name.replace(
                    ".npy", ".json"
                )
                with open(json_path, "w") as f:
                    json.dump(fmasks.to_dict(), f)

        del inference_state, video_segments
        self._image_predictor.reset_predictor()
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "objects_count": objects_count,
            "frame_dir": frame_dir,
            "mask_data_dir": mask_data_dir,
            "json_data_dir": json_data_dir,
        }


    def _extract_frames(self, video: Video, frame_dir: Path) -> list[str]:
        """Decode every frame of *video* and write numbered JPEGs.

        Returns the sorted list of file names (``"000000.jpg"``, ...).
        """
        return self._extract_frames_from_path(
            str(video.path), video.video_id, frame_dir
        )

    # ------------------------------------------------------------------
    # NPZ helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask_from_npz(npz_path: Path, frame_stem: str) -> np.ndarray | None:
        """Load a single frame's mask array from a mask_data.npz file.

        Parameters
        ----------
        npz_path : Path
            Path to the ``mask_data.npz`` archive.
        frame_stem : str
            The key inside the archive, e.g. ``"mask_000042"``.

        Returns ``None`` if the archive or the key does not exist.
        """
        if not npz_path.exists():
            return None
        with zipfile.ZipFile(str(npz_path), "r") as zf:
            key = frame_stem + ".npy"
            if key not in zf.namelist():
                return None
            with zf.open(key) as f:
                return np.load(io.BytesIO(f.read()))

    @staticmethod
    def _is_segmented(vid_out: Path) -> bool:
        """Return True if this video has already been segmented."""
        return (vid_out / "mask_data.npz").exists()

    @staticmethod
    def _load_cached_result(vid_out: Path, video_id: str) -> dict:
        """Build a result dict from already-present segmentation data.

        Reads ``mask_data.npz`` to determine the maximum person ID so that
        sequential processing can continue with a non-colliding ID offset.
        """
        npz_path = vid_out / "mask_data.npz"
        objects_count = 0
        with zipfile.ZipFile(str(npz_path), "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()))
                max_val = int(arr.max())
                if max_val > objects_count:
                    objects_count = max_val

        return {
            "video_id": video_id,
            "objects_count": objects_count,
            "frame_dir": str(vid_out / "frames"),
            "mask_data_dir": str(vid_out / "mask_data"),
            "json_data_dir": str(vid_out / "json_data"),
        }

    # ------------------------------------------------------------------
    # Cross-video appearance matching
    # ------------------------------------------------------------------

    def _match_across_videos(
        self,
        scene: Scene,
        video_results: dict[str, dict],
        scene_dir: Path,
    ) -> dict[str, dict[int, int]] | None:
        """Build a global person-ID mapping using colour-histogram similarity.

        Uses the temporally-middle frame of each video as the reference
        for computing per-person appearance features, then greedily matches
        identities from every video to those of the first (reference) video.

        Returns ``{video_id: {local_id: global_id}}`` or ``None``.
        """
        video_features: dict[str, dict[int, np.ndarray]] = {}

        for video in scene.videos:
            vid_id = video.video_id
            res = video_results[vid_id]
            json_dir = Path(res["json_data_dir"])
            frame_dir = Path(res["frame_dir"])
            npz_path = Path(res["json_data_dir"]).parent / "mask_data.npz"

            json_files = sorted(json_dir.glob("*.json"))
            if not json_files:
                continue

            mid_json = json_files[len(json_files) // 2]
            with open(mid_json) as f:
                meta = json.load(f)

            mask_img = PersonSegmenter._load_mask_from_npz(npz_path, mid_json.stem)
            if mask_img is None:
                continue

            frame_idx_str = mid_json.stem.replace("mask_", "")
            frame_path = frame_dir / f"{frame_idx_str}.jpg"
            if not frame_path.exists():
                continue
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue

            features: dict[int, np.ndarray] = {}
            for str_id in meta.get("labels", {}):
                pid = int(str_id)
                person_mask = mask_img == pid
                if person_mask.sum() < 100:
                    continue
                features[pid] = self._appearance_feature(frame_bgr, person_mask)

            video_features[vid_id] = features

        if len(video_features) < 2:
            return None

        ref_vid_id = list(video_features.keys())[0]
        ref_feats = video_features[ref_vid_id]
        if not ref_feats:
            return None

        id_mapping: dict[str, dict[int, int]] = {
            ref_vid_id: {pid: pid for pid in ref_feats}
        }

        for vid_id, feats in video_features.items():
            if vid_id == ref_vid_id or not feats:
                continue
            id_mapping[vid_id] = self._greedy_match(ref_feats, feats)

        mapping_path = scene_dir / "cross_video_id_mapping.json"
        serialisable = {
            k: {str(a): b for a, b in v.items()}
            for k, v in id_mapping.items()
        }
        with open(mapping_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Cross-video mapping saved to {mapping_path}")

        return id_mapping

    @staticmethod
    def _appearance_feature(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """HSV colour-histogram descriptor for the masked region."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = mask.astype(np.uint8)
        h = cv2.calcHist([hsv], [0], m, [32], [0, 180])
        s = cv2.calcHist([hsv], [1], m, [32], [0, 256])
        v = cv2.calcHist([hsv], [2], m, [16], [0, 256])
        feat = np.concatenate([h.ravel(), s.ravel(), v.ravel()])
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @staticmethod
    def _greedy_match(
        ref: dict[int, np.ndarray],
        query: dict[int, np.ndarray],
    ) -> dict[int, int]:
        """Greedy cosine-similarity matching of *query* IDs to *ref* IDs."""
        ref_ids = list(ref.keys())
        query_ids = list(query.keys())
        if not ref_ids or not query_ids:
            return {}

        sim = np.zeros((len(query_ids), len(ref_ids)))
        for i, qid in enumerate(query_ids):
            for j, rid in enumerate(ref_ids):
                sim[i, j] = float(np.dot(query[qid], ref[rid]))

        mapping: dict[int, int] = {}
        used: set[int] = set()
        for flat in np.argsort(-sim.ravel()):
            qi, rj = divmod(int(flat), len(ref_ids))
            qid, rid = query_ids[qi], ref_ids[rj]
            if qid in mapping or rid in used:
                continue
            mapping[qid] = rid
            used.add(rid)
            if len(mapping) == min(len(query_ids), len(ref_ids)):
                break

        # Unmatched queries keep their original local IDs.
        for qid in query_ids:
            if qid not in mapping:
                mapping[qid] = qid
        return mapping

    # ------------------------------------------------------------------
    # Apply cross-video remapping
    # ------------------------------------------------------------------

    def _apply_id_mapping(
        self,
        id_mapping: dict[str, dict[int, int]],
        scene_dir: Path,
        scene: Scene,
    ) -> None:
        """Rewrite mask_data.npz and .json metadata with unified IDs."""
        for video in scene.videos:
            vid_id = video.video_id
            mapping = id_mapping.get(vid_id)
            if mapping is None or all(k == v for k, v in mapping.items()):
                continue

            npz_path = scene_dir / vid_id / "mask_data.npz"
            json_dir = scene_dir / vid_id / "json_data"

            # Stream-remap every frame in the archive without loading all into RAM.
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
                        for old_id, new_id in mapping.items():
                            new_mask[mask_img == old_id] = new_id
                        unmapped = set(np.unique(mask_img)) - {0} - set(mapping.keys())
                        for uid in unmapped:
                            new_mask[mask_img == uid] = uid
                        buf = io.BytesIO()
                        np.save(buf, new_mask)
                        zf_out.writestr(name, buf.getvalue())
                tmp_path.replace(npz_path)

            for json_path in sorted(json_dir.glob("*.json")):
                with open(json_path) as f:
                    data = json.load(f)
                if "labels" in data:
                    new_labels = {}
                    for str_id, info in data["labels"].items():
                        old_id = int(str_id)
                        new_id = mapping.get(old_id, old_id)
                        info["instance_id"] = new_id
                        new_labels[str(new_id)] = info
                    data["labels"] = new_labels
                with open(json_path, "w") as f:
                    json.dump(data, f)

        print("Cross-video ID remapping applied.")

    # ------------------------------------------------------------------
    # Mask compaction
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_mask_data(video_dir: Path) -> None:
        """Merge per-frame .npy masks into one compressed .npz, then delete the folder.

        Replaces ``mask_data/*.npy`` (one uncompressed file per frame) with a
        single ``mask_data.npz`` whose keys are the original file stems
        (e.g. ``"mask_000000"``).  For sparse uint16 masks the compression
        ratio is typically 20-50x.

        Must be called **after** cross-video ID remapping and visualisation,
        since those steps still read the individual .npy files.
        """
        mask_data_dir = video_dir / "mask_data"
        if not mask_data_dir.exists():
            return

        npy_files = sorted(mask_data_dir.glob("*.npy"))
        if not npy_files:
            shutil.rmtree(str(mask_data_dir))
            return

        original_mb = sum(f.stat().st_size for f in npy_files) / 1024 / 1024
        npz_path = video_dir / "mask_data.npz"

        with zipfile.ZipFile(str(npz_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in npy_files:
                arr = np.load(str(f))
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f.stem + ".npy", buf.getvalue())

        shutil.rmtree(str(mask_data_dir))

        compressed_mb = npz_path.stat().st_size / 1024 / 1024
        print(
            f"  mask_data: {len(npy_files)} .npy files, {original_mb:.0f} MB"
            f" → mask_data.npz {compressed_mb:.0f} MB"
            f" ({100 * compressed_mb / original_mb:.0f}% of original)"
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _visualize(self, video: Video, video_dir: Path) -> None:
        """Render annotated frames and encode an mp4.

        Temporarily unpacks ``mask_data.npz`` into a ``mask_data/`` directory
        so the visualisation utility can read individual frame files, then
        removes it once rendering is complete.
        """
        frame_dir = video_dir / "frames"
        npz_path = video_dir / "mask_data.npz"
        json_dir = video_dir / "json_data"
        result_dir = video_dir / "result"
        result_dir.mkdir(exist_ok=True)

        # Unpack .npz → mask_data/ for the visualisation utility.
        mask_dir = video_dir / "mask_data"
        mask_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(str(npz_path), "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()))
                np.save(str(mask_dir / name), arr)

        try:
            CommonUtils.draw_masks_and_box_with_supervision(
                str(frame_dir), str(mask_dir), str(json_dir), str(result_dir),
            )
        finally:
            shutil.rmtree(str(mask_dir))

        out_video = video_dir / "segmentation.mp4"
        create_video_from_images(
            str(result_dir), str(out_video), frame_rate=int(video.fps),
        )
        print(f"Visualisation saved: {out_video}")
