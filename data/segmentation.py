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
                mask_data/       .npy uint16 masks  (pixel value = person ID)
                json_data/       .json per-frame instance metadata
                result/          (optional) annotated visualisation frames
                segmentation.mp4 (optional) visualisation video
            cross_video_id_mapping.json   (if match_across_videos=True)
"""

from __future__ import annotations

import copy
import gc
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.gdsam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from sam2.gdsam2_utils.common_utils import CommonUtils
from sam2.gdsam2_utils.video_utils import create_video_from_images
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

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

    def segment_scene(
        self,
        scene: Scene,
        output_dir: str | Path,
        vis: bool = False,
        match_across_videos: bool = True,
    ) -> dict[str, Path]:
        """Segment every video in *scene* and (optionally) unify IDs.

        Parameters
        ----------
        scene : Scene
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
        dict mapping ``video_id``  ``Path`` to that video's output folder.
        """
        if not self._models_ready:
            self._init_models()

        scene_dir = Path(output_dir) / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        video_results: dict[str, dict] = {}
        global_objects_count = 0

        # --- per-video segmentation ---
        for video in tqdm(scene.videos, desc=f"Segmenting {scene.scene_id}"):
            vid_out = scene_dir / video.video_id
            result = self._segment_video(video, vid_out, global_objects_count)
            video_results[video.video_id] = result
            global_objects_count = result["objects_count"]

            del result
            gc.collect()
            torch.cuda.empty_cache()

        # --- cross-video ID matching ---
        if match_across_videos and len(scene.videos) > 1:
            id_mapping = self._match_across_videos(
                scene, video_results, scene_dir
            )
            if id_mapping:
                self._apply_id_mapping(id_mapping, scene_dir, scene)

        # --- optional visualisation ---
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
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Re-use already extracted frames if present.
        existing = sorted(
            p.name for p in frame_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg")
        )
        if existing:
            print(f"  Reusing {len(existing)} existing frames in {frame_dir}")
            return existing

        frame_names: list[str] = []
        vr = video._make_reader()
        total = len(vr)

        for idx in tqdm(
            range(total),
            desc=f"  Extracting {video.video_id}",
            leave=False,
        ):
            frame_np = vr[idx].asnumpy()  # (H, W, C) RGB — single frame
            name = f"{idx:06d}.jpg"
            cv2.imwrite(
                str(frame_dir / name),
                cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR),
            )
            frame_names.append(name)
            del frame_np  # free immediately

        del vr
        return frame_names

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
            mask_dir = Path(res["mask_data_dir"])
            frame_dir = Path(res["frame_dir"])

            json_files = sorted(json_dir.glob("*.json"))
            if not json_files:
                continue

            mid_json = json_files[len(json_files) // 2]
            with open(mid_json) as f:
                meta = json.load(f)

            mask_path = mask_dir / (mid_json.stem + ".npy")
            if not mask_path.exists():
                continue
            mask_img = np.load(str(mask_path))

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
        """Rewrite saved .npy masks and .json metadata with unified IDs."""
        for video in scene.videos:
            vid_id = video.video_id
            mapping = id_mapping.get(vid_id)
            if mapping is None or all(k == v for k, v in mapping.items()):
                continue

            mask_dir = scene_dir / vid_id / "mask_data"
            json_dir = scene_dir / vid_id / "json_data"

            for mask_path in sorted(mask_dir.glob("*.npy")):
                mask_img = np.load(str(mask_path))
                new_mask = np.zeros_like(mask_img)
                for old_id, new_id in mapping.items():
                    new_mask[mask_img == old_id] = new_id
                unmapped = set(np.unique(mask_img)) - {0} - set(mapping.keys())
                for uid in unmapped:
                    new_mask[mask_img == uid] = uid
                np.save(str(mask_path), new_mask)

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
    # Visualisation
    # ------------------------------------------------------------------

    def _visualize(self, video: Video, video_dir: Path) -> None:
        """Render annotated frames and encode an mp4."""
        frame_dir = video_dir / "frames"
        mask_dir = video_dir / "mask_data"
        json_dir = video_dir / "json_data"
        result_dir = video_dir / "result"
        result_dir.mkdir(exist_ok=True)

        CommonUtils.draw_masks_and_box_with_supervision(
            str(frame_dir), str(mask_dir), str(json_dir), str(result_dir),
        )
        out_video = video_dir / "segmentation.mp4"
        create_video_from_images(
            str(result_dir), str(out_video), frame_rate=int(video.fps),
        )
        print(f"Visualisation saved: {out_video}")