"""Person segmentation and tracking for multi-view scenes.

Uses SAM3 for zero-shot text-prompted person detection, mask prediction,
and temporal tracking.  SAM3 combines detection and tracking in one model,
replacing the previous Grounding DINO + SAM2 pipeline.  Provides
consistent person IDs within each video and across videos
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

import functools
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

from sam3.model.sam3_video_predictor import Sam3VideoPredictor  # type: ignore
from sam3.visualization_utils import render_masklet_frame       # type: ignore

from decord import VideoReader, cpu

from data.video_dataset import Scene, Video


class PersonSegmenter:
    """Segment and track people across all videos in a :class:`Scene`.

    Uses SAM3 which jointly handles text-prompted detection, mask
    prediction, and temporal tracking in a single model.

    Models are loaded lazily on the first call to :meth:`segment_scene`.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to a SAM3 checkpoint file.  When ``None``, the checkpoint
        is automatically downloaded from HuggingFace.
    device : str
        ``"cuda"`` or ``"cpu"``.
    text_prompt : str
        Text query for SAM3 detection (e.g. ``"person"``).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        text_prompt: str = "person",
        redetect_interval: int = 15,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.text_prompt = text_prompt
        self.redetect_interval = redetect_interval

        # Populated by _init_models()
        self._predictor: Sam3VideoPredictor | None = None
        self._models_ready = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        """Load SAM3 into GPU memory (once)."""
        if self._models_ready:
            return

        import importlib.util as _ilu
        import os as _os
        # pkg_resources.resource_filename fails when sam3 is a namespace package
        # (no __file__). Locate the asset via importlib instead.
        _mb_spec = _ilu.find_spec("sam3.model_builder")
        _sam3_dir = _os.path.dirname(_os.path.abspath(_mb_spec.origin))
        _bpe_path = _os.path.join(_sam3_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")
        self._predictor = Sam3VideoPredictor(
            checkpoint_path=self.checkpoint_path,
            bpe_path=_bpe_path,
        )
        # Use async (lazy) frame loading + CPU-offloaded frames to avoid OOM.
        # See _segment_video_on_gpu for the full explanation.
        self._predictor.async_loading_frames = True
        self._predictor.model.init_state = functools.partial(
            self._predictor.model.init_state, offload_video_to_cpu=True
        )

        self._models_ready = True
        print(f"PersonSegmenter: SAM3 model loaded on {self.device}")

    def _free_models(self) -> None:
        """Release GPU models so child processes can use the VRAM."""
        if self._predictor is not None:
            self._predictor.shutdown()
        self._predictor = None
        self._models_ready = False
        gc.collect()
        torch.cuda.empty_cache()

    def segment_scene(
        self,
        scene: Scene,
        output_dir: str | Path,
        vis: bool = False,
        _objects_count_start: int = 0,
    ) -> dict[str, Path]:
        """Segment every video in scene independently.

        Videos are processed in parallel across all available GPUs.
        Each video is assigned to a GPU and runs in its own process with
        its own model instances.

        Parameters
        ----------
        scene : Scene
            Scene containing one or more :class:`Video` objects.
        output_dir : str | Path
            Root directory where results are written.
        vis : bool
            If ``True``, render annotated frames and an mp4 per video.

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
                scene, output_dir, vis, _objects_count_start
            )

        print(f"Parallel segmentation: {num_videos} videos across {num_gpus} GPUs")

        # Build worker arguments — each video gets a GPU (round-robin)
        # and a large ID offset so person IDs don't collide.
        video_results: dict[str, dict] = {}
        worker_args = []
        for vi, video in enumerate(scene.videos):
            vid_out = scene_dir / video.video_id
            # If a video has already been segmented, do not do it again
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
                self.checkpoint_path,
                self.text_prompt,
                self.redetect_interval,
            ))

        newly_segmented = bool(worker_args)
        if newly_segmented:
            # Free the main-process models before spawning children
            self._free_models()

            # Group tasks by GPU so each worker process handles all of its
            # GPU's videos sequentially.  This prevents two workers from
            # loading a SAM3 model onto the same GPU simultaneously (which
            # would exceed VRAM and cause an OOM).
            gpu_batches: dict[int, list] = {}
            for args in worker_args:
                gpu_id = args[0]
                gpu_batches.setdefault(gpu_id, []).append(args)
            gpu_batch_list = list(gpu_batches.values())

            mp.set_start_method("spawn", force=True)
            with mp.Pool(processes=len(gpu_batch_list)) as pool:
                batch_results = pool.map(PersonSegmenter._segment_gpu_batch, gpu_batch_list)
            results = [r for batch in batch_results for r in batch]

            for r in results:
                video_results[r["video_id"]] = r

            # --- compact per-frame .npy files into a single compressed .npz to save space ---
            for video in scene.videos:
                PersonSegmenter._compact_mask_data(scene_dir / video.video_id)

        # --- optional visualisation (unpacks .npz temporarily) ---
        if vis:
            for video in scene.videos:
                self._visualize(video, scene_dir / video.video_id)

        return {v.video_id: scene_dir / v.video_id for v in scene.videos}

    # ------------------------------------------------------------------
    # Sequential fallback (single GPU)
    # ------------------------------------------------------------------

    def _segment_scene_sequential(
        self,
        scene: Scene,
        output_dir: str | Path,
        vis: bool = False,
        objects_count_start: int = 0,
    ) -> dict[str, Path]:
        """Original sequential fallback for single-GPU environments."""

        # Create a directory for the scene
        scene_dir = Path(output_dir) / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        video_results: dict[str, dict] = {}
        newly_segmented = False

        # Segment each video individually
        for video in tqdm(scene.videos, desc=f"Segmenting {scene.scene_id}"):
            vid_out = scene_dir / video.video_id
            if PersonSegmenter._is_segmented(vid_out):
                # If a video is already segmented, avoid repeating
                print(f"  {video.video_id}: already segmented, loading cached data")
                result = PersonSegmenter._load_cached_result(vid_out, video.video_id)
                video_results[video.video_id] = result
                continue

            if not self._models_ready:
                self._init_models()

            result = self._segment_video(video, vid_out)
            video_results[video.video_id] = result
            newly_segmented = True

            del result
            gc.collect()
            torch.cuda.empty_cache()

        if newly_segmented:
            # --- compact per-frame .npy files into a single compressed .npz to save space---
            for video in scene.videos:
                PersonSegmenter._compact_mask_data(scene_dir / video.video_id)

        if vis:
            # optionally save the segmented video as a .mp4 file
            for video in scene.videos:
                self._visualize(video, scene_dir / video.video_id)

        return {v.video_id: scene_dir / v.video_id for v in scene.videos}


    def _segment_video(
        self,
        video: Video,
        output_dir: Path,
    ) -> dict:
        """Run the SAM3 detection + tracking pipeline on a single video."""

        # Create output folders
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_data_dir = output_dir / "mask_data"
        json_data_dir = output_dir / "json_data"
        mask_data_dir.mkdir(exist_ok=True)
        json_data_dir.mkdir(exist_ok=True)

        # SAM3 accepts a directory of numbered JPEGs or an MP4 file.
        frame_dir = output_dir / "frames"
        frame_names = self._extract_frames(video, frame_dir)

        print(f"  {video.video_id}: {len(frame_names)} frames")

        # Start a SAM3 session on the video frames
        response = self._predictor.handle_request(
            request=dict(type="start_session", resource_path=str(frame_dir))
        )
        session_id = response["session_id"]

        try:
            # Add text prompt at regular intervals so people entering after
            # frame 0 are also detected and tracked.
            keyframes = range(0, len(frame_names), self.redetect_interval)
            for kf in keyframes:
                self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=kf,
                        text=self.text_prompt,
                    )
                )

            # Propagate through the entire video
            objects_count = 0
            for response in self._predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]
                outputs = response["outputs"]

                obj_ids = outputs["out_obj_ids"]       # (N,)
                masks = outputs["out_binary_masks"]    # (N, H, W) bool
                boxes_xywh = outputs["out_boxes_xywh"] # (N, 4) normalised
                scores = outputs["out_probs"]           # (N,)

                frame_name = frame_names[frame_idx]
                base_name = frame_name.split(".")[0]
                mask_name = f"mask_{base_name}.npy"

                if len(obj_ids) == 0:
                    # No detections — save empty mask and metadata
                    h, w = Image.open(frame_dir / frame_name).size[::-1]
                    empty_mask = np.zeros((h, w), dtype=np.uint16)
                    np.save(str(mask_data_dir / mask_name), empty_mask)
                    meta = {
                        "mask_name": mask_name,
                        "mask_height": h,
                        "mask_width": w,
                        "labels": {},
                    }
                    json_path = json_data_dir / mask_name.replace(".npy", ".json")
                    with open(json_path, "w") as f:
                        json.dump(meta, f)
                    continue

                H, W = masks.shape[1], masks.shape[2]

                # Build combined uint16 mask (pixel value = person ID)
                mask_img = np.zeros((H, W), dtype=np.uint16)
                labels_meta: dict[str, dict] = {}

                for i, obj_id in enumerate(obj_ids):
                    oid = int(obj_id)
                    binary = masks[i]  # (H, W) bool
                    mask_img[binary] = oid

                    # Compute bounding box from binary mask
                    ys, xs = np.where(binary)
                    if len(ys) > 0:
                        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    else:
                        x1 = y1 = x2 = y2 = 0

                    labels_meta[str(oid)] = {
                        "instance_id": oid,
                        "class_name": self.text_prompt,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "score": float(scores[i]),
                    }

                    if oid > objects_count:
                        objects_count = oid

                np.save(str(mask_data_dir / mask_name), mask_img)

                meta = {
                    "mask_name": mask_name,
                    "mask_height": H,
                    "mask_width": W,
                    "labels": labels_meta,
                }
                json_path = json_data_dir / mask_name.replace(".npy", ".json")
                with open(json_path, "w") as f:
                    json.dump(meta, f)
        finally:
            # Always close the session to free GPU memory
            self._predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "objects_count": objects_count,
            "frame_dir": frame_dir,
            "mask_data_dir": mask_data_dir,
            "json_data_dir": json_data_dir,
        }


    # ------------------------------------------------------------------
    # Parallel GPU worker
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_gpu_batch(task_list: list) -> list:
        """Process multiple videos sequentially on a single GPU.

        Accepts a list of arg-tuples for :meth:`_segment_video_on_gpu` that
        all share the same ``gpu_id``.  Running them inside one process
        guarantees only one SAM3 model exists on that GPU at a time.
        """
        results = []
        for task_args in task_list:
            results.append(PersonSegmenter._segment_video_on_gpu(*task_args))
        return results

    @staticmethod
    def _segment_video_on_gpu(
        gpu_id: int,
        video_path: str,
        video_id: str,
        output_dir: str,
        checkpoint_path: str | None,
        text_prompt: str,
        redetect_interval: int = 30,
    ) -> dict:
        """Segment one video on a specific GPU (runs in a child process).

        Loads its own SAM3 model and processes the video end-to-end.
        """
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        # Load SAM3 model on this GPU
        import sam3.model_builder as _mb
        _sam3_bpe = str(Path(_mb.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        predictor = Sam3VideoPredictor(checkpoint_path=checkpoint_path, bpe_path=_sam3_bpe)
        # Use async (lazy) frame loading + CPU-offloaded frames to avoid OOM.
        #
        # async_loading_frames=True  → returns AsyncImageFrameLoader instead of
        #   a large tensor; copy_data_to_device skips it (unknown type → as-is).
        # offload_video_to_cpu=True  → the async loader keeps each frame on CPU;
        #   SAM3's _get_img_feats moves individual frames to GPU on-demand.
        #
        # Without both flags, all ~3k frames (~19 GiB) end up on the GPU
        # (either as a contiguous tensor, or accumulated by the async loader's
        # background thread), causing OOM alongside the ~5 GiB model.
        # start_session doesn't forward offload_video_to_cpu, so we inject it.
        predictor.async_loading_frames = True
        predictor.model.init_state = functools.partial(
            predictor.model.init_state, offload_video_to_cpu=True
        )
        print(f"  [GPU {gpu_id}] SAM3 model loaded for {video_id}")

        # Model loading leaves fragmented intermediate allocations in PyTorch's caching
        # allocator.  Clear them now before the first session starts.
        gc.collect()
        torch.cuda.empty_cache()

        # Set up output directories
        vid_out = Path(output_dir)
        vid_out.mkdir(parents=True, exist_ok=True)
        mask_data_dir = vid_out / "mask_data"
        json_data_dir = vid_out / "json_data"
        mask_data_dir.mkdir(exist_ok=True)
        json_data_dir.mkdir(exist_ok=True)

        # Extract frames directly with decord
        frame_dir = vid_out / "frames"
        frame_names = PersonSegmenter._extract_frames_from_path(
            video_path, video_id, frame_dir
        )

        print(f"  [GPU {gpu_id}] {video_id}: {len(frame_names)} frames")

        # Start a SAM3 session on the video frames
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(frame_dir))
        )
        session_id = response["session_id"]

        objects_count = 0
        try:
            # Add text prompt at regular intervals so people entering after
            # frame 0 are also detected and tracked.
            keyframes = range(0, len(frame_names), redetect_interval)
            for kf in keyframes:
                predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=kf,
                        text=text_prompt,
                    )
                )

            # Propagate through entire video
            for resp in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = resp["frame_index"]
                outputs = resp["outputs"]

                obj_ids = outputs["out_obj_ids"]
                masks = outputs["out_binary_masks"]
                scores = outputs["out_probs"]

                frame_name = frame_names[frame_idx]
                base_name = frame_name.split(".")[0]
                mask_name = f"mask_{base_name}.npy"

                if len(obj_ids) == 0:
                    h, w = Image.open(frame_dir / frame_name).size[::-1]
                    empty_mask = np.zeros((h, w), dtype=np.uint16)
                    np.save(str(mask_data_dir / mask_name), empty_mask)
                    meta = {
                        "mask_name": mask_name,
                        "mask_height": h,
                        "mask_width": w,
                        "labels": {},
                    }
                    json_path = json_data_dir / mask_name.replace(".npy", ".json")
                    with open(json_path, "w") as f:
                        json.dump(meta, f)
                    continue

                H, W = masks.shape[1], masks.shape[2]
                mask_img = np.zeros((H, W), dtype=np.uint16)
                labels_meta: dict[str, dict] = {}

                for i, obj_id in enumerate(obj_ids):
                    oid = int(obj_id)
                    binary = masks[i]
                    mask_img[binary] = oid

                    ys, xs = np.where(binary)
                    if len(ys) > 0:
                        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    else:
                        x1 = y1 = x2 = y2 = 0

                    labels_meta[str(oid)] = {
                        "instance_id": oid,
                        "class_name": text_prompt,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "score": float(scores[i]),
                    }

                    if oid > objects_count:
                        objects_count = oid

                np.save(str(mask_data_dir / mask_name), mask_img)

                meta = {
                    "mask_name": mask_name,
                    "mask_height": H,
                    "mask_width": W,
                    "labels": labels_meta,
                }
                json_path = json_data_dir / mask_name.replace(".npy", ".json")
                with open(json_path, "w") as f:
                    json.dump(meta, f)
        finally:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

        del predictor
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frames_from_path(
        video_path: str, video_id: str, frame_dir: Path
    ) -> list[str]:
        """
        Extract frames using decord directly from a video path and saves thm in the desired folder
        """
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


    def _extract_frames(self, video: Video, frame_dir: Path) -> list[str]:
        """
        Decode every frame of a Video object and write numbered JPEGs.

        Returns the sorted list of file names (``"000000.jpg"``, ...).
        """
        return self._extract_frames_from_path(
            str(video.path), video.video_id, frame_dir
        )

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
            logging.warning(f"The path {npz_path} doesn't exist")
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

        Unpacks ``mask_data.npz`` to reconstruct per-frame SAM3 output
        format, renders overlays, and writes an mp4 video.
        """
        frame_dir = video_dir / "frames"
        npz_path = video_dir / "mask_data.npz"
        json_dir = video_dir / "json_data"
        result_dir = video_dir / "result"
        result_dir.mkdir(exist_ok=True)

        frame_names = sorted(
            p.name for p in frame_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg")
        )
        if not frame_names:
            return

        # Read first frame to get video dimensions
        first_frame = cv2.imread(str(frame_dir / frame_names[0]))
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = video_dir / "segmentation.mp4"
        writer = cv2.VideoWriter(
            str(out_video), fourcc, int(video.fps), (width, height)
        )

        for frame_name in tqdm(frame_names, desc="Rendering visualisation", leave=False):
            base_name = frame_name.split(".")[0]
            mask_key = f"mask_{base_name}"

            # Load the frame image
            img = cv2.imread(str(frame_dir / frame_name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load mask from npz
            mask_arr = PersonSegmenter._load_mask_from_npz(npz_path, mask_key)

            # Load JSON metadata for bounding boxes
            json_path = json_dir / f"mask_{base_name}.json"
            labels_meta = {}
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                labels_meta = meta.get("labels", {})

            if mask_arr is None or int(mask_arr.max()) == 0:
                # No detections — write raw frame
                overlay = img_rgb.copy()
            else:
                # Reconstruct SAM3-style output dict for render_masklet_frame
                unique_ids = sorted(set(mask_arr.flatten()) - {0})
                obj_ids = np.array(unique_ids, dtype=np.int64)
                binary_masks = np.stack(
                    [(mask_arr == oid) for oid in unique_ids], axis=0
                )
                # Get scores and normalised boxes from JSON metadata
                probs = []
                boxes_xywh = []
                for oid in unique_ids:
                    info = labels_meta.get(str(oid), {})
                    probs.append(info.get("score", 1.0))
                    x1 = info.get("x1", 0)
                    y1 = info.get("y1", 0)
                    x2 = info.get("x2", 0)
                    y2 = info.get("y2", 0)
                    boxes_xywh.append([
                        x1 / width, y1 / height,
                        (x2 - x1) / width, (y2 - y1) / height,
                    ])

                sam3_outputs = {
                    "out_obj_ids": obj_ids,
                    "out_binary_masks": binary_masks,
                    "out_probs": np.array(probs),
                    "out_boxes_xywh": np.array(boxes_xywh) if boxes_xywh else np.zeros((0, 4)),
                }
                overlay = render_masklet_frame(img_rgb, sam3_outputs)

            # Save annotated frame
            cv2.imwrite(
                str(result_dir / frame_name),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Visualisation saved: {out_video}")
