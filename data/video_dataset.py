"""
EgoExo4D multi-view scene dataset — lazy video loading.

Expected directory layout::

    data_root/
        scene_001/
            cam01.mp4
            cam01/
                frames/      ← extracted JPEGs always live here, next to the video
            cam02.mp4
            cam02/
                frames/
        scene_002/
            ...

Videos are NEVER fully loaded into memory upfront.  ``__getitem__`` returns
a lightweight ``Scene`` object that holds ``Video`` handles.  Frames are
decoded on demand — one at a time, in slices, or in batches.

Extracted frames are always stored **co-located with the source data** at
``data_root/<scene_id>/<video_id>/frames/``.  If frames are found directly
inside ``<video_id>/`` (not in a ``frames/`` sub-directory) they are
automatically migrated into ``frames/``.
"""

from __future__ import annotations

import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from decord import VideoReader, cpu

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ======================================================================
# Video — wrapper around a single video
# ======================================================================

class Video:
    """Lazy handle to a single video or image sequence.

    Supports three construction modes:

    * **Video only** — ``Video(path="clip.mp4")``
      Frames are decoded on demand; :meth:`extract_frames` extracts JPEGs
      to a given directory, skipping if they already exist there.

    * **Images only** — ``Video(frames_dir="cam01/frames/")``
      No video file involved.  :meth:`extract_frames` returns the images
      in-place — nothing is copied.

    * **Both** — ``Video(path="clip.mp4", frames_dir="cam01/frames/")``
      Metadata (fps, resolution, …) is read from the video file, but
      :meth:`extract_frames` prefers the pre-extracted directory when it is
      non-empty, falling back to video extraction otherwise.

    Parameters
    ----------
    path : Path | None
        Path to a video file **or** to a directory of JPEG/PNG frames.
        At least one of *path* or *frames_dir* must be provided.
    resolution : tuple[int, int] | None
        Optional (H, W) to resize during video decoding.
    frames_dir : Path | None
        Pre-existing directory of extracted frames.  When provided and
        non-empty, :meth:`extract_frames` returns these frames directly.
    """

    def __init__(
        self,
        path: Path | None = None,
        resolution: tuple[int, int] | None = None,
        *,
        frames_dir: Path | None = None,
    ):
        if path is None and frames_dir is None:
            raise ValueError("At least one of 'path' or 'frames_dir' must be provided.")

        self.path = Path(path) if path is not None else None
        self.frames_dir = Path(frames_dir) if frames_dir is not None else None
        self._resolution = resolution

        # Determine video_id
        if self.path is not None:
            self.video_id: str = self.path.stem if self.path.is_file() else self.path.name
        else:
            self.video_id = self.frames_dir.name  # type: ignore[union-attr]

        # Gather metadata from the best available source
        if self.path is not None and self.path.is_file():
            # Video file → open VideoReader once for metadata, then close.
            # Readers are re-created on demand so the object stays picklable
            vr = self._make_reader()
            self.num_frames: int = len(vr)
            self.start: int = 0
            self.end: int = self.num_frames
            self.fps: float = float(vr.get_avg_fps())
            self.original_w, self.original_h = vr[0].shape[1], vr[0].shape[0]
            self.duration_s: float = self.num_frames / self.fps if self.fps > 0 else 0.0
            del vr
        else:
            # Image directory (path is a dir, or only frames_dir provided).
            img_source: Path = (
                self.path if (self.path is not None and self.path.is_dir())
                else self.frames_dir  # type: ignore[assignment]
            )
            frame_paths = sorted(
                p for p in img_source.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
            self.num_frames = len(frame_paths)
            self.start = 0
            self.end = self.num_frames
            self.fps = 30.0  # not embedded in images; assume 30 fps
            if frame_paths:
                img = cv2.imread(str(frame_paths[0]))
                self.original_h, self.original_w = img.shape[:2]
            else:
                self.original_h = self.original_w = 0
            self.duration_s = 0.0

        self.frame_resolution: tuple[int, int] = (
            resolution if resolution is not None else (self.original_h, self.original_w)
        )
        self.metadata: dict = {
            "video_id": self.video_id,
            "path": str(self.path) if self.path is not None else None,
            "frames_dir": str(self.frames_dir) if self.frames_dir is not None else None,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_s": self.duration_s,
            "original_h": self.original_h,
            "original_w": self.original_w,
        }

    def extract_frames(self, frame_dir: Path) -> tuple[Path, list[str]]:
        """Locate or extract frames for this video.

        Resolution order:

        1. If *frames_dir* was supplied at construction and is non-empty →
           return those frames directly (no extraction).
        2. If *path* is an image directory → return its images in-place.
        3. If *path* is a video file → write numbered JPEGs to *frame_dir*,
           skipping extraction when files already exist there.

        Parameters
        ----------
        frame_dir : Path
            Destination for extracted frames (used only in case 3 above).

        Returns
        -------
        (actual_frame_dir, sorted_frame_names)
            *actual_frame_dir* is the directory that actually holds the frames
            (may differ from *frame_dir* for cases 1 and 2).
        """
        # Case 1: pre-extracted frames directory given at construction.
        if self.frames_dir is not None and self.frames_dir.is_dir():
            frames = sorted(
                p.name for p in self.frames_dir.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
            if frames:
                return self.frames_dir, frames

        # Case 2: path is itself an image directory.
        if self.path is not None and self.path.is_dir():
            frames_subdir = self.path / "frames"
            # If already migrated, use the frames/ subdirectory.
            if frames_subdir.is_dir():
                frames = sorted(
                    p.name for p in frames_subdir.iterdir()
                    if p.suffix.lower() in _IMAGE_EXTS
                )
                if frames:
                    return frames_subdir, frames
            # Migrate images sitting directly in path into path/frames/.
            images_in_root = sorted(
                p for p in self.path.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
            if images_in_root:
                frames_subdir.mkdir(exist_ok=True)
                for img in images_in_root:
                    shutil.move(str(img), str(frames_subdir / img.name))
                print(
                    f"  {self.video_id}: migrated {len(images_in_root)} frames"
                    f" → {frames_subdir}"
                )
                frames = sorted(
                    p.name for p in frames_subdir.iterdir()
                    if p.suffix.lower() in _IMAGE_EXTS
                )
                return frames_subdir, frames
            # No images found anywhere.
            return self.path, []

        # Case 3: path is a video file — extract to frame_dir.
        if self.path is None:
            raise ValueError(
                f"Video '{self.video_id}': no video file or valid frames directory available."
            )
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p.name for p in frame_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        if existing:
            print(f"  {self.video_id}: reusing {len(existing)} existing frames in {frame_dir}")
            return frame_dir, existing

        vr = VideoReader(str(self.path), ctx=cpu(0))
        total = len(vr)
        frame_names: list[str] = []
        for idx in tqdm.tqdm(range(total), desc=f"  Extracting {self.video_id}", leave=False):
            frame_np = vr[idx].asnumpy()
            name = f"{idx:06d}.jpg"
            cv2.imwrite(
                str(frame_dir / name),
                cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR),
            )
            frame_names.append(name)
            del frame_np
        del vr
        return frame_dir, frame_names

    def get_frame(self, index: int) -> torch.Tensor:
        """Decode a single frame -> (C, H, W) uint8 tensor."""
        vr = self._make_reader()
        frame_np = vr[index].asnumpy()          # (H, W, C)
        del vr
        return torch.from_numpy(frame_np).permute(2, 0, 1)  # (C, H, W)

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Decode an arbitrary set of frame indices -> (T, C, H, W) uint8."""
        vr = self._make_reader()
        frames_np: np.ndarray = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        del vr
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2) # (T, C, H, W)

    def get_uniform_sample(self, num_frames: int) -> torch.Tensor:
        """Uniformly sample *num_frames* across the video -> (T, C, H, W)."""
        indices = np.linspace(
            0, self.num_frames - 1, num_frames, dtype=int
        ).tolist()
        return self.get_batch(indices)

    def get_clip(self, start: int, end: int, step: int = 1) -> torch.Tensor:
        """Decode frames[start:end:step] -> (T, C, H, W) uint8."""
        indices = list(range(start, min(end, self.num_frames), step))
        return self.get_batch(indices)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, key: int | slice) -> torch.Tensor:
        """Index a single frame (int) or a contiguous slice.

        Returns (C, H, W) for int, (T, C, H, W) for slice.
        """
        if isinstance(key, int):
            if key < 0:
                key += self.num_frames
            return self.get_frame(key)
        if isinstance(key, slice):
            indices = list(range(*key.indices(self.num_frames)))
            return self.get_batch(indices)
        raise TypeError(f"index must be int or slice, got {type(key)}")

    @property
    def frames_home(self) -> Path | None:
        """Canonical frames directory co-located with the source data.

        For a video file at ``data/scene1/cam01.mp4`` this returns
        ``data/scene1/cam01/frames/``.  For an image directory at
        ``data/scene1/cam01/`` the same path is returned.  When only
        *frames_dir* was supplied at construction, that directory is
        returned as-is (assumed canonical already).
        """
        if self.path is not None:
            return self.path.parent / self.video_id / "frames"
        if self.frames_dir is not None:
            return self.frames_dir
        return None

    def __repr__(self) -> str:
        return (
            f"Video({self.video_id!r}, frames={self.num_frames}, "
            f"fps={self.fps:.1f}, dur={self.duration_s:.1f}s)"
        )

    def _make_reader(self) -> VideoReader:
        if self.path is None or not self.path.is_file():
            raise RuntimeError(
                f"Video '{self.video_id}': VideoReader requires a video file path."
            )
        kwargs = {}
        if self._resolution is not None:
            kwargs["height"] = self._resolution[0]
            kwargs["width"] = self._resolution[1]
        return VideoReader(str(self.path), ctx=cpu(0), **kwargs)


# ======================================================================
# Scene — a collection of Videos belonging to one scene
# ======================================================================

class Scene:
    """A scene containing multiple camera views as lazy ``Video`` objects.

    Access videos by id (stem name) or by integer index.
    """

    def __init__(self, scene_id: str, videos: list[Video]):
        self.scene_id = scene_id
        self.videos = videos
        self._by_id: dict[str, Video] = {v.video_id: v for v in videos}
        self.video_ids = [v.video_id for v in self.videos]

    def __getitem__(self, key: int | str) -> Video:
        if isinstance(key, int):
            return self.videos[key]
        return self._by_id[key]

    def __len__(self) -> int:
        return len(self.videos)

    def __iter__(self):
        return iter(self.videos)

    def shuffle(self, max_shift: int = 30) -> dict[str, tuple[int, int]]:
        """Apply random temporal cropping to each video in this scene.

        Each video gets a random start point in [0, max_shift] and a random
        end point in [T - max_shift, T], where T is the total number of
        frames.  The effective frame range becomes [start, end).

        Parameters
        ----------
        max_shift : int
            Maximum shift for start/end boundaries.

        Returns
        -------
        dict mapping video_id -> (start, end).
        """
        shifts = {}
        for video in self.videos:
            start = random.randint(0, min(max_shift, video.num_frames - 1))
            end = random.randint(max(video.num_frames - max_shift, start + 1), video.num_frames)
            video.start = start
            video.end = end
            shifts[video.video_id] = (start, end)
        return shifts

    def get_shifted_frame(self, video_key: int | str, t: int) -> torch.Tensor:
        """Get frame *t* from a video, remapped to its [start, end) range.

        *t* is treated as a normalised index into the cropped range:
        actual_index = start + t, clamped to [start, end - 1].
        Returns (C, H, W) uint8.
        """
        video = self[video_key]
        idx = min(max(video.start + t, video.start), video.end - 1)
        return video[idx]

    def get_shifted_batch(self, video_key: int | str,
                          indices: list[int]) -> torch.Tensor:
        """Get multiple frames from a video, remapped to its [start, end) range.

        Returns (T, C, H, W) uint8.
        """
        video = self[video_key]
        shifted = [min(max(video.start + t, video.start), video.end - 1) for t in indices]
        return video.get_batch(shifted)

    def __repr__(self) -> str:
        return f"Scene({self.scene_id!r}, videos={len(self.videos)})"


# ======================================================================
# Dataset
# ======================================================================

class EgoExoSceneDataset(Dataset):
    """PyTorch dataset where each item is one EgoExo4D scene.

    All ``Scene`` and ``Video`` objects are built at init time so that
    metadata (fps, resolution, frame count, etc.) is immediately available.
    Actual frame pixels are still decoded lazily — nothing heavy is loaded
    until you explicitly request frames from a ``Video``.

    If *frames_root* is provided, frames are extracted for every video
    during ``__init__`` (skipped if already present) and each
    ``Video.frames_dir`` is set to the extraction target.  Downstream
    consumers (e.g. ``PersonSegmenter``) will then find pre-extracted
    frames immediately without doing any extraction themselves.  This also
    makes the pipeline work with image-only datasets that never had a
    video file.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing one sub-folder per scene.
    resolution : tuple[int, int] | None
        (H, W) to resize frames during decoding.  None keeps original.
    video_extensions : tuple[str, ...]
        File extensions treated as video files.
    preextract_frames : bool
        If ``True``, extract frames for every video during ``__init__``
        (skipped if already present).  Frames are always written to the
        canonical co-located path ``<data_root>/<scene>/<video_id>/frames/``.
    """

    def __init__(
        self,
        data_root: str | Path,
        resolution: tuple[int, int] | None = None,
        video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi"),
        slice: int | None = None,
        exclude_ego: bool = True,
        preextract_frames: bool = False,
    ):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.video_extensions = video_extensions
        self.exclude_ego = exclude_ego

        # Discover scene directories and build Scene objects up front.
        scene_dirs = sorted(
            p for p in self.data_root.iterdir()
            if p.is_dir() and self._has_videos(p)
        )
        if slice is not None:
            scene_dirs = scene_dirs[:slice]

        if len(scene_dirs) == 0:
            raise FileNotFoundError(
                f"No scene folders with videos found under {self.data_root}"
            )

        # Collect all (scene_dir, video_path) pairs for parallel loading.
        scene_video_paths: list[tuple[Path, list[Path]]] = []
        for scene_dir in scene_dirs:
            vid_paths = [
                p for p in self._list_videos(scene_dir)
                if not (self.exclude_ego and p.stem.startswith("aria"))
                and not p.stem.startswith("ego_preview")
            ]
            scene_video_paths.append((scene_dir, vid_paths))

        total_videos = sum(len(vps) for _, vps in scene_video_paths)
        print(f"Loading metadata for {total_videos} videos across {len(scene_dirs)} scenes...")

        # Load all Video objects in parallel using threads (I/O-bound).
        # Maps video_path -> Video object.
        all_video_paths = [
            vp for _, vps in scene_video_paths for vp in vps
        ]
        video_map: dict[Path, Video] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(all_video_paths) or 1)) as pool:
            futures = {
                pool.submit(Video, p, self.resolution): p
                for p in all_video_paths
            }
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Loading videos"
            ):
                path = futures[future]
                video_map[path] = future.result()

        # Assemble scenes in the original order.
        self.scenes: list[Scene] = []
        for scene_dir, vid_paths in scene_video_paths:
            videos = [video_map[p] for p in vid_paths]
            self.scenes.append(Scene(scene_id=scene_dir.name, videos=videos))

        print(f"Dataset created: {len(self.scenes)} scenes, "
              f"{sum(len(s) for s in self.scenes)} videos total")

        # Pre-extract frames to the canonical co-located path if requested.
        if preextract_frames:
            total_vids = sum(len(s) for s in self.scenes)
            print(f"Pre-extracting frames for {total_vids} videos (co-located with data) ...")
            for scene in tqdm.tqdm(self.scenes, desc="Scenes"):
                for video in tqdm.tqdm(scene.videos, desc=f"  {scene.scene_id}", leave=False):
                    actual_dir, _ = video.extract_frames(video.frames_home)
                    video.frames_dir = actual_dir
            print("Frame extraction complete.")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Scene:
        return self.scenes[idx]

    def get_scene_ids(self) -> list[str]:
        return [s.scene_id for s in self.scenes]

    def _list_videos(self, directory: Path) -> list[Path]:
        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in self.video_extensions
        )

    def _has_videos(self, directory: Path) -> bool:
        return any(
            p.suffix.lower() in self.video_extensions
            for p in directory.rglob("*")
            if p.is_file()
        )



if __name__ =="__main__":
    path = "/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"
    video_ds = EgoExoSceneDataset(path, slice=5)
    print(f"Found {len(video_ds)} scenes: {video_ds.get_scene_ids()}")

    scene = video_ds[0]
    print(f"\n--- Scene: {scene.scene_id} ({len(scene)} videos) ---")
    for v in scene:
        print(f"  {v}\n")

    # Test shuffle: apply random temporal cropping
    print("\n--- Testing shuffle ---")
    shifts = scene.shuffle(max_shift=30)
    for vid_id, (s, e) in shifts.items():
        print(f"  {vid_id}: start={s}, end={e}")

    # Verify shifts are stored on the Video objects
    for v in scene:
        s, e = shifts[v.video_id]
        assert v.start == s and v.end == e, f"Shift mismatch for {v.video_id}"
    print("  All shifts stored correctly on Video objects.")

    # Test shifted frame access
    video = scene[0]
    t = 50
    frame_normal = video[video.start + t]
    frame_shifted = scene.get_shifted_frame(0, t)
    print(f"\n--- Shifted frame access (t={t}, start={video.start}, end={video.end}) ---")
    print(f"  video[start+{t}]        -> shape={frame_normal.shape}")
    print(f"  get_shifted_frame(0,{t}) -> shape={frame_shifted.shape}")
    same = torch.equal(frame_normal, frame_shifted)
    print(f"  Frames identical: {same} (expected True)")