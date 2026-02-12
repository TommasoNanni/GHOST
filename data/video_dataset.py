"""
EgoExo4D multi-view scene dataset — lazy video loading.

Expected directory layout:
    data_root/
        scene_001/
            cam01.mp4
            cam02.mp4
            ...
        scene_002/
            cam01.mp4
            ...

Videos are NEVER fully loaded into memory upfront.  ``__getitem__`` returns
a lightweight ``Scene`` object that holds ``Video`` handles.  Frames are
decoded on demand — one at a time, in slices, or in batches.
"""

from __future__ import annotations

import random
from pathlib import Path
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from decord import VideoReader, cpu


# ======================================================================
# Video — thin wrapper around a single .mp4 on disk
# ======================================================================

class Video:
    """Lazy handle to a single video file.

    No frames are decoded until you ask for them.  Supports integer
    indexing, slicing, and explicit batch reads.

    Parameters
    ----------
    path : Path
        Path to the video file.
    resolution : tuple[int, int] | None
        Optional (H, W) to resize during decoding.
    """

    def __init__(self, path: Path, resolution: tuple[int, int] | None = None):
        self.path = path
        self.video_id: str = path.stem
        self._resolution = resolution

        # Open the reader once just to grab metadata, then close it.
        # Readers are re-created on demand so the object stays picklable
        # (important for DataLoader with num_workers > 0).
        vr = self._make_reader()
        self.num_frames: int = len(vr)
        self.shift: int | None = None
        self.fps: float = float(vr.get_avg_fps())
        # Get resolution from container metadata — no frame decoding needed.
        self.original_w, self.original_h = vr[0].shape[1], vr[0].shape[0]
        self.duration_s: float = self.num_frames / self.fps if self.fps > 0 else 0.0
        self.frame_resolution: tuple[int, int] = (
            resolution if resolution is not None else (self.original_h, self.original_w)
        )
        self.metadata: dict = {
            "video_id": self.video_id,
            "path": str(self.path),
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_s": self.duration_s,
            "original_h": self.original_h,
            "original_w": self.original_w,
        }
        del vr

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

    def __repr__(self) -> str:
        return (
            f"Video({self.video_id!r}, frames={self.num_frames}, "
            f"fps={self.fps:.1f}, dur={self.duration_s:.1f}s)"
        )

    def _make_reader(self) -> VideoReader:
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

    def shuffle(self, max_shift: int = 30) -> dict[str, int]:
        """Apply a random temporal shift to each video in this scene.

        Each video gets a random offset in [-max_shift, +max_shift] frames,
        clamped so indices stay within [0, num_frames).  The shifts are stored
        on each ``Video`` object as ``video.shift`` and also returned as a dict.

        Parameters
        ----------
        max_shift : int
            Maximum absolute frame shift (in either direction).

        Returns
        -------
        dict mapping video_id -> applied shift (int).
        """
        shifts = {}
        for video in self.videos:
            shift = random.randint(-max_shift, max_shift)
            video.shift = shift
            shifts[video.video_id] = shift
        return shifts

    def get_shifted_frame(self, video_key: int | str, t: int) -> torch.Tensor:
        """Get frame *t* from a video, accounting for its current shift.

        Actual decoded index = t + video.shift, clamped to valid range.
        Returns (C, H, W) uint8.
        """
        video = self[video_key]
        shift = getattr(video, "shift", 0)
        idx = min(max(t + shift, 0), video.num_frames - 1)
        return video[idx]

    def get_shifted_batch(self, video_key: int | str,
                          indices: list[int]) -> torch.Tensor:
        """Get multiple frames from a video with shift applied.

        Returns (T, C, H, W) uint8.
        """
        video = self[video_key]
        shift = getattr(video, "shift", 0)
        shifted = [min(max(t + shift, 0), video.num_frames - 1) for t in indices]
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

    Parameters
    ----------
    data_root : str | Path
        Root directory containing one sub-folder per scene.
    resolution : tuple[int, int] | None
        (H, W) to resize frames during decoding.  None keeps original.
    video_extensions : tuple[str, ...]
        File extensions treated as video files.
    """

    def __init__(
        self,
        data_root: str | Path,
        resolution: tuple[int, int] | None = None,
        video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi"),
        slice: int | None = None,
        exclude_ego: bool = True,
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

        self.scenes: list[Scene] = []
        for scene_dir in tqdm.tqdm(scene_dirs, desc="Loading scenes"):
            videos = [
                Video(p, resolution=self.resolution)
                for p in self._list_videos(scene_dir)
                if not (self.exclude_ego and p.stem.startswith("aria")) and not p.stem.startswith("ego_preview")
            ]
            self.scenes.append(Scene(scene_id=scene_dir.name, videos=videos))

        print(f"Dataset created: {len(self.scenes)} scenes, "
              f"{sum(len(s) for s in self.scenes)} videos total")

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

    # Test shuffle: apply random temporal shifts
    print("\n--- Testing shuffle ---")
    shifts = scene.shuffle(max_shift=30)
    for vid_id, s in shifts.items():
        print(f"  {vid_id}: shift={s}")

    # Verify shifts are stored on the Video objects
    for v in scene:
        assert v.shift == shifts[v.video_id], f"Shift mismatch for {v.video_id}"
    print("  All shifts stored correctly on Video objects.")

    # Test shifted frame access
    video = scene[0]
    t = 50
    frame_normal = video[t]
    frame_shifted = scene.get_shifted_frame(0, t)
    print(f"\n--- Shifted frame access (t={t}, shift={video.shift}) ---")
    print(f"  video[{t}]             -> shape={frame_normal.shape}")
    print(f"  get_shifted_frame(0,{t}) -> shape={frame_shifted.shape}")
    same = torch.equal(frame_normal, frame_shifted)
    print(f"  Frames identical: {same} (expected {video.shift == 0})")