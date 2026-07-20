"""Lag-free 3D scene viewer for the ghost multi-camera pipeline (Rerun, columnar).

This is a rewrite of :mod:`visualize.visualize`.  It produces the exact same
scene — animated SMPL-X meshes, camera frustums and synchronised video feeds —
but logs every entity's **entire time-series in one shot** via Rerun's columnar
API (:func:`rerun.send_columns`) instead of issuing one ``rr.log`` per frame.

Why this matters
----------------
The old viewer walked ``for t in range(T): rr.log(...)`` and paid a
Python→Rust round-trip for *every* mesh, transform and image of *every* frame.
On a long take that is tens of thousands of tiny messages, which is what makes
scrubbing and playback feel laggy.

Here, for each entity we build the whole column (all frames) in memory and send
it with a single :func:`rerun.send_columns` call:

* **Meshes** — topology (``triangle_indices``) and per-vertex colour are logged
  once as *static*; the per-frame ``vertex_positions`` are sent as one
  partitioned column (one ``V×3`` chunk per visible frame).
* **Cameras** — ``Pinhole`` intrinsics are static; a moving camera's
  ``Transform3D`` is one column; a static camera is one static log.
* **Video** — the whole MP4 is logged once as an ``AssetVideo`` and every frame
  is addressed by a single column of cheap ``VideoFrameReference`` timestamps
  (no pixel data per frame).  Frame directories are JPEG-encoded and shipped as
  one ``EncodedImage`` column.

The result: the recording is fully resident in the viewer immediately, and the
timeline scrubs with no per-frame data transfer.

Quick start
-----------
    from visualize.visualize_rerun import SceneViewer, CameraView

    viewer = SceneViewer(vertices, faces, cameras, fps=30.0)
    viewer.serve(port=9090)      # live (SSH-tunnel on a cluster)
    # or
    viewer.save("scene.rrd")     # replay: python -m rerun scene.rrd
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import rerun as rr
from tqdm import tqdm


# ── Colour palette (RGB uint8, one per person, cycles) ───────────────────────
_PALETTE: list[tuple[int, int, int]] = [
    (220,  80,  60),   # red
    ( 60, 200,  60),   # green
    ( 60,  80, 220),   # blue
    (210, 210,  40),   # yellow
    (210,  60, 210),   # magenta
    ( 40, 210, 210),   # cyan
    (220, 140,  40),   # orange
    (160,  60, 160),   # purple
]

_FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Rerun timeline names — shared by every entity so meshes, cameras and video
# scrub together.
_FRAME_TL = "frame"     # integer frame index
_TIME_TL  = "time_s"    # seconds, derived from fps


@dataclass
class CameraView:
    """Camera parameters and image source for one camera.

    Parameters
    ----------
    R : (3, 3) or (T, 3, 3) ndarray
        World-to-camera rotation.  ``x_cam = R @ x_world + t``.
    t : (3,) or (T, 3) ndarray
        World-to-camera translation.
    focal_length : float or (T,) ndarray
        Focal length in pixels (isotropic; shared for fx and fy).
    img_wh : (W, H)
        Image dimensions in pixels.
    frames_dir : Path, optional
        Directory of extracted JPEG/PNG frames named ``{idx:06d}.jpg`` (or
        similar; several zero-padding widths and extensions are tried).
    video_path : Path, optional
        MP4 video file.  Logged once as an ``AssetVideo``; frames are then
        referenced cheaply.
    principal_point : (cx, cy), optional
        Principal point in pixels.  Defaults to image centre.
    """

    R:               np.ndarray
    t:               np.ndarray
    focal_length:    Union[float, np.ndarray]
    img_wh:          tuple[int, int]
    frames_dir:      Optional[Path] = None
    video_path:      Optional[Path] = None
    principal_point: Optional[tuple[float, float]] = None

    @property
    def W(self) -> int:
        return self.img_wh[0]

    @property
    def H(self) -> int:
        return self.img_wh[1]

    @property
    def is_static(self) -> bool:
        """True when R / t describe a single, time-invariant pose."""
        return np.asarray(self.R).ndim == 2

    @classmethod
    def from_pipeline_params(
        cls,
        camera_params: np.ndarray,
        img_wh: tuple[int, int],
        frames_dir: Optional[Path] = None,
        video_path: Optional[Path] = None,
        principal_point: Optional[tuple[float, float]] = None,
    ) -> "CameraView":
        """Construct from the ghost pipeline's 8-D camera array.

        Parameters
        ----------
        camera_params : (T, 8) or (8,) ndarray
            ``[quat(4), trans(3), focal_raw(1)]`` — the raw 8-D output of
            ``SSTOutputHeads``, *before* softplus.  Do NOT pass the output
            of ``extract_cameras``: that function already applies softplus
            and returns decomposed ``R, t, K``; in that case construct
            ``CameraView(R=..., t=..., focal_length=...)`` directly.
        img_wh : (W, H)
        frames_dir / video_path : image source (at least one required).
        """
        import torch
        import torch.nn.functional as F
        from pytorch3d.transforms import quaternion_to_matrix

        params = np.asarray(camera_params, dtype=np.float32)
        scalar_input = params.ndim == 1
        if scalar_input:
            params = params[np.newaxis]           # (1, 8)

        R = quaternion_to_matrix(
            torch.from_numpy(params[:, :4])
        ).numpy()                                 # (T, 3, 3)
        t     = params[:, 4:7]                    # (T, 3)
        focal = F.softplus(
            torch.from_numpy(params[:, 7])
        ).numpy()                                 # (T,) — guaranteed > 0

        if scalar_input:
            R     = R[0]
            t     = t[0]
            focal = float(focal[0])

        return cls(
            R=R, t=t, focal_length=focal, img_wh=img_wh,
            frames_dir=frames_dir, video_path=video_path,
            principal_point=principal_point,
        )


class SceneViewer:
    """Lag-free interactive 3D scene viewer (columnar Rerun logging).

    Builds a Rerun recording containing animated SMPL-X meshes, camera frustums
    and synchronised video feeds, all sharing a single timeline.  Unlike the
    original :class:`visualize.visualize.SceneViewer`, every entity is sent to
    Rerun as a single columnar batch rather than frame-by-frame.

    Parameters
    ----------
    vertices : (T, P, V, 3) ndarray
        Mesh vertex positions in world coordinates (first-camera space).
        ``(T, V, 3)`` is also accepted for a single-person scene.  Frames where
        a person is invisible should have rows set to ``NaN`` — those frames are
        simply omitted from that person's column.
    faces : (F, 3) ndarray
        Face topology (e.g. SMPL-X body model faces, 10475 vertices).
    cameras : dict[str, CameraView]
        Ordered mapping from camera name to parameters and image source.
    fps : float
        Frame rate used to build the seconds axis.
    person_colors : list of (R, G, B) uint8 tuples, optional
        Per-person mesh colour.  Cycles a default palette when omitted.
    recording_id : str
        Rerun recording identifier shown in the viewer title bar.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cameras: dict[str, CameraView],
        fps: float = 30.0,
        person_colors: Optional[list[tuple[int, int, int]]] = None,
        recording_id: str = "ghost_scene",
    ) -> None:
        self.vertices     = np.asarray(vertices)
        self.faces        = np.asarray(faces, dtype=np.uint32)
        self.cameras      = cameras
        self.fps          = float(fps)
        self.recording_id = recording_id

        # Accept (T, V, 3) for a single-person scene
        if self.vertices.ndim == 3:
            self.vertices = self.vertices[:, np.newaxis]   # (T, 1, V, 3)

        self.T, self.P = self.vertices.shape[:2]

        self.person_colors = (
            person_colors
            if person_colors is not None
            else [_PALETTE[i % len(_PALETTE)] for i in range(self.P)]
        )

    # ── Public entry points ──────────────────────────────────────────────────

    def serve(
        self,
        port: int = 9090,
        grpc_port: int = 9876,
        open_browser: bool = False,
        server_memory_limit: str = "8GiB",
    ) -> None:
        """Build the recording, serve it over gRPC + web, and block.

        Rerun 0.32 serves in two parts: a gRPC data sink (:func:`serve_grpc`)
        and a web viewer (:func:`serve_web_viewer`) that connects to it.  We
        push the whole columnar recording once, then idle.

        Parameters
        ----------
        port : int
            HTTP port for the Rerun web viewer.
        grpc_port : int
            Port for the gRPC data server the viewer connects to.
        open_browser : bool
            Auto-open a browser tab.  Leave ``False`` on headless machines.
        server_memory_limit : str
            Memory budget for the gRPC server's recording buffer.  Bump this
            for long takes logged with ``AssetVideo`` (whole MP4 is buffered).
        """
        rr.init(self.recording_id, spawn=False)

        server_uri = rr.serve_grpc(
            grpc_port=grpc_port, server_memory_limit=server_memory_limit,
            default_blueprint=_default_blueprint(self.fps),
        )
        rr.serve_web_viewer(
            web_port=port, open_browser=open_browser, connect_to=server_uri
        )

        self._build_recording()

        _print_serve_help(port, grpc_port)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    def save(self, path: Union[str, Path]) -> Path:
        """Build the recording and write a ``.rrd`` file for later replay.

        Open anywhere with ``python -m rerun scene.rrd``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rr.init(self.recording_id, spawn=False)
        rr.save(str(path), default_blueprint=_default_blueprint(self.fps))
        self._build_recording()
        print(f"Recording saved  →  {path}")
        return path

    # ── Recording construction (all columnar) ────────────────────────────────

    def _build_recording(self) -> None:
        # World coordinate system: RDF = right / down / forward (OpenCV).
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        self._send_people()
        self._send_cameras()

    def _send_people(self) -> None:
        """Send every person's mesh as: static topology/colour + one column."""
        for p_idx in tqdm(range(self.P), desc="People", unit="person"):
            entity = f"world/person_{p_idx}"
            color  = np.asarray(self.person_colors[p_idx], dtype=np.uint8)

            verts_t = self.vertices[:, p_idx]                 # (T, V, 3)
            # A frame is "visible" when no vertex is NaN.
            visible = ~np.isnan(verts_t).any(axis=(1, 2))     # (T,)
            frames  = np.nonzero(visible)[0].astype(np.int64)
            if frames.size == 0:
                continue

            verts   = verts_t[frames].astype(np.float32)      # (N, V, 3)
            n, V    = verts.shape[:2]

            # Static: topology + per-vertex colour (constant across time).
            rr.log(
                entity,
                rr.Mesh3D.from_fields(
                    triangle_indices=self.faces,
                    vertex_colors=np.broadcast_to(color, (V, 3)).copy(),
                ),
                static=True,
            )

            # Temporal: one V×3 chunk of vertex positions per visible frame.
            rr.send_columns(
                entity,
                indexes=self._time_columns(frames),
                columns=rr.Mesh3D.columns(
                    vertex_positions=verts.reshape(n * V, 3)
                ).partition(np.full(n, V, dtype=np.int64)),
            )

    def _send_cameras(self) -> None:
        for cam_name, cam in self.cameras.items():
            entity = f"world/{cam_name}"

            # Pinhole intrinsics — always static.
            _log_pinhole(entity, cam)

            # Extrinsics — static log or one columnar batch.
            R = np.asarray(cam.R)
            t = np.asarray(cam.t)
            if cam.is_static:
                _log_static_transform(entity, R, t)
            else:
                self._send_camera_transforms(entity, R, t)

            # Image source — pick the cheapest available path.
            self._send_camera_images(entity, cam)

    def _send_camera_transforms(
        self, entity: str, R: np.ndarray, t: np.ndarray
    ) -> None:
        """Send a moving camera's full camera-to-world pose track as one column.

        Pipeline stores world-to-camera (``x_cam = R x_world + t``); Rerun wants
        camera-to-world: ``rotation = Rᵀ``, ``translation = -Rᵀ t``.
        """
        R = R.astype(np.float64)                              # (T, 3, 3)
        t = t.astype(np.float64)                              # (T, 3)
        RT    = np.transpose(R, (0, 2, 1))                    # (T, 3, 3)
        trans = -np.einsum("tij,tj->ti", RT, t)               # (T, 3)
        frames = np.arange(R.shape[0], dtype=np.int64)

        rr.send_columns(
            entity,
            indexes=self._time_columns(frames),
            columns=rr.Transform3D.columns(
                translation=trans.astype(np.float32),
                mat3x3=RT.astype(np.float32),
            ),
        )

    def _send_camera_images(self, entity: str, cam: CameraView) -> None:
        """Attach the camera's video feed in one columnar batch."""
        # Fast path: log the MP4 once, reference every frame by timestamp.
        if cam.video_path is not None and _send_video_asset(
            entity, cam.video_path, self.T, self._time_columns
        ):
            return

        # Frame directory → batch-encode to JPEG and ship one column.
        if cam.frames_dir is not None:
            self._send_encoded_images_from_dir(entity, cam.frames_dir)
            return

        # Last resort: decode an MP4 with decord and batch-encode.
        if cam.video_path is not None:
            self._send_encoded_images_from_video(entity, cam.video_path)

    def _send_encoded_images_from_dir(
        self, entity: str, frames_dir: Path
    ) -> None:
        blobs: list[bytes] = []
        frames: list[int] = []
        for t_idx in tqdm(
            range(self.T), desc=f"Encode {entity}", unit="frame", leave=False
        ):
            bgr = _read_frame_from_dir(frames_dir, t_idx)
            if bgr is None:
                continue
            blob = _jpeg(bgr)
            if blob is None:
                continue
            blobs.append(blob)
            frames.append(t_idx)
        self._send_encoded_column(entity, frames, blobs)

    def _send_encoded_images_from_video(
        self, entity: str, video_path: Path
    ) -> None:
        try:
            from decord import VideoReader
        except Exception as exc:                              # pragma: no cover
            print(f"  [{entity}] decord unavailable: {exc}")
            return
        try:
            vr = VideoReader(str(video_path))
        except Exception as exc:
            print(f"  [{entity}] cannot open video: {exc}")
            return

        n = min(self.T, len(vr))
        blobs: list[bytes] = []
        frames: list[int] = []
        for t_idx in tqdm(
            range(n), desc=f"Encode {entity}", unit="frame", leave=False
        ):
            try:
                rgb = vr[t_idx].asnumpy()
            except Exception:
                continue
            blob = _jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if blob is None:
                continue
            blobs.append(blob)
            frames.append(t_idx)
        self._send_encoded_column(entity, frames, blobs)

    def _send_encoded_column(
        self, entity: str, frames: list[int], blobs: list[bytes]
    ) -> None:
        if not blobs:
            return
        rr.send_columns(
            entity,
            indexes=self._time_columns(np.asarray(frames, dtype=np.int64)),
            columns=rr.EncodedImage.columns(
                blob=blobs, media_type=["image/jpeg"] * len(blobs)
            ),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _time_columns(self, frames: np.ndarray) -> list:
        """Build the (frame, seconds) index columns shared by every entity."""
        return _time_columns(frames, self.fps)


# ── Module-level Rerun helpers ───────────────────────────────────────────────

def _log_pinhole(entity: str, cam: CameraView) -> None:
    """Log Pinhole intrinsics (always static — focal length assumed fixed)."""
    fl = cam.focal_length
    focal = float(fl) if np.isscalar(fl) else float(np.asarray(fl).mean())
    cx, cy = cam.principal_point or (cam.W / 2.0, cam.H / 2.0)

    rr.log(
        entity,
        rr.Pinhole(
            focal_length=focal,
            principal_point=(cx, cy),
            width=cam.W,
            height=cam.H,
            camera_xyz=rr.ViewCoordinates.RDF,   # OpenCV: right-down-forward
            image_plane_distance=0.3,
        ),
        static=True,
    )


def _log_static_transform(entity: str, R: np.ndarray, t: np.ndarray) -> None:
    """Log a fixed camera's camera-to-world Transform3D as static.

    world-to-camera ``x_cam = R x_world + t`` →  camera-to-world is
    ``rotation = Rᵀ``, ``translation = -Rᵀ t``.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    rr.log(
        entity,
        rr.Transform3D(translation=-(R.T @ t), mat3x3=R.T),
        static=True,
    )


def _send_video_asset(
    entity: str, video_path: Path, T: int, time_columns
) -> bool:
    """Log the MP4 once as an ``AssetVideo`` and send all frame references.

    Returns True on success.  The whole file is stored once; every frame is a
    tiny ``VideoFrameReference`` timestamp — no pixel data per frame.
    """
    try:
        asset = rr.AssetVideo(path=str(video_path))
        rr.log(entity, asset, static=True)
        ts = asset.read_frame_timestamps_nanos()             # (n_frames,)
    except Exception as exc:
        print(f"  [{entity}] AssetVideo failed ({exc}); falling back.")
        return False

    n = min(int(T), len(ts))
    if n == 0:
        return False

    timestamps = [
        rr.datatypes.VideoTimestamp(timestamp_ns=int(ts[i])) for i in range(n)
    ]
    rr.send_columns(
        entity,
        indexes=time_columns(np.arange(n, dtype=np.int64)),
        columns=rr.VideoFrameReference.columns(timestamp=timestamps),
    )
    return True


def _jpeg(frame_bgr: np.ndarray, quality: int = 85) -> Optional[bytes]:
    """JPEG-encode a BGR frame, or None on failure."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else None


def _read_frame_from_dir(frames_dir: Path, t_idx: int) -> Optional[np.ndarray]:
    """Return a BGR frame loaded from a frames directory, or None."""
    for stem in (
        f"{t_idx:06d}", f"{t_idx:05d}", f"{t_idx:04d}",
        f"{t_idx:03d}", str(t_idx),
    ):
        for ext in _FRAME_EXTS:
            p = frames_dir / f"{stem}{ext}"
            if p.exists():
                return cv2.imread(str(p))
    return None


# ── Reusable columnar senders (shared by SceneViewer and the fusion entry) ────

def _time_columns(frames: np.ndarray, fps: float) -> list:
    """Build the (frame, seconds) index columns shared by every entity."""
    frames = np.asarray(frames, dtype=np.int64)
    return [
        rr.TimeColumn(_FRAME_TL, sequence=frames),
        rr.TimeColumn(_TIME_TL, duration=frames.astype(np.float64) / fps),
    ]


def _send_mesh_column(
    entity: str,
    faces: np.ndarray,
    color: tuple[int, int, int],
    frames: np.ndarray,
    verts: np.ndarray,        # (N, V, 3) — only the visible frames
    fps: float,
) -> None:
    """Static topology+colour once, then one partitioned vertex-position column."""
    n, V = verts.shape[:2]
    if n == 0:
        return
    rr.log(
        entity,
        rr.Mesh3D.from_fields(
            triangle_indices=np.asarray(faces, dtype=np.uint32),
            vertex_colors=np.broadcast_to(
                np.asarray(color, dtype=np.uint8), (V, 3)
            ).copy(),
        ),
        static=True,
    )
    rr.send_columns(
        entity,
        indexes=_time_columns(frames, fps),
        columns=rr.Mesh3D.columns(
            vertex_positions=verts.reshape(n * V, 3).astype(np.float32)
        ).partition(np.full(n, V, dtype=np.int64)),
    )


def _send_transform_column(
    entity: str,
    R_w2c: np.ndarray,        # (T, 3, 3) world-to-camera
    t_w2c: np.ndarray,        # (T, 3)
    frames: np.ndarray,
    fps: float,
) -> None:
    """Send a moving camera's full camera-to-world pose track as one column."""
    R = np.asarray(R_w2c, dtype=np.float64)
    t = np.asarray(t_w2c, dtype=np.float64)
    RT    = np.transpose(R, (0, 2, 1))                    # (T, 3, 3)
    trans = -np.einsum("tij,tj->ti", RT, t)               # (T, 3)
    rr.send_columns(
        entity,
        indexes=_time_columns(frames, fps),
        columns=rr.Transform3D.columns(
            translation=trans.astype(np.float32),
            mat3x3=RT.astype(np.float32),
        ),
    )


def _log_pinhole_simple(entity: str, focal: float, W: int, H: int) -> None:
    """Log static Pinhole intrinsics from a bare focal length + image size."""
    rr.log(
        entity,
        rr.Pinhole(
            focal_length=float(focal),
            principal_point=(W / 2.0, H / 2.0),
            width=W,
            height=H,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.3,
        ),
        static=True,
    )


# ── Fusion-prediction entry point (drop-in for visualize_fusion.py) ──────────

def _build_smplx_vertices(
    pose:  np.ndarray,   # (T, P, J, 6)  world-frame 6D rotations
    shape: np.ndarray,   # (P, 10) or (T, P, 10)
    trans: np.ndarray,   # (T, P, 3)     world-frame root translation
    smplx_model_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SMPL-X forward, returning (vertices (T,P,V,3) float32, faces (F,3))."""
    import torch
    import smplx as smplx_lib
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

    def _6d_to_aa(p6: np.ndarray) -> np.ndarray:
        m = rotation_6d_to_matrix(torch.from_numpy(p6.astype(np.float32)))
        return matrix_to_axis_angle(m).numpy()

    T, P, J, _ = pose.shape
    if shape.ndim == 2:                       # (P, 10) — constant over time
        shape = np.broadcast_to(shape[None], (T, P, 10)).copy()

    p = Path(smplx_model_dir)
    create_kwargs: dict = {"model_type": "smplx", "model_path": str(p)}
    if p.is_file():
        create_kwargs["ext"] = p.suffix.lstrip(".")
    model = smplx_lib.create(
        **create_kwargs, gender="neutral", use_pca=False, num_betas=10,
        flat_hand_mean=True, batch_size=T * P,
    )
    model.eval()

    global_orient_aa = _6d_to_aa(pose[:, :, 0, :])
    body_pose_aa     = _6d_to_aa(pose[:, :, 1:22, :])

    def _t(x):
        return torch.from_numpy(x.reshape(T * P, -1).astype(np.float32))

    with torch.no_grad():
        out = model(
            global_orient=_t(global_orient_aa),
            body_pose=_t(body_pose_aa),
            betas=_t(shape),
            transl=_t(trans),
            return_verts=True,
        )
    V = out.vertices.shape[1]
    return out.vertices.numpy().reshape(T, P, V, 3), model.faces.copy()


def _load_rich_frame(
    rich_data_root: Path,
    scene_name: str,
    cam_idx: int,
    rich_frame_idx: int,
    frames_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load one RICH frame as BGR uint8, or None.  Mirrors visualize_fusion."""
    scene_root = frames_dir if frames_dir is not None else rich_data_root / scene_name
    cam_dir = scene_root / f"cam_{cam_idx:02d}"
    search_dirs = [cam_dir, cam_dir / "frames"]
    for stem in (f"{rich_frame_idx:05d}_{cam_idx:02d}", f"{rich_frame_idx:05d}"):
        for sd in search_dirs:
            for ext in _FRAME_EXTS:
                q = sd / f"{stem}{ext}"
                if q.exists():
                    return cv2.imread(str(q))
    return None


def _decode_cameras(camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,K,8) [quat(4), trans(3), focal(1)] → (R_w2c (T,K,3,3), t (T,K,3), focal (K,))."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix

    T, K = camera.shape[:2]
    q = camera[..., :4]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    R = quaternion_to_matrix(
        torch.from_numpy(q.reshape(-1, 4).astype(np.float32))
    ).numpy().reshape(T, K, 3, 3)
    return R, camera[..., 4:7], camera[0, :, 7]


def _default_blueprint(fps: float):
    """Blueprint that makes ``frame`` the active timeline (auto layout otherwise).

    Without this the viewer opens on the auto ``log_time`` timeline, where all
    ``send_columns`` rows collapse onto a single instant — so meshes/depth show
    no timeline data and vanish on play, while only the *static* camera frustums
    remain.  Defaulting to ``frame`` makes scrubbing and play animate everything.
    """
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/world", name="world", background=[255, 255, 255],
        ),
        rrb.TimePanel(timeline=_FRAME_TL, fps=float(fps), state="expanded"),
        collapse_panels=False,
    )


def run_fusion(
    predictions:     Path,
    scene_dir:       Path,
    rich_data_root:  Path,
    smplx_model_dir: Path = Path("body_models/SMPLX_NEUTRAL.pkl"),
    frame_start:     int  = 0,
    show_gt:         bool = True,
    show_depth:      bool = False,
    fps:             float = 30.0,
    depth_stride:    int  = 1,
    depth_conf_thr:  float = 0.5,
    depth_voxel:     float = 0.02,
    depth_time_stride: int = 1,
    depth_mode:      str  = "image",
    point_radius:    float = 0.01,
    port:            int  = 9090,
    grpc_port:       int  = 9876,
    server_memory_limit: str = "8GiB",
    save:            Optional[Path] = None,
    frames_dir:      Optional[Path] = None,
) -> None:
    """Lag-free Rerun viewer for ghost fusion predictions.

    Drop-in replacement for ``visualize_fusion.py`` (which uses viser).  Every
    entity — predicted/GT meshes, predicted/GT camera poses, the video feeds and
    the VGGT depth point clouds — is streamed to Rerun as a single columnar
    batch, so scrubbing/playback has no per-frame data transfer.

    Parameters mirror ``visualize_fusion.run``.  ``save`` writes a ``.rrd`` for
    offline replay instead of serving live.
    """
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name

    # ── load predictions ──────────────────────────────────────────────────────
    print(f"Loading {predictions} …")
    d = dict(np.load(predictions, allow_pickle=True))
    pose              = d["pose"]               # (T, P, J, 6)
    shape             = d["shape"]              # (P, 10)
    camera            = d["camera"]             # (T, K, 8)
    body_transl_world = d["body_transl_world"]  # (T, P, 3)
    T, P, J, _ = pose.shape
    K = camera.shape[1]
    print(f"  T={T} frames, P={P} persons, K={K} cameras")

    gt_body_pose         = d.get("gt_body_pose")
    gt_body_shape        = d.get("gt_body_shape")
    gt_body_transl_world = d.get("gt_body_transl_world")
    if gt_body_transl_world is not None:
        gt_valid = ~np.all(gt_body_transl_world == 0, axis=-1)   # (T, P)
    else:
        gt_valid = np.zeros((T, P), dtype=bool)

    # ── SMPL-X meshes (world frame == Rerun RDF, no axis flip) ────────────────
    print("Running SMPL-X forward pass (predictions) …")
    pred_verts, faces = _build_smplx_vertices(
        pose, shape, body_transl_world, smplx_model_dir
    )
    gt_verts = None
    if show_gt and gt_body_pose is not None and gt_body_transl_world is not None:
        print("Running SMPL-X forward pass (GT) …")
        gt_shape_arr = gt_body_shape if gt_body_shape is not None else shape
        gt_verts, _ = _build_smplx_vertices(
            gt_body_pose, gt_shape_arr, gt_body_transl_world, smplx_model_dir
        )

    # ── decode cameras ────────────────────────────────────────────────────────
    R_w2c, t_w2c, focal = _decode_cameras(camera)
    gt_camera = d.get("gt_camera")
    gt_R = gt_t = gt_focal = None
    if gt_camera is not None:
        gt_R, gt_t, gt_focal = _decode_cameras(gt_camera)

    # ── image size from first available frame ─────────────────────────────────
    sample = _load_rich_frame(rich_data_root, scene_name, 0, frame_start, frames_dir)
    H, W = (sample.shape[:2] if sample is not None else (1080, 1920))

    # ── optional VGGT depth ───────────────────────────────────────────────────
    depth_ctx = None
    if show_depth:
        depth_ctx = _load_depth_context(scene_dir, camera, T, K)
        if depth_ctx is None:
            show_depth = False

    # ── build the recording ───────────────────────────────────────────────────
    def _build() -> None:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        all_frames = np.arange(T, dtype=np.int64)

        # Predicted meshes (always shown, all frames).
        for p in tqdm(range(P), desc="Pred meshes", unit="person"):
            _send_mesh_column(
                f"world/person_{p}/pred", faces, _PALETTE[p % len(_PALETTE)],
                all_frames, pred_verts[:, p], fps,
            )

        # GT meshes (light grey; only annotated frames).
        if gt_verts is not None:
            for p in tqdm(range(P), desc="GT meshes", unit="person"):
                fr = np.nonzero(gt_valid[:, p])[0].astype(np.int64)
                if fr.size:
                    _send_mesh_column(
                        f"world/person_{p}/gt", faces, (200, 200, 200),
                        fr, gt_verts[fr, p], fps,
                    )

        # Predicted cameras: intrinsics + pose track + video frames.
        for k in tqdm(range(K), desc="Cameras", unit="cam"):
            ent = f"world/cam_{k:02d}"
            _log_pinhole_simple(ent, focal[k], W, H)
            _send_transform_column(ent, R_w2c[:, k], t_w2c[:, k], all_frames, fps)
            _send_camera_media(
                ent, k, scene_name, rich_data_root, frames_dir, frame_start,
                T, W, H, fps, depth_ctx if show_depth else None, depth_stride,
                depth_conf_thr, scene_dir, depth_voxel, depth_time_stride,
                depth_mode, point_radius,
            )

        # GT cameras (static frustums) — suppressed together with GT meshes.
        if show_gt and gt_R is not None:
            for k in range(gt_R.shape[1]):
                ent = f"world/cam_gt_{k:02d}"
                _log_pinhole_simple(ent, gt_focal[k], W, H)
                _log_static_transform(ent, gt_R[0, k], gt_t[0, k])

    # ── serve or save ─────────────────────────────────────────────────────────
    rr.init("ghost_fusion", spawn=False)
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save), default_blueprint=_default_blueprint(fps))
        _build()
        print(f"Recording saved  →  {save}")
        return

    server_uri = rr.serve_grpc(
        grpc_port=grpc_port,
        server_memory_limit=server_memory_limit,
        default_blueprint=_default_blueprint(fps),
    )
    rr.serve_web_viewer(web_port=port, open_browser=False, connect_to=server_uri)
    _build()
    _print_serve_help(port, grpc_port)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def _print_serve_help(port: int, grpc_port: int) -> None:
    """Print the dual-port SSH tunnel and the viewer URL with data source baked in.

    Rerun 0.32 serves the viewer (``port``) and the gRPC data stream
    (``grpc_port``) separately, so BOTH must be tunnelled, and the browser must
    open the URL that carries the gRPC endpoint as a ``?url=`` parameter — a bare
    ``localhost:<port>`` shows the empty "Welcome to Rerun" page.
    """
    from urllib.parse import quote
    data_uri = f"rerun+http://localhost:{grpc_port}/proxy"
    viewer_url = f"http://localhost:{port}/?url={quote(data_uri, safe='')}"
    print("\n" + "=" * 70)
    print("Rerun viewer ready. On your laptop, tunnel BOTH ports:")
    print(f"  ssh -L {port}:localhost:{port} -L {grpc_port}:localhost:{grpc_port} <host>")
    print("\nThen open this exact URL (it carries the data source):")
    print(f"  {viewer_url}")
    print("\n(A bare http://localhost:%d shows the empty welcome page.)" % port)
    print("=" * 70 + "\nPress Ctrl+C to stop.\n")


def _load_depth_context(scene_dir: Path, camera: np.ndarray, T: int, K: int):
    """Load VGGT depth + camera npz and compute per-frame metric scale, or None."""
    depth_path = scene_dir / "vggt_depth_centered.npz"
    cam_path   = scene_dir / "vggt_cameras_centered.npz"
    if not (depth_path.exists() and cam_path.exists()):
        print(f"WARNING: --show-depth requested but {depth_path.name} / "
              f"{cam_path.name} not found in {scene_dir} — depth disabled.")
        return None

    depth_npz = np.load(depth_path, mmap_mode="r")
    cam_npz   = np.load(cam_path)
    extr = cam_npz["extrinsics"]       # (T, K, 3, 4) cam-from-world
    cam_valid = cam_npz["valid"]       # (T, K) bool
    if not cam_valid.any():
        cam_valid = ~np.isnan(extr[:, :, 0, 0])
    names = [n.decode() if isinstance(n, bytes) else n for n in cam_npz["camera_names"]]

    # Per-frame metric scale: median ||t_pred|| / ||t_vggt|| over valid cams.
    scale = np.full(T, np.nan)
    for ts in range(min(T, extr.shape[0])):
        ratios = []
        for ks in range(extr.shape[1]):
            if not cam_valid[ts, ks]:
                continue
            n_raw  = np.linalg.norm(extr[ts, ks, :3, 3])
            n_pred = np.linalg.norm(camera[ts, ks, 4:7]) if ks < K else 0.0
            if n_raw > 1e-6 and n_pred > 1e-6:
                ratios.append(n_pred / n_raw)
        if ratios:
            scale[ts] = np.median(ratios)
    med = np.nanmedian(scale)
    scale = np.where(np.isfinite(scale), scale, med if np.isfinite(med) else 1.0)
    print(f"Depth point clouds enabled (median scale = {np.median(scale):.4f} m/unit)")

    return {
        "depth_mm":   depth_npz["depth"],        # (T, K, h, w) uint16
        "depth_conf": depth_npz["depth_conf"],   # (T, K, h, w) float16
        "depth_valid": depth_npz["depth_valid"], # (T, K) bool
        "extr": extr, "intr": cam_npz["intrinsics"],
        "oc": cam_npz["original_coords"],        # (T, K, 4)
        "cam_valid": cam_valid, "names": names, "scale": scale,
    }


def _send_camera_media(
    entity, k, scene_name, rich_data_root, frames_dir, frame_start,
    T, W, H, fps, depth_ctx, depth_stride, depth_conf_thr, scene_dir,
    depth_voxel=0.0, depth_time_stride=1, depth_mode="image", point_radius=0.01,
) -> None:
    """Single pass over a camera's frames: ship the JPEG video column and,
    when depth is enabled, the depth cloud.

    ``depth_mode='image'`` streams a masked metric ``DepthImage`` per frame and
    lets Rerun back-project it on the GPU — fast to build, small payload, but the
    3D cloud is colour-mapped by distance.  ``depth_mode='points'`` unprojects on
    the CPU into an RGB-coloured ``Points3D`` cloud — slow to build, large
    payload, but keeps the true scene colour.
    """
    from rerun import datatypes as _rd

    blobs: list[bytes] = []
    img_frames: list[int] = []
    # Points3D accumulators.
    d_pos: list[np.ndarray] = []
    d_col: list[np.ndarray] = []
    d_cnt: list[int] = []
    d_frames: list[int] = []
    # DepthImage accumulators.
    di_buf: list[bytes] = []
    di_fmt: list = []
    di_intr: list[np.ndarray] = []
    di_res: list = []
    di_trans: list[np.ndarray] = []
    di_mat: list[np.ndarray] = []
    di_frames: list[int] = []
    mask_cache: dict = {}
    want_depth = depth_ctx is not None

    for t in tqdm(range(T), desc=f"{entity}", unit="frame", leave=False):
        bgr = _load_rich_frame(rich_data_root, scene_name, k, frame_start + t, frames_dir)
        if bgr is not None:
            blob = _jpeg(bgr)
            if blob is not None:
                blobs.append(blob)
                img_frames.append(t)

        if want_depth and (t % depth_time_stride == 0):
            if depth_mode == "points":
                res = _depth_cloud(
                    depth_ctx, t, k, frame_start, scene_dir, bgr, W, H,
                    depth_stride, depth_conf_thr, mask_cache, depth_voxel,
                )
                if res is not None:
                    pts, cols = res
                    d_pos.append(pts)
                    d_col.append(cols)
                    d_cnt.append(len(pts))
                    d_frames.append(t)
            else:
                res = _depth_image_frame(
                    depth_ctx, t, k, frame_start, scene_dir,
                    depth_stride, depth_conf_thr, mask_cache,
                )
                if res is not None:
                    dimg, intr, R, tvec = res
                    hd, wd = dimg.shape
                    di_buf.append(dimg.astype(np.float16).tobytes())
                    di_fmt.append(_rd.ImageFormat(
                        width=wd, height=hd,
                        channel_datatype=_rd.ChannelDatatype.F16,
                    ))
                    di_intr.append(intr)
                    di_res.append([float(wd), float(hd)])
                    di_trans.append(-(R.T @ tvec))
                    di_mat.append(R.T)
                    di_frames.append(t)

    if blobs:
        rr.send_columns(
            entity,
            indexes=_time_columns(np.asarray(img_frames, dtype=np.int64), fps),
            columns=rr.EncodedImage.columns(
                blob=blobs, media_type=["image/jpeg"] * len(blobs)
            ),
        )

    if d_frames:                                   # points mode
        dent = f"world/depth/cam_{k:02d}"
        # Single static radius broadcasts to every point — no per-point array,
        # so no size bloat.  Positive value == metres (world scale).
        rr.log(dent, rr.Points3D.from_fields(radii=[float(point_radius)]),
               static=True)
        rr.send_columns(
            dent,
            indexes=_time_columns(np.asarray(d_frames, dtype=np.int64), fps),
            columns=rr.Points3D.columns(
                positions=np.concatenate(d_pos).astype(np.float32),
                colors=np.concatenate(d_col).astype(np.uint8),
            ).partition(np.asarray(d_cnt, dtype=np.int64)),
        )

    if di_frames:                                  # image mode
        dent = f"world/depth/cam_{k:02d}"
        # Intrinsics/axes are ~fixed per camera → log the Pinhole once, static.
        rr.log(
            dent,
            rr.Pinhole(
                image_from_camera=di_intr[0],
                resolution=di_res[0],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.15,
            ),
            static=True,
        )
        idx = _time_columns(np.asarray(di_frames, dtype=np.int64), fps)
        # Per-frame camera pose (captures VGGT's per-frame extrinsic jitter).
        rr.send_columns(
            dent, indexes=idx,
            columns=rr.Transform3D.columns(
                translation=np.asarray(di_trans, dtype=np.float32),
                mat3x3=np.asarray(di_mat, dtype=np.float32),
            ),
        )
        # Per-frame depth map — Rerun back-projects it through the Pinhole.
        rr.send_columns(
            dent, indexes=idx,
            columns=rr.DepthImage.columns(
                buffer=di_buf, format=di_fmt, meter=[1.0] * len(di_frames),
            ),
        )


def _voxel_downsample(
    pts: np.ndarray, cols: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per ``voxel``-sized cell (first hit) — thins redundant
    same-surface points while preserving geometry and RGB.  Pure numpy, so the
    per-frame cloud that gets streamed is a fraction of the raw stride-1 grid."""
    if len(pts) == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], cols[idx]


def _depth_cloud(
    ctx, t, k, frame_start, scene_dir, bgr, W, H, stride, conf_thr, mask_cache,
    voxel=0.0,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Unproject camera k's depth at frame t into world space (RDF, no flip)."""
    depth_mm   = ctx["depth_mm"]
    if (t >= depth_mm.shape[0] or k >= depth_mm.shape[1]
            or not ctx["depth_valid"][t, k] or not ctx["cam_valid"][t, k]):
        return None

    h_full, w_full = depth_mm[t, k].shape
    d = depth_mm[t, k][::stride, ::stride].astype(np.float32) / 1000.0 * float(ctx["scale"][t])
    conf = ctx["depth_conf"][t, k][::stride, ::stride].astype(np.float32)
    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32) * stride
    x1, y1, x2, y2 = ctx["oc"][t, k]

    # Exclude pixels covered by any person (background-only cloud).
    bg = _person_bg_mask(ctx, k, frame_start + t, h_full, w_full, scene_dir, mask_cache)
    bg = True if bg is None else bg[::stride, ::stride]

    mask = (
        (d > 1e-4) & (conf >= conf_thr)
        & (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2) & bg
    )
    if not mask.any():
        return None
    u, v, z = uu[mask], vv[mask], d[mask]

    intr = ctx["intr"][t, k]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

    R_d   = ctx["extr"][t, k, :3, :3]
    t_vec = ctx["extr"][t, k, :3, 3] * float(ctx["scale"][t])
    pts_world = (pts_cam - t_vec) @ R_d        # world == Rerun RDF, no flip

    # Colours from the (BGR→RGB) RICH frame.
    if bgr is not None:
        fh, fw = bgr.shape[:2]
        iu = np.clip(((u - x1) * W / (x2 - x1)).astype(np.int32), 0, fw - 1)
        iv = np.clip(((v - y1) * H / (y2 - y1)).astype(np.int32), 0, fh - 1)
        colors = bgr[iv, iu][:, ::-1]          # BGR → RGB
    else:
        colors = np.full((len(z), 3), 128, dtype=np.uint8)
    pts_world = pts_world.astype(np.float32)
    if voxel and voxel > 0.0:
        pts_world, colors = _voxel_downsample(pts_world, colors, voxel)
    return pts_world, colors


def _depth_image_frame(
    ctx, t, k, frame_start, scene_dir, stride, conf_thr, mask_cache,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build one masked metric-depth map (metres) plus the VGGT intrinsics and
    extrinsics Rerun needs to back-project it on the GPU.

    Invalid pixels (low confidence, outside the valid crop, or covered by a
    person) are set to 0 — Rerun does not back-project zero-depth pixels, so the
    cloud contains only the real background surface.  Returns
    ``(depth (h,w) float32 metres, intr_ds (3,3), R (3,3), t_scaled (3,))`` or
    ``None`` when the frame/camera has no usable depth.
    """
    depth_mm = ctx["depth_mm"]
    if (t >= depth_mm.shape[0] or k >= depth_mm.shape[1]
            or not ctx["depth_valid"][t, k] or not ctx["cam_valid"][t, k]):
        return None

    h_full, w_full = depth_mm[t, k].shape
    s = float(ctx["scale"][t])
    d = depth_mm[t, k][::stride, ::stride].astype(np.float32) / 1000.0 * s
    conf = ctx["depth_conf"][t, k][::stride, ::stride].astype(np.float32)
    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d]
    u_full, v_full = uu * stride, vv * stride
    x1, y1, x2, y2 = ctx["oc"][t, k]

    bg = _person_bg_mask(ctx, k, frame_start + t, h_full, w_full, scene_dir, mask_cache)
    bg = np.ones((h_d, w_d), bool) if bg is None else bg[::stride, ::stride]

    valid = (
        (d > 1e-4) & (conf >= conf_thr)
        & (u_full >= x1) & (u_full < x2) & (v_full >= y1) & (v_full < y2) & bg
    )
    if not valid.any():
        return None
    d[~valid] = 0.0                              # 0 == no measurement

    intr = ctx["intr"][t, k].astype(np.float64).copy()
    intr[0, 0] /= stride; intr[1, 1] /= stride   # fx, fy → downsampled pixels
    intr[0, 2] /= stride; intr[1, 2] /= stride   # cx, cy → downsampled pixels
    R = ctx["extr"][t, k, :3, :3].astype(np.float64)
    t_scaled = ctx["extr"][t, k, :3, 3].astype(np.float64) * s
    return d, intr, R, t_scaled


def _person_bg_mask(
    ctx, k, abs_frame, h_full, w_full, scene_dir, mask_cache,
) -> Optional[np.ndarray]:
    """Return (h_full, w_full) bool, True where NO person is present, or None."""
    names = ctx["names"]
    cam_name = names[k] if k < len(names) else f"cam_{k:02d}"
    if k not in mask_cache:
        mpath = scene_dir / cam_name / "mask_data.npz"
        mask_cache[k] = np.load(mpath, mmap_mode="r") if mpath.exists() else None
    npz = mask_cache[k]
    if npz is None:
        return None
    cam_idx = int(cam_name.split("_")[-1])
    key = f"mask_{abs_frame:05d}_{cam_idx:02d}"
    if key not in npz:
        return None
    raw = npz[key].astype(np.uint16)
    if raw.shape != (h_full, w_full):
        from PIL import Image as _PIL
        raw = np.array(_PIL.fromarray(raw).resize((w_full, h_full), _PIL.NEAREST))
    return raw == 0


# ── Example / CLI entry point ────────────────────────────────────────────────

def demo_body_data() -> None:
    import torch
    import smplx

    # ── Config — edit these ──────────────────────────────────────────────────
    VIDEO_DIR       = Path("test_outputs/reid_logging_segmentation_test/BBQ_001_guitar/cam_00")
    SMPLX_MODEL_DIR = Path("body_models")
    FRAMES_DIR      = None          # None → VIDEO_DIR / "frames"
    FPS             = 30.0
    PORT            = 9090
    SAVE            = None          # Path("scene.rrd") to save instead of serving
    # ────────────────────────────────────────────────────────────────────────

    body_dir     = VIDEO_DIR / "body_data"
    person_files = sorted(body_dir.glob("person_*.npz"))
    if not person_files:
        raise FileNotFoundError(f"No person_*.npz files found in {body_dir}")

    person_data = [dict(np.load(p, allow_pickle=True)) for p in person_files]
    T = int(max(d["frame_indices"].max() for d in person_data)) + 1
    P = len(person_data)
    print(f"Found {P} person(s) across {T} frames.")

    # ── SMPL-X forward pass ──────────────────────────────────────────────────
    smplx_pkl = SMPLX_MODEL_DIR / "SMPLX_NEUTRAL.pkl"
    smplx_model = smplx.SMPLX(str(smplx_pkl), use_pca=False, num_betas=10)
    smplx_model.eval()

    n_verts  = smplx_model.get_num_verts()
    faces    = smplx_model.faces.copy()                          # (F, 3)
    vertices = np.full((T, P, n_verts, 3), np.nan, dtype=np.float32)

    with torch.no_grad():
        for p_idx, data in enumerate(person_data):
            frame_indices = data["frame_indices"]                # (N,)
            out = smplx_model(
                betas         = torch.from_numpy(data["smplx_betas"]),
                body_pose     = torch.from_numpy(data["smplx_body_pose"]),
                global_orient = torch.from_numpy(data["smplx_global_orient"]),
                transl        = torch.from_numpy(data["smplx_transl"]),
                return_verts  = True,
            )
            verts_np = out.vertices.numpy()                      # (N, V, 3)
            for i, fi in enumerate(frame_indices):
                vertices[int(fi), p_idx] = verts_np[i]
            print(f"  Person {p_idx} ({person_files[p_idx].name}): {len(frame_indices)} frames.")

    # ── Camera ───────────────────────────────────────────────────────────────
    # SMPL-X vertices are in camera space; camera sits at the origin.
    all_focals = np.concatenate([d["focal_length"] for d in person_data])
    focal      = float(np.median(all_focals))
    frames_dir = FRAMES_DIR or (VIDEO_DIR / "frames")

    # Detect image dimensions from the first readable frame.
    W, H = 1920, 1080
    for stem in (f"{0:06d}", f"{0:05d}", f"{0:04d}", "0"):
        found = False
        for ext in _FRAME_EXTS:
            p = frames_dir / f"{stem}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    H, W = img.shape[:2]
                found = True
                break
        if found:
            break

    cam = CameraView(
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        focal_length=focal,
        img_wh=(W, H),
        frames_dir=frames_dir,
    )

    viewer = SceneViewer(
        vertices=vertices,
        faces=faces,
        cameras={VIDEO_DIR.name: cam},
        fps=FPS,
    )

    if SAVE is not None:
        viewer.save(SAVE)
    else:
        print(f"\nStarting Rerun viewer on port {PORT}.")
        print(f"On a cluster, run on your laptop:  ssh -L {PORT}:localhost:{PORT} <host>")
        print(f"Then open:  http://localhost:{PORT}\n")
        viewer.serve(port=PORT)


if __name__ == "__main__":
    import tyro
    tyro.cli(run_fusion)
