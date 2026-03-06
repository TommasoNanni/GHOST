"""Interactive 3D scene viewer for the ghost multi-camera pipeline.

Uses Rerun (https://rerun.io) to build an interactive scene that can be
explored in a browser at ``http://localhost:<port>``.

What you get
------------
* SMPL-X body meshes animated per frame, in world coordinates.
* Camera frustums positioned in 3D world space.
* The video feed behind each frustum — click a camera in the 3D view to
  look through it and see the synchronised video.
* A shared timeline: scrubbing syncs meshes and all video streams.

Quick start
-----------
    from visualize.visualize import SceneViewer, CameraView

    viewer = SceneViewer(
        vertices=vertices,   # (T, P, V, 3) – world coords, one mesh per person
        faces=faces,         # (F, 3)        – SMPL-X face topology
        cameras={
            "cam01": CameraView.from_pipeline_params(
                cam_params_8d,   # (T, 8) from the ghost pipeline
                img_wh=(W, H),
                frames_dir=Path("/data/scene/cam01/frames"),
            ),
            "cam02": CameraView.from_pipeline_params(
                cam_params_8d,
                img_wh=(W, H),
                video_path=Path("/data/scene/cam02.mp4"),
            ),
        },
        fps=30.0,
    )

    # Live viewer — on a cluster:
    #   ssh -L 9090:localhost:9090 <host>
    # then open http://localhost:9090
    viewer.serve(port=9090)

    # Or save to a .rrd file and replay anywhere:
    viewer.save("scene.rrd")
    # then:  python -m rerun scene.rrd
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
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
        MP4 video file.  When Rerun >= 0.19 is available the whole file is
        logged once as an ``AssetVideo`` and frames are referenced cheaply;
        otherwise frames are decoded and JPEG-encoded per-frame.
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
    """Interactive 3D scene viewer for the ghost multi-camera pipeline.

    Builds a Rerun recording that contains:

    * SMPL-X body meshes in world coordinates, animated per frame.
    * Camera frustums placed in 3D world space.
    * Synchronised video feeds — one per camera, displayed behind the
      corresponding frustum so you can click it and look through it.
    * A shared timeline that scrubs meshes and all video feeds together.

    Parameters
    ----------
    vertices : (T, P, V, 3) ndarray
        Mesh vertex positions in world coordinates (first-camera space).
        ``(T, V, 3)`` is also accepted for a single-person scene.
        Frames where a person is invisible should have rows set to ``NaN``.
    faces : (F, 3) ndarray
        Face topology (e.g. SMPL-X body model faces, 10475 vertices).
    cameras : dict[str, CameraView]
        Ordered mapping from camera name to parameters and image source.
        Names appear as entity path segments in the Rerun UI — keep them
        short and alphanumeric (e.g. ``"cam01"``).
    fps : float
        Frame rate used to build the time axis.
    person_colors : list of (R, G, B) uint8 tuples, optional
        Per-person mesh colour.  Cycles through a default palette when omitted.
    recording_id : str
        Rerun recording identifier shown in the viewer title bar.

    Examples
    --------
    Live viewer (SSH-tunnel on a cluster)::

        viewer = SceneViewer(vertices, faces, cameras, fps=30)
        viewer.serve(port=9090)
        # laptop:  ssh -L 9090:localhost:9090 <cluster_host>
        # browser: http://localhost:9090

    Save and replay on any machine::

        viewer.save("scene.rrd")
        # python -m rerun scene.rrd
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
        self.fps          = fps
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


    def serve(self, port: int = 9090, open_browser: bool = False) -> None:
        """Log all data and start a WebSocket server; blocks until Ctrl+C.

        Parameters
        ----------
        port : int
            HTTP port for the Rerun web viewer.
        open_browser : bool
            Auto-open a browser tab.  Leave ``False`` on headless machines.
        """
        import rerun as rr

        rr.init(self.recording_id, spawn=False)

        # rr.serve_web added in 0.19; older builds expose rr.serve
        try:
            rr.serve_web(open_browser=open_browser, web_port=port)
        except AttributeError:
            rr.serve(open_browser=open_browser, web_port=port)

        self._build_recording()

        print(f"\nRerun viewer ready  →  http://localhost:{port}")
        print(f"On a cluster run:      ssh -L {port}:localhost:{port} <host>")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    def save(self, path: Union[str, Path]) -> Path:
        """Log all data and write a ``.rrd`` file for later replay.

        Open anywhere with::

            python -m rerun scene.rrd

        Parameters
        ----------
        path : str or Path
            Output path.  ``.rrd`` extension recommended.

        Returns
        -------
        Path
        """
        import rerun as rr

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rr.init(self.recording_id, spawn=False)
        rr.save(str(path))
        self._build_recording()
        print(f"Recording saved  →  {path}")
        return path


    def _build_recording(self) -> None:
        import rerun as rr

        # World coordinate system: RDF = right / down / forward (OpenCV).
        # This matches the first-camera convention used throughout the pipeline.
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        self._log_static_cameras()
        self._log_frames()

    def _log_static_cameras(self) -> None:
        for cam_name, cam in self.cameras.items():
            # Pinhole intrinsics are always static — focal length assumed fixed.
            _log_pinhole(cam_name, cam)
            # Extrinsics are static only for fixed (non-moving) cameras.
            if cam.is_static:
                _log_camera_transform(cam_name, cam.R, cam.t, static=True)


    def _log_frames(self) -> None:
        import rerun as rr

        # Cache arrays to avoid repeated attribute lookups inside the hot loop
        R_arr     = {n: np.asarray(c.R)            for n, c in self.cameras.items()}
        t_arr     = {n: np.asarray(c.t)            for n, c in self.cameras.items()}

        # Decord VideoReaders — opened once, closed when the loop exits
        video_readers: dict[str, Any] = {}
        for cam_name, cam in self.cameras.items():
            if cam.video_path is not None and cam.frames_dir is None:
                try:
                    from decord import VideoReader
                    video_readers[cam_name] = VideoReader(str(cam.video_path))
                except Exception as exc:
                    print(f"  [{cam_name}] Cannot open video: {exc}")

        # For cameras with a video file, try the efficient AssetVideo path
        # (Rerun >= 0.19).  Falls back silently to per-frame decoding.
        asset_cameras: set[str] = set()
        for cam_name, cam in self.cameras.items():
            if cam.video_path is not None:
                if _try_log_video_asset(cam_name, cam.video_path):
                    asset_cameras.add(cam_name)

        for t_idx in tqdm(range(self.T), desc="Logging frames", unit="frame"):
            rr.set_time_sequence("frame", t_idx)
            rr.set_time_seconds("time_s", t_idx / self.fps)

            for p_idx in range(self.P):
                verts = self.vertices[t_idx, p_idx]   # (V, 3)
                if np.any(np.isnan(verts)):
                    continue
                color = np.array(self.person_colors[p_idx], dtype=np.uint8)
                rr.log(
                    f"world/person_{p_idx}",
                    rr.Mesh3D(
                        vertex_positions=verts.astype(np.float32),
                        triangle_indices=self.faces,
                        # broadcast the scalar colour to every vertex
                        vertex_colors=np.broadcast_to(
                            color, (len(verts), 3)
                        ).copy(),
                    ),
                )

            for cam_name, cam in self.cameras.items():
                R = R_arr[cam_name]
                t = t_arr[cam_name]

                if R.ndim == 3:    # time-varying: (T, 3, 3)
                    _log_camera_transform(
                        cam_name, R[t_idx], t[t_idx], static=False
                    )

                if cam_name in asset_cameras:
                    # Fast path: VideoFrameReference into the already-logged
                    # video asset — no image data is written per frame.
                    _log_video_frame_reference(cam_name, t_idx, self.fps)

                elif cam.frames_dir is not None:
                    frame_bgr = _read_frame_from_dir(cam.frames_dir, t_idx)
                    if frame_bgr is not None:
                        _log_encoded_image(cam_name, frame_bgr)

                elif cam_name in video_readers:
                    try:
                        frame_rgb = video_readers[cam_name][t_idx].asnumpy()
                        _log_encoded_image(
                            cam_name,
                            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                        )
                    except Exception:
                        pass


def _log_pinhole(cam_name: str, cam: CameraView) -> None:
    """Log Pinhole intrinsics (always logged as static)."""
    import rerun as rr

    fl = cam.focal_length
    focal = float(fl) if np.isscalar(fl) else np.asarray(fl).mean().item()
    cx, cy = cam.principal_point or (cam.W / 2.0, cam.H / 2.0)

    rr.log(
        f"world/{cam_name}",
        rr.Pinhole(
            focal_length=focal,
            principal_point=(cx, cy),
            width=cam.W,
            height=cam.H,
            camera_xyz=rr.ViewCoordinates.RDF,   # OpenCV: right-down-forward
        ),
        static=True,
    )


def _log_camera_transform(
    cam_name: str,
    R: np.ndarray,   # (3, 3) world-to-camera rotation
    t: np.ndarray,   # (3,)   world-to-camera translation
    static: bool = False,
) -> None:
    """Log the camera-to-world Transform3D.

    Rerun's Transform3D expresses the pose of an entity in its parent space
    (i.e. camera-to-world for a top-level camera).

    Our pipeline stores world-to-camera:  x_cam = R @ x_world + t
    Camera-to-world (the inverse):        translation = -Rᵀ t,  rotation = Rᵀ
    """
    import rerun as rr

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    rr.log(
        f"world/{cam_name}",
        rr.Transform3D(translation=-(R.T @ t), mat3x3=R.T),
        static=static,
    )


def _try_log_video_asset(cam_name: str, video_path: Path) -> bool:
    """Log the full video file as a Rerun ``AssetVideo`` (requires >= 0.19).

    Returns True on success, False when the API is absent or the file cannot
    be read.  The whole MP4 is stored once in the recording; per-frame image
    data is then a tiny ``VideoFrameReference`` instead of a JPEG blob.
    """
    import rerun as rr

    if not hasattr(rr, "AssetVideo"):
        return False
    try:
        with open(video_path, "rb") as fh:
            blob = fh.read()
        rr.log(f"world/{cam_name}", rr.AssetVideo(blob=blob), static=True)
        return True
    except Exception as exc:
        print(f"  [{cam_name}] AssetVideo logging failed: {exc}")
        return False


def _log_video_frame_reference(
    cam_name: str, t_idx: int, fps: float
) -> None:
    """Log a ``VideoFrameReference`` at the current Rerun timestamp (>= 0.19).

    Rerun seeks into the ``AssetVideo`` at the same entity path using the
    ``seconds`` timestamp.
    """
    import rerun as rr

    if not hasattr(rr, "VideoFrameReference"):
        return
    try:
        rr.log(
            f"world/{cam_name}",
            rr.VideoFrameReference(seconds=t_idx / fps),
        )
    except Exception:
        pass


def _log_encoded_image(cam_name: str, frame_bgr: np.ndarray) -> None:
    """JPEG-encode a BGR frame and log it as a Rerun ``EncodedImage``."""
    import rerun as rr

    _, jpeg = cv2.imencode(
        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    rr.log(
        f"world/{cam_name}",
        rr.EncodedImage(contents=bytes(jpeg), media_type="image/jpeg"),
    )


def _read_frame_from_dir(
    frames_dir: Path, t_idx: int
) -> Optional[np.ndarray]:
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
