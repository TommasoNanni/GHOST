"""Fusion dataset for the SST model.

Provides a base class :class:`FusionDataset` that handles windowing, lookup
tables, and the uniform ``(inputs, targets)`` contract expected by
:class:`~fusion.fusion_module.SSTNetwork`, plus per-dataset subclasses that
each implement three hooks:

* ``load_body_data()``   -- populate ``self._raw`` from segmentation outputs
* ``load_cameras()``     -- populate ``self._cameras`` from calibration files
* ``load_ground_truth()``-- populate ``self._gt`` from dataset GT annotations

The output tensors always follow the same schema regardless of the source
dataset::

    inputs = dict(
        pose       = Tensor[T, K, P, J, 6],
        shape      = Tensor[T, K, P, 10],
        camera     = Tensor[T, K, 8],
        joint_mask = Tensor[T, K, P, J],
    )
    targets = dict(
        pose       = Tensor[T, K, P, J, 6],
        shape      = Tensor[T, K, P, 10],
        camera     = Tensor[T, K, 8],
        keypoints_3d = Tensor[T, K, P, 70, 3],
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class FusionDataset(Dataset, ABC):
    """Abstract multi-view temporal-window dataset for SST fusion.

    Subclasses **must** implement:
        - :meth:`load_body_data`
        - :meth:`load_cameras`
        - :meth:`load_ground_truth`

    And may override:
        - :meth:`convert_pose`
        - :meth:`convert_camera`
        - :meth:`build_gt_targets`
    """

    # Default raw fields when loading from the ghost pipeline npz files.
    # MHR-native keys (body_pose_params, shape_params, global_rot, …) are
    # intentionally absent: the pipeline now saves only SMPL-X params.
    _NPZ_FIELDS = (
        "frame_indices",
        "pred_cam_t",
        "focal_length",
        "pred_keypoints_2d",
        "pred_keypoints_3d",
        "smplx_betas",
        "smplx_body_pose",
        "smplx_global_orient",
        "smplx_transl",
        "smplx_left_hand_pose",
        "smplx_right_hand_pose",
        "smplx_expression",
        "bbox",
    )

    def __init__(
        self,
        scene_dir: str | Path,
        window_size: int = 128,
        window_stride: int = 64,
        num_joints: int = 55,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.window_size = window_size
        self.window_stride = window_stride
        self.num_joints = num_joints

        # ---- Populated by subclass hooks --------------------------------
        # _cam_dirs[cam_idx] = Path to each camera directory
        self._cam_dirs: list[Path] = []
        # _raw[cam_idx][person_id] = {field_name: np.ndarray, ...}
        self._raw: list[dict[int, dict[str, np.ndarray]]] = []
        # _cameras[cam_idx] = {arbitrary calibration data}
        self._cameras: list[dict[str, Any]] = []
        # _gt[cam_idx][person_id] = {arbitrary GT data}
        self._gt: list[dict[int, dict[str, np.ndarray]]] = []

        # ---- Call subclass hooks ----------------------------------------
        self.load_body_data(**kwargs)
        self.load_cameras(**kwargs)
        self.load_ground_truth(**kwargs)
        self._transform_to_world_frame()

        # ---- Derived book-keeping (dataset-agnostic) --------------------
        self.num_cameras: int = len(self._cam_dirs)
        if self.num_cameras == 0:
            raise FileNotFoundError(
                f"No camera directories found in {self.scene_dir}"
            )

        self.max_persons: int = (
            max(len(cam) for cam in self._raw) if self._raw else 0
        )

        self._frame_start: int = 0
        self._frame_end: int = 0
        self._compute_frame_range()

        self._lookup: list[list[dict[int, int]]] = []
        self._pid_order: list[list[int]] = []
        self._build_lookup()

        self._windows: list[int] = []
        self._build_windows()

        logger.info(
            f"{type(self).__name__}: {self.num_cameras} cameras, "
            f"{self.max_persons} max persons, "
            f"frames [{self._frame_start}, {self._frame_end}), "
            f"{len(self._windows)} windows "
            f"(T={window_size}, stride={window_stride})"
        )

    # ------------------------------------------------------------------
    # Abstract hooks -- every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_body_data(self, **kwargs: Any) -> None:
        """Populate ``self._cam_dirs`` and ``self._raw``.

        After this method returns:
        - ``self._cam_dirs`` is a list of camera directory Paths.
        - ``self._raw[cam_idx][person_id]`` maps to dicts whose keys
          **must** include ``"frame_indices"`` (1-D int array) and may
          include any of ``_NPZ_FIELDS``.
        """
        ...

    @abstractmethod
    def load_cameras(self, **kwargs: Any) -> None:
        """Populate ``self._cameras`` with per-camera predicted calibration data.

        ``self._cameras[cam_idx]`` can hold any dict the subclass needs
        (e.g. intrinsics, extrinsics).  The base :meth:`convert_camera`
        will use whatever is stored here.
        """
        ...

    @abstractmethod
    def load_ground_truth(self, **kwargs: Any) -> None:
        """Populate ``self._gt`` with per-camera, per-person GT data.

        ``self._gt[cam_idx][person_id]`` can hold any dict the subclass
        needs (e.g. GT SMPL params, GT keypoints).  Used in
        :meth:`__getitem__` to build the ``targets`` dict.

        If no GT is available (self-supervised), leave ``self._gt`` empty.
        """
        ...

    def _transform_to_world_frame(self) -> None:
        """Re-express all body parameters in a common world coordinate frame.

        The default implementation is a no-op.  Subclasses that have
        calibrated camera extrinsics should override this to transform
        ``self._raw`` (predictions) and ``self._gt`` (ground truth) from
        their native coordinate frames into the shared world frame.

        Called automatically by ``__init__`` after all three data-loading
        hooks complete.
        """

    # ------------------------------------------------------------------
    # Conversion hooks (override per-dataset if needed)
    # ------------------------------------------------------------------

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        """Convert raw pose parameters to ``[J, 6]`` (6-D rotation).

        Default: zeros.  Override per-dataset.
        """
        return np.zeros((self.num_joints, 6), dtype=np.float32)

    def convert_camera(
        self,
        pred_cam_t: np.ndarray,
        focal_length: float,
        global_rot: np.ndarray,
        cam_calib: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Convert raw camera params to ``[8]`` = ``[quat(4), trans(3), focal_raw(1)]``.

        ``cam_calib`` is ``self._cameras[cam_idx]`` -- available for
        subclasses that have proper extrinsics.
        Default: identity rotation + pred_cam_t + softplus(focal_raw)=focal_length.
        """
        cam = np.zeros(8, dtype=np.float32)
        cam[3] = 1.0  # qw = 1 -> identity rotation
        cam[4:7] = pred_cam_t
        # Store softplus_inv(focal) so that extract_cameras can recover the
        # true focal length by applying softplus.  For f >> 1, softplus_inv(f) ≈ f.
        cam[7] = float(np.log(np.exp(float(focal_length)) - 1.0 + 1e-6))
        return cam

    def build_gt_targets(
        self,
        cam_idx: int,
        person_slot: int,
        pid: int,
        local_idx: int,
        t: int,
        pose_out: np.ndarray,
        shape_out: np.ndarray,
        camera_out: np.ndarray,
        kp3d_out: np.ndarray,
    ) -> None:
        """Fill GT target arrays for one person / one frame.

        By default mirrors the predicted values (self-supervised).
        Override to fill from ``self._gt`` when GT is available.

        Parameters
        ----------
        cam_idx, person_slot, pid, local_idx, t : indexing context
        pose_out   : writable view into targets["pose"][t, cam_idx, person_slot]
        shape_out  : writable view into targets["shape"][t, cam_idx, person_slot]
        camera_out : writable view into targets["camera"][t, cam_idx]
        kp3d_out   : writable view into targets["keypoints_3d"][t, cam_idx, person_slot]
        """
        # Default: no-op -- targets will be set equal to inputs in __getitem__

    # ------------------------------------------------------------------
    # Internal book-keeping (not overridden)
    # ------------------------------------------------------------------

    def _compute_frame_range(self) -> None:
        """Determine [start, end) across all cameras and people."""
        lo: float = float("inf")
        hi: float = float("-inf")
        for cam_persons in self._raw:
            for pdata in cam_persons.values():
                fi = pdata["frame_indices"]
                lo = min(lo, int(fi.min()))
                hi = max(hi, int(fi.max()))
        self._frame_start = int(lo) if lo != float("inf") else 0
        self._frame_end = int(hi) + 1 if hi != float("-inf") else 0

    def _build_lookup(self) -> None:
        """Pre-build per-camera per-person frame -> local-npz-index maps."""
        for cam_idx, cam_persons in enumerate(self._raw):
            pids = sorted(cam_persons.keys())
            self._pid_order.append(pids)
            cam_lookups: list[dict[int, int]] = []
            for pid in pids:
                fi = cam_persons[pid]["frame_indices"]
                cam_lookups.append({int(f): i for i, f in enumerate(fi)})
            while len(cam_lookups) < self.max_persons:
                cam_lookups.append({})
            self._lookup.append(cam_lookups)

    def _build_windows(self) -> None:
        """Create list of start-frame indices for overlapping windows."""
        total = self._frame_end - self._frame_start
        if total <= 0:
            return
        if total <= self.window_size:
            self._windows.append(self._frame_start)
            return
        start = self._frame_start
        while start + self.window_size <= self._frame_end:
            self._windows.append(start)
            start += self.window_stride
        if (
            not self._windows
            or self._windows[-1] + self.window_size < self._frame_end
        ):
            self._windows.append(self._frame_end - self.window_size)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        start = self._windows[idx]
        T = self.window_size
        K = self.num_cameras
        P = self.max_persons
        J = self.num_joints
        frames = np.arange(start, start + T)

        pose = np.zeros((T, K, P, J, 6), dtype=np.float32)
        shape = np.zeros((T, K, P, 10), dtype=np.float32)
        camera = np.zeros((T, K, 8), dtype=np.float32)
        joint_mask = np.zeros((T, K, P, J), dtype=np.float32)
        kp3d = np.zeros((T, K, P, 70, 3), dtype=np.float32)

        # Target arrays (may be overwritten by build_gt_targets).
        gt_pose = np.zeros_like(pose)
        gt_shape = np.zeros_like(shape)
        gt_camera = np.zeros_like(camera)
        gt_kp3d = np.zeros_like(kp3d)

        for k in range(K):
            cam_persons = self._raw[k]
            pids = self._pid_order[k]
            cam_calib = self._cameras[k] if k < len(self._cameras) else None
            cam_camera_filled = np.zeros(T, dtype=bool)

            for p_slot, pid in enumerate(pids):
                if p_slot >= P:
                    break
                pdata = cam_persons[pid]
                lut = self._lookup[k][p_slot]

                for t, gf in enumerate(frames):
                    gf_int = int(gf)
                    if gf_int not in lut:
                        continue
                    li = lut[gf_int]

                    # --- Pose ---
                    bpp = pdata.get("smplx_body_pose")
                    lhp = pdata.get("smplx_left_hand_pose")
                    rhp = pdata.get("smplx_right_hand_pose")
                    gr = pdata.get("smplx_global_orient")
                    if bpp is not None and gr is not None:
                        # Concatenate both hands into a single (90,) array so
                        # convert_pose can place them at the correct joint slots.
                        if lhp is not None and rhp is not None:
                            hpp_li: np.ndarray | None = np.concatenate(
                                [lhp[li], rhp[li]]
                            )
                        elif lhp is not None:
                            hpp_li = lhp[li]
                        elif rhp is not None:
                            hpp_li = rhp[li]
                        else:
                            hpp_li = None
                        pose[t, k, p_slot] = self.convert_pose(
                            bpp[li], hpp_li, gr[li]
                        )

                    # --- Shape ---
                    sp = pdata.get("smplx_betas")
                    if sp is not None:
                        n_betas = min(10, sp.shape[1] if sp.ndim > 1 else sp.shape[0])
                        shape[t, k, p_slot, :n_betas] = (
                            sp[li, :n_betas] if sp.ndim > 1 else sp[:n_betas]
                        )

                    # --- Joint mask ---
                    joint_mask[t, k, p_slot, :] = 1.0

                    # --- Camera ---
                    if not cam_camera_filled[t]:
                        pct = pdata.get("pred_cam_t")
                        fl = pdata.get("focal_length")
                        if pct is not None and gr is not None:
                            camera[t, k] = self.convert_camera(
                                pct[li],
                                float(fl[li]) if fl is not None else 1000.0,
                                gr[li],
                                cam_calib=cam_calib,
                            )
                            cam_camera_filled[t] = True

                    # --- 3D keypoints ---
                    kp = pdata.get("pred_keypoints_3d")
                    if kp is not None:
                        kp3d[t, k, p_slot] = kp[li]

                    # --- GT targets (subclass fills if available) ---
                    self.build_gt_targets(
                        cam_idx=k,
                        person_slot=p_slot,
                        pid=pid,
                        local_idx=li,
                        t=t,
                        pose_out=gt_pose[t, k, p_slot],
                        shape_out=gt_shape[t, k, p_slot],
                        camera_out=gt_camera[t, k],
                        kp3d_out=gt_kp3d[t, k, p_slot],
                    )

        # Default self-supervised targets: mirror inputs where GT wasn't set.
        _has_gt = any(len(g) > 0 for g in self._gt) if self._gt else False
        if not _has_gt:
            gt_pose = pose.copy()
            gt_shape = shape.copy()
            gt_camera = camera.copy()
            gt_kp3d = kp3d.copy()

        inputs = {
            "pose": torch.from_numpy(pose),
            "shape": torch.from_numpy(shape),
            "camera": torch.from_numpy(camera),
            "joint_mask": torch.from_numpy(joint_mask),
        }
        targets = {
            "pose": torch.from_numpy(gt_pose),
            "shape": torch.from_numpy(gt_shape),
            "camera": torch.from_numpy(gt_camera),
            "keypoints_3d": torch.from_numpy(gt_kp3d),
        }
        return inputs, targets

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cameras={self.num_cameras}, "
            f"max_persons={self.max_persons}, "
            f"frames=[{self._frame_start}, {self._frame_end}), "
            f"windows={len(self._windows)}, T={self.window_size})"
        )


# ======================================================================
# EgoExo subclass (ghost pipeline output, no GT)
# ======================================================================

class EgoExoFusionDataset(FusionDataset):
    """Loads body data from the ghost segmentation pipeline output.

    Self-supervised -- no GT available, targets mirror predictions.
    Camera params come from SAM3D's ``pred_cam_t``.
    """

    def load_body_data(self, **kwargs: Any) -> None:
        self._cam_dirs = sorted(
            d
            for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()
        )
        for cam_dir in self._cam_dirs:
            body_dir = cam_dir / "body_data"
            persons: dict[int, dict[str, np.ndarray]] = {}
            for npz_path in sorted(body_dir.glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                data = dict(np.load(str(npz_path), allow_pickle=False))
                persons[pid] = {
                    k: data[k] for k in self._NPZ_FIELDS if k in data
                }
            self._raw.append(persons)
            logger.debug(f"  {cam_dir.name}: loaded {len(persons)} person(s)")

    def load_cameras(self, **kwargs: Any) -> None:
        # No external calibration -- camera is derived per-frame from
        # SAM3D's pred_cam_t inside convert_camera.
        self._cameras = [{} for _ in self._cam_dirs]

    def load_ground_truth(self, **kwargs: Any) -> None:
        # Self-supervised: no GT.
        self._gt = [{} for _ in self._cam_dirs]

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        """Convert SMPL-X axis-angle params to ``[J, 6]`` (6-D rotation).

        Joint layout (matches SMPL-X full_pose ordering):
            0        : global_orient   (3,)
            1 – 21   : body_pose       (63,)  = 21 joints
            22 – 36  : left_hand_pose  (45,)  = 15 joints  ─┐ from
            37 – 51  : right_hand_pose (45,)  = 15 joints  ─┘ hand_pose_params
            52 – 54  : jaw / leye / reye       — zeros (not estimated)

        Parameters
        ----------
        body_pose_params : (63,) float32
            21 body joints × 3 axis-angle.
        hand_pose_params : (90,) | (45,) | None
            Left+right hand concatenated (90,), left only (45,), or None.
        global_rot : (3,) float32
            Global orientation axis-angle.
        """
        J = self.num_joints  # 55
        out = np.zeros((J, 6), dtype=np.float32)
        if body_pose_params is None or global_rot is None:
            return out

        try:
            from scipy.spatial.transform import Rotation as SciR
        except Exception:
            return out

        go = np.asarray(global_rot, dtype=np.float32).reshape(1, 3)   # (1, 3)
        bp = np.asarray(body_pose_params, dtype=np.float32).reshape(-1, 3)  # (21, 3)
        parts = [go, bp]  # (22, 3)
        if hand_pose_params is not None:
            hp = np.asarray(hand_pose_params, dtype=np.float32).reshape(-1, 3)
            parts.append(hp)  # (30, 3) when both hands present
        aa = np.concatenate(parts, axis=0)  # (22,) or (52,) joints

        # Pad remaining slots (face joints) with identity = zero axis-angle.
        if aa.shape[0] < J:
            aa = np.concatenate(
                [aa, np.zeros((J - aa.shape[0], 3), dtype=np.float32)], axis=0
            )

        try:
            mats = SciR.from_rotvec(aa).as_matrix()
        except Exception:
            return out

        sixd = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)
        out[:J] = sixd[:J]
        return out


# ======================================================================
# RICH subclass (RICH dataset with GT SMPL-X params + calibrated cameras)
# ======================================================================

class RICHFusionDataset(FusionDataset):
    """Loads body data from ghost pipeline + GT from RICH annotations.

    Expected RICH data layout (all under ``rich_data_root``) ::

        rich_data_root/
            {scene_name}/           <- video image sequences
                cam_00/             <- one dir per camera
                    00000_00.bmp
                    ...
                cam_01/
            scan_calibration/
                {location}/         <- e.g. 'BBQ', 'LectureHall'
                    calibration/
                        000.xml     <- OpenCV extrinsics + intrinsics
                        001.xml
            train_body/
                {scene_name}/
                    {frame}/        <- one dir per frame
                        001.pkl     <- SMPL-X params for person 001

    The ghost pipeline output (``scene_dir``) must have the same
    ``cam_00/body_data/person_*.npz`` layout as EgoExo.

    Constructor kwargs
    ------------------
    rich_data_root : str | Path
        Root of the RICH split (contains ``scan_calibration/``,
        ``train_body/``, and the scene video directories).
    """

    @staticmethod
    def _scene_stem(scene_name: str) -> str:
        """Extract the location prefix used in scan_calibration.

        Splits on the first ``_{3-digits}_`` separator::

            'BBQ_001_guitar'               -> 'BBQ'
            'LectureHall_018_wipingchairs1' -> 'LectureHall'
            'ParkingLot1_003_walking'       -> 'ParkingLot1'
        """
        m = re.match(r'^(.+?)_\d{3}_', scene_name)
        return m.group(1) if m else scene_name

    @staticmethod
    def _parse_calib_xml(xml_path: Path) -> dict[str, Any]:
        """Parse an OpenCV-format calibration XML.

        Returns a dict with:
        - ``'extrinsics'``: (3, 4) float32 world-to-camera matrix ``[R|t]``
        - ``'intrinsics'``: (3, 3) float32 camera matrix K
        - ``'distortion'``: (8,) float32 distortion coefficients
        """
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        def _parse(tag: str) -> np.ndarray | None:
            node = root.find(tag)
            if node is None:
                return None
            rows = int(node.findtext('rows', default='1'))
            cols = int(node.findtext('cols', default='1'))
            data = list(map(float, node.findtext('data', default='').split()))
            return np.array(data, dtype=np.float32).reshape(rows, cols)

        dist = _parse('Distortion')
        return {
            'extrinsics': _parse('CameraMatrix'),   # (3, 4)
            'intrinsics': _parse('Intrinsics'),     # (3, 3)
            'distortion': dist.reshape(-1) if dist is not None else None,  # (8,)
        }

    def load_body_data(self, **kwargs: Any) -> None:
        # Ghost pipeline output -- same structure as EgoExo.
        # Identify the per-video folders of predictions
        self._cam_dirs = sorted(
            d
            for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()
        )
        # Loop on videos
        for cam_dir in self._cam_dirs:
            body_dir = cam_dir / "body_data"
            persons: dict[int, dict[str, np.ndarray]] = {}
            # Loop on people
            for npz_path in sorted(body_dir.glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                data = dict(np.load(str(npz_path), allow_pickle=False))
                persons[pid] = {
                    k: data[k] for k in self._NPZ_FIELDS if k in data
                }
            self._raw.append(persons)
            logger.debug(f"  {cam_dir.name}: loaded {len(persons)} person(s)")

    def load_cameras(self, **kwargs: Any) -> None:
        """Parse OpenCV XML grounf truth calibration files from scan_calibration/.

        Stores raw calibration dicts in ``self._cameras`` (available to
        ``convert_camera`` for predicted-camera conversion if needed) and
        also pre-converts the GT extrinsics to ``[qx, qy, qz, qw, tx, ty, tz]``
        vectors in ``self._gt_camera_vecs`` for use in targets.
        """
        rich_data_root = Path(kwargs.get("rich_data_root", ""))
        scene_name = self.scene_dir.name
        stem = self._scene_stem(scene_name)
        calib_dir = rich_data_root / "scan_calibration" / stem / "calibration"

        self._cameras = []
        self._gt_camera_vecs: list[np.ndarray] = []  # (8,) per camera
        for i, cam_dir in enumerate(self._cam_dirs):
            xml_path = calib_dir / f"{i:03d}.xml"
            if xml_path.exists():
                calib = self._parse_calib_xml(xml_path)
                self._cameras.append(calib)
                # Pre-compute GT camera vector [qx, qy, qz, qw, tx, ty, tz, focal]
                # from the static (per-camera) extrinsic matrix.
                ext = calib.get("extrinsics")  # (3, 4)
                if ext is not None:
                    from scipy.spatial.transform import Rotation as R
                    q = R.from_matrix(ext[:3, :3]).as_quat()  # [qx,qy,qz,qw]
                    vec = np.zeros(8, dtype=np.float32)
                    vec[:4] = q
                    vec[4:7] = ext[:3, 3]
                    intr = calib.get("intrinsics")
                    vec[7] = float(intr[0, 0]) if intr is not None else 1000.0
                else:
                    vec = np.array([0., 0., 0., 1., 0., 0., 0., 1000.], dtype=np.float32)
                self._gt_camera_vecs.append(vec)
                logger.debug(
                    f"  {cam_dir.name}: loaded calibration from {xml_path.name}"
                )
            else:
                logger.warning(
                    f"No calibration found for {cam_dir.name} "
                    f"(expected {xml_path})"
                )
                self._cameras.append({})
                self._gt_camera_vecs.append(
                    np.array([0., 0., 0., 1., 0., 0., 0., 1000.], dtype=np.float32)
                )

    def load_ground_truth(self, **kwargs: Any) -> None:
        """Load per-frame SMPL-X GT params from train_body/.

        Each ``{frame:05d}/{person:03d}.pkl`` file contains one frame's
        SMPL-X parameters for one person (batch dim = 1, squeezed out).
        GT is world-space so it is replicated across all camera slots.
        """
        rich_data_root = Path(kwargs.get("rich_data_root", ""))
        scene_name = self.scene_dir.name
        gt_root = rich_data_root / "train_body" / scene_name

        # gt_accum[person_id][field] = list of per-frame arrays
        gt_accum: dict[int, dict[str, list]] = {}

        if gt_root.is_dir():
            for frame_dir in sorted(gt_root.iterdir()):
                if not frame_dir.is_dir():
                    continue
                try:
                    frame_idx = int(frame_dir.name)
                except ValueError:
                    continue
                for pkl_path in sorted(frame_dir.glob("*.pkl")):
                    person_id = int(pkl_path.stem)   # '001' -> 1
                    with open(pkl_path, "rb") as f:
                        data: dict[str, np.ndarray] = pickle.load(f)
                    if person_id not in gt_accum:
                        gt_accum[person_id] = {
                            k: [] for k in data
                        }
                        gt_accum[person_id]["frame_indices"] = []
                    gt_accum[person_id]["frame_indices"].append(frame_idx)
                    for k, v in data.items():
                        # Remove batch dim added by SMPL-X: (1, ...) -> (...)
                        gt_accum[person_id][k].append(
                            np.asarray(v, dtype=np.float32).squeeze(0)
                        )
        else:
            logger.warning(f"GT directory not found: {gt_root}")

        # Stack lists into arrays and build a frame-index LUT for fast lookup.
        gt_final: dict[int, dict[str, np.ndarray]] = {}
        for pid, fields in gt_accum.items():
            gt_final[pid] = {k: np.stack(v) for k, v in fields.items()}

        # frame_lut[person_id][global_frame_int] = array_index
        self._gt_frame_lut: dict[int, dict[int, int]] = {
            pid: {
                int(fi): i
                for i, fi in enumerate(gt_final[pid]["frame_indices"])
            }
            for pid in gt_final
        }

        # GT is world-space -- replicate the same dict across all cameras.
        n_cams = max(len(self._cam_dirs), 1)
        self._gt = [gt_final] * n_cams

    # ------------------------------------------------------------------
    # World-frame transform
    # ------------------------------------------------------------------

    def _transform_to_world_frame(self) -> None:
        """Re-express all body parameters in camera-0's coordinate frame.

        Camera-0 is treated as the world origin.  For each camera i the
        extrinsic ``[R_i | t_i]`` maps true-world → cam-i.  The transform
        to cam-0 (= world) is:

            x_world = R_0 @ R_i^T @ (x_cami - t_i) + t_0

        For rotations (axis-angle):

            R_body_world = R_0 @ R_i^T @ R_body_cami

        GT (RICH) is in true world coordinates so it maps as:

            x_world = R_0 @ x_true_world + t_0

        Fields transformed in ``self._raw``:
            ``smplx_global_orient``, ``smplx_transl``, ``pred_keypoints_3d``

        Fields transformed in ``self._gt``:
            ``global_orient``, ``transl``
        """
        from scipy.spatial.transform import Rotation as SciR

        if not self._cameras:
            return

        ext0 = self._cameras[0].get("extrinsics")   # (3, 4) world→cam0
        if ext0 is None:
            return
        R0 = ext0[:3, :3].astype(np.float64)        # (3, 3)
        t0 = ext0[:3, 3].astype(np.float64)         # (3,)

        # ---- Predictions: cam-i → cam-0 --------------------------------
        for cam_idx, (cam_calib, persons) in enumerate(
            zip(self._cameras, self._raw)
        ):
            ext_i = cam_calib.get("extrinsics")
            if ext_i is None:
                continue
            R_i = ext_i[:3, :3].astype(np.float64)
            t_i = ext_i[:3, 3].astype(np.float64)

            # Affine: x_world = R_i2w @ x_cami + d_i
            R_i2w = R0 @ R_i.T             # (3, 3)
            d_i = t0 - R_i2w @ t_i        # (3,)

            for pdata in persons.values():
                # smplx_global_orient: (N, 3) axis-angle
                go = pdata.get("smplx_global_orient")
                if go is not None:
                    R_body = SciR.from_rotvec(
                        go.astype(np.float64)
                    ).as_matrix()                              # (N, 3, 3)
                    R_body_w = R_i2w[None] @ R_body            # (N, 3, 3)
                    pdata["smplx_global_orient"] = (
                        SciR.from_matrix(R_body_w)
                        .as_rotvec()
                        .astype(np.float32)
                    )

                # smplx_transl: (N, 3)
                tr = pdata.get("smplx_transl")
                if tr is not None:
                    pdata["smplx_transl"] = (
                        tr.astype(np.float64) @ R_i2w.T + d_i
                    ).astype(np.float32)

                # pred_keypoints_3d: (N, J, 3)
                # Note: assumes keypoints are in absolute camera space
                # (not root-relative).  Adjust if SAM3D outputs root-relative.
                kp = pdata.get("pred_keypoints_3d")
                if kp is not None:
                    pdata["pred_keypoints_3d"] = (
                        kp.astype(np.float64) @ R_i2w.T + d_i
                    ).astype(np.float32)

        # ---- GT: true world → cam-0 ------------------------------------
        # self._gt is a list of references to the same dict object, so
        # transforming self._gt[0] covers all camera slots at once.
        if not self._gt:
            return
        gt_dict = self._gt[0]   # {pid: {field: array}}

        for gt in gt_dict.values():
            # global_orient: (N, 3) axis-angle in true world space
            go = gt.get("global_orient")
            if go is not None:
                R_body = SciR.from_rotvec(
                    go.astype(np.float64).reshape(-1, 3)
                ).as_matrix()                                  # (N, 3, 3)
                R_body_w = R0[None] @ R_body                   # (N, 3, 3)
                gt["global_orient"] = (
                    SciR.from_matrix(R_body_w)
                    .as_rotvec()
                    .astype(np.float32)
                    .reshape(go.shape)
                )

            # transl: (N, 3) in true world space
            tr = gt.get("transl")
            if tr is not None:
                gt["transl"] = (
                    tr.astype(np.float64) @ R0.T + t0
                ).astype(np.float32)

    # ------------------------------------------------------------------
    # Conversion overrides
    # ------------------------------------------------------------------
    # NOTE: convert_camera is NOT overridden here.
    # inputs["camera"] is always the *predicted* camera from SAM3D
    # (pred_cam_t + global_rot from the ghost pipeline npz).  The GT
    # calibrated camera goes into targets["camera"] via build_gt_targets.

    def build_gt_targets(
        self,
        cam_idx: int,
        person_slot: int,
        pid: int,
        local_idx: int,
        t: int,
        pose_out: np.ndarray,
        shape_out: np.ndarray,
        camera_out: np.ndarray,
        kp3d_out: np.ndarray,
    ) -> None:
        """Fill GT targets from RICH SMPL-X annotations.

        Resolves the global frame number from the body_data ``frame_indices``
        array, then looks it up in the GT frame LUT.  If no GT exists for
        this frame / person slot, the arrays are left as zeros.
        """
        if cam_idx >= len(self._gt) or pid not in self._gt[cam_idx]:
            return

        # Resolve global frame number from body_data.
        raw_fi = (
            self._raw[cam_idx].get(pid, {}).get("frame_indices")
            if cam_idx < len(self._raw)
            else None
        )
        if raw_fi is None or local_idx >= len(raw_fi):
            return
        gf_int = int(raw_fi[local_idx])

        # Look up GT array index for this global frame.
        gt_idx = self._gt_frame_lut.get(pid, {}).get(gf_int)
        if gt_idx is None:
            return   # No GT annotation for this frame

        gt = self._gt[cam_idx][pid]

        # GT camera: static per-camera extrinsic, same for every frame.
        if cam_idx < len(self._gt_camera_vecs):
            camera_out[:] = self._gt_camera_vecs[cam_idx]

        # Shape: betas (10,)
        if "betas" in gt:
            n = min(shape_out.shape[0], gt["betas"].shape[-1])
            shape_out[:n] = gt["betas"][gt_idx, :n]

        # Pose: build full 55-joint axis-angle matching the same layout used
        # by EgoExoFusionDataset.convert_pose:
        #   [global(1), body(21), lhand(15), rhand(15), jaw+eyes(3)]
        if "body_pose" in gt and "global_orient" in gt:
            from scipy.spatial.transform import Rotation as SciR
            go = gt["global_orient"][gt_idx].reshape(1, 3)   # (1, 3)
            bp = gt["body_pose"][gt_idx].reshape(-1, 3)       # (21, 3)
            parts = [go, bp]
            for key in ("left_hand_pose", "right_hand_pose"):
                if key in gt:
                    parts.append(gt[key][gt_idx].reshape(-1, 3))  # (15, 3)
            for key in ("jaw_pose", "leye_pose", "reye_pose"):
                if key in gt:
                    parts.append(gt[key][gt_idx].reshape(-1, 3))   # (1, 3)
            aa = np.concatenate(parts, axis=0)  # up to (55, 3)
            J_pose = pose_out.shape[0]          # 55
            if aa.shape[0] < J_pose:
                aa = np.concatenate(
                    [aa, np.zeros((J_pose - aa.shape[0], 3), dtype=np.float32)],
                    axis=0,
                )
            mats = SciR.from_rotvec(aa).as_matrix()      # (55, 3, 3)
            sixd = np.concatenate(
                [mats[:, :, 0], mats[:, :, 1]], axis=1
            )                                             # (55, 6)
            pose_out[:J_pose] = sixd[:J_pose]


# ======================================================================
# Helper
# ======================================================================

def build_fusion_dataloader(
    dataset_type: str,
    scene_dir: str | Path,
    window_size: int = 128,
    window_stride: int = 64,
    num_joints: int = 55,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Convenience builder that picks the right subclass by name.

    Parameters
    ----------
    dataset_type : str
        One of ``"egoexo"`` or ``"rich"``.
    dataset_kwargs
        Extra kwargs forwarded to the subclass constructor.
        RICH requires ``rich_data_root`` (path to the RICH split root).
    """
    _REGISTRY: dict[str, type[FusionDataset]] = {
        "egoexo": EgoExoFusionDataset,
        "rich": RICHFusionDataset,
    }
    cls = _REGISTRY.get(dataset_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    ds = cls(
        scene_dir=scene_dir,
        window_size=window_size,
        window_stride=window_stride,
        num_joints=num_joints,
        **dataset_kwargs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
