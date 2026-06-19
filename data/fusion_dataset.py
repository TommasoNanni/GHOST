"""Fusion dataset for the SST model.

 Provides a base class :class:`FusionDatapoint` that handles windowing, lookup
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
        camera     = Tensor[K, 8],
        joint_mask = Tensor[T, K, P, J],
    )
    targets = dict(
        pose       = Tensor[T, K, P, J, 6],
        shape      = Tensor[T, K, P, 10],
        camera     = Tensor[K, 8],
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
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utilities.smplx_utilities import get_smplx_joints

logger = logging.getLogger(__name__)


class FusionDatapoint(Dataset, ABC):
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
        "pred_joint_confidence",
        "bbox",
    )

    def __init__(
        self,
        scene_dir: str | Path,
        num_joints: int = 55,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_joints = num_joints
        # When False, GT matching is still performed (for evaluation overlays)
        # but does NOT restrict the batch to GT-matched persons: all foreground
        # ghost pids are kept (e.g. to visualize background people).
        self._restrict_to_gt_persons: bool = bool(
            kwargs.pop("restrict_to_gt_persons", True)
        )

        # _cam_dirs[cam_idx] = Path to each camera directory
        self._cam_dirs: list[Path] = []
        # _raw[cam_idx][person_id] = {field_name: np.ndarray, ...}, parameters per person per camera (in camera frame as of now)
        self._raw: list[dict[int, dict[str, np.ndarray]]] = []
        # _cameras[cam_idx] = {arbitrary calibration data}
        self._cameras: list[dict[str, Any]] = []
        # _gt[cam_idx][person_id] = {arbitrary GT data}
        self._gt: list[dict[int, dict[str, np.ndarray]]] = []
        # Ghost person IDs that were matched to a RICH GT person.
        # None = inference mode (no GT loaded) → all ghost persons are included.
        # Set[int] = training mode → only these PIDs enter the batch; background
        # people (no GT annotation) are silently discarded.
        self._gt_matched_ghost_pids: set[int] | None = None

        self.load_body_data(**kwargs)       # fills the _raw and the _cam_dirs fields
        self._apply_temporal_offsets()      # shift frame_indices to global timeline if offsets available
        self.load_cameras(**kwargs)         # fills the _cameras with predicted cameras
        self.load_ground_truth(**kwargs)    # fills the _gt with GT data
        self._transform_to_world_frame()    # Projects the 3D quantities in the "world" (=cam0) system
        self._match_persons_to_gt()         # Remaps _gt keys from dataset pids to ghost pids

        self.num_cameras: int = len(self._cam_dirs)
        if self.num_cameras == 0:
            raise FileNotFoundError(
                f"No camera directories found in {self.scene_dir}"
            )

        # Person slots are global: slot p maps to the same ghost pid in every
        # camera, so max_persons = number of distinct (kept) pids scene-wide.
        self.max_persons: int = (
            len({
                pid for cam in self._raw for pid in cam
                if self._gt_matched_ghost_pids is None or pid in self._gt_matched_ghost_pids
            }) if self._raw else 0
        )

        self._frame_start: int = 0
        self._frame_end: int = 0
        self._compute_frame_range()

        self._lookup: list[list[dict[int, int]]] = []
        self._pid_order: list[list[int]] = []
        self._build_lookup()
        self._img_size: tuple[int, int] | None = None

        logger.info(
            f"{type(self).__name__}: {self.num_cameras} cameras, "
            f"{self.max_persons} max persons, "
            f"frames [{self._frame_start}, {self._frame_end}), "
            f"T={self._frame_end - self._frame_start}"
        )

    @property
    def num_frames(self) -> int:
        return self._frame_end - self._frame_start

    @property
    def has_gt(self) -> bool:
        return bool(self._gt) and any(len(g) > 0 for g in self._gt)

    @property
    def img_size(self) -> tuple[int, int]:
        """(H, W) of the source images, read lazily from the first mask frame."""
        if self._img_size is None:
            for cam_dir in self._cam_dirs:
                mask_path = cam_dir / "mask_data.npz"
                if mask_path.exists():
                    data = np.load(str(mask_path))
                    first_key = next(iter(data))
                    h, w = data[first_key].shape[:2]
                    self._img_size = (h, w)
                    break
            if self._img_size is None:
                self._img_size = (1080, 1920)  # safe fallback
        return self._img_size

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

    def _match_persons_to_gt(self) -> None:
        """Remap ``self._gt`` keys from dataset person IDs to ghost person IDs.

        Called after ``_transform_to_world_frame()`` so both predictions and
        GT are in the same world coordinate frame.  Default is a no-op.
        Override in subclasses that have GT with independent person IDs.
        Automatically skipped when ``self._gt`` is empty (inference mode).
        """

    # Hooks to be overridden

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
        body_transl_cam: np.ndarray,
        focal_length: float,
        global_rot: np.ndarray,
        cam_calib: dict[str, Any] | None = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Convert raw camera params to ``[8]`` = ``[quat(4), trans(3), focal_raw(1)]``.

        ``cam_calib`` is ``self._cameras[cam_idx]`` -- available for
        subclasses that have proper extrinsics.
        Default: identity rotation + body_transl_cam (body root in cam frame) + softplus(focal_raw)=focal_length.
        """
        cam = np.zeros(8, dtype=np.float32)
        cam[0] = 1.0  # pytorch3d [w,x,y,z]: w=1 -> identity rotation
        cam[4:7] = body_transl_cam
        # Store softplus_inv(focal) so that extract_cameras can recover the
        # true focal length by applying softplus.  For f >> 1, softplus_inv(f) ≈ f.
        _f = float(focal_length)
        cam[7] = _f if _f > 20.0 else float(np.log(np.expm1(_f) + 1e-6))
        return cam

    @staticmethod
    def _get_est_ext(cam_calib: dict | None) -> np.ndarray | None:
        """Return estimated extrinsics as (3, 4), or None if unavailable.

        Camera 0 (reference) returns identity. Other cameras use the
        single Kabsch-estimated (R, t) stored by the alignment-loading step.
        """
        if cam_calib is None:
            return None
        if cam_calib.get("est_ext_is_reference"):
            ext = np.zeros((3, 4), dtype=np.float64)
            ext[:3, :3] = np.eye(3)
            return ext
        if "est_ext_R" not in cam_calib:
            return None
        ext = np.zeros((3, 4), dtype=np.float64)
        ext[:3, :3] = cam_calib["est_ext_R"]
        ext[:3, 3] = cam_calib["est_ext_t"]
        return ext

    def _fill_static_gt_cameras(self, gt_camera: np.ndarray) -> None:
        """Fill gt_camera with static per-camera extrinsics across all T frames.

        Called once after the per-frame data loop. Default is a no-op; subclasses
        with fixed calibration (e.g. RICH) override this to stamp their calibration
        vectors for every frame so that cameras with no detected person in a given
        frame still carry a valid quaternion (norm > 0) in targets["camera"].
        """

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
        transl_out: np.ndarray,
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
        transl_out : writable view into targets["trans"][t, cam_idx, person_slot]
        """

    # Internal book-keeping (not overridden)

    def _apply_temporal_offsets(self) -> None:
        """Shift each camera's frame_indices by its temporal offset.

        Reads temporal_offsets.json from scene_dir (written by the synchronizer
        for genuinely async inputs). Each camera's frame_indices are shifted so
        all cameras share a common physical timeline — camera B with offset 10
        goes from [0..99] to [10..109], so _build_sample correctly aligns it
        with camera A's frames. No-op if the file is missing (synchronized data).
        """
        offsets_path = self.scene_dir / "temporal_offsets.json"
        if not offsets_path.exists():
            return
        with open(offsets_path) as f:
            offsets: dict[str, int] = json.load(f)
        for cam_idx, cam_dir in enumerate(self._cam_dirs):
            offset = offsets.get(cam_dir.name, 0)
            if offset == 0:
                continue
            for pdata in self._raw[cam_idx].values():
                pdata["frame_indices"] = pdata["frame_indices"] + offset

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
        """Pre-build per-camera per-person frame -> local-npz-index maps.

        Person slots are GLOBAL: slot p corresponds to the same ghost pid in
        every camera; cameras that never see that pid get an empty lookup
        (person_mask stays False there).
        """
        global_pids = sorted({
            pid for cam in self._raw for pid in cam
            if self._gt_matched_ghost_pids is None or pid in self._gt_matched_ghost_pids
        })
        for cam_idx, cam_persons in enumerate(self._raw):
            self._pid_order.append(global_pids)
            cam_lookups: list[dict[int, int]] = []
            for pid in global_pids:
                if pid in cam_persons:
                    fi = cam_persons[pid]["frame_indices"]
                    cam_lookups.append({int(f): i for i, f in enumerate(fi)})
                else:
                    cam_lookups.append({})
            self._lookup.append(cam_lookups)

    def _build_sample(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        T = self._frame_end - self._frame_start
        K = self.num_cameras
        P = self.max_persons
        J = self.num_joints
        frames = np.arange(self._frame_start, self._frame_end)

        pose = np.zeros((T, K, P, J, 6), dtype=np.float32)
        pose_cam = np.zeros((T, K, P, J, 6), dtype=np.float32)
        translation = np.zeros((T, K, P, 3), dtype=np.float32)
        body_transl_cam = np.zeros((T, K, P, 3), dtype=np.float32)  # body root in each camera's local frame
        shape = np.zeros((T, K, P, 10), dtype=np.float32)
        camera = np.zeros((K, 8), dtype=np.float32)
        camera[:, 0] = 1.0  # identity quaternion (w=1); prevents NaN in model forward for undetected cameras
        joint_mask = np.zeros((T, K, P, J), dtype=np.float32)
        # Binary presence mask: 1 when person p was detected in camera k at frame t.
        # Stored as bool to save memory; shape (T, K, P).
        person_mask = np.zeros((T, K, P), dtype=bool)
        kp3d  = np.zeros((T, K, P, 70, 3), dtype=np.float32)
        kp2d  = np.zeros((T, K, P, 70, 2), dtype=np.float32)

        # Target arrays (may be overwritten by build_gt_targets).
        gt_body_pose = np.zeros_like(pose)                              # ground-truth body pose (T, K, P, J, 6)
        gt_body_shape = np.zeros_like(shape)                            # ground-truth SMPL-X betas (T, K, P, 10)
        gt_camera = np.zeros((K, 8), dtype=np.float32)                 # (K, 8) — one extrinsic per camera, static
        # gt_camera stays at zero for cameras not filled by build_gt_targets.
        # Zero quaternion (norm=0) is the "no GT" sentinel used by cam_valid
        # checks (norm > 0.5) in all losses to skip unsupervised cameras.
        gt_kp3d = np.zeros_like(kp3d)
        gt_body_transl_world = np.zeros((T, K, P, 3), dtype=np.float32)  # ground-truth body root in world (cam-0) frame

        for k in range(K):
            cam_persons = self._raw[k]
            pids = self._pid_order[k]
            cam_calib = self._cameras[k] if k < len(self._cameras) else None
            cam_camera_filled = False  # filled once per camera (static extrinsic)

            for p_slot, pid in enumerate(pids):
                if p_slot >= P:
                    break
                if pid not in cam_persons:
                    continue   # this camera never sees this person
                pdata = cam_persons[pid]
                lut = self._lookup[k][p_slot]

                for t, gf in enumerate(frames):
                    gf_int = int(gf)
                    if gf_int not in lut:
                        continue
                    li = lut[gf_int]
                    person_mask[t, k, p_slot] = True

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
                        # Camera-frame pose: same body/hand joints, but global
                        # orient in the camera's own frame (before world transform).
                        gr_cam_arr = pdata.get("smplx_global_orient_cam")
                        gr_cam_li = gr_cam_arr[li] if gr_cam_arr is not None else gr[li]
                        pose_cam[t, k, p_slot] = self.convert_pose(
                            bpp[li], hpp_li, gr_cam_li
                        )

                    # --- Translation ---
                    tr = pdata.get("smplx_transl")
                    if tr is not None:
                        translation[t, k, p_slot] = tr[li]
                    pct = pdata.get("pred_cam_t")
                    if pct is not None:
                        body_transl_cam[t, k, p_slot] = pct[li]

                    # --- Shape ---
                    sp = pdata.get("smplx_betas")
                    if sp is not None:
                        n_betas = min(10, sp.shape[1] if sp.ndim > 1 else sp.shape[0])
                        shape[t, k, p_slot, :n_betas] = (
                            sp[li, :n_betas] if sp.ndim > 1 else sp[:n_betas]
                        )

                    # --- Joint mask (per-joint confidence from occlusion estimation) ---
                    jc = pdata.get("pred_joint_confidence")
                    if jc is not None:
                        jc_frame = jc[li] if jc.ndim > 1 else jc
                        joint_mask[t, k, p_slot, :min(J, len(jc_frame))] = jc_frame[:J]
                    else:
                        joint_mask[t, k, p_slot, :] = 1.0

                    # --- Camera (static: fill once per camera from first detection) ---
                    if not cam_camera_filled:
                        pct = pdata.get("pred_cam_t")  # body root in camera frame (NPZ key)
                        fl = pdata.get("focal_length")
                        if pct is not None and gr is not None:
                            camera[k] = self.convert_camera(
                                pct[li],
                                float(fl[li]) if fl is not None else 1000.0,
                                gr[li],
                                cam_calib=cam_calib,
                                frame_index=gf_int,
                            )
                            cam_camera_filled = True

                    # --- 3D keypoints ---
                    kp = pdata.get("pred_keypoints_3d")
                    if kp is not None:
                        kp3d[t, k, p_slot] = kp[li]

                    # --- 2D keypoints (pixel space, used by EpipolarLoss) ---
                    kp2 = pdata.get("pred_keypoints_2d")
                    if kp2 is not None:
                        kp2d[t, k, p_slot] = kp2[li]

                    # --- GT targets (subclass fills if available) ---
                    self.build_gt_targets(
                        cam_idx=k,
                        person_slot=p_slot,
                        pid=pid,
                        local_idx=li,
                        t=t,
                        pose_out=gt_body_pose[t, k, p_slot],
                        shape_out=gt_body_shape[t, k, p_slot],
                        camera_out=gt_camera[k],
                        kp3d_out=gt_kp3d[t, k, p_slot],
                        transl_out=gt_body_transl_world[t, k, p_slot],
                    )

        # Give subclasses a chance to fill static (per-camera, not per-frame) GT camera
        # vectors for all T frames.  Must happen before the self-supervised fallback below
        # so that cameras present in calibration but absent from person detections still
        # carry a valid quaternion (norm > 0).
        self._fill_static_gt_cameras(gt_camera)

        # Default self-supervised targets: mirror inputs where GT wasn't set.
        # gt_camera is intentionally not mirrored: _fill_static_gt_cameras stamped
        # the XML calibration just above, and losses gate on quaternion norm > 0.5.
        _has_gt = any(len(g) > 0 for g in self._gt) if self._gt else False
        if not _has_gt:
            gt_body_pose = pose.copy()
            gt_body_shape = shape.copy()
            gt_kp3d = kp3d.copy()

        inputs = {
            "pose": torch.from_numpy(pose_cam),                      # joint 0 (root orient) in camera k's own frame;
            "body_transl_cam_in": torch.from_numpy(body_transl_cam), # body_transl_cam_in: body root position in camera k's local frame (from pred_cam_t).
            "shape": torch.from_numpy(shape),
            "camera": torch.from_numpy(camera),
            "joint_mask": torch.from_numpy(joint_mask),
            "person_mask": torch.from_numpy(person_mask),            # (T, K, P) bool
            "kp2d": torch.from_numpy(kp2d),                         # (T, K, P, 70, 2) SAM3D 2D keypoints, pixel space
        }
        # gt_valid: True for frames that have real GT annotations.
        # Frames with all-zero gt_body_transl_world have no RICH annotation and must be
        # excluded from supervised losses (pose MSE, shape MSE, etc.).
        gt_body_transl_world_cam0 = gt_body_transl_world[:, 0]    # (T, P, 3) — cam-0 copy (world frame)
        gt_valid = ~np.all(                                        # (T, P) bool
            gt_body_transl_world_cam0.reshape(T, max(P, 1), 3) == 0, axis=-1
        )
        targets = {
            "pose": torch.from_numpy(gt_body_pose[:, 0]),           # [T, P, J, 6]  ground-truth body pose
            "shape": torch.from_numpy(gt_body_shape[:, 0]),         # [T, P, 10]    ground-truth SMPL-X betas
            "camera": torch.from_numpy(gt_camera),                  # [K, 8] — static extrinsic per camera
            "keypoints_3d": torch.from_numpy(gt_kp3d[:, 0]),        # [T, P, 70, 3]
            "trans": torch.from_numpy(gt_body_transl_world_cam0),   # [T, P, 3]     ground-truth body root in world frame
            "gt_valid": torch.from_numpy(gt_valid),                 # [T, P] bool
            "scene_name": self.scene_dir.name,                      # str — for logging
        }

        # Precompute GT joint positions once at data-loading time so that
        # JointPositionLoss does not need to run FK on the GT every training step.
        # Shape of gt_joints: (T, P, 55, 3) — body joints only, local frame (no root translation).
        try:
            with torch.no_grad():
                gt_joints = get_smplx_joints(
                    targets["pose"].unsqueeze(0).float(),    # (1, T, P, J, 6)
                    targets["shape"].unsqueeze(0).float(),   # (1, T, P, 10)
                ).squeeze(0)                                  # (T, P, Jout, 3)
            targets["gt_joints"] = gt_joints[..., :55, :]   # (T, P, 55, 3)
        except Exception as e:
            logger.debug("Skipping gt_joints precomputation: %s", e)

        return inputs, targets

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cameras={self.num_cameras}, "
            f"max_persons={self.max_persons}, "
            f"frames=[{self._frame_start}, {self._frame_end}), "
            f"T={self._frame_end - self._frame_start})"
        )


# ======================================================================
# EgoExo subclass (ghost pipeline output, no GT)
# ======================================================================

# ======================================================================
# EgoExo4D GT-aware subclass
# ======================================================================

_EGOEXO4D_COCO17 = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
]


class EgoExo4DGTFusionDatapoint(FusionDatapoint):
    """GT-aware EgoExo4D fusion datapoint.

    Data sources (produced by ``utilities/download_egoexo4d_gt.py``)::

        gt_dir/<take_name>/keypoints_gt.json
            {frame_idx_str: {joint_name: {x, y, z}, ...}, ...}
            One annotated person (the ego/aria wearer); keypoints in world frame.

        gt_dir/<take_name>/gopro_calibs.csv
            Camera calibration.  Each row covers a frame range.  Relevant cols:
              cam_uid, tx_world_cam, ty_world_cam, tz_world_cam,
              qx_world_cam, qy_world_cam, qz_world_cam, qw_world_cam,
              intrinsics_0 (fx), intrinsics_1 (fy), intrinsics_2 (cx), intrinsics_3 (cy)
            Convention: T_world_cam (camera-to-world); we invert to get extrinsics.

    Ghost pipeline output (scene_dir)::

        scene_dir/
            cam01/body_data/person_*.npz
            cam02/body_data/person_*.npz
            ...

    Constructor kwargs
    ------------------
    gt_dir : str | Path
        Base directory whose sub-folders are take names.
    take_name : str, optional
        Override the take name (default: ``scene_dir.name``).
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
                persons[pid] = {k: data[k] for k in self._NPZ_FIELDS if k in data}
            self._raw.append(persons)
            logger.debug(f"  {cam_dir.name}: loaded {len(persons)} person(s)")

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        J = self.num_joints
        out = np.zeros((J, 6), dtype=np.float32)
        if body_pose_params is None or global_rot is None:
            return out
        try:
            from scipy.spatial.transform import Rotation as SciR
        except Exception:
            return out
        go = np.asarray(global_rot, dtype=np.float32).reshape(1, 3)
        bp = np.asarray(body_pose_params, dtype=np.float32).reshape(-1, 3)
        parts = [go, bp]
        if hand_pose_params is not None:
            parts.append(np.asarray(hand_pose_params, dtype=np.float32).reshape(-1, 3))
        aa = np.concatenate(parts, axis=0)
        if aa.shape[0] < J:
            aa = np.concatenate([aa, np.zeros((J - aa.shape[0], 3), dtype=np.float32)], axis=0)
        try:
            mats = SciR.from_rotvec(aa).as_matrix()
        except Exception:
            return out
        sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
        out[:J] = sixd[:J]
        return out

    def load_cameras(self, **kwargs: Any) -> None:
        """Parse ``gopro_calibs.csv`` → per-camera extrinsics + K.

        Builds ``self._cameras`` (list of calibration dicts, one per cam_dir)
        and ``self._gt_camera_vecs`` (8-vectors relative to cam-0).
        """
        import csv
        from scipy.spatial.transform import Rotation as SciR

        gt_dir = Path(kwargs.get("gt_dir", ""))
        take_name = kwargs.get("take_name", self.scene_dir.name)
        calib_csv = gt_dir / take_name / "gopro_calibs.csv"

        # Parse CSV into per-cam-uid list of calibration rows.
        cam_rows: dict[str, list[dict]] = {}
        if calib_csv.exists():
            with open(calib_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    uid = row["cam_uid"].strip()
                    cam_rows.setdefault(uid, []).append(row)
        else:
            logger.warning(f"gopro_calibs.csv not found at {calib_csv}")

        def _best_row(rows: list[dict], frame_idx: int) -> dict:
            """Pick the calibration row whose time range covers frame_idx."""
            for r in rows:
                try:
                    s = int(r.get("start_frame_idx", 0) or 0)
                    e = int(r.get("end_frame_idx", 10**9) or 10**9)
                    if s <= frame_idx <= e:
                        return r
                except (ValueError, KeyError):
                    pass
            return rows[0]  # fall back to first row

        def _parse_extrinsics(row: dict) -> tuple[np.ndarray, np.ndarray]:
            """Camera-to-world T_world_cam → world-to-cam extrinsics (4×4)."""
            tx = float(row["tx_world_cam"])
            ty = float(row["ty_world_cam"])
            tz = float(row["tz_world_cam"])
            qx = float(row["qx_world_cam"])
            qy = float(row["qy_world_cam"])
            qz = float(row["qz_world_cam"])
            qw = float(row["qw_world_cam"])
            t_c2w = np.array([tx, ty, tz], dtype=np.float64)
            R_c2w = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            ext = np.eye(4, dtype=np.float64)
            ext[:3, :3] = R_w2c
            ext[:3, 3] = t_w2c
            return ext, R_c2w

        def _parse_intrinsics(row: dict) -> np.ndarray:
            fx = float(row["intrinsics_0"])
            fy = float(row["intrinsics_1"])
            cx = float(row["intrinsics_2"])
            cy = float(row["intrinsics_3"])
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Derive the target frame index from the GT keypoints file (frame key).
        # Used only for calibration row selection; falls back to 0.
        gt_dir_path = gt_dir / take_name
        gt_frame_idx = 0
        kp_file = gt_dir_path / "keypoints_gt.json"
        if kp_file.exists():
            with open(kp_file) as fh:
                kp_data = json.load(fh)
            if kp_data:
                gt_frame_idx = int(next(iter(kp_data)))

        # Match each ghost cam_dir to a CSV cam_uid (strip non-alphanumeric to be safe).
        import re as _re
        self._cameras = []
        self._gt_camera_vecs: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, cam_dir in enumerate(self._cam_dirs):
            cam_name = cam_dir.name  # e.g. "cam01", "gp01"
            rows = cam_rows.get(cam_name)
            if rows is None:
                # Try stripping leading zeros / normalising
                for uid, r in cam_rows.items():
                    if _re.sub(r"[^a-z0-9]", "", uid) == _re.sub(r"[^a-z0-9]", "", cam_name):
                        rows = r
                        break
            if rows is None:
                logger.warning(f"No calibration for {cam_name} — skipping camera")
                continue

            row = _best_row(rows, gt_frame_idx)
            ext, _ = _parse_extrinsics(row)
            K = _parse_intrinsics(row)
            calib = {"extrinsics": ext.astype(np.float32), "intrinsics": K}
            self._cameras.append(calib)
            valid_indices.append(i)

            vec = np.zeros(8, dtype=np.float32)
            q_xyzw = SciR.from_matrix(ext[:3, :3]).as_quat()
            vec[:4] = q_xyzw[[3, 0, 1, 2]]   # [qw, qx, qy, qz]
            vec[4:7] = ext[:3, 3].astype(np.float32)
            vec[7] = float(K[0, 0])
            self._gt_camera_vecs.append(vec)

        # Drop cam_dirs with no calibration match.
        if len(valid_indices) < len(self._cam_dirs):
            self._cam_dirs = [self._cam_dirs[i] for i in valid_indices]
            self._raw = [self._raw[i] for i in valid_indices]

        # Re-express camera vecs relative to cam-0.
        if self._cameras:
            ext0 = self._cameras[0]["extrinsics"].astype(np.float64)
            R0, t0 = ext0[:3, :3], ext0[:3, 3]
            self._gt_camera_vecs[0][:4] = np.array([1., 0., 0., 0.], dtype=np.float32)
            self._gt_camera_vecs[0][4:7] = np.zeros(3, dtype=np.float32)
            for i in range(1, len(self._cameras)):
                ext_i = self._cameras[i]["extrinsics"].astype(np.float64)
                R_i, t_i = ext_i[:3, :3], ext_i[:3, 3]
                R_rel = R_i @ R0.T
                t_rel = t_i - R_rel @ t0
                q_xyzw = SciR.from_matrix(R_rel).as_quat()
                self._gt_camera_vecs[i][:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
                self._gt_camera_vecs[i][4:7] = t_rel.astype(np.float32)

    def load_ground_truth(self, **kwargs: Any) -> None:
        """Load 17 COCO keypoints from ``keypoints_gt.json``.

        Stores the single annotated person (GT person ID 0) with
        ``keypoints_3d`` shape ``(1, 17, 3)`` in world frame.
        After ``_match_persons_to_gt`` the key will be remapped to the
        matching ghost pid.
        """
        gt_dir = Path(kwargs.get("gt_dir", ""))
        take_name = kwargs.get("take_name", self.scene_dir.name)
        kp_file = gt_dir / take_name / "keypoints_gt.json"

        self._gt = [{}]
        self._gt_frame_lut: dict[int, dict[int, int]] = {}
        self._gt_kp_world: np.ndarray | None = None  # (17, 3) world frame

        if not kp_file.exists():
            logger.warning(f"keypoints_gt.json not found at {kp_file}")
            return

        with open(kp_file) as fh:
            raw = json.load(fh)

        if not raw:
            return

        frame_key, ann = next(iter(raw.items()))
        frame_idx = int(frame_key)

        kp = np.zeros((17, 3), dtype=np.float32)
        for j, name in enumerate(_EGOEXO4D_COCO17):
            if name in ann:
                entry = ann[name]
                kp[j] = [entry["x"], entry["y"], entry["z"]]

        # Store raw world-frame keypoints for use in _transform_to_world_frame.
        self._gt_kp_world = kp

        # GT person id = 0 (single ego person).
        self._gt = [
            {0: {"frame_indices": np.array([frame_idx], dtype=np.int64),
                 "keypoints_3d": kp[np.newaxis].copy()}}  # (1, 17, 3) world frame
        ]
        self._gt_frame_lut = {0: {frame_idx: 0}}

    def _transform_to_world_frame(self) -> None:
        """Convert GT keypoints from EgoExo4D world frame to cam-0 frame.

        Called by the base class after ``load_ground_truth`` and ``load_cameras``.
        Modifies ``self._gt[0][pid]["keypoints_3d"]`` in-place.
        """
        if not self._gt or not self._gt[0]:
            return
        if not self._cameras:
            return

        ext0 = self._cameras[0]["extrinsics"].astype(np.float64)
        R0, t0 = ext0[:3, :3], ext0[:3, 3]

        # x_cam0 = R0 @ x_world + t0
        for pid, pdata in self._gt[0].items():
            kp = pdata.get("keypoints_3d")
            if kp is None:
                continue
            # kp: (N, 17, 3) — transform each frame
            kp_world = kp.reshape(-1, 3).T.astype(np.float64)  # (3, N*17)
            kp_cam0 = (R0 @ kp_world + t0[:, None]).T.astype(np.float32)
            pdata["keypoints_3d"] = kp_cam0.reshape(kp.shape)

    def _match_persons_to_gt(self) -> None:
        """Match the single GT person to the closest ghost-detected person.

        Uses mean GT keypoint position (cam-0 frame) vs mean ghost position
        (``smplx_transl`` or ``pred_cam_t``, transformed to cam-0 frame).
        """
        from scipy.spatial.transform import Rotation as SciR

        if not self._gt or not self._gt[0] or not self._cameras:
            return

        ext0 = self._cameras[0]["extrinsics"].astype(np.float64)
        R0, t0 = ext0[:3, :3], ext0[:3, 3]

        # GT person center in cam-0 frame (mean of 17 joints).
        gt_data = self._gt[0].get(0)
        if gt_data is None:
            return
        kp3 = gt_data.get("keypoints_3d")
        if kp3 is None:
            return
        gt_center = kp3.reshape(-1, 3).mean(axis=0).astype(np.float64)  # (3,)

        # Collect ghost person mean positions in cam-0 frame.
        all_ghost_pids: set[int] = set()
        for cam_persons in self._raw:
            all_ghost_pids.update(cam_persons.keys())
        ghost_pids = sorted(all_ghost_pids)
        if not ghost_pids:
            return

        ghost_centers: dict[int, list[np.ndarray]] = {gpid: [] for gpid in ghost_pids}
        for cam_idx, cam_persons in enumerate(self._raw):
            if cam_idx >= len(self._cameras):
                continue
            ext_i = self._cameras[cam_idx]["extrinsics"].astype(np.float64)
            R_i, t_i = ext_i[:3, :3], ext_i[:3, 3]
            # cam-i → world → cam-0
            R_i2w = R0 @ R_i.T
            d_i = t0 - R_i2w @ t_i

            for gpid, pdata in cam_persons.items():
                tr = pdata.get("smplx_transl") if pdata.get("smplx_transl") is not None \
                    else pdata.get("pred_cam_t")
                if tr is None:
                    continue
                tr_cam0 = (tr.astype(np.float64) @ R_i2w.T + d_i)
                ghost_centers[gpid].append(tr_cam0.mean(axis=0))

        # Match closest.
        best_gpid: int | None = None
        best_dist = float("inf")
        for gpid in ghost_pids:
            pts = ghost_centers[gpid]
            if not pts:
                continue
            center = np.mean(pts, axis=0)
            dist = float(np.linalg.norm(gt_center - center))
            if dist < best_dist:
                best_dist = dist
                best_gpid = gpid

        if best_gpid is None:
            logger.warning(f"No ghost person found for GT in {self.scene_dir.name}")
            self._gt = [{}]
            return

        logger.info(
            f"{self.scene_dir.name}: GT ego person → ghost pid {best_gpid} "
            f"(dist={best_dist:.3f} m)"
        )
        # Remap GT person 0 → matched ghost pid.
        self._gt = [{best_gpid: self._gt[0][0]}]
        self._gt_frame_lut = {best_gpid: self._gt_frame_lut[0]}
        if self._restrict_to_gt_persons:
            self._gt_matched_ghost_pids = {best_gpid}

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
        transl_out: np.ndarray,
    ) -> None:
        """Fill GT camera token and 17 COCO keypoints.

        Camera token: static COLMAP/gopro_calibs extrinsic (same for all frames).
        Keypoints: 17 COCO joints written to ``kp3d_out[..., :17, :]``.
        Pose/shape/betas: not available from EgoExo4D — left at zero.
        """
        if cam_idx < len(self._gt_camera_vecs) and camera_out is not None:
            camera_out[:] = self._gt_camera_vecs[cam_idx]

        # Keypoints only stored in cam-0.
        if cam_idx != 0:
            return

        if cam_idx >= len(self._gt) or pid not in self._gt[cam_idx]:
            return

        # Map local_idx → global frame → GT array index.
        raw_fi = (
            self._raw[cam_idx].get(pid, {}).get("frame_indices")
            if cam_idx < len(self._raw)
            else None
        )
        if raw_fi is None or local_idx >= len(raw_fi):
            return

        global_frame = int(raw_fi[local_idx])
        gt_lut = self._gt_frame_lut.get(pid, {})
        gt_idx = gt_lut.get(global_frame)
        if gt_idx is None:
            return

        kp3 = self._gt[cam_idx][pid].get("keypoints_3d")
        if kp3 is not None and kp3d_out is not None:
            n = min(kp3d_out.shape[-2], kp3[gt_idx].shape[0])
            kp3d_out[..., :n, :] = kp3[gt_idx][:n]


# ======================================================================
# RICH subclass (RICH dataset with GT SMPL-X params + calibrated cameras)
# ======================================================================

class RICHFusionDatapoint(FusionDatapoint):
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

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        """Convert SMPL-X axis-angle params to [J, 6] 6D rotation."""
        J = self.num_joints
        out = np.zeros((J, 6), dtype=np.float32)
        if body_pose_params is None or global_rot is None:
            return out
        try:
            from scipy.spatial.transform import Rotation as SciR
        except Exception:
            return out

        go = np.asarray(global_rot, dtype=np.float32).reshape(1, 3)
        bp = np.asarray(body_pose_params, dtype=np.float32).reshape(-1, 3)
        parts = [go, bp]
        if hand_pose_params is not None:
            hp = np.asarray(hand_pose_params, dtype=np.float32).reshape(-1, 3)
            parts.append(hp)
        aa = np.concatenate(parts, axis=0)

        if aa.shape[0] < J:
            aa = np.concatenate(
                [aa, np.zeros((J - aa.shape[0], 3), dtype=np.float32)], axis=0
            )
        try:
            mats = SciR.from_rotvec(aa).as_matrix()
        except Exception:
            return out

        sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
        out[:J] = sixd[:J]
        return out

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
        exclude_cameras: list[str] = kwargs.get("exclude_cameras", [])
        self._cam_dirs = sorted(
            d
            for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()
            and d.name not in exclude_cameras
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

        # Keep only ghost persons that appear in at least (num_cameras - 1) views.
        # Foreground people are tracked consistently across nearly all cameras;
        # background/spurious tracks appear in fewer views and are dropped here
        # so they never pollute the GT matching step.
        if self._raw:
            num_cams = len(self._raw)
            min_cams = kwargs.get("min_foreground_cams")
            if min_cams is None:
                min_cams = max(1, num_cams - 1)
            cam_count: dict[int, int] = {}
            for cam in self._raw:
                for pid in cam:
                    cam_count[pid] = cam_count.get(pid, 0) + 1
            foreground_pids = {pid for pid, cnt in cam_count.items() if cnt >= min_cams}
            self._raw = [
                {pid: data for pid, data in cam.items() if pid in foreground_pids}
                for cam in self._raw
            ]
            logger.info(f"  foreground ghost pids (>={min_cams}/{num_cams} cams): {sorted(foreground_pids)}")

    def load_cameras(self, **kwargs: Any) -> None:
        """Parse OpenCV XML grounf truth calibration files from scan_calibration/.

        Stores raw calibration dicts in ``self._cameras`` (available to
        ``convert_camera`` for predicted-camera conversion if needed) and
        also pre-converts the GT extrinsics to ``[qx, qy, qz, qw, tx, ty, tz]``
        vectors in ``self._gt_camera_vecs`` for use in targets.
        """
        rich_gt_dir = Path(kwargs.get("rich_gt_dir", kwargs.get("rich_data_root", "")))
        scene_name = self.scene_dir.name
        stem = self._scene_stem(scene_name)
        calib_dir = rich_gt_dir / "scan_calibration" / stem / "calibration"

        self._cameras = []
        self._gt_camera_vecs: list[np.ndarray] = []  # (8,) per camera
        valid_cam_indices: list[int] = []             # indices into self._cam_dirs with valid GT calibration

        # STEP 1: Load GT cameras in rich world frame coordinates
        for i, cam_dir in enumerate(self._cam_dirs):
            # Extract camera number from directory name (e.g. "cam_10" → 10 → "010.xml")
            # rather than using the sequential index, which would give the wrong file
            # when camera numbers are non-contiguous (e.g. cam_00, cam_01, cam_10).
            _m = re.search(r"\d+", cam_dir.name)
            _cam_num = int(_m.group()) if _m else i
            xml_path = calib_dir / f"{_cam_num:03d}.xml"
            if xml_path.exists():
                calib = self._parse_calib_xml(xml_path)
                self._cameras.append(calib)
                valid_cam_indices.append(i)
                # Pre-compute GT camera vector [qx, qy, qz, qw, tx, ty, tz, focal]
                # from the static (per-camera) extrinsic matrix.
                ext = calib.get("extrinsics")  # (3, 4)
                if ext is not None:
                    from scipy.spatial.transform import Rotation as R
                    q_xyzw = R.from_matrix(ext[:3, :3]).as_quat()  # scipy [qx,qy,qz,qw]
                    vec = np.zeros(8, dtype=np.float32)
                    vec[:4] = q_xyzw[[3, 0, 1, 2]]  # pytorch3d [qw,qx,qy,qz]
                    vec[4:7] = ext[:3, 3]
                    intr = calib.get("intrinsics")
                    vec[7] = float(intr[0, 0]) if intr is not None else 1000.0
                else:
                    vec = np.array([1., 0., 0., 0., 0., 0., 0., 1000.], dtype=np.float32)
                self._gt_camera_vecs.append(vec)
                logger.debug(
                    f"  {cam_dir.name}: loaded calibration from {xml_path.name}"
                )
            else:
                logger.warning(
                    f"No GT calibration for {cam_dir.name} "
                    f"(expected {os.path.abspath(xml_path)}) — camera will be ignored"
                )

        # Drop cameras with no GT calibration from _cam_dirs and _raw so the
        # rest of the pipeline only sees cameras with valid extrinsics.
        if len(valid_cam_indices) < len(self._cam_dirs):
            self._cam_dirs = [self._cam_dirs[i] for i in valid_cam_indices]
            self._raw      = [self._raw[i]      for i in valid_cam_indices]

        # STEP 2: 
        # Re-express GT camera vectors relative to cam-0 so that they are in
        # the same frame as the body predictions (which are mapped to cam-0 in
        # _transform_to_world_frame).  The cam-0→cam-i relative transform is:
        #   R_rel = R_i @ R_0^T
        #   t_rel = t_i - R_rel @ t_0

        ext0 = self._cameras[0].get("extrinsics") if self._cameras else None
        if ext0 is not None:
            from scipy.spatial.transform import Rotation as SciR
            R0 = ext0[:3, :3].astype(np.float64)
            t0 = ext0[:3, 3].astype(np.float64)
            for i, calib in enumerate(self._cameras):
                if i == 0:
                    # Camera 0 is the world origin by definition — set exactly to identity.
                    self._gt_camera_vecs[0][:4] = np.array([1., 0., 0., 0.], dtype=np.float32)
                    self._gt_camera_vecs[0][4:7] = np.zeros(3, dtype=np.float32)
                    continue
                ext_i = calib.get("extrinsics")
                if ext_i is None:
                    continue
                R_i = ext_i[:3, :3].astype(np.float64)
                t_i = ext_i[:3, 3].astype(np.float64)
                R_rel = R_i @ R0.T
                t_rel = t_i - R_rel @ t0
                q_rel_xyzw = SciR.from_matrix(R_rel).as_quat()  # scipy [qx, qy, qz, qw]
                self._gt_camera_vecs[i][:4] = q_rel_xyzw[[3, 0, 1, 2]].astype(np.float32)  # → pytorch3d [qw,qx,qy,qz]
                self._gt_camera_vecs[i][4:7] = t_rel.astype(np.float32)

        # STEP 3:
        # Load estimated camera poses from vggt_cameras.npz (produced by the
        # VGGT preprocessing step).  Extrinsics are (T, K, 3, 4) world-to-camera
        # transforms; we average over time to get a single static estimate per
        # camera, then express each camera relative to cam-0 (same convention
        # as the GT camera vectors computed in STEP 2 above).
        vggt_path = self.scene_dir / "vggt_cameras.npz"
        if vggt_path.exists():
            vggt = np.load(str(vggt_path))
            vggt_exts   = vggt["extrinsics"].astype(np.float64)   # (T, K, 3, 4)
            vggt_names  = [n.decode() if isinstance(n, bytes) else n
                           for n in vggt["camera_names"]]          # list[str], len K

            # Build name→index map for VGGT cameras
            vggt_name_to_idx = {name: k for k, name in enumerate(vggt_names)}

            # Average extrinsics over frames (valid frames only where available)
            valid_mask = vggt.get("valid")  # (T, K) bool or None
            mean_exts: dict[int, np.ndarray] = {}
            for k in range(vggt_exts.shape[1]):
                if valid_mask is not None:
                    frames_k = vggt_exts[valid_mask[:, k], k]
                else:
                    frames_k = vggt_exts[:, k]
                if len(frames_k) == 0:
                    frames_k = vggt_exts[:, k]
                mean_exts[k] = frames_k.mean(axis=0)  # (3, 4)

            # Get cam-0 mean extrinsic for relative computation
            cam0_name = self._cam_dirs[0].name if self._cam_dirs else None
            cam0_vggt_idx = vggt_name_to_idx.get(cam0_name)
            if cam0_vggt_idx is None:
                logger.warning(
                    f"vggt_cameras.npz: cam0 '{cam0_name}' not found — "
                    "falling back to pred_cam_t"
                )
            else:
                ext0 = mean_exts[cam0_vggt_idx]
                R0, t0 = ext0[:3, :3], ext0[:3, 3]
                for i, cam_dir in enumerate(self._cam_dirs):
                    if i == 0:
                        self._cameras[i]["est_ext_is_reference"] = True
                        continue
                    vggt_idx = vggt_name_to_idx.get(cam_dir.name)
                    if vggt_idx is None:
                        logger.warning(
                            f"vggt_cameras.npz: '{cam_dir.name}' not found"
                        )
                        continue
                    ext_i = mean_exts[vggt_idx]
                    R_i, t_i = ext_i[:3, :3], ext_i[:3, 3]
                    R_rel = R_i @ R0.T
                    t_rel = t_i - R_rel @ t0
                    self._cameras[i]["est_ext_R"] = R_rel.astype(np.float32)
                    self._cameras[i]["est_ext_t"] = t_rel.astype(np.float32)
        else:
            logger.warning(
                f"vggt_cameras.npz not found in {self.scene_dir} — "
                "input camera tokens will fall back to pred_cam_t"
            )

    def load_ground_truth(self, **kwargs: Any) -> None:
        """Load per-frame SMPL-X GT params from train_body/.

        Each ``{frame:05d}/{person:03d}.pkl`` file contains one frame's
        SMPL-X parameters for one person (batch dim = 1, squeezed out).
        GT is world-space so it is replicated across all camera slots.
        """
        rich_gt_dir = Path(kwargs.get("rich_gt_dir", kwargs.get("rich_data_root", "")))
        scene_name = self.scene_dir.name
        body_split = kwargs.get("body_split", "train_body")
        gt_root = rich_gt_dir / body_split / scene_name

        # Load SMPL-X hand PCA basis so 12-dim GT hand coefficients can be
        # decoded to 45-dim axis-angle (15 joints × 3).
        from configuration import CONFIG as _CONFIG
        smplx_model_path = kwargs.get("smplx_model_path", _CONFIG.data.smplx_model_path)
        self._hand_pca: dict[str, np.ndarray] = {}
        try:
            with open(smplx_model_path, "rb") as _f:
                _smplx = pickle.load(_f, encoding="latin1")
            self._hand_pca = {
                "mean_l":       np.asarray(_smplx["hands_meanl"],       dtype=np.float32),   # (45,)
                "mean_r":       np.asarray(_smplx["hands_meanr"],       dtype=np.float32),   # (45,)
                "components_l": np.asarray(_smplx["hands_componentsl"], dtype=np.float32),   # (45, 45)
                "components_r": np.asarray(_smplx["hands_componentsr"], dtype=np.float32),   # (45, 45)
            }
        except Exception as e:
            logger.warning(f"Could not load SMPL-X hand PCA from {smplx_model_path}: {e}. "
                           f"Hand GT will remain zeros.")

        # gt_accum[person_id][field] = list of per-frame arrays with information
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

        # GT is world-space — no camera dimension needed. Single entry list keeps
        # the same self._gt[0] indexing pattern used throughout the class.
        self._gt = [gt_final]


    def _transform_to_world_frame(self) -> None:
        """Re-express GT body parameters in camera-0's coordinate frame.

        Camera-0 is our world origin. GT (RICH) is in true world coordinates,
        so it maps as:
            x_cam0 = R_0 @ x_true_world + t_0

        Predictions stay in their native cam-i frame. The cam-i → cam-0
        transform is applied locally inside _match_persons_to_gt() where it
        is actually needed (person proximity matching).

        Fields transformed in ``self._gt``:
            ``global_orient``, ``transl``
        """
        from scipy.spatial.transform import Rotation as SciR

        if not self._cameras or not self._gt:
            return

        ext0 = self._cameras[0].get("extrinsics")  # (3, 4) true-world → cam-0
        if ext0 is None:
            return
        R0 = ext0[:3, :3].astype(np.float64)  # true world → cam-0
        t0 = ext0[:3, 3].astype(np.float64)

        # self._gt is a list of references to the same dict, so transforming
        # self._gt[0] covers all camera slots at once.
        gt_dict = self._gt[0]  # {pid: {field: array}}
        for gt in gt_dict.values():
            # global_orient: (N, 3) axis-angle in true world space
            go = gt.get("global_orient")
            if go is not None:
                R_body = SciR.from_rotvec(
                    go.astype(np.float64).reshape(-1, 3)
                ).as_matrix()                          # (N, 3, 3)
                gt["global_orient"] = (
                    SciR.from_matrix(R0[None] @ R_body)
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
    def convert_camera(
        self,
        body_transl_cam: np.ndarray,
        focal_length: float,
        global_rot: np.ndarray,
        cam_calib: dict | None = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Use the Kabsch-estimated cam-0→cam-i extrinsic for inputs["camera"].

        Position 7 carries the GT focal length from calibration instead of the
        predicted focal — it is used as fixed context, not predicted by the network.

        Falls back to identity rotation + body_transl_cam if no Kabsch extrinsics
        are available for this camera/frame.
        """
        intr = cam_calib.get("intrinsics") if cam_calib else None
        focal_gt = float(intr[0, 0]) if intr is not None else float(focal_length)

        est = self._get_est_ext(cam_calib)
        if est is not None:
            from scipy.spatial.transform import Rotation as SciR
            q_xyzw = SciR.from_matrix(est[:3, :3]).as_quat()
            cam = np.zeros(8, dtype=np.float32)
            cam[:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
            cam[4:7] = est[:3, 3]
            cam[7] = focal_gt
            return cam

        cam = np.zeros(8, dtype=np.float32)
        cam[0] = 1.0
        cam[4:7] = body_transl_cam
        cam[7] = focal_gt
        return cam

    def _match_persons_to_gt(self) -> None:
        """Match ghost person IDs to RICH GT person IDs by 3D proximity.

        Both are already in cam-0 world space after _transform_to_world_frame().
        For each RICH GT person, finds the ghost person whose world-frame
        translation is closest (mean distance over common frames), then remaps
        _gt and _gt_frame_lut to use ghost pids as keys.

        Skipped automatically when _gt is empty (inference mode).
        Unmatched ghost persons have no _gt entry → build_gt_targets returns
        zeros → gt_valid is False → masked by the loss.
        """
        if not self._gt or not self._raw:
            return

        gt_dict = self._gt[0]   # {rich_pid: {field: array}}
        rich_pids = list(gt_dict.keys())
        if not rich_pids:
            return

        # Collect cam-0-frame translations for every ghost person across ALL cameras.
        # Predictions are in cam-i frame; transform locally here using GT extrinsics.
        # Using GT cameras for matching is acceptable: this is a one-time preprocessing
        # step, not model input.
        if not self._cameras:
            return
        ext0 = self._cameras[0].get("extrinsics")  # (3, 4) true-world → cam-0
        if ext0 is None:
            return
        R0 = ext0[:3, :3].astype(np.float64)
        t0 = ext0[:3, 3].astype(np.float64)

        all_ghost_pids: set[int] = set()
        for cam_persons in self._raw:
            all_ghost_pids.update(cam_persons.keys())
        ghost_pids = sorted(all_ghost_pids)
        if not ghost_pids:
            return

        # accum[ghost_pid][frame_idx] = list of (3,) translations in cam-0 frame (one per camera)
        accum: dict[int, dict[int, list[np.ndarray]]] = {gpid: {} for gpid in ghost_pids}
        for cam_idx, cam_persons in enumerate(self._raw):
            ext_i = self._cameras[cam_idx].get("extrinsics") if cam_idx < len(self._cameras) else None
            if ext_i is None:
                continue
            R_i = ext_i[:3, :3].astype(np.float64)
            t_i = ext_i[:3, 3].astype(np.float64)
            R_i2w = R0 @ R_i.T       # cam-i → cam-0 rotation
            d_i   = t0 - R_i2w @ t_i  # cam-i → cam-0 translation offset
            for gpid, pdata in cam_persons.items():
                tr = pdata.get("smplx_transl")   # (N, 3) cam-i frame
                fi = pdata.get("frame_indices")  # (N,)
                if tr is None or fi is None:
                    continue
                tr_cam0 = (tr.astype(np.float64) @ R_i2w.T + d_i).astype(np.float32)
                for i, f in enumerate(fi):
                    accum[gpid].setdefault(int(f), []).append(tr_cam0[i])

        # Average across cameras per frame
        ghost_transl: dict[int, dict[int, np.ndarray]] = {}
        for gpid in ghost_pids:
            ghost_transl[gpid] = {
                f: np.mean(views, axis=0)
                for f, views in accum[gpid].items()
            }

        # For each RICH GT person pick the closest ghost person
        rich_to_ghost: dict[int, int] = {}
        for rpid in rich_pids:
            gt_lut   = self._gt_frame_lut.get(rpid, {})
            gt_transl = gt_dict[rpid].get("transl")   # (N, 3) world frame
            if gt_transl is None:
                continue

            best_gpid: int | None = None
            best_dist = float("inf")
            for gpid in ghost_pids:
                common = set(gt_lut.keys()) & set(ghost_transl[gpid].keys())
                if not common:
                    continue
                dists = [
                    np.linalg.norm(gt_transl[gt_lut[f]] - ghost_transl[gpid][f])
                    for f in common
                ]
                mean_dist = float(np.mean(dists))
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_gpid = gpid

            if best_gpid is not None:
                rich_to_ghost[rpid] = best_gpid
                logger.info(
                    f"GT person {rpid} → ghost person {best_gpid} "
                    f"(mean dist={best_dist:.3f} m)"
                )
            else:
                logger.warning(f"No ghost match found for GT person {rpid} — will be skipped")

        # Remap _gt and _gt_frame_lut: replace RICH pids with ghost pids
        new_gt_dict: dict[int, dict[str, np.ndarray]] = {
            rich_to_ghost[rpid]: gt_dict[rpid]
            for rpid in rich_to_ghost
        }
        new_lut: dict[int, dict[int, int]] = {
            rich_to_ghost[rpid]: self._gt_frame_lut[rpid]
            for rpid in rich_to_ghost
        }
        self._gt = [new_gt_dict]
        self._gt_frame_lut = new_lut
        if self._restrict_to_gt_persons:
            self._gt_matched_ghost_pids = set(new_gt_dict.keys())
        self._rich_to_ghost: dict[int, int] = rich_to_ghost  # rich_pid → ghost_pid

        unmatched = set(ghost_pids) - set(rich_to_ghost.values())
        if unmatched:
            logger.info(
                f"Ghost persons {sorted(unmatched)} have no GT match "
                f"— excluded from batch"
            )

    def _fill_static_gt_cameras(self, gt_camera: np.ndarray) -> None:
        for k, vec in enumerate(self._gt_camera_vecs):
            if k < gt_camera.shape[0]:
                gt_camera[k] = vec

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
        transl_out: np.ndarray,
    ) -> None:
        """Fill GT targets from RICH SMPL-X annotations.

        Resolves the global frame number from the body_data ``frame_indices``
        array, then looks it up in the GT frame LUT.  If no GT exists for
        this frame / person slot, the arrays are left as zeros.
        """
        # GT camera: static per-camera extrinsic — fill for every camera.
        if cam_idx < len(self._gt_camera_vecs):
            camera_out[:] = self._gt_camera_vecs[cam_idx]

        # Pose/shape GT lives in world (cam-0) frame — only supervise cam-0.
        if cam_idx != 0:
            return

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

        # Shape: betas (10,)
        if "betas" in gt:
            n = min(shape_out.shape[0], gt["betas"].shape[-1])
            shape_out[:n] = gt["betas"][gt_idx, :n]

        # Pose: build full 55-joint axis-angle matching the same layout used
        # by EgoExoFusionDatapoint.convert_pose:
        #   [global(1), body(21), lhand(15), rhand(15), jaw+eyes(3)]
        if "body_pose" in gt and "global_orient" in gt:
            from scipy.spatial.transform import Rotation as SciR
            go = gt["global_orient"][gt_idx].reshape(1, 3)   # (1, 3)
            bp = gt["body_pose"][gt_idx].reshape(-1, 3)       # (21, 3)
            parts = [go, bp]
            for key, pca_mean_key, pca_comp_key in (
                ("left_hand_pose",  "mean_l", "components_l"),
                ("right_hand_pose", "mean_r", "components_r"),
            ):
                if key in gt:
                    raw = gt[key][gt_idx].reshape(-1)   # (12,) PCA or (45,) axis-angle
                    if raw.shape[0] != 45 and self._hand_pca:
                        # Decode PCA coefficients → 45-dim axis-angle
                        n = raw.shape[0]
                        aa45 = (self._hand_pca[pca_mean_key]
                                + self._hand_pca[pca_comp_key][:n].T @ raw)  # (45,)
                        parts.append(aa45.reshape(15, 3))
                    else:
                        parts.append(raw.reshape(-1, 3))  # (15, 3)
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
                [mats[:, 0, :], mats[:, 1, :]], axis=1
            )                                             # (55, 6)
            pose_out[:J_pose] = sixd[:J_pose]

        # Translation: world-frame root position (3,)
        if "transl" in gt:
            transl_out[:] = gt["transl"][gt_idx]


# ======================================================================
# DNA-Rendering subclass (ghost pipeline output + SMC calibration, no GT)
# ======================================================================

class DNARenderingFusionDatapoint(FusionDatapoint):
    """Loads body data from ghost pipeline + cameras from DNA-Rendering annots.

    Expected ghost pipeline output::

        scene_dir/
            <cam_id>/
                body_data/
                    person_1.npz
                    ...
            camera_alignment.npz    (optional Kabsch alignment)

    Calibration is read from a ``<scene>_annots.smc`` HDF5 file using
    :func:`~data.video_dataset.dna_load_calibration`.  No GT body parameters
    are loaded yet — targets mirror predictions (self-supervised mode).

    Constructor kwargs
    ------------------
    annots_path : str | Path
        Path to the ``<scene>_annots.smc`` annotation file.  A warning is
        emitted if absent; camera tokens fall back to ``pred_cam_t``.
    """

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        """Convert SMPL-X axis-angle params → [J, 6] 6D rotation.

        Identical layout to :class:`RICHFusionDatapoint`.
        """
        J = self.num_joints
        out = np.zeros((J, 6), dtype=np.float32)
        if body_pose_params is None or global_rot is None:
            return out
        try:
            from scipy.spatial.transform import Rotation as SciR
        except Exception:
            return out

        go = np.asarray(global_rot, dtype=np.float32).reshape(1, 3)
        bp = np.asarray(body_pose_params, dtype=np.float32).reshape(-1, 3)
        parts = [go, bp]
        if hand_pose_params is not None:
            hp = np.asarray(hand_pose_params, dtype=np.float32).reshape(-1, 3)
            parts.append(hp)
        aa = np.concatenate(parts, axis=0)
        if aa.shape[0] < J:
            aa = np.concatenate(
                [aa, np.zeros((J - aa.shape[0], 3), dtype=np.float32)], axis=0
            )
        try:
            mats = SciR.from_rotvec(aa).as_matrix()
        except Exception:
            return out
        sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
        out[:J] = sixd[:J]
        return out

    def load_body_data(self, **kwargs: Any) -> None:
        """Load ghost pipeline NPZ predictions (same layout as RICH/EgoExo)."""
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
                persons[pid] = {k: data[k] for k in self._NPZ_FIELDS if k in data}
            self._raw.append(persons)
            logger.debug(f"  {cam_dir.name}: loaded {len(persons)} person(s)")

    def load_cameras(self, **kwargs: Any) -> None:
        """Load GT calibration from ``_annots.smc`` and optional Kabsch estimates.

        Steps:
        1. Parse R/T/K for each camera from the SMC annotation file.
        2. Drop cameras with no calibration from ``_cam_dirs`` / ``_raw``.
        3. Re-express GT cameras relative to cam-0 (same convention as RICH).
        4. Attach Kabsch-estimated extrinsics from ``camera_alignment.npz`` as
           ``estimated_extrinsics`` so :meth:`convert_camera` can use them.
        """
        from scipy.spatial.transform import Rotation as SciR
        from data.video_dataset import dna_load_calibration

        annots_path = kwargs.get("annots_path")
        raw_calib: dict[str, dict[str, np.ndarray]] = {}
        if annots_path is not None and Path(annots_path).exists():
            raw_calib = dna_load_calibration(Path(annots_path))
        else:
            if annots_path is not None:
                logger.warning(
                    f"Annotation file not found: {annots_path} — "
                    "cameras will fall back to pred_cam_t"
                )

        self._cameras = []
        self._gt_camera_vecs: list[np.ndarray] = []
        valid_indices: list[int] = []

        # STEP 1 — match ghost camera dirs to DNA calibration.
        # Primary: exact name match (ghost dir name == HDF5 camera key).
        # Fallback: positional match by sorted index when no name matches at
        #           all (e.g. ghost dirs named "cam_00" but HDF5 keys are "0").
        calib_ids_sorted = sorted(raw_calib.keys())
        any_name_matched = any(cd.name in raw_calib for cd in self._cam_dirs)
        if raw_calib and not any_name_matched:
            logger.warning(
                f"No ghost camera dir name matched any DNA calibration key "
                f"(ghost: {[cd.name for cd in self._cam_dirs]}, "
                f"DNA keys: {calib_ids_sorted}). "
                f"Falling back to positional matching."
            )

        for i, cam_dir in enumerate(self._cam_dirs):
            # Try exact name match first.
            calib = raw_calib.get(cam_dir.name)
            matched_by = "name"
            # Positional fallback: use the i-th sorted calibration entry.
            if calib is None and not any_name_matched and i < len(calib_ids_sorted):
                calib = raw_calib[calib_ids_sorted[i]]
                matched_by = f"position (DNA key '{calib_ids_sorted[i]}')"

            if calib is not None:
                self._cameras.append(calib)
                valid_indices.append(i)
                ext = calib["extrinsics"]  # (3, 4) world → camera
                q_xyzw = SciR.from_matrix(ext[:3, :3].astype(np.float64)).as_quat()
                vec = np.zeros(8, dtype=np.float32)
                vec[:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)  # pytorch3d [qw,qx,qy,qz]
                vec[4:7] = ext[:3, 3]
                vec[7] = float(calib["K"][0, 0])  # fx
                self._gt_camera_vecs.append(vec)
                logger.debug(f"  {cam_dir.name}: loaded DNA calibration by {matched_by}")
            else:
                logger.warning(
                    f"  No DNA calibration found for camera '{cam_dir.name}' — excluding"
                )

        if len(valid_indices) < len(self._cam_dirs):
            self._cam_dirs = [self._cam_dirs[i] for i in valid_indices]
            self._raw      = [self._raw[i]      for i in valid_indices]

        # STEP 2 — re-express GT cameras relative to cam-0.
        if self._cameras:
            ext0 = self._cameras[0]["extrinsics"]
            R0 = ext0[:3, :3].astype(np.float64)
            t0 = ext0[:3, 3].astype(np.float64)
            for i, calib in enumerate(self._cameras):
                if i == 0:
                    self._gt_camera_vecs[0][:4] = np.array([1., 0., 0., 0.], dtype=np.float32)
                    self._gt_camera_vecs[0][4:7] = np.zeros(3, dtype=np.float32)
                    continue
                ext_i = calib["extrinsics"]
                R_rel = ext_i[:3, :3].astype(np.float64) @ R0.T
                t_rel = ext_i[:3, 3].astype(np.float64) - R_rel @ t0
                q_xyzw = SciR.from_matrix(R_rel).as_quat()
                self._gt_camera_vecs[i][:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
                self._gt_camera_vecs[i][4:7] = t_rel.astype(np.float32)

        # STEP 3 — optional Kabsch-estimated extrinsics for predicted camera token.
        align_path = self.scene_dir / "camera_alignment.npz"
        if align_path.exists():
            from preprocessing.camera_alignment import CameraAlignment
            alignment = CameraAlignment.load(align_path)
            cam0_name = self._cam_dirs[0].name if self._cam_dirs else None
            for i, cam_dir in enumerate(self._cam_dirs):
                cam_i_name = cam_dir.name
                if i == 0:
                    self._cameras[i]["est_ext_is_reference"] = True
                elif (cam0_name, cam_i_name) in alignment:
                    R_arr, t_arr = alignment[(cam0_name, cam_i_name)]
                    self._cameras[i]["est_ext_R"] = R_arr
                    self._cameras[i]["est_ext_t"] = t_arr
                elif (cam_i_name, cam0_name) in alignment:
                    R_arr, t_arr = alignment[(cam_i_name, cam0_name)]
                    R_inv = R_arr.T
                    t_inv = -R_inv @ t_arr
                    self._cameras[i]["est_ext_R"] = R_inv
                    self._cameras[i]["est_ext_t"] = t_inv
                else:
                    logger.warning(f"No Kabsch alignment for {cam_i_name} ↔ {cam0_name}")
        else:
            logger.warning(
                f"camera_alignment.npz not found in {self.scene_dir} — "
                "input camera tokens will fall back to pred_cam_t"
            )

    def load_ground_truth(self, **kwargs: Any) -> None:
        """No GT body annotations available for DNA-Rendering yet.

        Targets mirror predictions (self-supervised mode).
        Override this method once GT annotations become available.
        """
        self._gt = [{} for _ in self._cam_dirs]

    def convert_camera(
        self,
        body_transl_cam: np.ndarray,
        focal_length: float,
        global_rot: np.ndarray,
        cam_calib: dict | None = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Build the ``[8]`` camera token: ``[qw, qx, qy, qz, tx, ty, tz, fx]``.

        Uses Kabsch-estimated extrinsics when available; falls back to
        identity rotation + pred_cam_t.
        """
        focal_gt = (
            float(cam_calib["K"][0, 0])
            if cam_calib is not None and "K" in cam_calib
            else float(focal_length)
        )
        est = self._get_est_ext(cam_calib)
        if est is not None:
            from scipy.spatial.transform import Rotation as SciR
            q_xyzw = SciR.from_matrix(est[:3, :3]).as_quat()
            cam = np.zeros(8, dtype=np.float32)
            cam[:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
            cam[4:7] = est[:3, 3]
            cam[7] = focal_gt
            return cam

        cam = np.zeros(8, dtype=np.float32)
        cam[0] = 1.0
        cam[4:7] = body_transl_cam
        cam[7] = focal_gt
        return cam

    def _fill_static_gt_cameras(self, gt_camera: np.ndarray) -> None:
        for k, vec in enumerate(self._gt_camera_vecs):
            if k < gt_camera.shape[0]:
                gt_camera[k] = vec

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
        transl_out: np.ndarray,
    ) -> None:
        """Fill the GT camera slot from calibration (other GT fields are zero).

        The camera extrinsic from the annotation file is constant per camera,
        so we write it unconditionally on every call for this (frame, camera).
        """
        if cam_idx < len(self._gt_camera_vecs) and camera_out is not None:
            camera_out[:] = self._gt_camera_vecs[cam_idx]


# ======================================================================
# EgoHumans subclass
# ======================================================================

class EgoHumansFusionDatapoint(FusionDatapoint):
    """Fusion datapoint for the EgoHumans dataset.

    Reads ghost pipeline predictions as input and GT SMPL params + calibrated
    exo cameras from the EgoHumans data release.

    Expected layout after extracting a sequence archive::

        egohumans_seq_dir/
            ego/
                aria01/
                    images/rgb/<frame:05d>.jpg
                    calib/<frame:05d>.txt    (per-frame fisheye intrinsics+extrinsics)
                aria02/, aria03/
            exo/
                cam01/images/<frame:05d>.jpg
                ...cam08/
            colmap/workplace/
                cameras.txt   (OPENCV_FISHEYE intrinsics, 11 cameras)
                images.txt    (per-image extrinsics + name → camera_id mapping)
            processed_data/
                smpl/<frame:05d>.npy          {ariaXX: {global_orient(3), transl(3), body_pose(69), betas(10), ...}}
                refine_poses3d/<frame:05d>.npy  {ariaXX: (17, 4) COCO-17 3D + conf}
                poses3d/<frame:05d>.npy          fallback when refine_poses3d absent

    Ghost pipeline output (scene_dir)::

        scene_dir/
            cam01/body_data/person_*.npz
            ...cam08/

    Constructor kwargs
    ------------------
    egohumans_seq_dir : str | Path
        Path to the extracted EgoHumans sequence directory (see layout above).
        Cameras and GT are loaded from here.  If absent, runs in self-supervised
        mode (no GT, no calibrated cameras).
    """

    # Mapping: aria person name → integer GT person ID used throughout
    _ARIA_NAME_TO_ID = {"aria01": 0, "aria02": 1, "aria03": 2}

    def load_body_data(self, **kwargs: Any) -> None:
        """Load ghost pipeline NPZ predictions (same layout as RICH/EgoExo)."""
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
                persons[pid] = {k: data[k] for k in self._NPZ_FIELDS if k in data}
            self._raw.append(persons)
            logger.debug(f"  {cam_dir.name}: loaded {len(persons)} person(s)")

    def load_cameras(self, **kwargs: Any) -> None:
        """Parse COLMAP cameras.txt + images.txt for GT exo camera calibration.

        Steps:
        1. Parse OPENCV_FISHEYE intrinsics from cameras.txt.
        2. Parse per-image extrinsics from images.txt; for exo cameras
           (cam01-cam08) average across frames to get a static extrinsic.
        3. Map each ghost pipeline cam_dir name to a COLMAP camera entry.
        4. Drop ghost cam_dirs with no matching COLMAP calibration.
        5. Build ``_gt_camera_vecs`` relative to cam-0 (same convention as RICH).
        """
        from scipy.spatial.transform import Rotation as SciR

        seq_dir = Path(kwargs.get("egohumans_seq_dir", ""))
        colmap_dir = seq_dir / "colmap" / "workplace"

        # Read COLMAP intrinsics: camera_id → {fx, fy, cx, cy, distortion}
        colmap_intrinsics: dict[int, np.ndarray] = {}  # camera_id → (3, 3) K
        cameras_txt = colmap_dir / "cameras.txt"
        if cameras_txt.exists():
            with open(cameras_txt) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    cam_id = int(parts[0])
                    # model, width, height, fx, fy, cx, cy, k1..k4
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    colmap_intrinsics[cam_id] = K
        else:
            logger.warning(f"COLMAP cameras.txt not found at {cameras_txt}")

        # Read COLMAP images.txt: name_prefix → list of (R, t) extrinsics
        # Each line pair: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        #                 POINTS2D[] (ignored)
        # COLMAP convention: x_cam = R @ x_world + t  (world-to-cam).
        # We accumulate per-camera-prefix lists; exo cameras get averaged.
        from collections import defaultdict
        extrinsics_by_name: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = defaultdict(list)
        images_txt = colmap_dir / "images.txt"
        if images_txt.exists():
            with open(images_txt) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            i = 0
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) < 9:
                    i += 1
                    continue
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz      = float(parts[5]), float(parts[6]), float(parts[7])
                cam_id  = int(parts[8])
                name    = parts[9]                     # e.g. "aria01/00126.jpg" or "cam01/images/00001.jpg"
                prefix  = name.split("/")[0]           # "aria01", "cam01", …
                R = SciR.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float64)
                t = np.array([tx, ty, tz], dtype=np.float64)
                extrinsics_by_name[prefix].append((R, t, cam_id))
                i += 2  # skip POINTS2D line
        else:
            logger.warning(f"COLMAP images.txt not found at {images_txt}")

        # Build one representative (R, t) per camera prefix.
        # Ego cameras (aria) are dynamic; we still compute a mean to have some
        # fallback, but note their GT camera tokens are filled as NaN/identity
        # in build_gt_targets (ego cameras are not used as static views).
        def _median_extrinsic(entries: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray, int]:
            Rs   = np.stack([e[0] for e in entries])   # (N, 3, 3)
            ts   = np.stack([e[1] for e in entries])   # (N, 3)
            cam_id = entries[0][2]
            t_mean = ts.mean(axis=0)
            # Use the rotation closest to the mean rotation as representative
            R_mean_approx = Rs.mean(axis=0)
            U, _, Vt = np.linalg.svd(R_mean_approx)
            R_rep = U @ Vt
            if np.linalg.det(R_rep) < 0:
                U[:, -1] *= -1
                R_rep = U @ Vt
            return R_rep, t_mean, cam_id

        colmap_extrinsics: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for prefix, entries in extrinsics_by_name.items():
            colmap_extrinsics[prefix] = _median_extrinsic(entries)

        # Match ghost cam_dirs to COLMAP entries, drop unmatched ones.
        self._cameras = []
        self._gt_camera_vecs: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, cam_dir in enumerate(self._cam_dirs):
            name = cam_dir.name   # e.g. "cam01", "cam03", "aria01"
            if name in colmap_extrinsics:
                R, t, cam_id = colmap_extrinsics[name]
                K = colmap_intrinsics.get(cam_id, np.eye(3, dtype=np.float32))
                ext = np.zeros((3, 4), dtype=np.float32)
                ext[:3, :3] = R
                ext[:3, 3]  = t
                calib = {"extrinsics": ext, "intrinsics": K}
                self._cameras.append(calib)
                valid_indices.append(i)
                # GT camera vector [qw,qx,qy,qz, tx,ty,tz, fx]
                q_xyzw = SciR.from_matrix(R).as_quat()
                vec = np.zeros(8, dtype=np.float32)
                vec[:4]  = q_xyzw[[3, 0, 1, 2]].astype(np.float32)  # scipy→pytorch3d quat order
                vec[4:7] = t.astype(np.float32)
                vec[7]   = float(K[0, 0])
                self._gt_camera_vecs.append(vec)
            else:
                logger.warning(
                    f"No COLMAP calibration for cam_dir '{name}' — camera dropped"
                )

        if len(valid_indices) < len(self._cam_dirs):
            self._cam_dirs = [self._cam_dirs[i] for i in valid_indices]
            self._raw      = [self._raw[i]      for i in valid_indices]

        if not self._cameras:
            # Self-supervised fallback: no calibration available
            self._cameras  = [{} for _ in self._cam_dirs]
            self._gt_camera_vecs = []
            return

        # Re-express GT cameras relative to cam-0 (same convention as RICH).
        ext0 = self._cameras[0].get("extrinsics")
        if ext0 is not None:
            R0 = ext0[:3, :3].astype(np.float64)
            t0 = ext0[:3, 3].astype(np.float64)
            self._gt_camera_vecs[0][:4] = np.array([1., 0., 0., 0.], dtype=np.float32)
            self._gt_camera_vecs[0][4:7] = np.zeros(3, dtype=np.float32)
            for i in range(1, len(self._cameras)):
                ext_i = self._cameras[i].get("extrinsics")
                if ext_i is None:
                    continue
                R_i = ext_i[:3, :3].astype(np.float64)
                t_i = ext_i[:3, 3].astype(np.float64)
                R_rel = R_i @ R0.T
                t_rel = t_i - R_rel @ t0
                q_xyzw = SciR.from_matrix(R_rel).as_quat()
                self._gt_camera_vecs[i][:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
                self._gt_camera_vecs[i][4:7] = t_rel.astype(np.float32)

        # Attach Kabsch-estimated extrinsics if available (same as RICH/DNA).
        align_path = self.scene_dir / "camera_alignment.npz"
        if align_path.exists():
            from preprocessing.camera_alignment import CameraAlignment
            alignment = CameraAlignment.load(align_path)
            cam0_name = self._cam_dirs[0].name if self._cam_dirs else None
            for i, cam_dir in enumerate(self._cam_dirs):
                cam_name = cam_dir.name
                if i == 0:
                    self._cameras[i]["est_ext_is_reference"] = True
                elif (cam0_name, cam_name) in alignment:
                    R_arr, t_arr = alignment[(cam0_name, cam_name)]
                    self._cameras[i]["est_ext_R"] = R_arr
                    self._cameras[i]["est_ext_t"] = t_arr
                elif (cam_name, cam0_name) in alignment:
                    R_arr, t_arr = alignment[(cam_name, cam0_name)]
                    R_inv = R_arr.T
                    t_inv = -R_inv @ t_arr
                    self._cameras[i]["est_ext_R"] = R_inv
                    self._cameras[i]["est_ext_t"] = t_inv
                else:
                    logger.warning(f"No alignment for {cam_name} relative to {cam0_name}")

    def load_ground_truth(self, **kwargs: Any) -> None:
        """Load per-frame GT SMPL params and 3D poses from processed_data/.

        GT SMPL (processed_data/smpl/) is keyed by aria person name.  We
        convert SMPL body_pose (69-dim, 23 joints) to the SMPL-X convention
        (63-dim, 21 joints) by dropping the last two hand-proxy joints.

        GT 3D poses come from processed_data/refine_poses3d/ (or poses3d/ as
        fallback), giving 17 COCO-format keypoints per person per frame.

        GT is in COLMAP world frame; _transform_to_world_frame() will later
        re-express it in cam-0 frame exactly as done in RICHFusionDatapoint.
        """
        seq_dir = Path(kwargs.get("egohumans_seq_dir", ""))
        if not seq_dir.is_dir():
            logger.warning(
                f"egohumans_seq_dir not provided or missing — running without GT"
            )
            self._gt = [{} for _ in self._cam_dirs]
            return

        smpl_dir     = seq_dir / "processed_data" / "smpl"
        refine3d_dir = seq_dir / "processed_data" / "refine_poses3d"
        poses3d_dir  = seq_dir / "processed_data" / "poses3d"
        gt3d_dir     = refine3d_dir if refine3d_dir.is_dir() else poses3d_dir

        # gt_accum[person_id][field] = list of per-frame arrays
        gt_accum: dict[int, dict[str, list]] = {}

        def _ensure_pid(pid: int) -> None:
            if pid not in gt_accum:
                gt_accum[pid] = {
                    "frame_indices":  [],
                    "global_orient":  [],
                    "transl":         [],
                    "body_pose":      [],
                    "betas":          [],
                    "keypoints_3d":   [],
                }

        smpl_frames = sorted(smpl_dir.glob("*.npy")) if smpl_dir.is_dir() else []
        for npy_path in smpl_frames:
            try:
                frame_idx = int(npy_path.stem)
            except ValueError:
                continue

            smpl_data = np.load(str(npy_path), allow_pickle=True).item()
            # Load matching 3D pose file if available
            kp3d_data: dict[str, np.ndarray] = {}
            kp3d_path = gt3d_dir / npy_path.name if gt3d_dir.is_dir() else None
            if kp3d_path is not None and kp3d_path.exists():
                raw = np.load(str(kp3d_path), allow_pickle=True)
                kp3d_data = raw.item() if raw.shape == () else {}

            for person_name, params in smpl_data.items():
                if not isinstance(params, dict):
                    continue
                pid = self._ARIA_NAME_TO_ID.get(person_name)
                if pid is None:
                    continue
                _ensure_pid(pid)

                go  = np.asarray(params["global_orient"], dtype=np.float32).reshape(3)
                tr  = np.asarray(params["transl"],        dtype=np.float32).reshape(3)
                # SMPL body_pose: (69,) = 23 joints × 3; truncate to 21 joints
                # to match SMPL-X body_pose layout used by the rest of the pipeline.
                bp  = np.asarray(params["body_pose"], dtype=np.float32).flatten()[:63]
                bt  = np.asarray(params["betas"],     dtype=np.float32).flatten()[:10]

                gt_accum[pid]["frame_indices"].append(frame_idx)
                gt_accum[pid]["global_orient"].append(go)
                gt_accum[pid]["transl"].append(tr)
                gt_accum[pid]["body_pose"].append(bp)
                gt_accum[pid]["betas"].append(bt)

                # 3D keypoints: (17, 4) → take xyz, ignore confidence
                kp3d = kp3d_data.get(person_name)
                if kp3d is not None:
                    kp3d = np.asarray(kp3d, dtype=np.float32)[:, :3]  # (17, 3)
                else:
                    kp3d = np.zeros((17, 3), dtype=np.float32)
                gt_accum[pid]["keypoints_3d"].append(kp3d)

        # Stack into arrays
        gt_final: dict[int, dict[str, np.ndarray]] = {}
        for pid, fields in gt_accum.items():
            if not fields["frame_indices"]:
                continue
            gt_final[pid] = {k: np.stack(v) for k, v in fields.items()}

        self._gt_frame_lut: dict[int, dict[int, int]] = {
            pid: {int(fi): i for i, fi in enumerate(gt_final[pid]["frame_indices"])}
            for pid in gt_final
        }
        # World-space GT replicated as single entry (same pattern as RICH).
        self._gt = [gt_final]

    def _transform_to_world_frame(self) -> None:
        """Re-express GT body params from COLMAP world frame into cam-0 frame.

        COLMAP world-to-cam-0 transform: x_cam0 = R0 @ x_world + t0.
        Applied to global_orient (rotation composition) and transl (affine).
        Identical to RICHFusionDatapoint._transform_to_world_frame.
        """
        from scipy.spatial.transform import Rotation as SciR

        if not self._cameras or not self._gt:
            return
        ext0 = self._cameras[0].get("extrinsics")  # (3, 4) world → cam-0
        if ext0 is None:
            return
        R0 = ext0[:3, :3].astype(np.float64)
        t0 = ext0[:3, 3].astype(np.float64)

        gt_dict = self._gt[0]
        for gt in gt_dict.values():
            go = gt.get("global_orient")
            if go is not None:
                R_body = SciR.from_rotvec(go.astype(np.float64).reshape(-1, 3)).as_matrix()
                gt["global_orient"] = (
                    SciR.from_matrix(R0[None] @ R_body)
                    .as_rotvec()
                    .astype(np.float32)
                    .reshape(go.shape)
                )
            tr = gt.get("transl")
            if tr is not None:
                gt["transl"] = (tr.astype(np.float64) @ R0.T + t0).astype(np.float32)
            kp3d = gt.get("keypoints_3d")
            if kp3d is not None:
                # (N, 17, 3) → transform each point
                gt["keypoints_3d"] = (
                    kp3d.astype(np.float64).reshape(-1, 3) @ R0.T + t0
                ).astype(np.float32).reshape(kp3d.shape)

    def _match_persons_to_gt(self) -> None:
        """Match ghost integer PIDs to EgoHumans GT aria person IDs by 3D proximity.

        Same algorithm as RICHFusionDatapoint: for each GT person compute the
        mean distance to each ghost person's world-frame translation, pick the
        closest, remap ``_gt`` and ``_gt_frame_lut`` to ghost PIDs.
        """
        if not self._gt or not self._raw:
            return

        gt_dict = self._gt[0]
        if not gt_dict:
            return

        if not self._cameras:
            return
        ext0 = self._cameras[0].get("extrinsics")
        if ext0 is None:
            return
        R0 = ext0[:3, :3].astype(np.float64)
        t0 = ext0[:3, 3].astype(np.float64)

        all_ghost_pids: set[int] = set()
        for cam_persons in self._raw:
            all_ghost_pids.update(cam_persons.keys())
        ghost_pids = sorted(all_ghost_pids)
        if not ghost_pids:
            return

        # Transform ghost translations to cam-0 frame
        accum: dict[int, dict[int, list[np.ndarray]]] = {gpid: {} for gpid in ghost_pids}
        for cam_idx, cam_persons in enumerate(self._raw):
            ext_i = self._cameras[cam_idx].get("extrinsics") if cam_idx < len(self._cameras) else None
            if ext_i is None:
                continue
            R_i = ext_i[:3, :3].astype(np.float64)
            t_i = ext_i[:3, 3].astype(np.float64)
            R_i2w = R0 @ R_i.T
            d_i   = t0 - R_i2w @ t_i
            for gpid, pdata in cam_persons.items():
                tr = pdata.get("smplx_transl")
                fi = pdata.get("frame_indices")
                if tr is None or fi is None:
                    continue
                tr_cam0 = (tr.astype(np.float64) @ R_i2w.T + d_i).astype(np.float32)
                for i, f in enumerate(fi):
                    accum[gpid].setdefault(int(f), []).append(tr_cam0[i])

        ghost_transl: dict[int, dict[int, np.ndarray]] = {
            gpid: {f: np.mean(vs, axis=0) for f, vs in accum[gpid].items()}
            for gpid in ghost_pids
        }

        ego_pids = list(gt_dict.keys())
        ego_to_ghost: dict[int, int] = {}
        used_ghost: set[int] = set()
        for epid in ego_pids:
            gt_lut    = self._gt_frame_lut.get(epid, {})
            gt_transl = gt_dict[epid].get("transl")
            if gt_transl is None:
                continue
            best_gpid: int | None = None
            best_dist = float("inf")
            for gpid in ghost_pids:
                if gpid in used_ghost:
                    continue
                common = set(gt_lut.keys()) & set(ghost_transl[gpid].keys())
                if not common:
                    continue
                dists = [
                    np.linalg.norm(gt_transl[gt_lut[f]] - ghost_transl[gpid][f])
                    for f in common
                ]
                mean_dist = float(np.mean(dists))
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_gpid = gpid
            if best_gpid is not None:
                ego_to_ghost[epid] = best_gpid
                used_ghost.add(best_gpid)
                logger.info(
                    f"EgoHumans GT person {epid} → ghost person {best_gpid} "
                    f"(mean dist={best_dist:.3f} m)"
                )
            else:
                logger.warning(f"No ghost match for GT person {epid} — skipped")

        new_gt: dict[int, dict[str, np.ndarray]] = {
            ego_to_ghost[epid]: gt_dict[epid] for epid in ego_to_ghost
        }
        new_lut: dict[int, dict[int, int]] = {
            ego_to_ghost[epid]: self._gt_frame_lut[epid] for epid in ego_to_ghost
        }
        self._gt = [new_gt]
        self._gt_frame_lut = new_lut
        if self._restrict_to_gt_persons:
            self._gt_matched_ghost_pids = set(new_gt.keys())

        unmatched = set(ghost_pids) - set(ego_to_ghost.values())
        if unmatched:
            logger.info(f"Ghost persons {sorted(unmatched)} have no GT — excluded")

    def _fill_static_gt_cameras(self, gt_camera: np.ndarray) -> None:
        for k, vec in enumerate(self._gt_camera_vecs):
            if k < gt_camera.shape[0]:
                gt_camera[k] = vec

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
        transl_out: np.ndarray,
    ) -> None:
        """Fill GT targets from EgoHumans SMPL annotations.

        GT camera: static COLMAP extrinsic (same for every frame, every person).
        GT pose/shape: from the matched aria SMPL params, cam-0 frame only.
        """
        if cam_idx < len(self._gt_camera_vecs) and camera_out is not None:
            camera_out[:] = self._gt_camera_vecs[cam_idx]

        if cam_idx != 0:
            return

        if cam_idx >= len(self._gt) or pid not in self._gt[cam_idx]:
            return

        raw_fi = (
            self._raw[cam_idx].get(pid, {}).get("frame_indices")
            if cam_idx < len(self._raw)
            else None
        )
        if raw_fi is None or local_idx >= len(raw_fi):
            return

        global_frame = int(raw_fi[local_idx])
        gt_lut  = self._gt_frame_lut.get(pid, {})
        gt_idx  = gt_lut.get(global_frame)
        if gt_idx is None:
            return

        gt = self._gt[cam_idx][pid]
        go  = gt.get("global_orient")
        bp  = gt.get("body_pose")
        bt  = gt.get("betas")
        tr  = gt.get("transl")
        kp3 = gt.get("keypoints_3d")

        if go is not None and bp is not None:
            pose_out[:] = self.convert_pose(bp[gt_idx], None, go[gt_idx])
        if bt is not None:
            shape_out[:] = bt[gt_idx]
        if tr is not None and t < transl_out.shape[0]:
            transl_out[t] = tr[gt_idx]
        if kp3 is not None and kp3d_out is not None:
            n = min(kp3d_out.shape[-2], kp3[gt_idx].shape[0])
            kp3d_out[..., :n, :] = kp3[gt_idx][:n]

    def convert_pose(
        self,
        body_pose_params: np.ndarray,
        hand_pose_params: np.ndarray | None,
        global_rot: np.ndarray,
    ) -> np.ndarray:
        """Convert SMPL-X / SMPL axis-angle → [J, 6] 6D rotation.

        Identical to RICHFusionDatapoint.convert_pose.
        """
        J = self.num_joints
        out = np.zeros((J, 6), dtype=np.float32)
        if body_pose_params is None or global_rot is None:
            return out
        try:
            from scipy.spatial.transform import Rotation as SciR
        except Exception:
            return out

        go = np.asarray(global_rot, dtype=np.float32).reshape(1, 3)
        bp = np.asarray(body_pose_params, dtype=np.float32).reshape(-1, 3)
        parts = [go, bp]
        if hand_pose_params is not None:
            parts.append(np.asarray(hand_pose_params, dtype=np.float32).reshape(-1, 3))
        aa = np.concatenate(parts, axis=0)
        if aa.shape[0] < J:
            aa = np.concatenate(
                [aa, np.zeros((J - aa.shape[0], 3), dtype=np.float32)], axis=0
            )
        try:
            mats = SciR.from_rotvec(aa).as_matrix()
        except Exception:
            return out
        sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
        out[:J] = sixd[:J]
        return out

    def convert_camera(
        self,
        body_transl_cam: np.ndarray,
        focal_length: float,
        global_rot: np.ndarray,
        cam_calib: dict | None = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Build [8] camera token using Kabsch-estimated extrinsics when available."""
        intr = cam_calib.get("intrinsics") if cam_calib else None
        focal_gt = float(intr[0, 0]) if intr is not None else float(focal_length)
        est = self._get_est_ext(cam_calib)
        if est is not None:
            from scipy.spatial.transform import Rotation as SciR
            q_xyzw = SciR.from_matrix(est[:3, :3]).as_quat()
            cam = np.zeros(8, dtype=np.float32)
            cam[:4] = q_xyzw[[3, 0, 1, 2]].astype(np.float32)
            cam[4:7] = est[:3, 3]
            cam[7] = focal_gt
            return cam
        cam = np.zeros(8, dtype=np.float32)
        cam[0] = 1.0
        cam[4:7] = body_transl_cam
        cam[7] = focal_gt
        return cam


# ======================================================================
# Dataset (collection of datapoints)
# ======================================================================

def _swap_reference_camera(
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    ref: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Data augmentation: change the world reference frame to camera ``ref``.

    The model assumes camera 0 is always the world origin (identity extrinsic).
    Without augmentation it always sees the same 50 fixed camera layouts and
    memorises them rather than learning generalizable relative geometry.

    This function produces an equivalent sample where camera ``ref`` takes the
    role of camera 0:
      - Camera ``ref`` becomes identity in the new world frame (by definition).
      - All other cameras are re-expressed relative to camera ``ref`` instead
        of camera 0.  Concretely, for camera k with old extrinsic (R_k, t_k):
            new_R_k = R_k @ R_ref^T
            new_t_k = t_k - new_R_k @ t_ref
      - The indices 0 and ``ref`` are then swapped in the K dimension of every
        K-shaped tensor so that the new reference sits at position 0, matching
        the model's convention.
      - World-frame GT quantities (body translation, 3D keypoints, root
        orientation) are rotated and translated into the new frame:
            P_new = R_ref @ P_old + t_ref

    The transform is defined using GT camera extrinsics (not predicted ones) so
    the augmented targets are always geometrically correct regardless of the
    current model state.

    Over a training epoch with K cameras per scene this produces K distinct
    views of each scene, forcing the model to learn geometry that is invariant
    to which camera happens to be designated as the reference.
    """
    from pytorch3d.transforms import (
        quaternion_to_matrix, matrix_to_quaternion,
        rotation_6d_to_matrix, matrix_to_rotation_6d,
    )

    cam_gt = targets["camera"]   # (K, 8)  [qw,qx,qy,qz, tx,ty,tz, focal_raw]
    K, _ = cam_gt.shape

    # Skip augmentation if GT camera for ``ref`` is missing (zero quaternion).
    if cam_gt[ref, :4].norm().item() < 0.5:
        return inputs, targets

    q_ref = cam_gt[ref, :4]    # (4,)
    t_ref = cam_gt[ref, 4:7]   # (3,)
    R_ref   = quaternion_to_matrix(q_ref.unsqueeze(0)).squeeze(0)   # (3, 3)
    R_ref_T = R_ref.t()                                               # (3, 3)

    def _retransform_cameras(cam: torch.Tensor) -> torch.Tensor:
        """Re-express a (K, 8) camera tensor relative to the new reference."""
        cam = cam.clone()
        for k in range(K):
            R_k   = quaternion_to_matrix(cam[k, :4].unsqueeze(0)).squeeze(0)  # (3, 3)
            t_k   = cam[k, 4:7]                                                 # (3,)
            new_R = R_k @ R_ref_T                                               # (3, 3)
            new_t = t_k - new_R @ t_ref                                         # (3,)
            cam[k, :4] = matrix_to_quaternion(new_R.unsqueeze(0)).squeeze(0)
            cam[k, 4:7] = new_t
        return cam

    def _swap_k(t: torch.Tensor) -> torch.Tensor:
        """Swap cameras 0 and ref along the K dimension (dim 1 for T,K,... tensors)."""
        t = t.clone()
        t[:, [0, ref]] = t[:, [ref, 0]]
        return t

    # ── Transform + swap camera tensors (K, 8) ─────────────────────────
    new_cam_in = _retransform_cameras(inputs["camera"])
    new_cam_in[[0, ref]] = new_cam_in[[ref, 0]]
    new_cam_gt = _retransform_cameras(cam_gt)
    new_cam_gt[[0, ref]] = new_cam_gt[[ref, 0]]

    # ── Swap K dim (dim 1) for all other K-shaped (T, K, ...) inputs ───
    new_inputs = {
        k: (_swap_k(v) if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == K else v)
        for k, v in inputs.items()
    }
    new_inputs["camera"] = new_cam_in

    new_targets = dict(targets)
    new_targets["camera"] = new_cam_gt
    if "kp2d" in new_targets and isinstance(new_targets["kp2d"], torch.Tensor) and new_targets["kp2d"].shape[1] == K:
        new_targets["kp2d"] = _swap_k(new_targets["kp2d"])

    # ── Transform world-frame GT quantities: P_new = R_ref @ P_old + t_ref ──
    if "trans" in new_targets:
        trans = new_targets["trans"]                                  # (T, P, 3)
        new_targets["trans"] = (
            torch.einsum("ij,tpj->tpi", R_ref, trans) + t_ref[None, None, :]
        )

    if "keypoints_3d" in new_targets:
        kp3d = new_targets["keypoints_3d"]                           # (T, P, 70, 3)
        new_targets["keypoints_3d"] = (
            torch.einsum("ij,tpqj->tpqi", R_ref, kp3d) + t_ref[None, None, None, :]
        )

    if "pose" in new_targets:
        pose = new_targets["pose"]                                    # (T, P, J, 6)
        Tp, Pp = pose.shape[:2]
        R_root = rotation_6d_to_matrix(pose[:, :, 0, :].reshape(-1, 6)).reshape(Tp, Pp, 3, 3)
        new_R_root = torch.einsum("ij,tpjk->tpik", R_ref, R_root)
        new_pose = pose.clone()
        new_pose[:, :, 0, :] = matrix_to_rotation_6d(new_R_root.reshape(-1, 3, 3)).reshape(Tp, Pp, 6)
        new_targets["pose"] = new_pose

    return new_inputs, new_targets


class RICHFusionDataset(Dataset):
    """A simple dataset holding a list of :class:`RICHFusionDatapoint` objects.

    Each item is one scene; ``__getitem__`` builds the tensors lazily so only
    the currently-accessed scene is materialised in memory.

    When ``augment=True`` the dataset expands each scene into K variants, one
    per reference-camera choice, multiplying the effective dataset size by the
    average number of cameras per scene.
    """

    def __init__(self, datapoints: list[FusionDatapoint], augment: bool = False) -> None:
        self._datapoints = datapoints
        self._augment = augment
        if augment:
            self._aug_index: list[tuple[int, int]] = [
                (dp_idx, ref)
                for dp_idx, dp in enumerate(datapoints)
                for ref in range(dp.num_cameras)
            ]

    def __len__(self) -> int:
        if self._augment:
            return len(self._aug_index)
        return len(self._datapoints)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self._augment:
            dp_idx, ref = self._aug_index[idx]
            inputs, targets = self._datapoints[dp_idx]._build_sample()
            if ref > 0:
                inputs, targets = _swap_reference_camera(inputs, targets, ref)
            return inputs, targets
        return self._datapoints[idx]._build_sample()

class MixedFusionDataset(Dataset):
    """A dataset that combines multiple named sources of :class:`FusionDatapoint` objects.

    Each sample is one complete scene (same contract as :class:`RICHFusionDataset`).
    Sources with fewer scenes can be up-sampled via ``weights`` so that every
    dataset contributes roughly equally to each epoch regardless of size.

    Parameters
    ----------
    sources : list[tuple[str, list[FusionDatapoint]]]
        Named scene pools, e.g.::

            [
                ("rich", rich_train_datapoints),
                ("dna",  dna_train_datapoints),
            ]

    weights : list[float] | None
        Multiplier applied to each source when building the flat index.
        ``None`` (default) weights each source by 1.0 — all scenes appear
        exactly once per epoch regardless of pool size.

        Example: ``weights=[1.0, 3.0]`` replicates every DNA scene 3× so
        that a small DNA pool (say 10 scenes) contributes as often as a
        large RICH pool (say 30 scenes).

        Fractional weights are rounded to the nearest integer (minimum 1).

    shuffle : bool
        If ``True``, the flat index is shuffled once at construction time.
        For epoch-level shuffling use a ``DataLoader`` with ``shuffle=True``.

    How indexing works
    ------------------
    ``__init__`` builds a flat ``_index`` list of ``(source_idx, local_scene_idx)``
    pairs by repeating each source's scene indices ``round(weight)`` times::

        source 0  →  [0, 1, 2]          weight=1  →  [(0,0),(0,1),(0,2)]
        source 1  →  [0, 1]             weight=2  →  [(1,0),(1,1),(1,0),(1,1)]
        combined  →  [(0,0),(0,1),(0,2),(1,0),(1,1),(1,0),(1,1)]

    ``dataset[i]`` looks up ``_index[i]``, resolves the correct datapoint,
    and calls ``_build_sample()`` on it.
    """

    def __init__(
        self,
        sources: list[tuple[str, list[FusionDatapoint]]],
        weights: list[float] | None = None,
        shuffle: bool = False,
    ) -> None:
        if not sources:
            raise ValueError("MixedFusionDataset requires at least one source.")
        if weights is not None and len(weights) != len(sources):
            raise ValueError(
                f"len(weights)={len(weights)} must match len(sources)={len(sources)}"
            )

        self._sources = sources
        self._weights = weights if weights is not None else [1.0] * len(sources)

        # Build the flat index.
        self._index: list[tuple[int, int]] = []
        for src_i, (name, datapoints) in enumerate(sources):
            repeat = max(1, round(self._weights[src_i]))
            for _ in range(repeat):
                for local_i in range(len(datapoints)):
                    self._index.append((src_i, local_i))

        if shuffle:
            import random
            random.shuffle(self._index)

        # Summary for logging.
        parts = [
            f"{name}×{max(1, round(self._weights[i]))} ({len(dps)} scenes)"
            for i, (name, dps) in enumerate(sources)
        ]
        logger.info(
            f"MixedFusionDataset: {len(self._index)} total samples — "
            + ", ".join(parts)
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        src_i, local_i = self._index[idx]
        return self._sources[src_i][1][local_i]._build_sample()

    @property
    def source_names(self) -> list[str]:
        return [name for name, _ in self._sources]

    def source_datapoints(self, name: str) -> list[FusionDatapoint]:
        """Return the datapoint list for a given source name."""
        for src_name, dps in self._sources:
            if src_name == name:
                return dps
        raise KeyError(f"No source named {name!r}. Available: {self.source_names}")

    def __repr__(self) -> str:
        parts = [
            f"{name}: {len(dps)} scenes × {max(1, round(self._weights[i]))}"
            for i, (name, dps) in enumerate(self._sources)
        ]
        return f"MixedFusionDataset({len(self._index)} samples — " + ", ".join(parts) + ")"


def build_fusion_dataloader(
    dataset_type: str,
    scene_dir: str | Path,
    num_joints: int = 55,
    batch_size: int = 1,
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
    _REGISTRY: dict[str, type[FusionDatapoint]] = {
        "egoexo4d_gt": EgoExo4DGTFusionDatapoint,
        "rich": RICHFusionDatapoint,
        "dna": DNARenderingFusionDatapoint,
    }
    cls = _REGISTRY.get(dataset_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    dp = cls(
        scene_dir=scene_dir,
        num_joints=num_joints,
        **dataset_kwargs,
    )
    ds = RICHFusionDataset([dp])
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
