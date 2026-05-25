from pathlib import Path

import numpy as np
import torch


def _dense_fill(stored: np.ndarray, frame_indices: np.ndarray | None, fill_shape: tuple) -> np.ndarray:
    """Expand a stored array into a dense (T_dense, ...) array using frame_indices."""
    if frame_indices is None:
        return stored
    T_dense = int(frame_indices[-1] - frame_indices[0] + 1)
    if len(frame_indices) >= T_dense:
        return stored
    dense = np.zeros((T_dense,) + fill_shape, dtype=stored.dtype)
    dense[frame_indices - frame_indices[0]] = stored
    return dense


def load_person_smplx_pose(
    npz_path: Path | str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load SMPL-X pose rotations and joint confidences from a person npz.

    Concatenates smplx_body_pose (T,63) + smplx_left_hand_pose (T,45) +
    smplx_right_hand_pose (T,45) → reshaped to (T_dense, 51, 3).
    Confidence is pred_joint_confidence[:, 1:52] → (T_dense, 51).

    Detection gaps (frames present in frame_indices but not stored) are
    filled with zero pose and zero confidence so they do not corrupt
    cross-correlation or DTW cost landscapes.

    Returns None if required keys are missing.
    """
    with np.load(str(npz_path)) as d:
        required = {"smplx_body_pose", "smplx_left_hand_pose",
                    "smplx_right_hand_pose", "pred_joint_confidence"}
        if not required.issubset(d.files):
            return None
        pose = np.concatenate([
            d["smplx_body_pose"],       # (T, 63)
            d["smplx_left_hand_pose"],  # (T, 45)
            d["smplx_right_hand_pose"], # (T, 45)
        ], axis=1).astype(np.float32)  # (T_stored, 153)
        raw_conf = d["pred_joint_confidence"][:, 1:52].astype(np.float32)  # (T_stored, 51)
        frame_indices = d["frame_indices"].astype(int) if "frame_indices" in d.files else None

    if frame_indices is not None and len(frame_indices) < (frame_indices[-1] - frame_indices[0] + 1):
        T_dense = int(frame_indices[-1] - frame_indices[0] + 1)
        dense_pose = np.zeros((T_dense, 153), dtype=np.float32)
        dense_conf = np.zeros((T_dense, 51),  dtype=np.float32)
        idx = frame_indices - frame_indices[0]
        dense_pose[idx] = pose
        dense_conf[idx] = raw_conf
        pose     = dense_pose
        raw_conf = dense_conf

    rotations = torch.from_numpy(pose).reshape(-1, 51, 3)
    conf      = torch.from_numpy(raw_conf)
    return rotations, conf


def load_person_keypoints(
    npz_path: Path | str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load 3D keypoints and joint confidences from a person npz.

    Returns pred_keypoints_3d as (T_dense, J, 3) and pred_joint_confidence
    as (T_dense, J).

    Detection gaps are filled with zero keypoints and zero confidence so
    they do not corrupt cost landscapes.

    Returns None if pred_keypoints_3d is missing or empty.
    """
    with np.load(str(npz_path), allow_pickle=False) as d:
        if "pred_keypoints_3d" not in d:
            return None
        kpts = d["pred_keypoints_3d"].astype(np.float32)  # (T_stored, J, 3)
        if len(kpts) == 0:
            return None
        T_stored, J = kpts.shape[:2]
        raw_conf = (
            d["pred_joint_confidence"].astype(np.float32)
            if "pred_joint_confidence" in d.files
            else np.ones((T_stored, J), dtype=np.float32)
        )
        frame_indices = d["frame_indices"].astype(int) if "frame_indices" in d.files else None

    if frame_indices is not None and len(frame_indices) < (frame_indices[-1] - frame_indices[0] + 1):
        T_dense = int(frame_indices[-1] - frame_indices[0] + 1)
        dense_kpts = np.zeros((T_dense, J, 3), dtype=np.float32)
        dense_conf = np.zeros((T_dense, J),    dtype=np.float32)
        idx = frame_indices - frame_indices[0]
        dense_kpts[idx] = kpts
        dense_conf[idx] = raw_conf
        kpts     = dense_kpts
        raw_conf = dense_conf

    joints = torch.from_numpy(kpts)
    conf   = torch.from_numpy(raw_conf)
    return joints, conf


def load_person_smplx_joints(
    npz_path: Path | str,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run SMPL-X FK on stored pose params and return root-relative joint positions.

    Returns (T, 51, 3) positions and (T, 51) confidence — joints 1-51 in SMPL-X
    ordering (body + hands, root excluded).  Global orient is zeroed so the output
    is camera-agnostic: only the relative joint configuration is encoded.

    Returns None if required pose keys are missing.
    """
    from utilities.smplx_utilities import _get_smplx_model

    with np.load(str(npz_path)) as d:
        required = {"smplx_body_pose", "smplx_left_hand_pose",
                    "smplx_right_hand_pose", "pred_joint_confidence"}
        if not required.issubset(d.files):
            return None
        body_pose  = d["smplx_body_pose"].astype(np.float32)       # (T_stored, 63)
        lhand_pose = d["smplx_left_hand_pose"].astype(np.float32)  # (T_stored, 45)
        rhand_pose = d["smplx_right_hand_pose"].astype(np.float32) # (T_stored, 45)
        betas_raw  = np.zeros((1, 10), dtype=np.float32)  # neutral shape: eliminates cross-camera inconsistency
        raw_conf   = d["pred_joint_confidence"][:, 1:52].astype(np.float32)  # (T_stored, 51)
        frame_indices = d["frame_indices"].astype(int) if "frame_indices" in d.files else None

    body_pose  = _dense_fill(body_pose,  frame_indices, (63,))
    lhand_pose = _dense_fill(lhand_pose, frame_indices, (45,))
    rhand_pose = _dense_fill(rhand_pose, frame_indices, (45,))
    raw_conf   = _dense_fill(raw_conf,   frame_indices, (51,))
    T = len(body_pose)

    # Betas: broadcast to T if stored as a single vector
    if betas_raw.ndim == 1:
        betas_raw = betas_raw[None]
    betas = np.broadcast_to(betas_raw[:1], (T, 10)).copy()

    bp  = torch.from_numpy(body_pose).to(device)
    lhp = torch.from_numpy(lhand_pose).to(device)
    rhp = torch.from_numpy(rhand_pose).to(device)
    bt  = torch.from_numpy(betas).to(device)
    go  = torch.zeros(T, 3, device=device)  # zero global orient → camera-agnostic

    with torch.no_grad():
        model  = _get_smplx_model(T, device, bp.dtype)
        output = model(
            global_orient=go,
            body_pose=bp,
            left_hand_pose=lhp,
            right_hand_pose=rhp,
            betas=bt,
        )

    joints = output.joints[:, :52, :]          # (T, 52, 3) — body + hands
    joints = joints - joints[:, 0:1, :]        # root-relative
    joints = joints[:, 1:, :]                  # drop root → (T, 51, 3)
    conf   = torch.from_numpy(raw_conf)        # (T, 51)
    return joints.cpu(), conf
