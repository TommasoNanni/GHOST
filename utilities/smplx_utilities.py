import smplx
import pickle
import torch
import torch.nn.functional as F
from configuration import CONFIG
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion
import numpy as np


def _rot_matrix_to_axis_angle_safe(R_flat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle without acos singularity.

    Uses matrix_to_quaternion (polynomial, no sqrt) then atan2-based angle
    with a safe norm so the backward is finite even at the identity rotation.

    Parameters
    ----------
    R_flat : (..., 3, 3)

    Returns
    -------
    axis_angle : (..., 3)
    """
    q        = matrix_to_quaternion(R_flat)               # (..., 4) [w, x, y, z]
    xyz      = q[..., 1:]                                 # (..., 3) imaginary part
    w        = q[..., :1].abs()                           # (..., 1) real part, keep w≥0
    xyz_norm = xyz.pow(2).sum(-1, keepdim=True).add(1e-8).sqrt()   # safe norm
    angle    = 2.0 * torch.atan2(xyz_norm, w)             # (..., 1) in [0, π]
    return (xyz / xyz_norm) * angle                       # (..., 3)

_smplx_cache: dict = {}


def _smplx_create_kwargs(model_path: str) -> dict:
    """Return model_path + ext kwargs for smplx.create, auto-detecting pkl vs npz.

    smplx.create() treats a file path directly (bypassing its internal
    model_type subdirectory logic), so we always pass the full file path.
    """
    from pathlib import Path as _Path
    p = _Path(model_path)
    if p.is_file():
        return {"model_path": str(p), "ext": p.suffix.lstrip(".")}
    return {"model_path": model_path, "ext": "pkl"}


def _get_smplx_model(batch_size: int, device: torch.device, dtype: torch.dtype):
    key = (batch_size, str(device), dtype)
    if key not in _smplx_cache:
        _smplx_cache[key] = smplx.create(
            **_smplx_create_kwargs(CONFIG.data.smplx_model_path),
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            batch_size=batch_size,
        ).to(device).to(dtype)
    return _smplx_cache[key]


def clear_smplx_cache() -> None:
    """Release all cached SMPL-X model instances from GPU memory.

    Each training sequence may have a different (B*T_chunk*P) batch size,
    so without clearing the cache accumulates one model instance per unique
    batch size seen — each instance holds GPU memory.  Call this once per
    training step (after backward) to keep the cache at most one entry.
    """
    _smplx_cache.clear()


def get_smplx_vertices(pose: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through SMPLX model to get vertices.

    Parameters
    ----------
    pose : (B, T, P, J, 6) or (B, T, K, P, J, 6) — 6D rotation representation.
           Joint 0 is the global orientation; joints 1..J-1 are body joints.
    shape : (B, T, P, 10) or (B, T, K, P, 10) — shape parameters.

    Returns
    -------
    vertices : (B, T, [K,] P, V, 3) — SMPLX vertices in the local frame.
    """
    device = pose.device
    dtype = pose.dtype
    has_camera_dim = (pose.ndim == 6)

    if has_camera_dim:
        B, T, K, P, J, _ = pose.shape
        pose  = pose.reshape(B*T*K, P, J, 6)
        shape = shape.reshape(B*T*K, P, 10)
    else:
        B, T, P, J, _ = pose.shape
        pose  = pose.reshape(B*T, P, J, 6)
        shape = shape.reshape(B*T, P, 10)

    N = pose.shape[0]  # B*T*K or B*T

    pose_aa = _rot_matrix_to_axis_angle_safe(
        rotation_6d_to_matrix(pose.reshape(N*P*J, 6))
    ).reshape(N*P, J*3)

    output = _get_smplx_model(N*P, device, dtype)(
        global_orient=pose_aa[:, :3],
        body_pose=pose_aa[:, 3:66],
        left_hand_pose=pose_aa[:, 66:111],
        right_hand_pose=pose_aa[:, 111:156],
        jaw_pose=pose_aa[:, 156:159],
        leye_pose=pose_aa[:, 159:162],
        reye_pose=pose_aa[:, 162:165],
        betas=shape.reshape(N*P, 10),
        return_verts=True,
    )

    V = output.vertices.shape[1]
    if has_camera_dim:
        return output.vertices.reshape(B, T, K, P, V, 3)
    else:
        return output.vertices.reshape(B, T, P, V, 3)


def get_smplx_joints(pose: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through SMPLX model to get joint positions.

    Parameters
    ----------
    pose : (B, T, P, J, 6) or (B, T, K, P, J, 6) — 6D rotation representation.
    shape : (B, T, P, 10) or (B, T, K, P, 10) — shape parameters.

    Returns
    -------
    joints : (B, T, [K,] P, J, 3) — SMPLX joint positions.
    """
    device = pose.device
    dtype = pose.dtype
    has_camera_dim = (pose.ndim == 6)

    if has_camera_dim:
        B, T, K, P, J, _ = pose.shape
        pose  = pose.reshape(B*T*K, P, J, 6)
        shape = shape.reshape(B*T*K, P, 10)
    else:
        B, T, P, J, _ = pose.shape
        pose  = pose.reshape(B*T, P, J, 6)
        shape = shape.reshape(B*T, P, 10)

    N = pose.shape[0]

    pose_aa = _rot_matrix_to_axis_angle_safe(
        rotation_6d_to_matrix(pose.reshape(N*P*J, 6))
    ).reshape(N*P, J*3)

    output = _get_smplx_model(N*P, device, dtype)(
        global_orient=pose_aa[:, :3],
        body_pose=pose_aa[:, 3:66],
        left_hand_pose=pose_aa[:, 66:111],
        right_hand_pose=pose_aa[:, 111:156],
        jaw_pose=pose_aa[:, 156:159],
        leye_pose=pose_aa[:, 159:162],
        reye_pose=pose_aa[:, 162:165],
        betas=shape.reshape(N*P, 10),
        return_verts=False,
    )

    Jout = output.joints.shape[1]
    if has_camera_dim:
        return output.joints.reshape(B, T, K, P, Jout, 3)
    else:
        return output.joints.reshape(B, T, P, Jout, 3)


def get_joint_parent_mapping():
    data = np.load(CONFIG.data.smplx_model_path, allow_pickle=True)

    print(data.keys())
    print(data["kintree_table"])


PARENTS_TABLE = [
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15,
     15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46,
     47, 21, 49, 50, 21, 52, 53],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
     48, 49, 50, 51, 52, 53, 54]
]