import smplx
import pickle
import torch
from configuration import CONFIG
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import numpy as np

_smplx_cache: dict = {}


def _get_smplx_model(batch_size: int, device: torch.device, dtype: torch.dtype):
    key = (batch_size, str(device), dtype)
    if key not in _smplx_cache:
        _smplx_cache[key] = smplx.create(
            model_path=CONFIG.data.smplx_model_path,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            batch_size=batch_size,
        ).to(device).to(dtype)
    return _smplx_cache[key]


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

    pose_aa = matrix_to_axis_angle(
        rotation_6d_to_matrix(pose.reshape(N*P*J, 6))
    ).reshape(N*P, J*3)

    output = _get_smplx_model(N*P, device, dtype)(
        global_orient=pose_aa[:, :3],
        body_pose=pose_aa[:, 3:],
        betas=shape.reshape(N*P, 10),
        return_verts=True,
    )

    V = output.vertices.shape[1]
    if has_camera_dim:
        return output.vertices.reshape(B, T, K, P, V, 3)
    else:
        return output.vertices.reshape(B, T, P, V, 3)


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