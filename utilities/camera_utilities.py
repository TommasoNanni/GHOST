import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix


def extract_cameras(camera_params, img_size):
    """
    Extract rotation, translation and intrinsics from camera parameters.

    Parameters
    ----------
    camera_params : (B, 8) or (B, T, 8) — [quat(4), trans(3), focal_raw(1)]
        (B, 8)    — static camera (one extrinsic per scene, no T).
        (B, T, 8) — dynamic camera (one extrinsic per frame).
    img_size : (H, W)

    Returns
    -------
    R     : (..., 3, 3)
    t     : (..., 3)
    K_mat : (..., 3, 3)  intrinsics
    """
    static = camera_params.dim() == 2  # True for (B, 8)

    quat      = camera_params[..., 0:4]
    trans     = camera_params[..., 4:7]
    focal_raw = camera_params[..., 7]

    R = quaternion_to_matrix(quat)
    t = trans
    focal = F.softplus(focal_raw)

    H_img, W_img = img_size
    K_mat = torch.zeros(*camera_params.shape[:-1], 3, 3,
                        device=camera_params.device, dtype=camera_params.dtype)
    K_mat[..., 0, 0] = focal
    K_mat[..., 1, 1] = focal
    K_mat[..., 0, 2] = W_img / 2
    K_mat[..., 1, 2] = H_img / 2
    K_mat[..., 2, 2] = 1

    return R, t, K_mat
