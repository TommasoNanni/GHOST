import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix


def extract_cameras(camera_params, img_size):
    """
    Extract rotation, translation and intrinsics from camera parameters.

    Parameters
    ----------
    camera_params : (B, T, 8) — [quat(4), trans(3), focal_raw(1)]
        focal_raw is an unconstrained scalar; softplus is applied internally
        to ensure the focal length is positive.
    img_size : (H, W)

    Returns
    -------
    R : (B, T, 3, 3)
    t : (B, T, 3)
    K : (B, T, 3, 3)
    """
    B, T = camera_params.shape[:2]

    quat      = camera_params[..., 0:4]
    trans     = camera_params[..., 4:7]
    focal_raw = camera_params[..., 7]

    R = quaternion_to_matrix(quat)
    t = trans
    focal = F.softplus(focal_raw)   # (B, T) — guaranteed positive

    H, W = img_size
    K = torch.zeros(B, T, 3, 3, device=camera_params.device, dtype=camera_params.dtype)
    K[:, :, 0, 0] = focal
    K[:, :, 1, 1] = focal
    K[:, :, 0, 2] = W / 2
    K[:, :, 1, 2] = H / 2
    K[:, :, 2, 2] = 1

    return R, t, K
