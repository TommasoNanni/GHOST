import torch
from pytorch3d.transforms import quaternion_to_matrix


def extract_cameras(camera_params, img_size, focal_length: float = 1000.0):
    """
    Extract rotation, translation and intrinsics from camera parameters.

    Parameters
    ----------
    camera_params : (B, T, 7) — [quat(4), trans(3)]
    img_size : (H, W)
    focal_length : float
        Focal length used to build the intrinsics matrix K. Defaults to 1000.0.

    Returns
    -------
    R : (B, T, 3, 3)
    t : (B, T, 3)
    K : (B, T, 3, 3)
    """
    B, T = camera_params.shape[:2]

    quat  = camera_params[..., 0:4]
    trans = camera_params[..., 4:7]

    R = quaternion_to_matrix(quat)
    t = trans

    H, W = img_size
    K = torch.zeros(B, T, 3, 3, device=camera_params.device, dtype=camera_params.dtype)
    K[:, :, 0, 0] = focal_length
    K[:, :, 1, 1] = focal_length
    K[:, :, 0, 2] = W / 2
    K[:, :, 1, 2] = H / 2
    K[:, :, 2, 2] = 1

    return R, t, K
