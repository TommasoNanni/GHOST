import torch
from pytorch3d.transforms import quaternion_to_matrix


def extract_cameras(camera_params, img_size):
    """
    Extract rotation, translation and intrinsics from camera parameters

    Parameters
    ----------
    camera_params : (B, T, 8) — [focal, quat(4), trans(3)]
    img_size : (H, W)
    
    Returns
    -------
    R : (B, T, 3, 3)
    t : (B, T, 3)
    K : (B, T, 3, 3)
    """
    B, T = camera_params.shape[:2]
    
    focal = camera_params[..., 0:1]
    quat  = camera_params[..., 1:5]
    trans = camera_params[..., 5:8]

    R = quaternion_to_matrix(quat)

    t = trans

    H, W = img_size
    K = torch.zeros(B, T, 3, 3, device = camera_params.device, dtype = camera_params.dtype)

    K[:, :, 0, 0] = focal
    K[:, :, 1, 1] = focal
    K[:, :, 0, 2] = W/2
    K[:, :, 1, 2] = H/2
    K[:, :, 2, 2] = 1

    return R, t, K
