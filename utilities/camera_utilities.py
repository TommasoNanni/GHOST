import torch


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

    focal = camera_params[..., 0:1]
    quat  = camera_params[..., 1:5]
    trans = camera_params[..., 5:8]

    from pytorch3d.transforms import quaternion_to_matrix
    R = quaternion_to_matrix(quat)

    t = trans

    H, W = img_size
    K = torch.zeros(B, T, 3, 3, device = camera_params.device, dtype = camera_params.dtype)
