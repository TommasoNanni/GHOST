import torch

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a vector of shape (B,T,3)
    """
    B, T = v.shape[:2]
    S = torch.zeros(B,T,3,3,device = v.device, dtype = v.dtype)
    S[:, :, 0, 1] = -v[:, :, 2]
    S[:, :, 0, 2] =  v[:, :, 1]
    S[:, :, 1, 0] =  v[:, :, 2]
    S[:, :, 1, 2] = -v[:, :, 0]
    S[:, :, 2, 0] = -v[:, :, 1]
    S[:, :, 2, 1] =  v[:, :, 0]
    return S

def project_to_2d(vertices, K):
    """
    Projects 3D vertices to 2D using the camera intrinsics K
    Parameters
    ----------
    vertices : (B, T, P, V, 3) in camera frame
    K : (B, T, 3, 3)
    
    Returns
    -------
    x : (B, T, P, V, 3) - homogeneous coordinates of the projected points
    """
    B,T,P,V = vertices.shape[:4]
    K_expanded = K.reshape(B,T,1,1,3,3).expand(B,T,P,V,3,3)
    vertices_flat = vertices.reshape(B*T*P*V, 3)
    K_flat = K_expanded.reshape(B*T*P*V, 3, 3)

    projected = torch.bmm(K_flat, vertices_flat).reshape(B,T,P,V,3)

    x = projected/(projected[..., 2:3] + 1e-8)

    return x