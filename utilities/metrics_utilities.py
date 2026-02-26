import numpy as np


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment (SE(3) or Sim(3)).

    Finds R, t, s that minimise ‖dst − (s*R*src + t)‖^2.

    Parameters
    ----------
    src, dst : ndarray, shape (N, 3)
    with_scale : bool
        If True solve for s (Sim(3)); if False fix s = 1 (SE(3)).

    Returns
    -------
    R : ndarray (3, 3)
    t : ndarray (3,)
    s : float
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    cov = (dst_c.T @ src_c) / n  # (3, 3)

    U, D, Vt = np.linalg.svd(cov)

    # Correct reflection
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        s = np.trace(np.diag(D) @ S) / var_src if var_src > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    return R, t, float(s)


def apply_alignment(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """Apply s*R*pts + t."""
    return (s * (pts @ R.T)) + t


def geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two 3×3 rotation matrices."""
    R_diff = R_a.T @ R_b
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def batch_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between batches of 3×3 rotation matrices.

    Parameters
    ----------
    R_a, R_b : ndarray, shape (..., 3, 3)

    Returns
    -------
    angles : ndarray, shape (...)
    """
    R_diff = R_a.swapaxes(-1, -2) @ R_b  # R_a^T @ R_b, (..., 3, 3)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = np.clip((np.clip(trace, -1.0, 3.0) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """
    Fréchet mean of rotation matrices on SO(3).
    Approximated as the SVD projection of the element-wise sum.
    Parameters
    ----------
    Rs : ndarray, shape (N, 3, 3)

    Returns
    -------
    R_mean : ndarray (3, 3)
    """
    R_sum = Rs.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt
