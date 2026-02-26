"""Loss base class for fusion training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from utilities.smplx_utilities import get_smplx_vertices
from utilities.camera_utilities import extract_cameras
from utilities.geometry import project_to_2d, skew_symmetric

class Loss(ABC):
    """Base class for all fusion training losses.

    Subclasses must implement :meth:`forward`, which receives the model
    predictions and targets and returns a scalar tensor.

    Parameters
    ----------
    name   : human-readable identifier used for logging.
    weight : scalar multiplier applied to this loss when summing into the
             total loss. Defaults to 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def forward(self, *args) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        preds   : model output (structure depends on the subclass).
        targets : ground-truth (structure depends on the subclass).

        Returns
        -------
        scalar torch.Tensor (no grad graph required for logging, but
        the returned tensor must support .backward() during training).
        """

    def __call__(self, **kwargs) -> torch.Tensor:
        return self.forward(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"



class EpipolarLoss(Loss):
    def __init__(
        self, name: str = "Epipolar Loss", weight: float = 1.0
    ) -> None:
        super().__init__(name, weight)

    def forward(
            self, 
            pose_stream: torch.Tensor, 
            shape_stream: torch.Tensor, 
            camera_stream: torch.Tensor,
            img_size: int = 224
        ) -> torch.Tensor:
        """
        computes the epipolar loss starting from body poses and shapes in 
        RELATIVE camera coordinates, so each one wrt its camera
        Input:
            pose_stream: a stream of 3D poses (B, T, K, P, J, 6)
            shape_stream: a stream of 3D shapes (B, T, K, P, S)
            camera_stream: a stream of camera parameters (B, T, K, 8)
        Output:
            loss that checks the alignment of the 3D poses with the epipolar geometry defined by the camera parameters
        """
        B, T, K, P, J, _ = pose_stream.shape
        num_pairs = 0

        for i in range(K):
            for j in range(i+1,K):
                pose_i = pose_stream[:, :, i] # (B, T, P, J, 6)
                pose_j = pose_stream[:, :, j] # (B, T, P, J, 6)
                        
                shape_i = shape_stream[:, :, i] # (B, T, P, S)
                shape_j = shape_stream[:, :, j] # (B, T, P, S)
                
                vertices_i = get_smplx_vertices(pose_i, shape_i) # (B, T, P, V, 3) in camera i
                vertices_j = get_smplx_vertices(pose_j, shape_j) # (B, T, P, V, 3) in camera j

                R_i, t_i, K_i = extract_camera(cameras[:, :, i], img_size)
                R_j, t_j, K_j = extract_camera(cameras[:, :, j], img_size)

                F = self.compute_fundamental_matrix(R_i, t_i, R_j, t_j, K_i, K_j)

                x_i = project_to_2d(vertices_i, K_i)
                x_j = project_to_2d(vertices_j, K_j)

                epipolar_errors = self.compute_epipolar_errors(x_i, x_j, F)

                visible = (vertices_i[..., 2] > 0) & (vertices_j[..., 2] > 0)

                loss_ij = (epipolar_errors**2)*visible.float()
                total_loss = loss_ij.sum() / (visible.sum() + 1e-8)
                num_pairs += 1
        
        return total_loss / num_pairs


    def compute_fundamental_matrix(self, R_i, t_i, R_j, t_j, K_i, K_j):
        """
        Computes the fundamental matrix defining the transformation from camera i to camera j
        Parameters
        ----------
        R_i, R_j : (B, T, 3, 3)
        t_i, t_j : (B, T, 3)
        K_i, K_j : (B, T, 3, 3)
        
        Returns
        -------
        F : (B, T, 3, 3)
        """
        B, T = R_i.shape[:2]

        R_rel = torch.bmm(
            R_j.reshape(B*T, 3, 3),
            R_i.reshape(B*T, 3, 3).transpose(-2, -1)
        ).reshape(B,T,3,3)

        t_rel = t_j - torch.bmm(
            R_rel.reshape(B*T, 3, 3),
            t_i.reshape(B*T, 3, 1)
        ).reshape(B,T,3)

        t_skew = skew_symmetric(t_rel)
        E = torch.bmm(
            t_skew.reshape(B*T,3,3),
            R_rel.reshape(B*T,3,3)
        ).reshape(B,T,3,3)

        K_i_inv = torch.inverse(K_i.reshape(B*T,3,3)).reshape(B,T,3,3)
        K_j_inv_T = torch.inverse(K_j.reshape(B*T,3,3)).transpose(-2,-1).reshape(B,T,3,3)

        F = torch.bmm(
            torch.bmm(
                K_j_inv_T.reshape(B*T,3,3),
                E.reshape(B*T, 3, 3)
            ),
            K_i_inv.reshape(B*T, 3, 3)
        ).reshape(B,T,3,3)

        return F

    def compute_epipolar_errors(self, x_i, x_j, F):
        """
        Computes the epipolar error between the two batches of corresponding points
        Parameters
        ----------
        x_i, x_j : (B, T, P, V, 3) — homogeneous 2D points
        F : (B, T, 3, 3) — fundamental matrix
        
        Returns
        -------
        error : (B, T, P, V) — scalar error per point
        """
        B, T, P, V = x_i.shape[:4]

        x_i_flat = x_i.reshape(B*T*P*V, 3, 1)
        x_j_flat = x_j.reshape(B*T*P*V, 1, 3)
        F_expanded = F.reshape(B,T,1,1,3,3).expand(B,T,P,V,3,3).reshape(B*T*P*V, 3, 3)

        error = torch.bmm(
            torch.bmm(x_j_flat, F_expanded),
            x_i_flat,
        ).reshape(B,T,P,V)

        return error
