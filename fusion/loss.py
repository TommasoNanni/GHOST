"""Loss base class for fusion training."""

from __future__ import annotations
import logging

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_quaternion

from utilities.smplx_utilities import get_smplx_vertices
from utilities.camera_utilities import extract_cameras
from utilities.geometry import project_to_2d, skew_symmetric
from configuration import CONFIG
from utilities.smplx_utilities import PARENTS_TABLE

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
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
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

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(*args, **kwargs)

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
            img_size: tuple[int, int] = (224, 224),
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
        total_loss = pose_stream.new_zeros([])

        for i in range(K):
            for j in range(i+1,K):
                pose_i = pose_stream[:, :, i] # (B, T, P, J, 6)
                pose_j = pose_stream[:, :, j] # (B, T, P, J, 6)
                        
                shape_i = shape_stream[:, :, i] # (B, T, P, S)
                shape_j = shape_stream[:, :, j] # (B, T, P, S)
                
                vertices_i = get_smplx_vertices(pose_i, shape_i) # (B, T, P, V, 3) in camera i
                vertices_j = get_smplx_vertices(pose_j, shape_j) # (B, T, P, V, 3) in camera j

                R_i, t_i, K_i = extract_cameras(camera_stream[:, :, i], img_size)
                R_j, t_j, K_j = extract_cameras(camera_stream[:, :, j], img_size)

                F = self.compute_fundamental_matrix(R_i, t_i, R_j, t_j, K_i, K_j)

                x_i = project_to_2d(vertices_i, K_i)
                x_j = project_to_2d(vertices_j, K_j)

                epipolar_errors = self.compute_epipolar_errors(x_i, x_j, F)

                visible = (vertices_i[..., 2] > 0) & (vertices_j[..., 2] > 0)

                loss_ij = (epipolar_errors**2)*visible.float()
                total_loss += loss_ij.sum() / (visible.sum() + 1e-8)
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

class MSELoss(Loss):
    def __init__(
        self, name: str = "MSE Loss", weight: float = 1.0
    ) -> None:
        super().__init__(name, weight)

    def forward(self, pred, true):
        loss_fn = nn.MSELoss()
        return loss_fn(pred, true)

class TemporalSmoothnessLoss(Loss):
    def __init__(
        self, name: str = "Temporal smoothness loss", weight: float = 1.0
    )-> None:
        super().__init__(name, weight)

    def forward(
        self,
        current: torch.Tensor, 
        previous1: torch.Tensor | None = None, 
        previous2: torch.Tensor | None = None,
    ):
        if previous1 is None and previous2 is None:
            if current is None:
                raise ValueError("Temporal Smoothness needs the current parameters not to be None")
            logging.warning("You called Temporal Smoothness loss with previous parameters None, returning norm of identity")
            return torch.norm(current)**2
        elif previous2 is None:
            logging.warning("You called Temporal Smoothness loss with previous2 parameters None, returning velocity constraint")
            return torch.norm(current - previous1)**2
        elif previous1 is not None and previous2 is not None and current is not None:
            return torch.norm(current - 2*previous1 + previous2)**2
        else:
            raise ValueError(
                f"""Invalid parameters configuration. Is None? \n
                 Current: {(current is None)}, Previous1 {(previous1 is None)}, Previous2 {(previous2 is None)}"""
            )

class VPoserLoss(Loss):
    def __init__(
        self,
        name: str = "VPoser Loss",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)

        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser

        self.vposer, _ = load_model(
            CONFIG.fusion.loss.VPoser_path,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        self.vposer.eval()

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Compute the VPoser KL-divergence prior on body pose.

        Parameters
        ----------
        pose : ``(*, J, 3)`` — SMPLX pose in **axis-angle** format.
            J must be 55.
            The tensor is flattened to (N, 162) before encoding, where
            162 = 54 joints × 3 axis-angle values.

        Returns
        -------
        Scalar KL loss: KL( q(z|x) ‖ N(0,I) )
        """
        # slice body joints 1-55 (skip root), flatten to (N, 54*3)
        smpl_pose = pose[..., 1:, :].reshape(-1, 162)

        dist = self.vposer.encode(smpl_pose)

        mean = dist.mean          # (N, latent_dim)
        std  = dist.scale         # (N, latent_dim)
        logvar = 2.0 * torch.log(std + 1e-8)

        kl_loss = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=-1)

        return kl_loss.mean()


class BoneLengthconsistencyLoss(Loss):
    def __init__(
        self, 
        name: str = "Bone Lenght Consistency Loss",
        weight: float = 1.0
    )-> None:
        super().__init__(name, weight)

    def forward(self, joints):
        """
        Encodes the fact that all joints must have the same length across views
        and over time

        Parameters
        ----------
        joints: (B, T, K, P, J, 3) — joints in axis-angle format

        Returns
        -------
        Scalar loss encouraging constant bone length across time and cameras.
        """
        parent_mapping = torch.tensor(
            PARENTS_TABLE, device = joints.device, dtype = joints.dtype
        )

        parents = parent_mapping[0, :]
        parent_joints = joints[:, :, :, :, parents, :] # Tensor of father joints

        bone_lengths = torch.norm(joints - parent_joints, p = 2, dim = -1) # Bone lengths
        indices = torch.arange(len(parents), device = joints.device)
        mask = parent_mapping != indices
        valid_lengths = bone_lengths[..., mask] # (B, T, K, P, J_valid)

        lengths_to_persist = valid_lengths.permute(0, 3, 4, 1, 2).contiguous().flatten(start_dim = -2) #(B, P, J_valid, T*K)
        bone_std = torch.std(lengths_to_persist, dim = -1) + 1e-6
        return bone_std.mean()        


class BetaConsistencyLoss(Loss):
    def __init__(
        self,
        name: str = "Beta Consistency Loss",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)

    def forward(self, beta_stream: torch.Tensor) -> torch.Tensor:
        """
        Compute the consistency loss on shape parameters across cameras.

        Parameters
        ----------
        beta_stream : (B, T, K, P, S) — stream of shape parameters for each camera

        Returns
        -------
        Scalar loss encouraging consistency of shape parameters across cameras.
        """

        beta_mean = beta_stream.mean(dim=2, keepdim=True)  # (B, T, 1, P, S)
        loss = torch.mean((beta_stream - beta_mean)**2)
        return loss


class CameraMSELoss(Loss):
    def __init__(
        self,
        name: str = "Camera error (geodesic + MSE) loss",
        weight: float = 1.0
    )-> None:
        super().__init__(name, weight)

    def forward(self, R, t, R_gt, t_gt):
        """
        Computes the distance between the predicted and the ground truth cameras
        Parameters
        ----------
        R : (B, T, K, 3, 3) — stream of rotation camera parameters
        t : (B, T, K, 3) - stream of translation camera parameters
        R_gt : (B, T, K, 3, 3) — stream of GT rotation camera parameters
        t_gt : (B, T, K, 3) - stream of GT translation camera parameters

        Returns
        -------
        Scalar loss encouraging adherence of camera parameters to the GT
        """
        R_flat = R.reshape(-1, 3, 3)
        R_gt_flat = R_gt.reshape(-1, 3, 3)

        q = matrix_to_quaternion(R_flat)
        q_gt = matrix_to_quaternion(R_gt_flat)

        dot_product = torch.sum(q * q_gt, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

        quaternion_loss = 1 - torch.abs(dot_product)
        quaternion_mean = quaternion_loss.mean()

        translation_loss = torch.norm(t - t_gt, dim=-1)**2
        translation_mean = translation_loss.mean()

        return translation_mean + quaternion_mean