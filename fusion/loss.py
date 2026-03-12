"""Loss base class for fusion training."""

from __future__ import annotations
import logging
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbp_path = os.path.join(project_root, 'human_body_prior')
if hbp_path not in sys.path:
    sys.path.insert(0, hbp_path)

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_axis_angle, quaternion_to_matrix

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
        self, name: str = "Epipolar Loss", weight: float = 1.0, img_size: tuple[int, int] = (224, 224)
    ) -> None:
        super().__init__(name, weight)
        self.img_size = img_size

    def forward(
            self,
            preds: tuple,
            targets: dict,
        ) -> torch.Tensor:
        """
        Computes the epipolar loss from per-camera pose/shape/camera predictions.
        Uses pose_per_cam and shape_per_cam so each body is in its own camera frame.

        preds: (pose_aggr, shape_aggr, camera, pose_per_cam, shape_per_cam)
        """
        _, _, camera_stream, pose_stream, shape_stream = preds
        if pose_stream.shape[2] < 2:
            return pose_stream.new_zeros([])
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

                R_i, t_i, K_i = extract_cameras(camera_stream[:, :, i], self.img_size)
                R_j, t_j, K_j = extract_cameras(camera_stream[:, :, j], self.img_size)

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

class PoseMSELoss(Loss):
    def __init__(self, name: str = "Pose MSE Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        pose_aggr, _, _, _, _ = preds
        return F.mse_loss(pose_aggr, targets["pose"])


class ShapeMSELoss(Loss):
    def __init__(self, name: str = "Shape MSE Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        _, shape_aggr, _, _, _ = preds
        return F.mse_loss(shape_aggr, targets["shape"])

class TemporalSmoothnessLoss(Loss):
    def __init__(
        self, name: str = "Temporal smoothness loss", weight: float = 1.0
    )-> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """Acceleration smoothness on the aggregated pose stream (B, T, P, J, 6)."""
        pose_aggr, _, _, _, _ = preds
        if pose_aggr.shape[1] < 3:
            return pose_aggr.new_zeros([])
        current   = pose_aggr[:, 2:]
        previous1 = pose_aggr[:, 1:-1]
        previous2 = pose_aggr[:, :-2]
        return torch.norm(current - 2 * previous1 + previous2) ** 2

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

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """
        Compute the VPoser KL-divergence prior on the aggregated body pose.

        Converts pose_aggr from 6D rotation to axis-angle, then encodes with VPoser.

        Returns
        -------
        Scalar KL loss: KL( q(z|x) ‖ N(0,I) )
        """
        pose_aggr, _, _, _, _ = preds
        # 6D rotation → rotation matrix → axis-angle
        rot_mat    = rotation_6d_to_matrix(pose_aggr)                        # (B, T, P, J, 3, 3)
        axis_angle = matrix_to_axis_angle(rot_mat.reshape(-1, 3, 3))         # (B*T*P*J, 3)
        pose       = axis_angle.reshape(*pose_aggr.shape[:-1], 3)            # (B, T, P, J, 3)

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

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """
        Penalises inconsistent bone lengths across time in the fused prediction.

        Runs one SMPL-X forward pass on (pose_aggr, shape_aggr) to get 3D joints,
        then checks that each bone length is stable over T.

        pose_aggr : (B, T, P, J, 6)
        shape_aggr: (B, T, P, 10)
        joints    : (B, T, P, J, 3)  — no K dimension (already fused)

        Returns
        -------
        Scalar loss encouraging constant bone length across time.
        """
        pose_aggr, shape_aggr, _, _, _ = preds
        joints = get_smplx_vertices(pose_aggr, shape_aggr)  # (B, T, P, J, 3)

        parent_mapping = torch.tensor(
            PARENTS_TABLE, device=joints.device, dtype=joints.dtype
        )
        parents = parent_mapping[0, :]
        parent_joints = joints[..., parents, :]                       # (B, T, P, J, 3)

        bone_lengths = torch.norm(joints - parent_joints, p=2, dim=-1)  # (B, T, P, J)
        indices = torch.arange(len(parents), device=joints.device)
        mask = parent_mapping != indices
        valid_lengths = bone_lengths[..., mask]                       # (B, T, P, J_valid)

        # check consistency across T only (no K dimension after fusion)
        lengths_to_persist = valid_lengths.permute(0, 2, 3, 1).contiguous()  # (B, P, J_valid, T)
        bone_std = torch.std(lengths_to_persist, dim=-1) + 1e-6
        return bone_std.mean()


class BetaConsistencyLoss(Loss):
    def __init__(
        self,
        name: str = "Beta Consistency Loss",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """
        Penalises inconsistent shape parameters across cameras.

        Uses shape_per_cam (B, T, K, P, 10): for each person, betas should be
        the same regardless of which camera observes them.

        Returns
        -------
        Scalar loss encouraging consistency of shape parameters across cameras.
        """
        _, _, _, _, shape_per_cam = preds
        beta_mean = shape_per_cam.mean(dim=2, keepdim=True)  # (B, T, 1, P, 10)
        return torch.mean((shape_per_cam - beta_mean) ** 2)


class CameraMSELoss(Loss):
    def __init__(
        self,
        name: str = "Camera error (geodesic + MSE) loss",
        weight: float = 1.0,
        img_size: tuple[int, int] = (224, 224),
    )-> None:
        super().__init__(name, weight)
        self.img_size = img_size

    def forward(
        self,
        preds: tuple,
        targets: dict,
    ) -> torch.Tensor:
        """
        Computes geodesic rotation + MSE translation between predicted and GT cameras.

        camera pred : (B, T, K, 8) — [quat(4), trans(3), focal_raw(1)]
        camera GT   : (B, T, K, 8) — same layout

        Returns
        -------
        Scalar loss encouraging adherence of camera parameters to the GT.
        """
        _, _, camera_pred, _, _ = preds
        cam_gt = targets["camera"]

        R_pred = quaternion_to_matrix(camera_pred[..., :4].reshape(-1, 4)).reshape(*camera_pred.shape[:-1], 3, 3)
        t_pred = camera_pred[..., 4:7]
        focal_pred = camera_pred[..., 7]
        R_gt   = quaternion_to_matrix(cam_gt[..., :4].reshape(-1, 4)).reshape(*cam_gt.shape[:-1], 3, 3)
        t_gt   = cam_gt[..., 4:7]
        focal_gt = cam_gt[..., 7]

        q      = matrix_to_quaternion(R_pred.reshape(-1, 3, 3))
        q_gt   = matrix_to_quaternion(R_gt.reshape(-1, 3, 3))

        dot    = torch.clamp(torch.sum(q * q_gt, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        rot_loss   = (1 - torch.abs(dot)).mean()
        trans_loss = torch.norm(t_pred - t_gt, dim=-1).pow(2).mean()

        fov_pred = 2*torch.atan(self.img_size[1]/(2*focal_pred))
        fov_gt   = 2*torch.atan(self.img_size[1]/(2*focal_gt))
        fov_loss = F.mse_loss(fov_pred, fov_gt)

        return rot_loss + trans_loss + fov_loss