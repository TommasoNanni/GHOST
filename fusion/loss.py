"""Loss base class for fusion training."""

from __future__ import annotations
import logging
import sys
import os

# Set to True to print memory/shape diagnostics without running full training
DEBUG_MEMORY = os.environ.get("GHOST_DEBUG_MEMORY", "0") == "1"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbp_path = os.path.join(project_root, 'human_body_prior')
if hbp_path not in sys.path:
    sys.path.insert(0, hbp_path)

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix

from utilities.camera_utilities import extract_cameras
from utilities.geometry import project_to_2d, skew_symmetric
from configuration import CONFIG
from utilities.smplx_utilities import PARENTS_TABLE, get_smplx_joints

logger = logging.getLogger(__name__)

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
        self,
        name: str = "Epipolar Loss",
        weight: float = 1.0,
        img_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__(name, weight)
        self.img_size = img_size

    def forward(
            self,
            preds: tuple,
            targets: dict,
        ) -> torch.Tensor:
        """
        Computes the epipolar loss using world-frame predictions.

        Joints are computed from pose_aggr + shape_aggr in world frame,
        translated by body_transl_world to absolute world positions, then projected
        into each camera's frame using the predicted camera extrinsics.

        Processes T in chunks of `chunk_size` to avoid OOM when running SMPL-X
        on long sequences.

        preds: (pose_aggr, shape_aggr, camera, body_transl_world)
        """
        pose_aggr, _, camera, _ = preds[:4]
        K = camera.shape[1]  # camera is (B, K, 8) — static
        if K < 2:
            return pose_aggr.new_zeros([])
        B, T, P = pose_aggr.shape[:3]

        kp2d_obs = targets.get("kp2d")
        if kp2d_obs is None:
            return pose_aggr.new_zeros([])

        num_pairs = 0
        total_loss = pose_aggr.new_zeros([])
        img_diag2 = float(self.img_size[0] ** 2 + self.img_size[1] ** 2)

        for i in range(K):
            for j in range(i + 1, K):
                # camera[:, i] is (B, 8) — static extrinsic for camera i.
                R_i, t_i, K_i = extract_cameras(camera[:, i], self.img_size)  # (B,3,3),(B,3),(B,3,3)
                R_j, t_j, K_j = extract_cameras(camera[:, j], self.img_size)
                # Unsqueeze T=1 so compute_fundamental_matrix gets its expected (B, T, ...) inputs.
                # The result is (B, 1, 3, 3) — same F for every frame since cameras are static.
                F_static = self.compute_fundamental_matrix(
                    R_i.unsqueeze(1), t_i.unsqueeze(1),
                    R_j.unsqueeze(1), t_j.unsqueeze(1),
                    K_i.unsqueeze(1), K_j.unsqueeze(1),
                )  # (B, 1, 3, 3)
                F = F_static.expand(-1, T, -1, -1)  # (B, T, 3, 3)

                # Observed 2D keypoints from SAM3D (pixel space) — independent of
                # the model's own camera/pose estimates, so the constraint is real.
                obs_i = kp2d_obs[:, :, i]   # (B, T, P, 70, 2)
                obs_j = kp2d_obs[:, :, j]   # (B, T, P, 70, 2)

                # Valid: detected in both cameras (undetected frames have kp2d=0).
                valid = (obs_i.abs().sum(-1) > 0) & (obs_j.abs().sum(-1) > 0)  # (B, T, P, 70)
                if not valid.any():
                    continue

                ones = obs_i.new_ones(*obs_i.shape[:-1], 1)
                x_i = torch.cat([obs_i, ones], dim=-1)   # (B, T, P, 70, 3) homogeneous
                x_j = torch.cat([obs_j, ones], dim=-1)

                errors = self.compute_epipolar_errors(x_i, x_j, F)   # (B, T, P, 70)
                safe_errors = torch.where(valid, errors, errors.new_zeros([]))
                total_loss = total_loss + (
                    safe_errors.sum() / img_diag2
                    / (valid.sum() + 1e-8)
                )
                num_pairs += 1

        if num_pairs == 0:
            return pose_aggr.new_zeros([])
        result = total_loss / num_pairs
        if not result.isfinite():
            logger.warning(f"[EpipolarLoss] loss is non-finite: {result.item()}")
        return result


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

        # Focal length is known from calibration — K must not receive gradients through
        # torch.inverse (its backward is -(K^{-1} ⊗ K^{-1}), which produces NaN when
        # focal is near zero early in training).  Detach before inverting so that
        # gradients for focal flow only via CameraMSELoss, not through F.
        K_i_inv = torch.inverse(K_i.detach().reshape(B*T,3,3).float()).to(K_i.dtype).reshape(B,T,3,3)
        K_j_inv_T = torch.inverse(K_j.detach().reshape(B*T,3,3).float()).to(K_j.dtype).transpose(-2,-1).reshape(B,T,3,3)

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
        Computes the Sampson distance between corresponding points under F.

        The Sampson distance is a first-order approximation to the geometric
        reprojection error and is self-normalised by the epipolar line norms,
        so it does not blow up with large focal lengths the way the raw
        algebraic error (x_j^T F x_i) does.

        Parameters
        ----------
        x_i, x_j : (B, T, P, V, 3) — homogeneous 2D points
        F : (B, T, 3, 3) — fundamental matrix

        Returns
        -------
        error : (B, T, P, V) — Sampson distance per point (always >= 0)
        """
        B, T, P, V = x_i.shape[:4]

        x_i_flat = x_i.reshape(B*T*P*V, 3, 1)
        x_j_flat = x_j.reshape(B*T*P*V, 1, 3)
        F_expanded = F.reshape(B,T,1,1,3,3).expand(B,T,P,V,3,3).reshape(B*T*P*V, 3, 3)

        # Algebraic error: scalar per point
        algebraic = torch.bmm(
            torch.bmm(x_j_flat, F_expanded),
            x_i_flat,
        ).reshape(B*T*P*V)                              # (N,)

        # Epipolar lines: F x_i and F^T x_j
        l_j = torch.bmm(F_expanded, x_i_flat).reshape(B*T*P*V, 3)   # (N, 3)
        l_i = torch.bmm(F_expanded.transpose(-2,-1), x_j_flat.transpose(-2,-1)).reshape(B*T*P*V, 3)  # (N, 3)

        # Sampson denominator: sum of squared norms of the first two coords of each line
        denom = l_j[:, 0]**2 + l_j[:, 1]**2 + l_i[:, 0]**2 + l_i[:, 1]**2 + 1e-8

        sampson = (algebraic ** 2) / denom              # (N,)
        return sampson.reshape(B, T, P, V)

class ShapeRegularizationLoss(Loss):
    """L2 regularization on predicted SMPL-X betas (pushes toward mean shape).

    Prevents the shape head from memorizing per-subject betas on the training
    set, which causes val/shape to remain high while train/shape collapses to 0.
    """

    def __init__(self, name: str = "Shape Regularization Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        _, shape_aggr, _, _ = preds[:4]   # (B, P, 10)
        return shape_aggr.pow(2).mean()


class TranslationSmoothnessLoss(Loss):
    """Acceleration smoothness on the aggregated world-frame root translation.

    Mirrors TemporalSmoothnessLoss but operates on body_transl_world (preds[3])
    instead of pose, encouraging the predicted trajectory to be physically smooth.
    """

    def __init__(self, name: str = "Translation Smoothness Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        body_transl_world = preds[3]   # (B, T, P, 3)
        if body_transl_world.shape[1] < 3:
            return body_transl_world.new_zeros([])
        accel = body_transl_world[:, 2:] - 2 * body_transl_world[:, 1:-1] + body_transl_world[:, :-2]
        return accel.pow(2).mean()


class PoseMSELoss(Loss):
    """Geodesic rotation loss on the predicted 6-D body pose.

    Converts both predicted and GT 6-D rotations to SO(3) matrices via
    Gram-Schmidt (rotation_6d_to_matrix), then computes the trace-based
    chordal distance:

        loss = (3 - trace(R_pred^T @ R_gt)) / 2

    This equals (1 - cos θ) where θ is the geodesic angle between the two
    rotations — 0 when identical, 1 when 90° apart, 2 when 180° apart.
    Gradient always points along the geodesic on SO(3), unlike 6-D MSE whose
    gradient direction can diverge from the shortest rotation path.

    NaN safety: uses only matrix multiply + trace — no acos, no sqrt on
    potentially-zero inputs. rotation_6d_to_matrix uses F.normalize with
    eps=1e-8 internally, so degenerate 6-D inputs give large-but-finite
    gradients. Result is clamped to [0, 2] against floating-point drift.
    """

    def __init__(self, name: str = "Pose MSE Loss", weight: float = 1.0, body_only: bool = False) -> None:
        super().__init__(name, weight)
        # When True, supervise only root + body joints (0-21) and ignore hands/face.
        self.body_only = body_only

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        pose_aggr    = preds[0]         # (B, T, P, J, 6)
        gt_body_pose = targets["pose"]  # (B, T, P, J, 6)

        if self.body_only:
            pose_aggr    = pose_aggr[..., 1:, :]
            gt_body_pose = gt_body_pose[..., 1:, :]

        if "gt_valid" in targets:
            mask = targets["gt_valid"]  # (B, T, P)
            if not mask.any():
                print("[LOSS] PoseMSELoss: gt_valid all-False → disconnected zero")
                return pose_aggr.new_zeros([])
        else:
            mask = None

        shape = pose_aggr.shape[:-1]    # (B, T, P, J)
        R_pred = rotation_6d_to_matrix(pose_aggr.reshape(-1, 6)).reshape(*shape, 3, 3)
        R_gt   = rotation_6d_to_matrix(gt_body_pose.reshape(-1, 6)).reshape(*shape, 3, 3)

        # trace(R_pred^T @ R_gt) — batched via einsum, avoids explicit matmul reshape
        trace = torch.einsum("...ij,...ij->...", R_pred, R_gt)          # (B, T, P, J)
        loss_per_joint = (3.0 - trace).clamp(min=0.0) / 2.0            # (B, T, P, J)

        # Build (B, T, P, J) mask: a joint is supervised only if at least one camera
        # observed it (joint_mask > 0 across the K camera dimension).
        # inputs["joint_mask"] is (B, T, K, P, J); targets carries it as "joint_mask".
        if "joint_mask" in targets:
            jm = targets["joint_mask"]           # (B, T, K, P, J)
            obs = jm.any(dim=2)                  # (B, T, P, J) — any camera observed it
            if mask is not None:
                obs = obs & mask.unsqueeze(-1)
            if not obs.any():
                print("[LOSS] PoseMSELoss: joint_mask all-False → disconnected zero")
                return pose_aggr.new_zeros([])
            return loss_per_joint[obs].mean()

        if mask is not None:
            m = mask.unsqueeze(-1).expand_as(loss_per_joint)            # (B, T, P, J)
            return loss_per_joint[m].mean()
        return loss_per_joint.mean()


class ShapeMSELoss(Loss):
    def __init__(self, name: str = "Shape MSE Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        # shape_aggr is (B, P, 10); target is (B, T, P, 10)
        _, shape_aggr, _, _ = preds[:4]
        shape_gt = targets["shape"]                # (B, T, P, 10)
        if "gt_valid" in targets:
            mask = targets["gt_valid"]             # (B, T, P)
            if not mask.any():
                return shape_aggr.new_zeros([])
            # average GT shape only over valid frames per person
            mask_f = mask.to(shape_gt.dtype).unsqueeze(-1)    # (B, T, P, 1)
            shape_gt_mean = (shape_gt * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            shape_gt_mean = shape_gt.mean(dim=1)   # (B, P, 10)
        return F.mse_loss(shape_aggr, shape_gt_mean)

class TemporalSmoothnessLoss(Loss):
    def __init__(
        self, name: str = "Temporal smoothness loss", weight: float = 1.0
    )-> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """Acceleration smoothness on the aggregated pose stream (B, T, P, J, 6)."""
        pose_aggr, _, _, _ = preds[:4]
        if pose_aggr.shape[1] < 3:
            return pose_aggr.new_zeros([])
        current   = pose_aggr[:, 2:]
        previous1 = pose_aggr[:, 1:-1]
        previous2 = pose_aggr[:, :-2]
        accel = current - 2 * previous1 + previous2
        return (accel ** 2).mean()

class VPoserLoss(Loss):
    def __init__(
        self,
        name: str = "VPoser Loss",
        weight: float = 1.0,
        chunk_size: int = 256,
    ) -> None:
        super().__init__(name, weight)
        self.chunk_size = chunk_size

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

        Converts pose_aggr from 6D rotation to axis-angle, then encodes with VPoser
        in chunks of `chunk_size` rows to avoid OOM on long sequences.

        Returns
        -------
        Scalar KL loss: KL( q(z|x) ‖ N(0,I) )
        """
        pose_aggr, _, _, _ = preds[:4]
        if next(self.vposer.parameters()).device != pose_aggr.device:
            self.vposer = self.vposer.to(pose_aggr.device)
        # 6D rotation → rotation matrix → axis-angle (safe: atan2 + safe norm, no acos/sqrt(0) singularity)
        from utilities.smplx_utilities import _rot_matrix_to_axis_angle_safe
        rot_mat = rotation_6d_to_matrix(pose_aggr)                           # (B, T, P, J, 3, 3)
        pose = _rot_matrix_to_axis_angle_safe(rot_mat)                       # (B, T, P, J, 3)

        # slice body joints 1-22 (skip root, skip hands/face), flatten to (N, 21*3=63)
        smpl_pose = pose[..., 1:22, :].reshape(-1, 63).float()  # VPoser is float32

        kl_chunks = []
        for i in range(0, smpl_pose.shape[0], self.chunk_size):
            chunk = smpl_pose[i : i + self.chunk_size]
            dist = self.vposer.encode(chunk)
            mean   = dist.mean
            std    = dist.scale
            logvar = 2.0 * torch.log(std + 1e-8)
            kl = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=-1)
            kl_chunks.append(kl)

        return torch.cat(kl_chunks).mean()


class BoneLengthconsistencyLoss(Loss):
    def __init__(
        self,
        name: str = "Bone Lenght Consistency Loss",
        weight: float = 1.0,
    )-> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """
        Penalises inconsistent bone lengths across time in the fused prediction.

        Uses precomputed joints_world from preds[8] (set by Trainer._append_smplx_joints).

        pose_aggr : (B, T, P, J, 6)
        shape_aggr: (B, T, P, 10)
        joints    : (B, T, P, J, 3)  — no K dimension (already fused)

        Returns
        -------
        Scalar loss encouraging constant bone length across time.
        """
        pose_aggr = preds[0]
        B, T, P = pose_aggr.shape[:3]

        # joints_world precomputed once per step in Trainer._append_smplx_joints
        # bone lengths are global-translation invariant, so world-frame joints are fine
        joints = preds[8][..., :55, :]     # (B, T, P, 55, 3)

        parent_mapping = torch.tensor(
            PARENTS_TABLE, device=joints.device, dtype=torch.long
        )
        parents = parent_mapping[0, :]
        parent_joints = joints[..., parents, :]                       # (B, T, P, J, 3)

        bone_lengths = (joints - parent_joints).pow(2).sum(-1).add(1e-8).sqrt()  # (B, T, P, J) — safe at root (zero-vector)
        indices = torch.arange(len(parents), device=joints.device)
        mask = parents != indices                                      # (55,) — False for root joints
        valid_lengths = bone_lengths[..., mask]                       # (B, T, P, J_valid)

        # check consistency across T only (no K dimension after fusion)
        lengths_to_persist = valid_lengths.permute(0, 2, 3, 1).contiguous()  # (B, P, J_valid, T)
        bone_std = torch.std(lengths_to_persist, dim=-1) + 1e-6
        return bone_std.mean()




class TriangulationLoss(Loss):
    """Cross-camera consistency loss on the spatial stream outputs.

    The model produces one world-frame estimate of root orientation and
    translation per camera, before aggregation (preds[4] and preds[5]).
    Ideally all K estimates should agree — this loss penalises pairwise
    disagreement, enforcing geometric consistency across views.

    For translation: MSE between every pair of world-frame estimates,
    normalised by the squared scene scale so the loss is unit-free.

    For root orientation: geodesic distance between every pair of world-frame
    rotation matrices (1 - |q_k · q_j|), analogous to CameraMSELoss.
    """

    def __init__(
        self,
        name: str = "Triangulation Loss",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        """
        preds[4] : body_orient_world_per_cam  (B, T, K, P, 6) — per-camera world-frame root orient
        preds[5] : body_transl_world_per_cam  (B, T, K, P, 3) — per-camera world-frame translation
        """
        if len(preds) < 6:
            return preds[0].new_zeros([])

        body_orient_world_per_cam  = preds[4]   # (B, T, K, P, 6)
        body_transl_world_per_cam  = preds[5]   # (B, T, K, P, 3)
        person_visible = preds[6].bool() if len(preds) > 6 else None  # (B, T, K, P)
        K = body_transl_world_per_cam.shape[2]

        if K < 2:
            return body_transl_world_per_cam.new_zeros([])

        trans_loss = body_transl_world_per_cam.new_zeros([])
        rot_loss   = body_transl_world_per_cam.new_zeros([])
        num_pairs  = 0

        from pytorch3d.transforms import rotation_6d_to_matrix

        for i in range(K):
            for j in range(i + 1, K):
                # Mask: person must be visible in both camera i and camera j.
                if person_visible is not None:
                    mask = (person_visible[:, :, i] & person_visible[:, :, j]).to(body_transl_world_per_cam.dtype)  # (B, T, P)
                    n = mask.sum().clamp(min=1e-8)
                else:
                    mask = None
                    n = float(body_transl_world_per_cam[:, :, i].shape[0] * body_transl_world_per_cam[:, :, i].shape[1] * body_transl_world_per_cam[:, :, i].shape[2])

                diff_t = body_transl_world_per_cam[:, :, i] - body_transl_world_per_cam[:, :, j]    # (B, T, P, 3)
                if mask is not None:
                    trans_loss = trans_loss + (diff_t.pow(2).sum(-1) * mask).sum() / n
                else:
                    trans_loss = trans_loss + diff_t.pow(2).mean()

                identity_6d = body_orient_world_per_cam.new_tensor([1., 0., 0., 0., 1., 0.])
                def _safe_6d(x6d, vis):  # vis: (B, T, P)
                    absent = (vis == 0).unsqueeze(-1).expand_as(x6d)
                    return torch.where(absent, identity_6d.expand_as(x6d), x6d)
                R_i = rotation_6d_to_matrix(_safe_6d(body_orient_world_per_cam[:, :, i], person_visible[:, :, i] if person_visible is not None else torch.ones_like(body_orient_world_per_cam[:, :, i, 0])).reshape(-1, 6))
                R_j = rotation_6d_to_matrix(_safe_6d(body_orient_world_per_cam[:, :, j], person_visible[:, :, j] if person_visible is not None else torch.ones_like(body_orient_world_per_cam[:, :, j, 0])).reshape(-1, 6))
                # Trace-based geodesic: cos(theta) = (trace(R_i @ R_j^T) - 1) / 2.
                # The old code used matrix_to_quaternion whose sqrt gradient is 1/(2√x) → ∞
                # at near-identity rotations (common early in training).
                # The gradient of trace(R_i @ R_j^T) w.r.t. R_i is R_j — constant, no singularity.
                R_rel     = torch.bmm(R_i, R_j.transpose(-2, -1))
                cos_angle = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
                geodesic  = (1 - cos_angle) / 2   # in [0, 1]; 0 = identical, 1 = 180° apart
                if mask is not None:
                    rot_loss = rot_loss + (geodesic * mask.reshape(-1)).sum() / n
                else:
                    rot_loss = rot_loss + geodesic.mean()

                num_pairs += 1

        result = (trans_loss + rot_loss) / num_pairs
        if not result.isfinite():
            logger.warning(f"[TriangulationLoss] loss is non-finite: {result.item()}")
        return result


class TranslationMSELoss(Loss):
    """Supervise per-camera world-frame root translation against GT world translation.

    Compares ``body_transl_world_per_cam`` (preds[5], (B,T,K,P,3)) — the
    per-camera back-projected world translation — against ``targets["trans"]``
    expanded over K cameras.

    Both sides are in world frame, so there is no frame mismatch.
    Gradient flows through the back-projection:

        body_transl_world_per_cam[k] = (body_transl_cam[k] - cam_transl_w2c[k]) @ cam_rot_w2c[k]

    to the translation head (body_transl_cam), and to predicted camera rotation
    and translation (cam_rot_w2c, cam_transl_w2c) for all K cameras.
    """

    def __init__(self, name: str = "Translation world-frame loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        body_transl_world_per_cam = preds[5]   # (B, T, K, P, 3) — world frame, per camera
        person_visible            = preds[6]   # (B, T, K, P)
        gt_transl_world           = targets["trans"]   # (B, T, P, 3)
        cam_gt                    = targets["camera"]  # (B, K, 8) — static cameras

        gt_valid = targets.get("gt_valid")  # (B, T, P) bool or None

        B, T, K, P, _ = body_transl_world_per_cam.shape

        # Identify cameras with real GT (non-zero quaternion sentinel).
        cam_valid = cam_gt[..., :4].norm(dim=-1) > 0.5  # (B, K)

        # Expand GT world translation over K cameras.
        gt_transl_world_exp = gt_transl_world.unsqueeze(2).expand(B, T, K, P, 3)

        # Build mask: camera must have GT, person must be visible, frame must have GT annotation.
        vis_mask = person_visible > 0
        # cam_valid (B, K) → (B, 1, K, 1) → broadcast over (B, T, K, P)
        vis_mask = vis_mask & cam_valid.unsqueeze(1).unsqueeze(-1).expand_as(vis_mask)
        if gt_valid is not None:
            vis_mask = vis_mask & gt_valid.unsqueeze(2).expand_as(vis_mask)

        if not vis_mask.any():
            print("[LOSS] TranslationMSELoss: vis_mask all-False → disconnected zero")
            return body_transl_world_per_cam.new_zeros([])

        # vis_mask is (B,T,K,P); indexing a (B,T,K,P,3) tensor with it yields (N,3)
        return F.mse_loss(body_transl_world_per_cam[vis_mask], gt_transl_world_exp[vis_mask])


class CameraMSELossVGGT(Loss):
    """Camera loss following VGGT: Huber loss on [quat(4), trans_normalised(3)].

    Quaternion double cover is resolved by flipping the predicted quaternion sign
    so it always lies in the same hemisphere as the GT before computing the loss.
    Translation is normalised by a fixed physical scale (metres) set explicitly in
    config — not computed dynamically from the camera layout. This keeps a 1 m
    error at a predictable normalised magnitude (1/trans_scale) so the gradient
    stays meaningful throughout training.
    """

    def __init__(
        self,
        name: str = "Camera VGGT loss",
        weight: float = 1.0,
        huber_delta: float = 0.1,
        trans_scale: float = 10.0,
    ) -> None:
        super().__init__(name, weight)
        self.huber_delta = huber_delta
        self.trans_scale = trans_scale

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        _, _, camera_pred, _ = preds[:4]
        cam_gt = targets["camera"]

        cam_valid = cam_gt[..., :4].norm(dim=-1) > 0.5  # (B, K) — static cameras
        if not cam_valid.any():
            return camera_pred.new_zeros([])

        camera_pred_v = camera_pred[cam_valid]   # (N, 8)
        cam_gt_v      = cam_gt[cam_valid]         # (N, 8)

        q     = F.normalize(camera_pred_v[..., :4], dim=-1)
        q_gt  = F.normalize(cam_gt_v[..., :4],      dim=-1)

        # Resolve quaternion double cover: flip predicted sign so q · q_gt >= 0.
        dot = (q * q_gt).sum(dim=-1, keepdim=True)
        q   = torch.where(dot < 0, -q, q)

        t_pred_n = camera_pred_v[..., 4:7] / self.trans_scale
        t_gt_n   = cam_gt_v[..., 4:7]      / self.trans_scale

        g_pred = torch.cat([q,    t_pred_n], dim=-1)
        g_gt   = torch.cat([q_gt, t_gt_n],  dim=-1)

        return F.huber_loss(g_pred, g_gt, delta=self.huber_delta)


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

        camera pred : (B, K, 8) — [quat(4), trans(3), focal_raw(1)], static
        camera GT   : (B, K, 8) — same layout

        Returns
        -------
        Scalar loss encouraging adherence of camera parameters to the GT.
        """
        _, _, camera_pred, _ = preds[:4]
        cam_gt = targets["camera"]

        # Only supervise cameras where GT is available (valid quaternion norm > 0.5).
        # Cameras with no person detection stay at zero quaternion in gt_camera.
        cam_valid = cam_gt[..., :4].norm(dim=-1) > 0.5  # (B, K) — static cameras
        if not cam_valid.any():
            return camera_pred.new_zeros([])

        camera_pred_v = camera_pred[cam_valid]  # (N, 8)
        cam_gt_v      = cam_gt[cam_valid]        # (N, 8)

        cam_transl_w2c    = camera_pred_v[..., 4:7]
        gt_cam_rot_w2c    = quaternion_to_matrix(cam_gt_v[..., :4])  # needed for scene_scale below
        gt_cam_transl_w2c = cam_gt_v[..., 4:7]

        # Compare quaternions directly — no matrix round-trip.
        # The old code did quat → R → quat (via quaternion_to_matrix then matrix_to_quaternion)
        # to "clean up" the predicted quaternion, but matrix_to_quaternion uses sqrt whose
        # gradient is 1/(2√x) → ∞ at near-identity rotations.  F.normalize is sufficient.
        q    = F.normalize(camera_pred_v[..., :4], dim=-1)
        q_gt = F.normalize(cam_gt_v[..., :4], dim=-1)

        dot      = torch.clamp(torch.sum(q * q_gt, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        rot_loss = (1 - torch.abs(dot)).mean()

        # Normalise by squared scene scale so trans_loss lives in [0, 1] like rot_loss.
        # gt_cam_rot_w2c: (N, 3, 3), gt_cam_transl_w2c: (N, 3) → centres: (N, 3)
        gt_cam_centres = -torch.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)
        # scene_scale only needs the K camera positions at one time step — using all T*K
        # observations builds an (N, N, 3) tensor with N=T*K which OOMs at large T.
        K = cam_valid.shape[-1]
        centres_one_step = gt_cam_centres[:K]                                      # (K, 3)
        diff = centres_one_step.unsqueeze(-2) - centres_one_step.unsqueeze(-3)     # (K, K, 3)
        scene_scale = diff.pow(2).sum(-1).add(1e-8).sqrt().mean().clamp(min=1e-3).detach()
        trans_loss = (cam_transl_w2c - gt_cam_transl_w2c).pow(2).sum(-1).mean() / (scene_scale ** 2)

        return rot_loss + trans_loss


class JointPositionLoss(Loss):
    """FK-based joint position loss.

    Runs SMPL-X FK on the predicted (pose_aggr, shape_aggr) and supervises
    against GT joint positions precomputed at data-loading time
    (targets["gt_joints"]).  Both sides are made root-relative before the MSE
    so the loss is translation-invariant and directly penalises wrong bone
    lengths caused by poor shape prediction.

    By default only body joints 0-21 are used (``body_only=True``), which
    avoids noise from hand/face joints that many datasets do not annotate well.

    Parameters
    ----------
    chunk_size : int
        Number of time steps processed per SMPL-X call.  Tune to stay within
        GPU memory; 64 is safe for sequences up to T=300 with P=2.
    """

    def __init__(
        self,
        name: str = "Joint Position Loss",
        weight: float = 1.0,
        body_only: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__(name, weight)
        self.body_only = body_only
        self.chunk_size = chunk_size

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        if "gt_joints" not in targets:
            return preds[0].new_zeros([])

        pose_aggr  = preds[0]              # (B, T, P, J, 6)
        shape_aggr = preds[1]              # (B, P, 10)
        gt_joints  = targets["gt_joints"]  # (B, T, P, 55, 3)  precomputed, local frame

        B, T, P, J, _ = pose_aggr.shape

        gt_valid = targets.get("gt_valid")  # (B, T, P) bool or None
        if gt_valid is not None and not gt_valid.any():
            return pose_aggr.new_zeros([])

        # Expand static shape to every time step
        shape_exp = shape_aggr.unsqueeze(1).expand(B, T, P, 10)

        # Run FK on predictions in time chunks to keep memory bounded
        chunks = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            chunk_joints = get_smplx_joints(
                pose_aggr[:, t0:t1],     # (B, t, P, J, 6)
                shape_exp[:, t0:t1],     # (B, t, P, 10)
            )                             # (B, t, P, Jout, 3)
            chunks.append(chunk_joints[..., :55, :])
        pred_joints = torch.cat(chunks, dim=1)  # (B, T, P, 55, 3)

        # Root-relative: subtract joint 0 so the loss is invariant to global translation
        pred_rel = pred_joints - pred_joints[..., :1, :]
        gt_rel   = gt_joints   - gt_joints[..., :1, :]

        if self.body_only:
            pred_rel = pred_rel[..., :22, :]
            gt_rel   = gt_rel[..., :22, :]

        if gt_valid is not None:
            # expand mask to (B, T, P, J_sup, 3)
            mask = gt_valid.unsqueeze(-1).unsqueeze(-1).expand_as(pred_rel)
            if not mask.any():
                return pose_aggr.new_zeros([])
            return F.mse_loss(pred_rel[mask], gt_rel[mask])

        return F.mse_loss(pred_rel, gt_rel)