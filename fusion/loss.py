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

from utilities.smplx_utilities import get_smplx_joints
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
        self,
        name: str = "Epipolar Loss",
        weight: float = 1.0,
        img_size: tuple[int, int] = (224, 224),
        chunk_size: int = 32,
    ) -> None:
        super().__init__(name, weight)
        self.img_size = img_size
        self.chunk_size = chunk_size

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
        pose_aggr, shape_aggr, camera, body_transl_world = preds[:4]
        K = camera.shape[2]
        if K < 2:
            return pose_aggr.new_zeros([])
        B, T, P, J, _ = pose_aggr.shape
        num_pairs = 0
        total_loss = pose_aggr.new_zeros([])

        for i in range(K):
            for j in range(i + 1, K):
                cam_i = camera[:, :, i]   # (B, T, 8)
                cam_j = camera[:, :, j]

                pair_numerator   = pose_aggr.new_zeros([])
                pair_visible_sum = 0

                for t0 in range(0, T, self.chunk_size):
                    t1 = min(t0 + self.chunk_size, T)

                    # Joints in world frame: pelvis-centered → add body_transl_world for absolute position
                    # shape_aggr is (B, P, 10) — expand to (B, t_chunk, P, 10) for get_smplx_joints
                    t_chunk = t1 - t0
                    shape_chunk = shape_aggr.unsqueeze(1).expand(B, t_chunk, P, 10)
                    joints_world = get_smplx_joints(pose_aggr[:, t0:t1], shape_chunk)  # (B, t_chunk, P, J, 3)
                    joints_world = joints_world + body_transl_world[:, t0:t1].unsqueeze(-2)

                    R_i, t_i, K_i = extract_cameras(cam_i[:, t0:t1], self.img_size)
                    R_j, t_j, K_j = extract_cameras(cam_j[:, t0:t1], self.img_size)
                    F = self.compute_fundamental_matrix(R_i, t_i, R_j, t_j, K_i, K_j)

                    # Project world joints into each camera frame: v_cam = R @ v_world + t
                    Jsmplx = joints_world.shape[3]  # SMPL-X outputs more joints than pose_aggr input dim
                    flat = joints_world.reshape(B * t_chunk, P * Jsmplx, 3)
                    vi_cam = (torch.bmm(flat, R_i.reshape(B * t_chunk, 3, 3).transpose(-2, -1))
                              + t_i.reshape(B * t_chunk, 1, 3)).reshape(B, t_chunk, P, Jsmplx, 3)
                    vj_cam = (torch.bmm(flat, R_j.reshape(B * t_chunk, 3, 3).transpose(-2, -1))
                              + t_j.reshape(B * t_chunk, 1, 3)).reshape(B, t_chunk, P, Jsmplx, 3)

                    x_i = project_to_2d(vi_cam, K_i)
                    x_j = project_to_2d(vj_cam, K_j)

                    errors  = self.compute_epipolar_errors(x_i, x_j, F)
                    visible = (vi_cam[..., 2] > 0) & (vj_cam[..., 2] > 0)

                    img_diag2 = float(self.img_size[0] ** 2 + self.img_size[1] ** 2)
                    pair_numerator   = pair_numerator + (errors / img_diag2 * visible.to(errors.dtype)).sum()
                    pair_visible_sum = pair_visible_sum + visible.sum().item()

                total_loss = total_loss + pair_numerator / (pair_visible_sum + 1e-8)
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

        # torch.inverse does not support low-precision dtypes — compute in float32
        K_i_inv = torch.inverse(K_i.reshape(B*T,3,3).float()).to(K_i.dtype).reshape(B,T,3,3)
        K_j_inv_T = torch.inverse(K_j.reshape(B*T,3,3).float()).to(K_j.dtype).transpose(-2,-1).reshape(B,T,3,3)

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

class PoseMSELoss(Loss):
    def __init__(self, name: str = "Pose MSE Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        pose_aggr    = preds[0]                       # (B, T, P, J, 6)
        gt_body_pose = targets["pose"]                # (B, T, P, J, 6)
        if "gt_valid" in targets:
            mask = targets["gt_valid"]             # (B, T, P)
            if not mask.any():
                return pose_aggr.new_zeros([])
            # expand mask to (B, T, P, J, 6) and index
            m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pose_aggr)
            return F.mse_loss(pose_aggr[m], gt_body_pose[m])
        return F.mse_loss(pose_aggr, gt_body_pose)


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
        pose_aggr, _, _, _ = preds[:4]
        if next(self.vposer.parameters()).device != pose_aggr.device:
            self.vposer = self.vposer.to(pose_aggr.device)
        # 6D rotation → rotation matrix → axis-angle
        rot_mat    = rotation_6d_to_matrix(pose_aggr)                        # (B, T, P, J, 3, 3)
        axis_angle = matrix_to_axis_angle(rot_mat.reshape(-1, 3, 3))         # (B*T*P*J, 3)
        pose       = axis_angle.reshape(*pose_aggr.shape[:-1], 3)            # (B, T, P, J, 3)

        # slice body joints 1-22 (skip root, skip hands/face), flatten to (N, 21*3=63)
        smpl_pose = pose[..., 1:22, :].reshape(-1, 63).float()  # VPoser is float32

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
        pose_aggr, shape_aggr, _, _ = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        shape_exp = shape_aggr.unsqueeze(1).expand(B, T, P, 10)
        joints = get_smplx_joints(pose_aggr, shape_exp)[..., :55, :]  # (B, T, P, 55, 3)

        parent_mapping = torch.tensor(
            PARENTS_TABLE, device=joints.device, dtype=torch.long
        )
        parents = parent_mapping[0, :]
        parent_joints = joints[..., parents, :]                       # (B, T, P, J, 3)

        bone_lengths = torch.norm(joints - parent_joints, p=2, dim=-1)  # (B, T, P, J)
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

        from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion

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

                R_i = rotation_6d_to_matrix(body_orient_world_per_cam[:, :, i].reshape(-1, 6))  # (B*T*P, 3, 3)
                R_j = rotation_6d_to_matrix(body_orient_world_per_cam[:, :, j].reshape(-1, 6))
                q_i = matrix_to_quaternion(R_i)
                q_j = matrix_to_quaternion(R_j)
                dot = torch.clamp(torch.sum(q_i * q_j, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
                geodesic = 1 - torch.abs(dot)  # (B*T*P,)
                if mask is not None:
                    rot_loss = rot_loss + (geodesic * mask.reshape(-1)).sum() / n
                else:
                    rot_loss = rot_loss + geodesic.mean()

                num_pairs += 1

        return (trans_loss + rot_loss) / num_pairs


class TranslationMSELoss(Loss):
    """Direct MSE supervision of world-frame body root translation.

    Two terms are summed:
    1. ``body_transl_world`` (preds[3], (B,T,P,3)) vs GT — supervises the
       confidence-weighted mean across cameras.
    2. ``body_transl_world_per_cam`` (preds[5], (B,T,K,P,3)) vs GT — supervises each
       per-camera back-projected translation independently, masked by
       ``person_visible`` (preds[6]).  This forces all cameras to agree
       on the same world-frame position, coupling camera and body
       translation optimisation.

    Frames with no RICH annotation are masked out via ``targets["gt_valid"]``.
    Both terms are normalised by squared scene scale to keep the loss unit-free.
    """

    def __init__(self, name: str = "Translation MSE Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        body_transl_world          = preds[3]   # (B, T, P, 3)    — aggregated world frame
        body_transl_world_per_cam  = preds[5]   # (B, T, K, P, 3) — per-camera world frame
        person_visible             = preds[6]   # (B, T, K, P)    — visibility weights
        gt_body_transl_world       = targets["trans"]   # (B, T, P, 3)

        gt_valid = targets.get("gt_valid")  # (B, T, P) bool or None

        def _masked_mse(pred, gt, mask):
            m = mask.unsqueeze(-1).expand_as(pred)
            if not m.any():
                return pred.new_zeros([])
            return F.mse_loss(pred[m], gt[m])

        # 1. Aggregated translation loss
        if gt_valid is not None:
            if not gt_valid.any():
                return body_transl_world.new_zeros([])
            loss = _masked_mse(body_transl_world, gt_body_transl_world, gt_valid)
        else:
            loss = F.mse_loss(body_transl_world, gt_body_transl_world)

        # 2. Per-camera translation loss
        gt_body_transl_world_per_cam = gt_body_transl_world.unsqueeze(2).expand_as(body_transl_world_per_cam)
        vis_mask = person_visible > 0                              # (B, T, K, P) bool
        if gt_valid is not None:
            vis_mask = vis_mask & gt_valid.unsqueeze(2).expand_as(vis_mask)
        loss = loss + _masked_mse(body_transl_world_per_cam, gt_body_transl_world_per_cam, vis_mask)

        # Normalise by squared scene scale so the loss is unit-free.
        if "camera" in targets:
            cam_gt = targets["camera"]     # (B, T, K, 8)
            gt_cam_rot_w2c   = quaternion_to_matrix(cam_gt[..., :4].reshape(-1, 4)).reshape(*cam_gt.shape[:-1], 3, 3)
            gt_cam_transl_w2c = cam_gt[..., 4:7]
            gt_cam_centres = -torch.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)
            diff = gt_cam_centres.unsqueeze(-2) - gt_cam_centres.unsqueeze(-3)
            scene_scale = diff.norm(dim=-1).mean().clamp(min=1e-3)
            loss = loss / (scene_scale ** 2)

        return loss


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
        _, _, camera_pred, _ = preds[:4]
        cam_gt = targets["camera"]

        cam_rot_w2c    = quaternion_to_matrix(camera_pred[..., :4].reshape(-1, 4)).reshape(*camera_pred.shape[:-1], 3, 3)
        cam_transl_w2c = camera_pred[..., 4:7]
        gt_cam_rot_w2c    = quaternion_to_matrix(cam_gt[..., :4].reshape(-1, 4)).reshape(*cam_gt.shape[:-1], 3, 3)
        gt_cam_transl_w2c = cam_gt[..., 4:7]

        q      = matrix_to_quaternion(cam_rot_w2c.reshape(-1, 3, 3))
        q_gt   = matrix_to_quaternion(gt_cam_rot_w2c.reshape(-1, 3, 3))

        dot    = torch.clamp(torch.sum(q * q_gt, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        rot_loss   = (1 - torch.abs(dot)).mean()

        # Normalise by squared scene scale so trans_loss lives in [0, 1] like rot_loss.
        gt_cam_centres = -torch.einsum("...ji,...j->...i", gt_cam_rot_w2c, gt_cam_transl_w2c)  # (..., K, 3)
        diff = gt_cam_centres.unsqueeze(-2) - gt_cam_centres.unsqueeze(-3)  # (..., K, K, 3)
        scene_scale = diff.norm(dim=-1).mean().clamp(min=1e-3)
        trans_loss = torch.norm(cam_transl_w2c - gt_cam_transl_w2c, dim=-1).pow(2).mean() / (scene_scale ** 2)

        return rot_loss + trans_loss