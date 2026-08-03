"""Losses for PoseFusionModule (train_rich_v2.py).

preds = (pose_aggr,)
  preds[0] : (B, T, P, J, 6)  — fused 6-D body pose

All losses are self-contained; no imports from loss.py.
"""

from __future__ import annotations

import logging
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbp_path = os.path.join(project_root, "human_body_prior")
if hbp_path not in sys.path:
    sys.path.insert(0, hbp_path)

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix

from configuration import CONFIG
from utilities.smplx_utilities import get_smplx_joints

logger = logging.getLogger(__name__)


class Loss(ABC):
    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def forward(self, preds: tuple, targets: dict) -> torch.Tensor: ...

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"


# ─────────────────────────────────────────────────────────────────────────────
# Pose
# ─────────────────────────────────────────────────────────────────────────────

class PoseMSELoss(Loss):
    """Geodesic rotation loss on the predicted 6-D body pose.

    trace(R_pred^T @ R_gt) gives cos(θ) up to a scale; loss = (3 - trace) / 2
    equals (1 - cosθ), range [0, 2].  No acos → no NaN on degenerate inputs.
    """

    # CHANGE 1 — per-group joint weights.
    #
    # Rationale: JointPositionLoss supervises POSITIONS after forward kinematics,
    # so a joint's influence is implicitly scaled by how many descendants it has.
    # The hand chains hold 30 of the 55 joints and absorbed ~70% of that loss's
    # magnitude, while the trunk got ~2%. Restricting L_joint to body joints
    # removes hand supervision from the position term, so it has to come back
    # here: the geodesic loss weights every joint equally by construction, which
    # makes it the right place to keep hands supervised.
    #
    # LAYOUT — indices below are in pose_aggr space (PACKED, root ALREADY
    # stripped), verified by scripts/verify_joint_layout.py:
    #     body  0..20   (SMPL-X 1..21)
    #     hands 21..50  (SMPL-X packed 22..51 = canonical 25..54)
    #     face  51..53  (jaw, left eye, right eye)
    # NOT the canonical order, where hands are 25..54 and face is 22..24.
    _BODY_SLICE  = slice(0, 21)
    _HANDS_SLICE = slice(21, 51)
    _FACE_SLICE  = slice(51, 54)

    def __init__(
        self,
        name: str = "Pose MSE Loss",
        weight: float = 1.0,
        body_only: bool = False,
        body_weight: float = 1.0,
        hand_weight: float = 1.0,
        face_weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)
        self.body_only   = body_only
        self.body_weight = body_weight
        self.hand_weight = hand_weight
        self.face_weight = face_weight
        self._weighted = not (body_weight == hand_weight == face_weight == 1.0)
        self._ratio_logged = False

    def _joint_weights(self, J: int, device, dtype) -> torch.Tensor:
        """(J,) per-joint weight vector in pose_aggr order, mean-normalised.

        Normalising by the mean keeps the loss magnitude comparable to the
        unweighted version, so lambda_pose does not have to be retuned.
        """
        w = torch.full((J,), float(self.body_weight), device=device, dtype=dtype)
        if J > self._HANDS_SLICE.start:
            w[self._HANDS_SLICE] = self.hand_weight
        if J > self._FACE_SLICE.start:
            w[self._FACE_SLICE] = self.face_weight
        return w / w.mean().clamp(min=1e-8)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        pose_aggr    = preds[0]         # (B, T, P, 54, 6)  — root stripped
        gt_body_pose = targets["pose"]  # (B, T, P, 55, 6)  — includes root

        # Strip root from GT to match pose_aggr which has root removed.
        gt_body_pose = gt_body_pose[..., 1:, :]

        if self.body_only:
            pose_aggr    = pose_aggr[..., :21, :]
            gt_body_pose = gt_body_pose[..., :21, :]

        if "gt_valid" in targets:
            mask = targets["gt_valid"]  # (B, T, P)
            if not mask.any():
                return pose_aggr.new_zeros([])
        else:
            mask = None

        shape  = pose_aggr.shape[:-1]
        R_pred = rotation_6d_to_matrix(pose_aggr.reshape(-1, 6)).reshape(*shape, 3, 3)
        R_gt   = rotation_6d_to_matrix(gt_body_pose.reshape(-1, 6)).reshape(*shape, 3, 3)
        trace  = torch.einsum("...ij,...ij->...", R_pred, R_gt)     # (B, T, P, J)
        loss_per_joint = (3.0 - trace).clamp(min=0.0) / 2.0

        # CHANGE 1 — per-joint weights, broadcast over (B, T, P) and applied as a
        # WEIGHTED MEAN over the selected entries. body_only already dropped the
        # non-body joints, so weighting is a no-op in that case.
        if self._weighted and not self.body_only:
            J = loss_per_joint.shape[-1]
            jw = self._joint_weights(J, loss_per_joint.device, loss_per_joint.dtype)
            jw = jw.expand_as(loss_per_joint)
        else:
            jw = None

        def _reduce(sel: torch.Tensor) -> torch.Tensor:
            """Weighted mean of loss_per_joint over the boolean selection `sel`."""
            if jw is None:
                return loss_per_joint[sel].mean()
            num = (loss_per_joint * jw)[sel].sum()
            den = jw[sel].sum().clamp(min=1e-8)
            out = num / den
            if not self._ratio_logged:
                self._ratio_logged = True
                plain = loss_per_joint[sel].mean()
                logger.info(
                    f"[{self.name}] joint weighting body={self.body_weight} "
                    f"hands={self.hand_weight} face={self.face_weight} -> "
                    f"weighted/unweighted = {(out / plain.clamp(min=1e-12)).item():.4f} "
                    f"(first batch; lambda_pose does not need retuning if this is ~1)"
                )
            return out

        if "joint_mask" in targets:
            jm  = targets["joint_mask"]   # (B, T, K, P, J)
            obs = jm.any(dim=2)           # (B, T, P, J)
            if mask is not None:
                obs = obs & mask.unsqueeze(-1)
            if not obs.any():
                return pose_aggr.new_zeros([])
            return _reduce(obs)

        if mask is not None:
            m = mask.unsqueeze(-1).expand_as(loss_per_joint)
            return _reduce(m)
        return _reduce(torch.ones_like(loss_per_joint, dtype=torch.bool))


class TemporalSmoothnessLoss(Loss):
    """Penalises angular acceleration of the fused pose, measured on SO(3).

    The decoder emits an unnormalised 6-D vector and 6-D -> SO(3) (Gram-Schmidt)
    is many-to-one: scaling the vector leaves the rotation unchanged.  A finite
    difference taken in 6-D space is therefore gauge-dependent and can be driven
    to zero by shrinking the decoder output without altering a single rotation.
    We convert to rotation matrices first and penalise the geodesic discrepancy
    between consecutive relative rotations delta_t = R_t @ R_{t-1}^T, which is
    invariant to that gauge.
    """

    def __init__(self, name: str = "Temporal Smoothness Loss", weight: float = 1.0) -> None:
        super().__init__(name, weight)

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        pose_aggr = preds[0]                  # (B, T, P, J, 6)
        if pose_aggr.shape[1] < 3:
            return pose_aggr.new_zeros([])

        shape = pose_aggr.shape[:-1]
        R = rotation_6d_to_matrix(pose_aggr.reshape(-1, 6)).reshape(*shape, 3, 3)

        # relative rotation between consecutive frames  (B, T-1, P, J, 3, 3)
        delta = torch.einsum("btpjxy,btpjzy->btpjxz", R[:, 1:], R[:, :-1])
        # change of angular velocity: geodesic between consecutive deltas
        trace = torch.einsum("...ij,...ij->...", delta[:, 1:], delta[:, :-1])
        accel = (3.0 - trace).clamp(min=0.0) / 2.0     # (B, T-2, P, J)

        gt_valid = targets.get("gt_valid")
        if gt_valid is not None:
            v = gt_valid[:, : pose_aggr.shape[1]]
            # a triplet counts only if all three of its frames are annotated,
            # so smoothness is never enforced across an annotation gap
            m = (v[:, 2:] & v[:, 1:-1] & v[:, :-2]).unsqueeze(-1).expand_as(accel)
            if not m.any():
                return pose_aggr.new_zeros([])
            return accel[m].mean()
        return accel.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Joint positions (FK-based)
# ─────────────────────────────────────────────────────────────────────────────

class JointPositionLoss(Loss):
    """FK-based joint position loss.

    Runs SMPL-X FK on pose_aggr (with GT betas) and supervises against
    precomputed GT joint positions (targets["gt_joints"]).  Both sides are
    made root-relative so the loss is translation-invariant.

    Parameters
    ----------
    body_only  : if True, supervise only body joints 0-21 (skip hands/face).
    chunk_size : number of time steps per SMPL-X call (memory bound).
    """

    def __init__(
        self,
        name: str = "Joint Position Loss",
        weight: float = 1.0,
        body_only: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__(name, weight)
        self.body_only  = body_only
        self.chunk_size = chunk_size

    def forward(self, preds: tuple, targets: dict) -> torch.Tensor:
        if "gt_joints" not in targets:
            return preds[0].new_zeros([])

        pose_aggr = preds[0]               # (B, T, P, J, 6)
        gt_joints = targets["gt_joints"]   # (B, T, P, 55, 3)

        B, T, P, J, _ = pose_aggr.shape

        gt_valid = targets.get("gt_valid")
        if gt_valid is not None and not gt_valid.any():
            return pose_aggr.new_zeros([])

        # PoseFusionModule outputs J=54 (root stripped). Prepend GT global_orient so
        # both pred and GT FK share the same root frame — after root-relative subtraction
        # the global rotation cancels and we compare body pose only.
        gt_root = targets["pose"][..., :1, :].float()[:, :T]  # (B, T, P, 1, 6)
        pose_full = torch.cat([gt_root, pose_aggr], dim=3)    # (B, T, P, 55, 6)

        betas = targets["shape"].float()[:, :T]

        chunks = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            chunk_joints = get_smplx_joints(
                pose_full[:, t0:t1],
                betas[:, t0:t1],
            )
            chunks.append(chunk_joints[..., :55, :])
        pred_joints = torch.cat(chunks, dim=1)   # (B, T, P, 55, 3)

        pred_rel = pred_joints - pred_joints[..., :1, :]
        gt_rel   = gt_joints   - gt_joints[..., :1, :]

        if self.body_only:
            pred_rel = pred_rel[..., :22, :]
            gt_rel   = gt_rel[..., :22, :]

        if gt_valid is not None:
            mask = gt_valid.unsqueeze(-1).unsqueeze(-1).expand_as(pred_rel)
            if not mask.any():
                return pose_aggr.new_zeros([])
            return F.mse_loss(pred_rel[mask], gt_rel[mask])

        return F.mse_loss(pred_rel, gt_rel)


# ─────────────────────────────────────────────────────────────────────────────
# VPoser prior
# ─────────────────────────────────────────────────────────────────────────────

class VPoserLoss(Loss):
    """KL-divergence prior from a pretrained VPoser body model.

    Encodes the predicted body pose (joints 1-21) with VPoser and penalises
    deviation from the N(0, I) prior.
    """

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
        pose_aggr = preds[0]   # (B, T, P, J, 6)

        if next(self.vposer.parameters()).device != pose_aggr.device:
            self.vposer = self.vposer.to(pose_aggr.device)

        from utilities.smplx_utilities import _rot_matrix_to_axis_angle_safe
        rot_mat  = rotation_6d_to_matrix(pose_aggr)       # (B, T, P, J, 3, 3)
        pose_aa  = _rot_matrix_to_axis_angle_safe(rot_mat)  # (B, T, P, J, 3)

        # VPoser expects the 21 SMPL-X body joints (1-21), flattened to 63 dims.
        # PoseFusionModule returns J=54 with the root already stripped, so index i
        # there is SMPL-X joint i+1; accept the un-stripped J=55 layout as well.
        J = pose_aa.shape[-2]
        if J == 55:
            body = pose_aa[..., 1:22, :]   # root at index 0
        elif J == 54:
            body = pose_aa[..., :21, :]    # root already stripped
        else:
            raise ValueError(
                f"VPoserLoss expects 54 (root-stripped) or 55 joints, got {J}"
            )
        smpl_pose = body.reshape(-1, 63).float()

        # Drop (frame, person) slots without GT annotation: padded person slots and
        # unannotated frames carry no supervision, so the prior would be the only
        # gradient acting there.
        gt_valid = targets.get("gt_valid")
        if gt_valid is not None:
            keep = gt_valid[:, : pose_aggr.shape[1]].reshape(-1)
            if not keep.any():
                return pose_aggr.new_zeros([])
            smpl_pose = smpl_pose[keep]

        kl_chunks = []
        for i in range(0, smpl_pose.shape[0], self.chunk_size):
            chunk = smpl_pose[i : i + self.chunk_size]
            dist  = self.vposer.encode(chunk)
            mean, std = dist.mean, dist.scale
            logvar = 2.0 * torch.log(std + 1e-8)
            kl = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=-1)
            kl_chunks.append(kl)

        return torch.cat(kl_chunks).mean()
