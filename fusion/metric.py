"""Evaluation metrics for *Humans and Structure from Motion* (HSfM).

Implements the metrics from
*"Reconstructing People, Places, and Cameras"*
(Müller, Choi et al., arXiv 2412.17806, CVPR 2025):

**Human metrics** (all in metres):
    W-MPJPE ↓  — world MPJPE (SE(3)-aligned cameras)
    GA-MPJPE ↓ — group-aligned MPJPE (Sim(3) over all humans)
    PA-MPJPE ↓ — Procrustes-aligned MPJPE (per-human Sim(3))

**Camera metrics**:
    TE ↓       — translation error (m) after SE(3) alignment
    s-TE ↓     — translation error (m) after Sim(3) alignment
    AE ↓       — mean pairwise angle error (°)
    RRA@τ ↑    — relative rotation accuracy (fraction ≤ τ°)
    CCA@τ ↑    — camera centre accuracy (fraction within τ% of scene scale, SE(3))
    s-CCA@τ ↑  — camera centre accuracy (Sim(3))

Every metric subclasses :class:`Metric` with a uniform
``update`` / ``compute`` / ``reset`` interface.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Alignment utilities
# ═══════════════════════════════════════════════════════════════════════════

def _umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment (SE(3) or Sim(3)).

    Finds ``R, t, s`` that minimise ``‖dst − (s·R·src + t)‖²``.

    Parameters
    ----------
    src, dst : ndarray, shape ``(N, 3)``
    with_scale : bool
        If ``True`` solve for ``s`` (Sim(3)); if ``False`` fix ``s = 1`` (SE(3)).

    Returns
    -------
    R : ndarray ``(3, 3)``
    t : ndarray ``(3,)``
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


def _apply_alignment(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """Apply ``s·R·pts + t``."""
    return (s * (pts @ R.T)) + t


def _geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two 3×3 rotation matrices."""
    R_diff = R_a.T @ R_b
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class Metric(ABC):
    """Base class for all HSfM evaluation metrics.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    higher_is_better : bool
        Direction of improvement.
    """

    def __init__(self, name: str, higher_is_better: bool) -> None:
        self.name = name
        self.higher_is_better = higher_is_better
        self._values: List[float] = []

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Accumulate one scene / sample."""

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return aggregated statistics (at least ``"mean"``)."""

    def reset(self) -> None:
        self._values.clear()

    def _record(self, value: float) -> None:
        self._values.append(value)

    def _aggregate(self) -> Dict[str, float]:
        if not self._values:
            return {"mean": float("nan"), "median": float("nan"),
                    "std": float("nan"), "min": float("nan"),
                    "max": float("nan"), "count": 0.0}
        arr = np.asarray(self._values, dtype=np.float64)
        return {
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "count":  float(len(arr)),
        }

    def __repr__(self) -> str:
        d = "↑" if self.higher_is_better else "↓"
        return f"{self.__class__.__name__}(name={self.name!r}, {d})"


# ═══════════════════════════════════════════════════════════════════════════
# Human metrics
# ═══════════════════════════════════════════════════════════════════════════

class WMPJPE(Metric):
    """World Mean Per-Joint Position Error (metres).

    Predicted human meshes are brought into the GT world coordinate system
    via **SE(3)** alignment of *camera positions* (pred → GT), then the
    per-joint Euclidean error is computed.

    Call :meth:`update` once per scene.
    """

    def __init__(self) -> None:
        super().__init__(name="W-MPJPE", higher_is_better=False)

    def update(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_joints : ``(H, J, 3)`` — predicted 3-D joints for *H* humans,
            *J* joints each, in the predicted world frame.
        gt_joints : ``(H, J, 3)`` — ground-truth joints.
        pred_cam_pos : ``(C, 3)`` — predicted camera centres.
        gt_cam_pos : ``(C, 3)`` — ground-truth camera centres.
        """
        R, t, _ = _umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        H, J, _ = pred_joints.shape
        aligned = _apply_alignment(pred_joints.reshape(-1, 3), R, t, 1.0)
        errs = np.linalg.norm(aligned - gt_joints.reshape(-1, 3), axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class GAMPJPE(Metric):
    """Group-Aligned MPJPE (metres).

    All humans' joints in a scene are **concatenated** and then
    **Sim(3)**-aligned to GT jointly.  Measures relative positioning
    among people in the scene.
    """

    def __init__(self) -> None:
        super().__init__(name="GA-MPJPE", higher_is_better=False)

    def update(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_joints : ``(H, J, 3)``
        gt_joints : ``(H, J, 3)``
        """
        src = pred_joints.reshape(-1, 3)
        dst = gt_joints.reshape(-1, 3)
        R, t, s = _umeyama(src, dst, with_scale=True)
        aligned = _apply_alignment(src, R, t, s)
        errs = np.linalg.norm(aligned - dst, axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class PAMPJPE(Metric):
    """Procrustes-Aligned MPJPE (metres).

    Each human is independently **Sim(3)**-aligned to its GT.
    Measures local pose accuracy irrespective of scale & location.
    """

    def __init__(self) -> None:
        super().__init__(name="PA-MPJPE", higher_is_better=False)

    def update(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_joints : ``(H, J, 3)``
        gt_joints : ``(H, J, 3)``
        """
        H = pred_joints.shape[0]
        scene_errs: List[float] = []
        for h in range(H):
            src = pred_joints[h]  # (J, 3)
            dst = gt_joints[h]
            R, t, s = _umeyama(src, dst, with_scale=True)
            aligned = _apply_alignment(src, R, t, s)
            errs = np.linalg.norm(aligned - dst, axis=-1)
            scene_errs.append(float(errs.mean()))
        # Average over humans in this scene
        self._record(float(np.mean(scene_errs)))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


# ═══════════════════════════════════════════════════════════════════════════
# Camera metrics
# ═══════════════════════════════════════════════════════════════════════════

class TranslationError(Metric):
    """Camera translation error (**TE**, metres) after SE(3) alignment.

    Mean Euclidean distance between predicted and GT camera centres
    after a rigid (SE(3)) alignment of the full camera set.
    """

    def __init__(self) -> None:
        super().__init__(name="TE", higher_is_better=False)

    def update(
        self,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_cam_pos, gt_cam_pos : ``(C, 3)``
        """
        R, t, _ = _umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        aligned = _apply_alignment(pred_cam_pos, R, t, 1.0)
        errs = np.linalg.norm(aligned - gt_cam_pos, axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class ScaledTranslationError(Metric):
    """Scale-aligned translation error (**s-TE**, metres) after Sim(3) alignment."""

    def __init__(self) -> None:
        super().__init__(name="s-TE", higher_is_better=False)

    def update(
        self,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_cam_pos, gt_cam_pos : ``(C, 3)``
        """
        R, t, s = _umeyama(pred_cam_pos, gt_cam_pos, with_scale=True)
        aligned = _apply_alignment(pred_cam_pos, R, t, s)
        errs = np.linalg.norm(aligned - gt_cam_pos, axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class AngleError(Metric):
    """Average pairwise camera **Angle Error** (AE, degrees).

    For each pair ``(i, j)`` of cameras, compute the relative rotation
    ``R_ij = R_i @ R_j^T`` for both prediction and ground truth, then
    measure the geodesic distance.  Report the mean over all pairs.
    """

    def __init__(self) -> None:
        super().__init__(name="AE", higher_is_better=False)

    def update(
        self,
        pred_rotations: np.ndarray,
        gt_rotations: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_rotations, gt_rotations : ``(C, 3, 3)``
        """
        C = pred_rotations.shape[0]
        errs: List[float] = []
        for i, j in itertools.combinations(range(C), 2):
            R_rel_pred = pred_rotations[i] @ pred_rotations[j].T
            R_rel_gt = gt_rotations[i] @ gt_rotations[j].T
            errs.append(_geodesic_deg(R_rel_pred, R_rel_gt))
        if errs:
            self._record(float(np.mean(errs)))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class RRA(Metric):
    """Relative Rotation Accuracy (**RRA@τ**).

    Fraction of camera pairs whose pairwise angular error is ≤ *τ* degrees.

    Parameters
    ----------
    threshold : float
        Angular threshold in degrees (e.g. 10 or 15).
    """

    def __init__(self, threshold: float = 15.0) -> None:
        super().__init__(name=f"RRA@{threshold:.0f}", higher_is_better=True)
        self.threshold = threshold

    def update(
        self,
        pred_rotations: np.ndarray,
        gt_rotations: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_rotations, gt_rotations : ``(C, 3, 3)``
        """
        C = pred_rotations.shape[0]
        correct = 0
        total = 0
        for i, j in itertools.combinations(range(C), 2):
            R_rel_pred = pred_rotations[i] @ pred_rotations[j].T
            R_rel_gt = gt_rotations[i] @ gt_rotations[j].T
            err = _geodesic_deg(R_rel_pred, R_rel_gt)
            if err <= self.threshold:
                correct += 1
            total += 1
        if total > 0:
            self._record(correct / total)

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class CCA(Metric):
    """Camera Centre Accuracy (**CCA@τ**) after SE(3) alignment.

    Fraction of cameras whose position error is within *τ*% of the
    overall scene scale (max distance from any GT camera to the GT centroid).

    Parameters
    ----------
    threshold : float
        Percentage threshold (e.g. 10 or 15).
    """

    def __init__(self, threshold: float = 15.0) -> None:
        super().__init__(name=f"CCA@{threshold:.0f}", higher_is_better=True)
        self.threshold = threshold

    def update(
        self,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_cam_pos, gt_cam_pos : ``(C, 3)``
        """
        R, t, _ = _umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        aligned = _apply_alignment(pred_cam_pos, R, t, 1.0)

        # Scene scale: max distance from GT camera to GT centroid
        centroid = gt_cam_pos.mean(axis=0)
        scene_scale = np.linalg.norm(gt_cam_pos - centroid, axis=-1).max()
        if scene_scale < 1e-8:
            return  # degenerate

        errs = np.linalg.norm(aligned - gt_cam_pos, axis=-1)
        frac = float((errs <= self.threshold / 100.0 * scene_scale).mean())
        self._record(frac)

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class ScaledCCA(Metric):
    """Scale-aligned Camera Centre Accuracy (**s-CCA@τ**) after Sim(3) alignment.

    Same as :class:`CCA` but cameras are Sim(3)-aligned before evaluation.

    Parameters
    ----------
    threshold : float
        Percentage threshold (e.g. 10 or 15).
    """

    def __init__(self, threshold: float = 15.0) -> None:
        super().__init__(name=f"s-CCA@{threshold:.0f}", higher_is_better=True)
        self.threshold = threshold

    def update(
        self,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_cam_pos, gt_cam_pos : ``(C, 3)``
        """
        R, t, s = _umeyama(pred_cam_pos, gt_cam_pos, with_scale=True)
        aligned = _apply_alignment(pred_cam_pos, R, t, s)

        centroid = gt_cam_pos.mean(axis=0)
        scene_scale = np.linalg.norm(gt_cam_pos - centroid, axis=-1).max()
        if scene_scale < 1e-8:
            return

        errs = np.linalg.norm(aligned - gt_cam_pos, axis=-1)
        frac = float((errs <= self.threshold / 100.0 * scene_scale).mean())
        self._record(frac)

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


# ═══════════════════════════════════════════════════════════════════════════
# Collection
# ═══════════════════════════════════════════════════════════════════════════

class MetricCollection:
    """Groups several :class:`Metric` instances.

    Example — create the full HSfM evaluation suite::

        mc = MetricCollection([
            WMPJPE(), GAMPJPE(), PAMPJPE(),
            TranslationError(), ScaledTranslationError(),
            AngleError(),
            RRA(threshold=10), RRA(threshold=15),
            CCA(threshold=10), CCA(threshold=15),
            ScaledCCA(threshold=10), ScaledCCA(threshold=15),
        ])
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        self.metrics = {m.name: m for m in metrics}

    def compute(self) -> Dict[str, Dict[str, float]]:
        return {name: m.compute() for name, m in self.metrics.items()}

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]

    def __repr__(self) -> str:
        inner = ", ".join(repr(m) for m in self.metrics.values())
        return f"MetricCollection([{inner}])"
