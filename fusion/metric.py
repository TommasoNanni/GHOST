"""Evaluation metrics for *Humans and Structure from Motion* (HSfM).

Implements the metrics from
*"Reconstructing People, Places, and Cameras"*
(Müller, Choi et al., arXiv 2412.17806, CVPR 2025):

**Human metrics — position-based (metres)**:
    W-MPJPE ↓  — world MPJPE (SE(3)-aligned cameras)
    GA-MPJPE ↓ — group-aligned MPJPE (Sim(3) over all humans)
    PA-MPJPE ↓ — Procrustes-aligned MPJPE (per-human Sim(3))

**Human metrics — rotation-based (degrees)**:
    W-MPJRE ↓  — world MPJRE (SE(3)-aligned cameras, root corrected)
    GA-MPJRE ↓ — group-aligned MPJRE (shared root alignment across all humans)
    PA-MPJRE ↓ — Procrustes-aligned MPJRE (per-human root alignment)

**Camera metrics**:
    TE ↓       — translation error (m) after SE(3) alignment
    s-TE ↓     — translation error (m) after Sim(3) alignment
    AE ↓       — mean pairwise angle error (°)
    RRA@τ ↑    — relative rotation accuracy (fraction ≤ τ°)
    CCA@τ ↑    — camera centre accuracy (fraction within τ% of scene scale, SE(3))
    s-CCA@τ ↑  — camera centre accuracy (Sim(3))

Position-based metrics take ``(P, J, 3)`` 3-D joint positions.
Rotation-based metrics take ``(P, J, 3, 3)`` relative joint rotation matrices
(joint 0 = root; all others are parent-relative).

Every metric subclasses :class:`Metric` with a uniform
``update`` / ``compute`` / ``reset`` interface.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np

from utilities.metrics_utilities import (
    umeyama,
    apply_alignment,
    geodesic_deg,
    batch_geodesic_deg,
    mean_rotation,
)


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
# Human metrics — position-based (metres)
# ═══════════════════════════════════════════════════════════════════════════

class WMPJPE(Metric):
    """World Mean Per-Joint Position Error (W-MPJPE, metres).

    Predicted human joints are brought into the GT world frame via SE(3)
    alignment of camera positions (pred → GT), then the per-joint Euclidean
    error is computed.

    Call update once per scene.
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
        pred_joints  : (P, J, 3)  predicted 3-D joints in predicted world frame.
        gt_joints    : (P, J, 3)  ground-truth 3-D joints.
        pred_cam_pos : (C, 3)     predicted camera centres.
        gt_cam_pos   : (C, 3)     ground-truth camera centres.
        """
        R, t, _ = umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        aligned = apply_alignment(pred_joints.reshape(-1, 3), R, t, 1.0)
        errs = np.linalg.norm(aligned - gt_joints.reshape(-1, 3), axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class GAMPJPE(Metric):
    """Group-Aligned MPJPE (GA-MPJPE, metres).

    All humans' joints in a scene are concatenated and Sim(3)-aligned to GT
    jointly.  Measures relative positioning among people in the scene.
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
        pred_joints : (P, J, 3)
        gt_joints   : (P, J, 3)
        """
        src = pred_joints.reshape(-1, 3)
        dst = gt_joints.reshape(-1, 3)
        R, t, s = umeyama(src, dst, with_scale=True)
        aligned = apply_alignment(src, R, t, s)
        errs = np.linalg.norm(aligned - dst, axis=-1)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class PAMPJPE(Metric):
    """Procrustes-Aligned MPJPE (PA-MPJPE, metres).

    Each human is independently Sim(3)-aligned to its GT.  Measures local pose
    accuracy irrespective of scale and location.
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
        pred_joints : (P, J, 3)
        gt_joints   : (P, J, 3)
        """
        P = pred_joints.shape[0]
        scene_errs: List[float] = []
        for person in range(P):
            src = pred_joints[person]   # (J, 3)
            dst = gt_joints[person]
            R, t, s = umeyama(src, dst, with_scale=True)
            aligned = apply_alignment(src, R, t, s)
            errs = np.linalg.norm(aligned - dst, axis=-1)
            scene_errs.append(float(errs.mean()))
        self._record(float(np.mean(scene_errs)))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


# ═══════════════════════════════════════════════════════════════════════════
# Human metrics — rotation-based (degrees)
# ═══════════════════════════════════════════════════════════════════════════

class WMPJRE(Metric):
    """
    World Mean Per-Joint Rotation Error (W-MPJRE, degrees).

    The predicted world frame is aligned to GT via SE(3) alignment of camera
    positions.  The resulting rotation R_world is applied to the root joint
    (joint 0) of every predicted person; non-root joints are parent-relative
    and need no correction.  Mean geodesic error is then computed across all
    joints and people.

    Call :meth:`update` once per scene.
    """

    def __init__(self) -> None:
        super().__init__(name="W-MPJRE", higher_is_better=False)

    def update(
        self,
        pred_rotations: np.ndarray,
        gt_rotations: np.ndarray,
        pred_cam_pos: np.ndarray,
        gt_cam_pos: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_rotations : (P, J, 3, 3)  predicted relative joint rotations.
        gt_rotations   : (P, J, 3, 3)  ground-truth relative joint rotations.
        pred_cam_pos   : (C, 3)        predicted camera centres.
        gt_cam_pos     : (C, 3)        ground-truth camera centres.
        """
        R_world, _, _ = umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        pred_aligned = pred_rotations.copy()
        # Correct root orientation: (3,3) @ (P,3,3) broadcasts correctly
        pred_aligned[:, 0] = R_world @ pred_rotations[:, 0]
        errs = batch_geodesic_deg(pred_aligned, gt_rotations)  # (P, J)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class GAMPJRE(Metric):
    """Group-Aligned MPJRE (GA-MPJRE, degrees).

    A single global rotation R_align is estimated from the root joints of all
    people in the scene as the Fréchet mean of
    ``{gt_root[p] @ pred_root[p]^T}``, then applied to every predicted root.
    Measures relative pose accuracy among people after removing a shared
    global orientation offset.
    """

    def __init__(self) -> None:
        super().__init__(name="GA-MPJRE", higher_is_better=False)

    def update(
        self,
        pred_rotations: np.ndarray,
        gt_rotations: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_rotations : ``(P, J, 3, 3)``
        gt_rotations   : ``(P, J, 3, 3)``
        """
        # R_diffs[p] = gt_root[p] @ pred_root[p]^T  →  (P, 3, 3)
        R_diffs = gt_rotations[:, 0] @ pred_rotations[:, 0].swapaxes(-1, -2)
        R_align = mean_rotation(R_diffs)  # (3, 3)
        pred_aligned = pred_rotations.copy()
        pred_aligned[:, 0] = R_align @ pred_rotations[:, 0]
        errs = batch_geodesic_deg(pred_aligned, gt_rotations)  # (P, J)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class PAMPJRE(Metric):
    """Procrustes-Aligned MPJRE (PA-MPJRE, degrees).

    Each predicted person's root joint is independently aligned to GT via
    ``R_align[p] = gt_root[p] @ pred_root[p]^T``.  Geodesic errors are then
    computed over all joints.  Removes global orientation ambiguity per person,
    measuring local / relative pose accuracy.
    """

    def __init__(self) -> None:
        super().__init__(name="PA-MPJRE", higher_is_better=False)

    def update(
        self,
        pred_rotations: np.ndarray,
        gt_rotations: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        pred_rotations : ``(P, J, 3, 3)``
        gt_rotations   : ``(P, J, 3, 3)``
        """
        # R_aligns[p] = gt_root[p] @ pred_root[p]^T  →  (P, 3, 3)
        R_aligns = gt_rotations[:, 0] @ pred_rotations[:, 0].swapaxes(-1, -2)
        pred_aligned = pred_rotations.copy()
        # (P,3,3) @ (P,3,3) → per-person batched matmul
        pred_aligned[:, 0] = R_aligns @ pred_rotations[:, 0]
        errs = batch_geodesic_deg(pred_aligned, gt_rotations)  # (P, J)
        # Mean per person, then over the scene
        self._record(float(errs.mean(axis=-1).mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


# ═══════════════════════════════════════════════════════════════════════════
# Camera metrics
# ═══════════════════════════════════════════════════════════════════════════

class TranslationError(Metric):
    """Camera translation error (TE, metres) after SE(3) alignment.

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
        R, t, _ = umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        aligned = apply_alignment(pred_cam_pos, R, t, 1.0)
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
        R, t, s = umeyama(pred_cam_pos, gt_cam_pos, with_scale=True)
        aligned = apply_alignment(pred_cam_pos, R, t, s)
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
            errs.append(geodesic_deg(R_rel_pred, R_rel_gt))
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
            err = geodesic_deg(R_rel_pred, R_rel_gt)
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
        R, t, _ = umeyama(pred_cam_pos, gt_cam_pos, with_scale=False)
        aligned = apply_alignment(pred_cam_pos, R, t, 1.0)

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
        R, t, s = umeyama(pred_cam_pos, gt_cam_pos, with_scale=True)
        aligned = apply_alignment(pred_cam_pos, R, t, s)

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
            # position-based human metrics
            WMPJPE(), GAMPJPE(), PAMPJPE(),
            # rotation-based human metrics
            WMPJRE(), GAMPJRE(), PAMPJRE(),
            # camera metrics
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
