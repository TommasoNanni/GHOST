"""Body metrics for FusionWithBetas evaluation.

Three metrics, no camera or alignment assumptions:

  RR-MPJPE   root-relative MPJPE (metres)     — subtract joint 0, compare positions
  RR-MPJRE   root-relative MPJRE (degrees)    — compare parent-relative rotations, skip root
  Betas-MSE  mean squared error on betas      — per person, averaged over scenes

update() is called once per (time step, person) for the joint metrics,
and once per person for BetasMSE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np

from utilities.metrics_utilities import batch_geodesic_deg


class Metric(ABC):
    def __init__(self, name: str, higher_is_better: bool) -> None:
        self.name = name
        self.higher_is_better = higher_is_better
        self._values: List[float] = []

    @abstractmethod
    def update(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def compute(self) -> Dict[str, float]: ...

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


class RootRelativeMPJPE(Metric):
    """Root-relative MPJPE (metres).

    Both pred and GT joints are made root-relative (subtract joint 0) before
    computing mean L2 error.  No global alignment is applied.

    update() once per (time step, person).
    """

    def __init__(self) -> None:
        super().__init__(name="RR-MPJPE", higher_is_better=False)

    def update(self, pred_joints: np.ndarray, gt_joints: np.ndarray) -> None:
        """
        Parameters
        ----------
        pred_joints, gt_joints : (J, 3)
        """
        pred_rel = pred_joints - pred_joints[:1]
        gt_rel   = gt_joints   - gt_joints[:1]
        self._record(float(np.linalg.norm(pred_rel - gt_rel, axis=-1).mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class RootRelativeMPJRE(Metric):
    """Root-relative MPJRE (degrees).

    Compares parent-relative joint rotations for joints 1+ only (joint 0 is
    in world frame and is ignored).  No alignment applied.

    update() once per (time step, person).
    """

    def __init__(self) -> None:
        super().__init__(name="RR-MPJRE", higher_is_better=False)

    def update(self, pred_rotations: np.ndarray, gt_rotations: np.ndarray) -> None:
        """
        Parameters
        ----------
        pred_rotations, gt_rotations : (J, 3, 3)  parent-relative rotation matrices
        """
        errs = batch_geodesic_deg(pred_rotations[1:], gt_rotations[1:])  # (J-1,)
        self._record(float(errs.mean()))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class BetasMSE(Metric):
    """Mean squared error between predicted and GT SMPL-X betas.

    update() once per person.
    """

    def __init__(self) -> None:
        super().__init__(name="Betas-MSE", higher_is_better=False)

    def update(self, pred_betas: np.ndarray, gt_betas: np.ndarray) -> None:
        """
        Parameters
        ----------
        pred_betas, gt_betas : (10,)
        """
        self._record(float(np.mean((pred_betas - gt_betas) ** 2)))

    def compute(self) -> Dict[str, float]:
        return self._aggregate()


class MetricCollection:
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
        return f"MetricCollection([{', '.join(repr(m) for m in self.metrics.values())}])"
