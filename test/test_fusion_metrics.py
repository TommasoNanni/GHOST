"""
Unit tests for fusion/metric.py

Tests cover all metric classes (position-based and rotation-based human metrics,
all camera metrics) and MetricCollection. No GPU, no model loading.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from fusion.metric import (
    Metric,
    WMPJPE, GAMPJPE, PAMPJPE,
    WMPJRE, GAMPJRE, PAMPJRE,
    TranslationError, ScaledTranslationError,
    AngleError, RRA, CCA, ScaledCCA,
    MetricCollection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _identity_rotations(P: int, J: int) -> np.ndarray:
    return np.tile(np.eye(3), (P, J, 1, 1))


def _square_cameras() -> np.ndarray:
    """4 camera centres arranged in a unit square (well-spread for Kabsch)."""
    return np.array([[1., 0., 0.], [-1., 0., 0.],
                     [0., 1., 0.], [0., -1., 0.]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Metric base: _aggregate and reset
# ---------------------------------------------------------------------------

class TestMetricBase:

    def test_aggregate_empty_returns_nan(self):
        m = WMPJPE()
        result = m._aggregate()
        assert np.isnan(result["mean"])
        assert np.isnan(result["median"])
        assert result["count"] == 0.0

    def test_aggregate_single_value(self):
        m = WMPJPE()
        m._values = [5.0]
        r = m._aggregate()
        assert r["mean"] == pytest.approx(5.0)
        assert r["min"] == pytest.approx(5.0)
        assert r["max"] == pytest.approx(5.0)
        assert r["count"] == 1.0

    def test_aggregate_known_values(self):
        m = WMPJPE()
        m._values = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = m._aggregate()
        assert r["mean"]   == pytest.approx(3.0)
        assert r["median"] == pytest.approx(3.0)
        assert r["min"]    == pytest.approx(1.0)
        assert r["max"]    == pytest.approx(5.0)
        assert r["count"]  == 5.0

    def test_reset_clears_values(self):
        m = WMPJPE()
        m._values = [1.0, 2.0, 3.0]
        m.reset()
        assert m._values == []

    def test_repr_contains_name(self):
        m = WMPJPE()
        assert "W-MPJPE" in repr(m)

    def test_compute_after_reset_is_nan(self):
        m = WMPJPE()
        m._values = [1.0]
        m.reset()
        assert np.isnan(m.compute()["mean"])


# ---------------------------------------------------------------------------
# WMPJPE
# ---------------------------------------------------------------------------

class TestWMPJPE:

    def test_perfect_prediction_zero_error(self):
        """pred == gt and cameras match → error = 0."""
        m = WMPJPE()
        gt_joints = np.array([[[1., 2., 3.], [4., 5., 6.]],
                               [[0., 1., 0.], [1., 1., 1.]]], dtype=np.float64)
        cam_pos = _square_cameras()
        m.update(gt_joints.copy(), gt_joints, cam_pos, cam_pos)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_accumulates_multiple_scenes(self):
        m = WMPJPE()
        gt = np.ones((2, 4, 3))
        cam = _square_cameras()
        m.update(gt, gt, cam, cam)
        m.update(gt, gt, cam, cam)
        assert m.compute()["count"] == 2.0

    def test_higher_is_better_is_false(self):
        assert WMPJPE().higher_is_better is False

    def test_rigid_world_shift_gives_zero_error(self):
        """When the entire predicted frame (cameras + joints) is rigidly translated,
        umeyama recovers the offset and the metric should still read 0."""
        m = WMPJPE()
        gt_joints = np.ones((1, 3, 3))
        cam_pos = _square_cameras()
        offset = np.array([5., 3., -2.])
        # Both pred cameras AND pred joints shifted by the same offset.
        # umeyama finds t = -offset; apply_alignment then cancels it exactly.
        m.update(gt_joints + offset, gt_joints, cam_pos + offset, cam_pos)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GAMPJPE
# ---------------------------------------------------------------------------

class TestGAMPJPE:

    def test_perfect_prediction_zero_error(self):
        m = GAMPJPE()
        rng = np.random.default_rng(0)
        gt = rng.standard_normal((3, 10, 3))
        m.update(gt.copy(), gt)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_scale_invariant(self):
        """Sim(3) alignment absorbs scale → scaled pred gives 0 error."""
        m = GAMPJPE()
        gt = np.array([[[1., 0., 0.], [0., 1., 0.]], [[2., 0., 0.], [0., 2., 0.]]])
        m.update(3.0 * gt, gt)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_multiple_updates(self):
        m = GAMPJPE()
        gt = np.ones((2, 5, 3))
        m.update(gt, gt)
        m.update(gt, gt)
        assert m.compute()["count"] == 2.0


# ---------------------------------------------------------------------------
# PAMPJPE
# ---------------------------------------------------------------------------

class TestPAMPJPE:

    def test_perfect_prediction_zero_error(self):
        m = PAMPJPE()
        rng = np.random.default_rng(1)
        gt = rng.standard_normal((4, 8, 3))
        m.update(gt.copy(), gt)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_per_person_scale_invariant(self):
        """Each person scaled independently → still 0 error."""
        m = PAMPJPE()
        gt = np.array([[[1., 0., 0.], [0., 1., 0.]], [[3., 0., 0.], [0., 3., 0.]]])
        pred = gt.copy()
        pred[0] *= 2.0   # person 0 scaled by 2
        pred[1] *= 0.5   # person 1 scaled by 0.5
        m.update(pred, gt)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# WMPJRE
# ---------------------------------------------------------------------------

class TestWMPJRE:

    def test_perfect_prediction_zero_error(self):
        m = WMPJRE()
        P, J = 3, 6
        gt_rots = _identity_rotations(P, J)
        cam = _square_cameras()
        m.update(gt_rots.copy(), gt_rots, cam, cam)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)

    def test_higher_is_better_is_false(self):
        assert WMPJRE().higher_is_better is False


# ---------------------------------------------------------------------------
# GAMPJRE
# ---------------------------------------------------------------------------

class TestGAMPJRE:

    def test_perfect_prediction_zero_error(self):
        m = GAMPJRE()
        P, J = 2, 5
        gt_rots = _identity_rotations(P, J)
        m.update(gt_rots.copy(), gt_rots)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)

    def test_shared_rotation_offset_cancelled(self):
        """All persons' roots rotated by same R → GA alignment cancels it."""
        m = GAMPJRE()
        P, J = 3, 4
        R_offset = _rot_z(45.)
        gt_rots = _identity_rotations(P, J)
        pred_rots = gt_rots.copy()
        pred_rots[:, 0] = R_offset  # all roots rotated by same amount
        m.update(pred_rots, gt_rots)
        # GA computes mean of gt_root @ pred_root^T = I @ R_offset^T = R_offset^T
        # then applies R_offset^T to pred root → I. Should be near 0.
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# PAMPJRE
# ---------------------------------------------------------------------------

class TestPAMPJRE:

    def test_perfect_prediction_zero_error(self):
        m = PAMPJRE()
        P, J = 4, 7
        gt_rots = _identity_rotations(P, J)
        m.update(gt_rots.copy(), gt_rots)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)

    def test_per_person_root_offset_cancelled(self):
        """Each person's root rotated by its own R → PA cancels each independently."""
        m = PAMPJRE()
        P, J = 2, 3
        gt_rots = _identity_rotations(P, J)
        pred_rots = gt_rots.copy()
        pred_rots[0, 0] = _rot_z(30.)
        pred_rots[1, 0] = _rot_z(90.)
        m.update(pred_rots, gt_rots)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# TranslationError
# ---------------------------------------------------------------------------

class TestTranslationError:

    def test_perfect_cameras_zero_error(self):
        m = TranslationError()
        cam = _square_cameras()
        m.update(cam.copy(), cam)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation_offset_cancelled(self):
        """SE(3) alignment fully absorbs a constant translation offset."""
        m = TranslationError()
        cam = _square_cameras()
        m.update(cam + np.array([5., 3., -2.]), cam)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_higher_is_better_is_false(self):
        assert TranslationError().higher_is_better is False

    def test_nonzero_error_for_perturbed_cameras(self):
        m = TranslationError()
        cam_gt = _square_cameras()
        cam_pred = cam_gt.copy()
        cam_pred[0] += np.array([0., 0., 1.])  # move one camera off
        m.update(cam_pred, cam_gt)
        assert m.compute()["mean"] > 0.0


# ---------------------------------------------------------------------------
# ScaledTranslationError
# ---------------------------------------------------------------------------

class TestScaledTranslationError:

    def test_perfect_cameras_zero_error(self):
        m = ScaledTranslationError()
        cam = _square_cameras()
        m.update(cam.copy(), cam)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_scaled_cameras_zero_error(self):
        """Sim(3) alignment absorbs uniform scale."""
        m = ScaledTranslationError()
        cam = _square_cameras()
        m.update(2.5 * cam, cam)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AngleError
# ---------------------------------------------------------------------------

class TestAngleError:

    def test_identical_rotations_zero_error(self):
        m = AngleError()
        rots = np.stack([np.eye(3), _rot_z(30.), _rot_z(60.)])
        m.update(rots.copy(), rots)
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-4)

    def test_higher_is_better_is_false(self):
        assert AngleError().higher_is_better is False

    def test_single_rotation_pair(self):
        """
        pred = [I, R90z], gt = [I, I].
        R_rel_pred(0,1) = I @ R90z^T = R(-90z)
        R_rel_gt(0,1) = I @ I^T = I
        geodesic(R(-90z), I) = 90°.
        """
        m = AngleError()
        R90 = _rot_z(90.)
        m.update(np.stack([np.eye(3), R90]), np.stack([np.eye(3), np.eye(3)]))
        assert m.compute()["mean"] == pytest.approx(90.0, abs=1e-3)


# ---------------------------------------------------------------------------
# RRA
# ---------------------------------------------------------------------------

class TestRRA:

    def test_identical_rotations_score_one(self):
        m = RRA(threshold=15.0)
        rots = np.stack([np.eye(3)] * 4)
        m.update(rots, rots)
        assert m.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_180_degree_error_scores_zero(self):
        """
        pred = [I, Rz(90)], gt = [I, Rz(-90)].
        Relative pred: I @ Rz(90)^T = Rz(-90)
        Relative gt:   I @ Rz(-90)^T = Rz(90)
        geodesic(Rz(-90), Rz(90)) = 180° >> threshold.
        """
        m = RRA(threshold=15.0)
        R90  = _rot_z(90.)
        Rm90 = _rot_z(-90.)
        m.update(np.stack([np.eye(3), R90]), np.stack([np.eye(3), Rm90]))
        assert m.compute()["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_threshold_boundary(self):
        """Angle exactly at threshold should pass; above should fail."""
        # Two cameras; one pair → error = 20°.
        m_tight = RRA(threshold=15.0)
        m_loose = RRA(threshold=25.0)
        R20 = _rot_z(20.)
        pred = np.stack([np.eye(3), R20])
        gt   = np.stack([np.eye(3), np.eye(3)])
        m_tight.update(pred, gt)
        m_loose.update(pred, gt)
        assert m_tight.compute()["mean"] == pytest.approx(0.0, abs=1e-6)
        assert m_loose.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_higher_is_better(self):
        assert RRA().higher_is_better is True


# ---------------------------------------------------------------------------
# CCA
# ---------------------------------------------------------------------------

class TestCCA:

    def test_perfect_cameras_score_one(self):
        m = CCA(threshold=15.0)
        cam = _square_cameras()
        m.update(cam.copy(), cam)
        assert m.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_pure_translation_offset_score_one(self):
        """SE(3) alignment cancels translation; all cameras should still be correct."""
        m = CCA(threshold=15.0)
        cam = _square_cameras()
        m.update(cam + np.array([10., 0., 0.]), cam)
        assert m.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_higher_is_better(self):
        assert CCA().higher_is_better is True


# ---------------------------------------------------------------------------
# ScaledCCA
# ---------------------------------------------------------------------------

class TestScaledCCA:

    def test_perfect_cameras_score_one(self):
        m = ScaledCCA(threshold=15.0)
        cam = _square_cameras()
        m.update(cam.copy(), cam)
        assert m.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_scaled_cameras_score_one(self):
        """Sim(3) alignment absorbs scale → should still get full score."""
        m = ScaledCCA(threshold=15.0)
        cam = _square_cameras()
        m.update(3.0 * cam, cam)
        assert m.compute()["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_higher_is_better(self):
        assert ScaledCCA().higher_is_better is True


# ---------------------------------------------------------------------------
# MetricCollection
# ---------------------------------------------------------------------------

class TestMetricCollection:

    def test_grouping_and_compute(self):
        mc = MetricCollection([WMPJPE(), TranslationError()])
        cam = _square_cameras()
        gt_joints = np.ones((2, 5, 3))
        mc["W-MPJPE"].update(gt_joints, gt_joints, cam, cam)
        mc["TE"].update(cam, cam)
        results = mc.compute()
        assert "W-MPJPE" in results and "TE" in results
        assert results["W-MPJPE"]["mean"] == pytest.approx(0.0, abs=1e-6)
        assert results["TE"]["mean"]      == pytest.approx(0.0, abs=1e-6)

    def test_reset_clears_all(self):
        mc = MetricCollection([WMPJPE(), TranslationError()])
        mc["W-MPJPE"]._values = [1.0, 2.0]
        mc["TE"]._values = [3.0]
        mc.reset()
        assert mc["W-MPJPE"]._values == []
        assert mc["TE"]._values == []

    def test_getitem_returns_correct_metric(self):
        rra = RRA(threshold=10.0)
        mc = MetricCollection([rra])
        assert mc[f"RRA@10"] is rra

    def test_full_suite_zero_error_perfect_predictions(self):
        """Build the full metric suite and verify everything returns 0 for perfect preds."""
        P, J, C = 2, 6, 4
        gt_joints = np.random.default_rng(7).standard_normal((P, J, 3))
        gt_rots   = _identity_rotations(P, J)
        cam_rots  = np.stack([np.eye(3)] * C)
        cam_pos   = _square_cameras()

        mc = MetricCollection([
            WMPJPE(), GAMPJPE(), PAMPJPE(),
            WMPJRE(), GAMPJRE(), PAMPJRE(),
            TranslationError(), ScaledTranslationError(),
            AngleError(),
            RRA(threshold=10.), RRA(threshold=15.),
            CCA(threshold=10.), CCA(threshold=15.),
            ScaledCCA(threshold=10.), ScaledCCA(threshold=15.),
        ])

        mc["W-MPJPE"].update(gt_joints.copy(), gt_joints, cam_pos, cam_pos)
        mc["GA-MPJPE"].update(gt_joints.copy(), gt_joints)
        mc["PA-MPJPE"].update(gt_joints.copy(), gt_joints)
        mc["W-MPJRE"].update(gt_rots.copy(), gt_rots, cam_pos, cam_pos)
        mc["GA-MPJRE"].update(gt_rots.copy(), gt_rots)
        mc["PA-MPJRE"].update(gt_rots.copy(), gt_rots)
        mc["TE"].update(cam_pos, cam_pos)
        mc["s-TE"].update(cam_pos, cam_pos)
        mc["AE"].update(cam_rots, cam_rots)
        mc["RRA@10"].update(cam_rots, cam_rots)
        mc["RRA@15"].update(cam_rots, cam_rots)
        mc["CCA@10"].update(cam_pos, cam_pos)
        mc["CCA@15"].update(cam_pos, cam_pos)
        mc["s-CCA@10"].update(cam_pos, cam_pos)
        mc["s-CCA@15"].update(cam_pos, cam_pos)

        results = mc.compute()

        zero_error_metrics = ["W-MPJPE", "GA-MPJPE", "PA-MPJPE",
                              "W-MPJRE", "GA-MPJRE", "PA-MPJRE",
                              "TE", "s-TE", "AE"]
        for name in zero_error_metrics:
            assert results[name]["mean"] == pytest.approx(0.0, abs=1e-4), \
                f"{name} expected 0 error, got {results[name]['mean']}"

        one_score_metrics = ["RRA@10", "RRA@15", "CCA@10", "CCA@15", "s-CCA@10", "s-CCA@15"]
        for name in one_score_metrics:
            assert results[name]["mean"] == pytest.approx(1.0, abs=1e-6), \
                f"{name} expected score 1.0, got {results[name]['mean']}"
