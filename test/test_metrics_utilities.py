"""
Unit tests for utilities/metrics_utilities.py

All tests are deterministic (fixed inputs, no GPU required).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from utilities.metrics_utilities import (
    umeyama,
    apply_alignment,
    geodesic_deg,
    batch_geodesic_deg,
    mean_rotation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_z(deg: float) -> np.ndarray:
    """3x3 rotation matrix about Z by deg degrees."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _rot_x(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


# ---------------------------------------------------------------------------
# umeyama
# ---------------------------------------------------------------------------

class TestUmeyama:

    def test_identity_transform(self):
        """When src == dst, should recover R=I, t=0, s=1."""
        src = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 0.]])
        R, t, s = umeyama(src, src.copy(), with_scale=True)
        assert np.allclose(R, np.eye(3), atol=1e-6)
        assert np.allclose(t, np.zeros(3), atol=1e-6)
        assert abs(s - 1.0) < 1e-6

    def test_pure_translation(self):
        """Translation-only: R=I, t=true_t, s=1."""
        src = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        true_t = np.array([3., -2., 5.])
        dst = src + true_t
        R, t, s = umeyama(src, dst, with_scale=False)
        assert np.allclose(R, np.eye(3), atol=1e-6)
        assert np.allclose(t, true_t, atol=1e-6)
        assert abs(s - 1.0) < 1e-6

    def test_pure_rotation_90_z(self):
        """90° rotation about Z: should recover R exactly."""
        R_true = _rot_z(90.)
        src = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
        dst = src @ R_true.T
        R, t, s = umeyama(src, dst, with_scale=False)
        assert np.allclose(R, R_true, atol=1e-6)
        assert np.allclose(t, np.zeros(3), atol=1e-6)

    def test_scale_recovery(self):
        """Sim(3): should recover scale factor."""
        src = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        s_true = 2.5
        dst = s_true * src
        R, t, s = umeyama(src, dst, with_scale=True)
        assert abs(s - s_true) < 1e-5
        assert np.allclose(R, np.eye(3), atol=1e-5)

    def test_with_scale_false_fixes_s_to_one(self):
        """SE(3) mode: s must always be 1."""
        src = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        dst = 3.0 * src
        _, _, s = umeyama(src, dst, with_scale=False)
        assert s == 1.0

    def test_full_rigid_body_reconstruction(self):
        """apply_alignment(src) ≈ dst for an exact rigid transform."""
        R_true = _rot_z(37.)
        t_true = np.array([1., -3., 2.])
        src = np.array([[2., 0., 1.], [0., 3., 0.], [1., 1., 2.], [0., 0., 3.]])
        dst = src @ R_true.T + t_true
        R, t, s = umeyama(src, dst, with_scale=False)
        aligned = apply_alignment(src, R, t, s)
        assert np.allclose(aligned, dst, atol=1e-6)

    def test_full_sim3_reconstruction(self):
        """apply_alignment(src) ≈ dst for an exact Sim(3) transform."""
        R_true = _rot_x(45.)
        t_true = np.array([2., 1., -1.])
        s_true = 1.5
        src = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])
        dst = s_true * (src @ R_true.T) + t_true
        R, t, s = umeyama(src, dst, with_scale=True)
        aligned = apply_alignment(src, R, t, s)
        assert np.allclose(aligned, dst, atol=1e-6)

    def test_r_is_rotation_matrix(self):
        """R must be a proper rotation matrix (orthogonal, det=1)."""
        rng = np.random.default_rng(42)
        src = rng.standard_normal((8, 3))
        dst = rng.standard_normal((8, 3))
        R, _, _ = umeyama(src, dst, with_scale=True)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert abs(np.linalg.det(R) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# apply_alignment
# ---------------------------------------------------------------------------

class TestApplyAlignment:

    def test_identity(self):
        pts = np.array([[1., 2., 3.], [4., 5., 6.]])
        result = apply_alignment(pts, np.eye(3), np.zeros(3), 1.0)
        assert np.allclose(result, pts)

    def test_translation_only(self):
        pts = np.array([[0., 0., 0.], [1., 0., 0.]])
        t = np.array([1., 2., 3.])
        result = apply_alignment(pts, np.eye(3), t, 1.0)
        assert np.allclose(result, pts + t)

    def test_scale_only(self):
        pts = np.array([[1., 2., 3.]])
        result = apply_alignment(pts, np.eye(3), np.zeros(3), 2.0)
        assert np.allclose(result, 2.0 * pts)

    def test_rotation_only_90_z(self):
        pts = np.array([[1., 0., 0.]])
        R = _rot_z(90.)
        result = apply_alignment(pts, R, np.zeros(3), 1.0)
        assert np.allclose(result, [[0., 1., 0.]], atol=1e-6)


# ---------------------------------------------------------------------------
# geodesic_deg
# ---------------------------------------------------------------------------

class TestGeodesicDeg:

    def test_identity_vs_identity_is_zero(self):
        assert geodesic_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0, abs=1e-6)

    def test_same_rotation_is_zero(self):
        R = _rot_z(60.)
        assert geodesic_deg(R, R) == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_rotation_z(self):
        R90 = _rot_z(90.)
        assert geodesic_deg(np.eye(3), R90) == pytest.approx(90.0, abs=1e-4)

    def test_180_degree_rotation_z(self):
        R180 = _rot_z(180.)
        assert geodesic_deg(np.eye(3), R180) == pytest.approx(180.0, abs=1e-4)

    def test_45_degree_rotation_x(self):
        R45 = _rot_x(45.)
        assert geodesic_deg(np.eye(3), R45) == pytest.approx(45.0, abs=1e-4)

    def test_symmetry(self):
        """geodesic(A, B) == geodesic(B, A)."""
        R = _rot_x(37.)
        assert geodesic_deg(np.eye(3), R) == pytest.approx(geodesic_deg(R, np.eye(3)), abs=1e-6)

    def test_non_negative(self):
        assert geodesic_deg(_rot_z(30.), _rot_z(70.)) >= 0.0


# ---------------------------------------------------------------------------
# batch_geodesic_deg
# ---------------------------------------------------------------------------

class TestBatchGeodesicDeg:

    def test_batch_identities_are_zero(self):
        Rs = np.stack([np.eye(3)] * 5)
        errs = batch_geodesic_deg(Rs, Rs)
        assert np.allclose(errs, 0.0, atol=1e-6)

    def test_batch_matches_scalar(self):
        """Each element should agree with the scalar version."""
        angles = [30., 60., 90., 120., 180.]
        Ra = np.stack([np.eye(3)] * len(angles))
        Rb = np.stack([_rot_z(a) for a in angles])
        errs = batch_geodesic_deg(Ra, Rb)
        for i, a in enumerate(angles):
            assert errs[i] == pytest.approx(a, abs=1e-3)

    def test_batch_symmetry(self):
        Ra = np.stack([_rot_z(10.), _rot_z(45.)])
        Rb = np.stack([_rot_x(20.), _rot_x(90.)])
        assert np.allclose(batch_geodesic_deg(Ra, Rb), batch_geodesic_deg(Rb, Ra), atol=1e-6)

    def test_output_shape_2d(self):
        B, C = 3, 4
        Ra = np.tile(np.eye(3), (B, C, 1, 1))
        Rb = np.tile(_rot_z(45.), (B, C, 1, 1))
        errs = batch_geodesic_deg(Ra, Rb)
        assert errs.shape == (B, C)


# ---------------------------------------------------------------------------
# mean_rotation
# ---------------------------------------------------------------------------

class TestMeanRotation:

    def test_single_rotation(self):
        """Mean of one rotation is that rotation."""
        R = _rot_z(72.)
        R_mean = mean_rotation(np.stack([R]))
        assert np.allclose(R_mean, R, atol=1e-6)

    def test_identical_rotations(self):
        """Mean of copies of R is R."""
        R = _rot_z(30.)
        Rs = np.stack([R, R, R, R])
        R_mean = mean_rotation(Rs)
        assert np.allclose(R_mean, R, atol=1e-6)

    def test_identity_stack(self):
        Rs = np.stack([np.eye(3)] * 6)
        R_mean = mean_rotation(Rs)
        assert np.allclose(R_mean, np.eye(3), atol=1e-6)

    def test_result_is_rotation_matrix(self):
        """R_mean @ R_mean^T == I and det == 1."""
        Rs = np.stack([_rot_z(0.), _rot_z(60.), _rot_z(120.)])
        R_mean = mean_rotation(Rs)
        assert np.allclose(R_mean @ R_mean.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R_mean) - 1.0) < 1e-6

    def test_symmetric_angles_near_identity(self):
        """Mean of +θ and -θ rotations about same axis should be close to I."""
        Rs = np.stack([_rot_z(30.), _rot_z(-30.)])
        R_mean = mean_rotation(Rs)
        assert np.allclose(R_mean, np.eye(3), atol=1e-6)
