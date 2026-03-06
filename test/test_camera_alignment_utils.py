"""
Unit tests for data/camera_alignment.py — static / pure methods only.

Tests cover:
  - CameraAlignment._kabsch
  - CameraAlignment._absolute_joints
  - CameraAlignment.relative_pose_to_matrix
  - CameraAlignment.camera_center_in_A

No file I/O or model loading involved.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from preprocessing.camera_alignment import CameraAlignment


def _rot_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _rot_x(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


class TestKabsch:

    def test_identity_transform(self):
        """pts_a == pts_b → R = I, t = 0."""
        pts = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 0.]])
        R, t = CameraAlignment._kabsch(pts, pts.copy())
        assert np.allclose(R, np.eye(3), atol=1e-8)
        assert np.allclose(t, np.zeros(3), atol=1e-8)

    def test_pure_translation(self):
        """Pure translation: R = I, t = true_t."""
        pts_a = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        true_t = np.array([3., -2., 5.])
        pts_b = pts_a + true_t
        R, t = CameraAlignment._kabsch(pts_a, pts_b)
        assert np.allclose(R, np.eye(3), atol=1e-8)
        assert np.allclose(t, true_t, atol=1e-8)

    def test_90_degree_rotation_z(self):
        """90° rotation about Z."""
        R_true = _rot_z(90.)
        pts_a = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
        pts_b = pts_a @ R_true.T
        R, t = CameraAlignment._kabsch(pts_a, pts_b)
        assert np.allclose(R, R_true, atol=1e-8)
        assert np.allclose(t, np.zeros(3), atol=1e-8)

    def test_45_degree_rotation_x(self):
        """45° rotation about X."""
        R_true = _rot_x(45.)
        pts_a = np.array([[1., 1., 0.], [0., 2., 1.], [2., 0., 1.], [1., 1., 1.]])
        pts_b = pts_a @ R_true.T
        R, t = CameraAlignment._kabsch(pts_a, pts_b)
        assert np.allclose(R, R_true, atol=1e-8)
        assert np.allclose(t, np.zeros(3), atol=1e-8)

    def test_rotation_and_translation(self):
        """Combined rotation and translation."""
        R_true = _rot_z(37.)
        t_true = np.array([1., -2., 3.])
        pts_a = np.array([[2., 0., 1.], [0., 3., 0.], [1., 1., 2.], [3., 2., 1.]])
        pts_b = pts_a @ R_true.T + t_true
        R, t = CameraAlignment._kabsch(pts_a, pts_b)
        pts_b_reconstructed = pts_a @ R.T + t
        assert np.allclose(pts_b_reconstructed, pts_b, atol=1e-8)

    def test_result_is_proper_rotation(self):
        """R @ R^T = I and det(R) = 1."""
        pts_a = np.array([[1., 2., 0.], [3., 0., 1.], [0., 2., 3.], [1., 1., 1.]])
        pts_b = np.array([[2., 1., 0.], [0., 3., 1.], [2., 0., 3.], [1., 1., 1.]])
        R, _ = CameraAlignment._kabsch(pts_a, pts_b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert abs(np.linalg.det(R) - 1.0) < 1e-8

    def test_raises_on_too_few_points(self):
        pts = np.array([[1., 0., 0.], [0., 1., 0.]])  # only 2 points
        with pytest.raises(ValueError, match="at least 3"):
            CameraAlignment._kabsch(pts, pts)

    def test_raises_on_shape_mismatch(self):
        pts_a = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        pts_b = np.array([[1., 0., 0.], [0., 1., 0.]])
        with pytest.raises(ValueError):
            CameraAlignment._kabsch(pts_a, pts_b)

    def test_raises_on_wrong_dim(self):
        pts = np.array([[1., 0.], [0., 1.], [0., 0.]])  # 2D, not 3D
        with pytest.raises(ValueError):
            CameraAlignment._kabsch(pts, pts)

    def test_minimum_three_points(self):
        """Three collinear-free points should still give a valid result."""
        pts_a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        R_true = _rot_z(60.)
        pts_b = pts_a @ R_true.T
        R, t = CameraAlignment._kabsch(pts_a, pts_b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
        assert np.allclose(pts_a @ R.T + t, pts_b, atol=1e-7)


class TestAbsoluteJoints:

    def test_returns_root_when_no_keypoints_key(self):
        """No 'pred_keypoints_3d' in data → return root as (1, 3)."""
        root = np.array([1., 2., 3.])
        data = {"pred_cam_t": np.array([root])}
        result = CameraAlignment._absolute_joints(data, 0)
        assert result is not None
        assert np.allclose(result, root[None, :])

    def test_adds_root_to_keypoints(self):
        root = np.array([10., 20., 30.])
        kpts = np.array([[1., 2., 3.], [4., 5., 6.]])
        data = {
            "pred_cam_t": np.array([root]),
            "pred_keypoints_3d": np.array([kpts]),
        }
        result = CameraAlignment._absolute_joints(data, 0)
        assert np.allclose(result, kpts + root[None, :])

    def test_returns_none_without_cam_t(self):
        data = {}
        result = CameraAlignment._absolute_joints(data, 0)
        assert result is None

    def test_selects_correct_row(self):
        """Should pick row=1, not row=0."""
        roots = np.array([[0., 0., 0.], [5., 6., 7.]])
        data = {"pred_cam_t": roots}
        result = CameraAlignment._absolute_joints(data, 1)
        assert np.allclose(result, np.array([[5., 6., 7.]]))

    def test_filters_all_zero_joints(self):
        """Joints that are zero after adding root should be filtered out."""
        root = np.array([0., 0., 0.])
        kpts = np.array([[1., 2., 3.], [0., 0., 0.], [4., 5., 6.]])
        data = {
            "pred_cam_t": np.array([root]),
            "pred_keypoints_3d": np.array([kpts]),
        }
        result = CameraAlignment._absolute_joints(data, 0)
        # abs_joints = kpts + 0 = kpts; zero joint at index 1 is filtered
        assert result.shape[0] == 2
        assert np.allclose(result[0], [1., 2., 3.])
        assert np.allclose(result[1], [4., 5., 6.])

    def test_all_zero_joints_falls_back_to_root(self):
        """If all joints are zero after filter, should return root."""
        root = np.array([1., 2., 3.])
        kpts = np.array([[0., 0., 0.], [0., 0., 0.]])  # abs = kpts + root ≠ 0
        # Wait — abs = kpts + root = root, which is non-zero for non-zero root.
        # To make all-zero abs_joints we need kpts = -root everywhere.
        kpts = -root[None, :] * np.ones((2, 3))  # abs = 0 for all
        data = {
            "pred_cam_t": np.array([root]),
            "pred_keypoints_3d": np.array([kpts]),
        }
        result = CameraAlignment._absolute_joints(data, 0)
        # valid.sum() == 0, so falls back to root[None, :]
        assert np.allclose(result, root[None, :])


# ---------------------------------------------------------------------------
# relative_pose_to_matrix
# ---------------------------------------------------------------------------

class TestRelativePoseToMatrix:

    def test_identity(self):
        T = CameraAlignment.relative_pose_to_matrix(np.eye(3), np.zeros(3))
        assert np.allclose(T, np.eye(4))

    def test_known_values(self):
        R = _rot_z(90.)
        t = np.array([1., 2., 3.])
        T = CameraAlignment.relative_pose_to_matrix(R, t)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], R)
        assert np.allclose(T[:3, 3], t)
        assert np.allclose(T[3, :], [0., 0., 0., 1.])

    def test_bottom_row_is_0001(self):
        R = _rot_x(45.)
        t = np.array([-1., 3., 2.])
        T = CameraAlignment.relative_pose_to_matrix(R, t)
        assert np.allclose(T[3, :], [0., 0., 0., 1.])

    def test_compose_with_inverse(self):
        """T @ T^{-1} == I."""
        R = _rot_z(60.)
        t = np.array([1., -1., 2.])
        T = CameraAlignment.relative_pose_to_matrix(R, t)
        assert np.allclose(T @ np.linalg.inv(T), np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# camera_center_in_A
# ---------------------------------------------------------------------------

class TestCameraCenterInA:

    def test_identity_rotation_is_neg_t(self):
        """With R=I, camera centre = -t."""
        t = np.array([1., 2., 3.])
        center = CameraAlignment.camera_center_in_A(np.eye(3), t)
        assert np.allclose(center, -t)

    def test_zero_translation(self):
        """Camera at origin regardless of R."""
        R = _rot_z(45.)
        center = CameraAlignment.camera_center_in_A(R, np.zeros(3))
        assert np.allclose(center, np.zeros(3))

    def test_90_degree_rotation_z(self):
        """
        R = Rz(90°), t = [1, 0, 0].
        Center = -R^T @ t = -Rz(-90°) @ [1,0,0] = -[0,-1,0] = [0, 1, 0].
        """
        R = _rot_z(90.)
        t = np.array([1., 0., 0.])
        center = CameraAlignment.camera_center_in_A(R, t)
        assert np.allclose(center, np.array([0., 1., 0.]), atol=1e-8)

    def test_roundtrip_with_relative_pose_matrix(self):
        """Camera centre derived from matrix T should equal camera_center_in_A(R, t)."""
        R = _rot_z(30.)
        t = np.array([2., -1., 3.])
        T = CameraAlignment.relative_pose_to_matrix(R, t)
        center_direct = CameraAlignment.camera_center_in_A(R, t)
        # Camera centre from homogeneous matrix: -R^T t = T[:3,:3].T @ (-T[:3,3])
        center_from_T = -(T[:3, :3].T @ T[:3, 3])
        assert np.allclose(center_direct, center_from_T, atol=1e-10)
