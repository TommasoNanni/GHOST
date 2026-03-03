"""
Unit tests for utilities/geometry.py

All tests are deterministic (fixed inputs, CPU only).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from utilities.geometry import skew_symmetric, project_to_2d


# ---------------------------------------------------------------------------
# skew_symmetric
# ---------------------------------------------------------------------------

class TestSkewSymmetric:

    def test_antisymmetric(self):
        """S + S^T must be zero for any input."""
        v = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])  # B=1, T=2
        S = skew_symmetric(v)
        assert torch.allclose(S + S.transpose(-1, -2), torch.zeros_like(S), atol=1e-6)

    def test_diagonal_is_zero(self):
        v = torch.tensor([[[7., -3., 2.]]])
        S = skew_symmetric(v)
        diag = torch.stack([S[0, 0, i, i] for i in range(3)])
        assert torch.allclose(diag, torch.zeros(3), atol=1e-6)

    def test_known_values_e1(self):
        """v = [1, 0, 0]  ->  S = [[0,0,0],[0,0,-1],[0,1,0]]."""
        v = torch.tensor([[[1., 0., 0.]]])  # B=1, T=1
        S = skew_symmetric(v)
        expected = torch.tensor([[[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]]])
        assert torch.allclose(S, expected, atol=1e-6)

    def test_known_values_e2(self):
        """v = [0, 1, 0]  ->  S = [[0,0,1],[0,0,0],[-1,0,0]]."""
        v = torch.tensor([[[0., 1., 0.]]])
        S = skew_symmetric(v)
        expected = torch.tensor([[[[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]]])
        assert torch.allclose(S, expected, atol=1e-6)

    def test_known_values_e3(self):
        """v = [0, 0, 1]  ->  S = [[0,-1,0],[1,0,0],[0,0,0]]."""
        v = torch.tensor([[[0., 0., 1.]]])
        S = skew_symmetric(v)
        expected = torch.tensor([[[[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]]])
        assert torch.allclose(S, expected, atol=1e-6)

    def test_cross_product_e1_cross_e2(self):
        """S(e1) @ e2 == e1 x e2 == e3 == [0, 0, 1]."""
        v = torch.tensor([[[1., 0., 0.]]])
        w = torch.tensor([0., 1., 0.])
        S = skew_symmetric(v)
        result = S[0, 0] @ w
        assert torch.allclose(result, torch.tensor([0., 0., 1.]), atol=1e-6)

    def test_cross_product_e2_cross_e3(self):
        """S(e2) @ e3 == e2 x e3 == e1 == [1, 0, 0]."""
        v = torch.tensor([[[0., 1., 0.]]])
        w = torch.tensor([0., 0., 1.])
        S = skew_symmetric(v)
        result = S[0, 0] @ w
        assert torch.allclose(result, torch.tensor([1., 0., 0.]), atol=1e-6)

    def test_output_shape(self):
        B, T = 3, 5
        v = torch.randn(B, T, 3)
        S = skew_symmetric(v)
        assert S.shape == (B, T, 3, 3)

    def test_zero_vector_gives_zero_matrix(self):
        v = torch.zeros(1, 1, 3)
        S = skew_symmetric(v)
        assert torch.allclose(S, torch.zeros(1, 1, 3, 3), atol=1e-6)

    def test_linear_in_v(self):
        """S(2v) == 2 * S(v)."""
        v = torch.tensor([[[3., -1., 2.]]])
        S_v  = skew_symmetric(v)
        S_2v = skew_symmetric(2 * v)
        assert torch.allclose(S_2v, 2 * S_v, atol=1e-6)


# ---------------------------------------------------------------------------
# project_to_2d
# ---------------------------------------------------------------------------

class TestProjectTo2D:

    def _identity_K(self, B, T):
        return torch.eye(3).reshape(1, 1, 3, 3).expand(B, T, 3, 3).clone()

    def test_output_shape(self):
        B, T, P, V = 2, 3, 4, 5
        vertices = torch.randn(B, T, P, V, 3).abs() + 0.5
        K = self._identity_K(B, T)
        out = project_to_2d(vertices, K)
        assert out.shape == (B, T, P, V, 3)

    def test_homogeneous_z_is_one(self):
        """After dividing by z, the last coordinate must always be 1."""
        B, T, P, V = 1, 2, 3, 4
        vertices = torch.randn(B, T, P, V, 3).abs() + 0.1  # ensure positive z
        K = self._identity_K(B, T)
        out = project_to_2d(vertices, K)
        assert torch.allclose(out[..., 2], torch.ones(B, T, P, V), atol=1e-5)

    def test_identity_intrinsics_axis_point(self):
        """Point at (0, 0, d) with K=I projects to (0, 0, 1)."""
        B, T, P, V = 1, 1, 1, 1
        vertices = torch.tensor([0., 0., 7.]).reshape(B, T, P, V, 3)
        K = self._identity_K(B, T)
        out = project_to_2d(vertices, K)
        assert torch.allclose(out[0, 0, 0, 0], torch.tensor([0., 0., 1.]), atol=1e-6)

    def test_pinhole_known_projection(self):
        """
        Point at (2, 3, 10) in camera frame with f=500, cx=320, cy=240:
            u = 2/10 * 500 + 320 = 420
            v = 3/10 * 500 + 240 = 390
        """
        B, T, P, V = 1, 1, 1, 1
        f, cx, cy = 500., 320., 240.
        K = torch.tensor([[f, 0., cx], [0., f, cy], [0., 0., 1.]]).reshape(1, 1, 3, 3)
        vertices = torch.tensor([2., 3., 10.]).reshape(B, T, P, V, 3)
        out = project_to_2d(vertices, K)
        assert out[0, 0, 0, 0, 0].item() == pytest.approx(420., abs=1e-3)
        assert out[0, 0, 0, 0, 1].item() == pytest.approx(390., abs=1e-3)
        assert out[0, 0, 0, 0, 2].item() == pytest.approx(1., abs=1e-6)

    def test_pinhole_on_principal_axis(self):
        """Point on optical axis (cx, cy, z) should project to (cx, cy, 1)."""
        B, T, P, V = 1, 1, 1, 1
        f, cx, cy = 800., 640., 480.
        K = torch.tensor([[f, 0., cx], [0., f, cy], [0., 0., 1.]]).reshape(1, 1, 3, 3)
        vertices = torch.tensor([cx, cy, 1.]).reshape(B, T, P, V, 3)  # z=1 simplifies things
        # projected = K @ [cx, cy, 1]^T = [f*cx+cx, f*cy+cy, 1] — wait, that's wrong
        # The vertex is in camera frame. x=cx means the ray has slope cx in pixels only when z=1.
        # Actually: proj = K @ [cx, cy, 1] then divide by z_proj
        # K @ [cx, cy, 1] = [f*cx + cx*1, f*cy + cy*1, 1] ... let me recalculate.
        # u = f*x + cx*z, so for x=cx, z=1: u = f*cx + cx → that's not on axis.
        # For a point ON the principal axis: x=0, y=0, z=d.
        # u = f*0/d + cx = cx, v = f*0/d + cy = cy.
        vertices = torch.tensor([0., 0., 5.]).reshape(B, T, P, V, 3)
        out = project_to_2d(vertices, K)
        assert out[0, 0, 0, 0, 0].item() == pytest.approx(cx, abs=1e-3)
        assert out[0, 0, 0, 0, 1].item() == pytest.approx(cy, abs=1e-3)

    def test_scaling_z_does_not_change_projection(self):
        """Scaling a point along the ray (multiply all coords by k) should give same projection."""
        B, T, P, V = 1, 1, 1, 1
        f, cx, cy = 300., 200., 150.
        K = torch.tensor([[f, 0., cx], [0., f, cy], [0., 0., 1.]]).reshape(1, 1, 3, 3)
        point = torch.tensor([1., 2., 4.]).reshape(B, T, P, V, 3)
        point_scaled = (3.0 * point)
        out1 = project_to_2d(point, K)
        out2 = project_to_2d(point_scaled, K)
        assert torch.allclose(out1, out2, atol=1e-5)
