"""Per-joint confidence estimation from SAM3D body outputs.

Two sources of occlusion are measured at mesh-vertex level and then
aggregated to the J body joints via K-nearest-vertex averaging:

1. External occlusion – a vertex is externally occluded if its 2-D
   projection does not lie within the person's SAM3 segmentation mask.
   This catches objects (or other people) blocking parts of the body.

2. Self-occlusion – a vertex is self-occluded when another vertex of the
   same mesh is closer to the camera at the same image pixel (z-buffer
   test).  This catches the back of the body, limbs crossing, etc.

The camera principal point is recovered analytically from the already-
projected 2-D keypoints and their 3-D camera-space counterparts, so no
extra intrinsics need to be stored.

Combined per-joint confidence is in [0, 1]:  1 = fully visible.
"""

from __future__ import annotations

import numpy as np


class ConfidenceEstimator:
    """Estimate per-joint confidence from SAM3D body outputs.

    Parameters
    ----------
    n_nearest_vertices : int
        Number of nearest mesh vertices to average per joint (KNN).
    depth_threshold : float
        Minimum z-depth difference (metres) to label a vertex as
        self-occluded.  A small slack avoids false positives from
        projection discretisation onto integer pixels.
    """

    def __init__(
        self,
        n_nearest_vertices: int = 8,
        depth_threshold: float = 0.005,
    ):
        self.n_nearest = n_nearest_vertices
        self.depth_threshold = depth_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        pred_vertices: np.ndarray,      # (V, 3)  camera space (or body-centric if pred_cam_t != 0)
        pred_keypoints_3d: np.ndarray,  # (J, 3)  same space as pred_vertices
        pred_keypoints_2d: np.ndarray | None,  # (J, 2)  full-image pixel coords; unused if cx_cy given
        pred_cam_t: np.ndarray,         # (3,)  added to vertices/keypoints to get camera space
        focal_length: float,
        person_mask: np.ndarray,        # (H, W) uint8 / bool
        img_h: int,
        img_w: int,
        cx_cy: tuple[float, float] | None = None,
    ) -> np.ndarray:                    # (J,) float32 in [0, 1]
        """Compute per-joint occlusion-based confidence.

        Parameters
        ----------
        pred_vertices : (V, 3)
            Mesh vertices.  Pass in camera space with ``pred_cam_t=zeros``,
            or in body-centric space with the corresponding ``pred_cam_t``.
        pred_keypoints_3d : (J, 3)
            3-D joint positions in the same coordinate system as
            ``pred_vertices``.  Used for KNN vertex-to-joint aggregation.
        pred_keypoints_2d : (J, 2) or None
            Pre-projected 2-D joint positions in full-image pixel coords.
            Used to recover the camera principal point analytically.
            Not required when ``cx_cy`` is provided.
        pred_cam_t : (3,)
            Camera translation that converts body-centric -> camera space:
            ``p_cam = p_body + pred_cam_t``.  Pass zeros if vertices are
            already in camera space.
        focal_length : float
            Focal length in pixels (fx = fy assumed).
        person_mask : (H, W)
            Binary segmentation mask from SAM3: non-zero where this person
            is visible in the frame.
        img_h, img_w : int
            Full image dimensions in pixels.
        cx_cy : (cx, cy) or None
            Pre-computed camera principal point in pixels.  When provided,
            skips the analytic recovery from ``pred_keypoints_2d``.

        Returns
        -------
        confidence : (J,) float32
            Per-joint confidence in [0, 1].
        """
        # Camera-space coordinates
        verts_cam = pred_vertices + pred_cam_t[None, :]     # (V, 3)
        kpts_cam  = pred_keypoints_3d + pred_cam_t[None, :] # (J, 3)

        # Recover camera principal point
        if cx_cy is not None:
            cx, cy = cx_cy
        else:
            cx, cy = self._recover_principal_point(
                kpts_cam, pred_keypoints_2d, focal_length
            )

        # Project vertices to image pixels
        verts_2d = self._project(verts_cam, focal_length, cx, cy)  # (V, 2)

        # --- Self-occlusion via z-buffer ---
        self_occ = self._self_occlusion(verts_cam, verts_2d, img_h, img_w)

        # --- External occlusion via SAM3 mask ---
        ext_vis = self._external_visibility(verts_2d, person_mask, img_h, img_w)

        # Vertex is visible if: in front of camera AND in mask AND not self-occluded
        in_front = verts_cam[:, 2] > 0
        vertex_vis = (in_front & ext_vis & ~self_occ).astype(np.float32)

        # Aggregate to joints via KNN over 3-D body space
        return self._aggregate_to_joints(vertex_vis, pred_vertices, pred_keypoints_3d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recover_principal_point(
        kpts_cam: np.ndarray,   # (J, 3)
        kpts_2d: np.ndarray,    # (J, 2)  known projections in image space
        focal_length: float,
    ) -> tuple[float, float]:
        """Analytically recover (cx, cy) from known (3D, 2D) correspondences.

        For each valid joint j (depth > 0):
            u_j = (x_cam_j / z_cam_j) * f + cx
            -> cx = u_j - (x_cam_j / z_cam_j) * f

        The median over all valid joints is returned for robustness.
        """
        valid = kpts_cam[:, 2] > 1e-5
        if not valid.any():
            return 0.0, 0.0
        z = kpts_cam[valid, 2]
        cx = float(np.median(kpts_2d[valid, 0] - kpts_cam[valid, 0] / z * focal_length))
        cy = float(np.median(kpts_2d[valid, 1] - kpts_cam[valid, 1] / z * focal_length))
        return cx, cy

    @staticmethod
    def _project(
        points_cam: np.ndarray,  # (N, 3)
        focal_length: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:             # (N, 2)  pixel coords (u, v)
        """Standard perspective projection with the recovered principal point."""
        z = np.maximum(points_cam[:, 2], 1e-6)
        u = points_cam[:, 0] / z * focal_length + cx
        v = points_cam[:, 1] / z * focal_length + cy
        return np.stack([u, v], axis=-1)

    def _self_occlusion(
        self,
        verts_cam: np.ndarray,   # (V, 3)  camera space
        verts_2d: np.ndarray,    # (V, 2)  pixel coords
        img_h: int,
        img_w: int,
    ) -> np.ndarray:             # (V,) bool — True = self-occluded
        """Z-buffer self-occlusion test.

        A vertex is self-occluded when another vertex of the same mesh is
        strictly closer to the camera at the same discrete pixel location.
        """
        z = verts_cam[:, 2]
        u = np.round(verts_2d[:, 0]).astype(np.int32)
        v = np.round(verts_2d[:, 1]).astype(np.int32)
        in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 0)

        # Build z-buffer: minimum depth at each pixel (vectorised scatter)
        depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)
        valid_u = u[in_bounds]
        valid_v = v[in_bounds]
        valid_z = z[in_bounds].astype(np.float32)
        np.minimum.at(depth_buffer, (valid_v, valid_u), valid_z)

        # A vertex is self-occluded if something closer exists at its pixel
        self_occ = np.zeros(len(z), dtype=bool)
        idx = np.where(in_bounds)[0]
        self_occ[idx] = valid_z > depth_buffer[valid_v, valid_u] + self.depth_threshold

        return self_occ

    @staticmethod
    def _external_visibility(
        verts_2d: np.ndarray,    # (V, 2)  pixel coords
        person_mask: np.ndarray, # (H, W)  non-zero = this person is visible
        img_h: int,
        img_w: int,
    ) -> np.ndarray:             # (V,) bool — True = vertex falls inside mask
        """Check each vertex's projected pixel against the SAM3 person mask.

        A vertex is externally visible if the segmentation mask is set at
        its projected location, meaning no external object blocks it.
        """
        u = np.round(verts_2d[:, 0]).astype(np.int32)
        v = np.round(verts_2d[:, 1]).astype(np.int32)
        in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        visible = np.zeros(len(u), dtype=bool)
        mask_bool = person_mask.astype(bool)
        idx = np.where(in_bounds)[0]
        visible[idx] = mask_bool[v[idx], u[idx]]
        return visible

    def _aggregate_to_joints(
        self,
        vertex_scores: np.ndarray,  # (V,) float32
        vertices_3d: np.ndarray,    # (V, 3)  body-centric space
        keypoints_3d: np.ndarray,   # (J, 3)  body-centric space
    ) -> np.ndarray:                # (J,) float32
        """Average vertex visibility scores over K nearest vertices per joint.

        Euclidean distance in body-centric 3-D space is used for assignment,
        so joints covering dense mesh regions (torso) and sparse regions
        (fingertips) are handled uniformly.
        """
        K = min(self.n_nearest, vertices_3d.shape[0])

        # Pairwise distances (J, V)
        diff = keypoints_3d[:, None, :] - vertices_3d[None, :, :]  # (J, V, 3)
        dists = np.linalg.norm(diff, axis=-1)                       # (J, V)

        # K nearest vertices per joint, then mean their visibility
        knn_idx = np.argpartition(dists, K, axis=1)[:, :K]         # (J, K)
        confidence = vertex_scores[knn_idx].mean(axis=1)            # (J,)
        return confidence.astype(np.float32)
