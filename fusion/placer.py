"""fusion/placer.py — estimate VGGT depth scale from SMPL-X metric bone depths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


# SMPL-X 55-joint kinematic tree: (proximal, distal) pairs for long bones.
# Using large bones that have consistent metric lengths and clear depth separation.
_LONG_BONES = [
    (16, 18),  # left humerus:  left shoulder → left elbow
    (17, 19),  # right humerus: right shoulder → right elbow
    (1,  4),   # left femur:    left hip → left knee
    (2,  5),   # right femur:   right hip → right knee
    (4,  7),   # left tibia:    left knee → left ankle
    (5,  8),   # right tibia:   right knee → right ankle
]


class BodyPlacer:
    """Estimate the metric scale factor of VGGT depth maps.

    VGGT depth is accurate up to an unknown global scale.  This class recovers
    that scale by comparing, for each long bone visible in a camera:

        Δz_smplx  — depth difference of the two endpoints from SMPL-X 3D
                    keypoints (metric, metres, translation-independent).
        Δz_vggt   — depth difference of the same endpoints sampled from the
                    VGGT depth map at the projected 2D positions.

        s = Δz_smplx / Δz_vggt

    The median of all valid (camera, person, frame, bone) samples is returned.

    Args:
        scene_output_dir: Scene output directory.  Must contain:
            - ``vggt_cameras.npz``  (extrinsics, intrinsics, original_coords,
              original_size [W,H], valid, camera_names)
            - ``vggt_depth.npz``    (depth uint16 mm, depth_conf float16,
              depth_valid bool)
            - Camera subdirectories with a ``body_data/`` folder, sorted in the
              same order as the K dimension in the VGGT arrays.
    """

    def __init__(self, scene_output_dir: str | Path) -> None:
        self.scene_dir = Path(scene_output_dir)

        cam_npz = np.load(self.scene_dir / "vggt_cameras.npz")
        depth_npz = np.load(self.scene_dir / "vggt_depth.npz")

        # (T, K, 3, 4) float32 — camera-from-world, OpenCV convention
        self.extrinsics = cam_npz["extrinsics"]
        # (T, K, 3, 3) float32
        self.intrinsics = cam_npz["intrinsics"]
        # (T, K, 4) float32 — [x1,y1,x2,y2] in 518-space corresponding to original image
        self.original_coords = cam_npz["original_coords"]
        # (T, K, 2) int32  — [W_orig, H_orig] of the frame before padding
        self.original_size = cam_npz["original_size"]
        # (T, K) bool
        self.cam_valid = cam_npz["valid"]
        # (K,) bytes
        self.camera_names = cam_npz["camera_names"]

        # (T, K, 518, 518) uint16 mm
        self.depth_mm = depth_npz["depth"]
        # (T, K, 518, 518) float16
        self.depth_conf = depth_npz["depth_conf"]
        # (T, K) bool
        self.depth_valid = depth_npz["depth_valid"]

        self.T, self.K = self.cam_valid.shape

        # Camera dirs sorted in the same order as the K axis
        self._cam_dirs: list[Path] = sorted(
            d for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_scale(
        self,
        conf_threshold: float = 0.5,
        min_delta_z: float = 0.05,
        joint_conf_threshold: float = 0.3,
    ) -> float:
        """Return the median VGGT depth scale factor (metres per VGGT unit).

        Args:
            conf_threshold: Minimum VGGT depth confidence to accept a sample.
            min_delta_z: Minimum |Δz_smplx| in metres required to use a bone.
            joint_conf_threshold: Minimum ``pred_joint_confidence`` for both
                endpoints of a bone to be considered visible in that frame.
                Ignored if the body npz has no ``pred_joint_confidence`` array.

        Returns:
            Scalar ``s`` such that ``depth_metric_m = s * depth_vggt``.

        Raises:
            RuntimeError: If no valid bone samples were found.
        """
        samples: list[float] = []

        for k, cam_dir in enumerate(self._cam_dirs):
            for body_file in sorted((cam_dir / "body_data").glob("person_*.npz")):
                samples.extend(
                    self._collect_scale_samples(
                        k, body_file, conf_threshold, min_delta_z, joint_conf_threshold
                    )
                )

        if not samples:
            raise RuntimeError(
                "No valid bone depth samples found. "
                "Check that body_data/ and vggt_depth.npz exist and overlap in frame indices."
            )

        scale = float(np.median(samples))
        return scale

    def estimate_root_translation(
        self,
        k: int,
        body_file: Path,
        scale: float,
        conf_threshold: float = 0.5,
        joint_conf_threshold: float = 0.3,
    ) -> np.ndarray:
        """Estimate the body root translation in camera space for every frame.

        For each visible joint j, back-projects its 2D position through the
        (already scaled) VGGT depth map to get its metric 3D camera-space
        position P_j, then recovers the root translation as:

            t_root_j = P_j − kp3d[j]

        where ``kp3d[j]`` is the body-centric 3D keypoint (pred_keypoints_3d).
        The coordinate-wise median over all valid per-joint estimates is
        returned for each frame.

        Args:
            k: Camera index (0-based, matching the K axis of VGGT arrays).
            body_file: Path to a ``person_*.npz`` file.
            scale: Metric scale factor obtained from :meth:`estimate_scale`.
            conf_threshold: Minimum VGGT depth confidence.
            joint_conf_threshold: Minimum ``pred_joint_confidence`` for a
                joint to be used.  Ignored if the key is absent.

        Returns:
            ``(T_local, 3)`` float32 root translations in world space (cam0
            frame after VGGT re-rooting).  Frames with no valid joints
            contain NaN.
        """
        data = np.load(body_file, allow_pickle=False)

        required = {"pred_keypoints_3d", "pred_keypoints_2d", "frame_indices"}
        if not required.issubset(data.files):
            raise KeyError(f"Missing required keys in {body_file}")

        kp3d = data["pred_keypoints_3d"]   # (T_local, J, 3) body-centric camera space
        kp2d = data["pred_keypoints_2d"]   # (T_local, J, 2+) original image pixels
        frame_indices = data["frame_indices"]
        joint_conf = (
            data["pred_joint_confidence"]
            if "pred_joint_confidence" in data.files
            else None
        )

        T_local = len(frame_indices)
        J = kp3d.shape[1]
        root_translations = np.full((T_local, 3), np.nan, dtype=np.float32)

        for local_t, global_t in enumerate(frame_indices):
            if global_t >= self.T:
                continue
            if not self.cam_valid[global_t, k]:
                continue
            if not self.depth_valid[global_t, k]:
                continue

            depth_frame = self.depth_mm[global_t, k].astype(np.float32) / 1000.0 * scale
            conf_frame  = self.depth_conf[global_t, k].astype(np.float32)

            intr = self.intrinsics[global_t, k]   # (3, 3) VGGT 518-space intrinsics
            fx, fy = float(intr[0, 0]), float(intr[1, 1])
            cx, cy = float(intr[0, 2]), float(intr[1, 2])

            oc = self.original_coords[global_t, k]
            os = self.original_size[global_t, k]
            W_orig, H_orig = float(os[0]), float(os[1])

            per_joint: list[np.ndarray] = []

            for j in range(J):
                if joint_conf is not None and joint_conf[local_t, j] < joint_conf_threshold:
                    continue

                u_518, v_518 = self._orig_to_518(kp2d[local_t, j], oc, W_orig, H_orig)
                if not self._in_bounds(u_518, v_518):
                    continue

                d = float(map_coordinates(depth_frame, [[v_518], [u_518]], order=1)[0])
                c = float(map_coordinates(conf_frame,  [[v_518], [u_518]], order=1)[0])

                if c < conf_threshold or d <= 0.0:
                    continue

                # Back-project to metric 3D camera space
                P_j = np.array(
                    [(u_518 - cx) / fx * d, (v_518 - cy) / fy * d, d],
                    dtype=np.float32,
                )
                per_joint.append(P_j - kp3d[local_t, j].astype(np.float32))

            if per_joint:
                t_root_camk = np.median(np.stack(per_joint, axis=0), axis=0)
                # Transform from camera-k space to world (cam0) space using
                # VGGT extrinsics: p_world = R_k^T @ (p_camk - t_k)
                R_k = self.extrinsics[global_t, k, :3, :3].astype(np.float32)
                t_k = self.extrinsics[global_t, k, :3, 3].astype(np.float32)
                root_translations[local_t] = R_k.T @ (t_root_camk - t_k)

        return root_translations

    def estimate_global_orient(
        self,
        body_files_per_cam: dict[int, Path],
        joint_conf_threshold: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate body orientation in world (cam0) space via triangulation.

        For each frame, the 2D projections of the root, hips and shoulders are
        triangulated across all cameras that observe the person.  A coordinate
        frame is then built from the resulting 3D hip/shoulder positions,
        giving the body orientation without relying on SAM3D's predicted
        global_orient.

        Joints used (SMPL-X indices):
            0  — root/pelvis
            1  — left hip,   2  — right hip
            16 — left shoulder, 17 — right shoulder

        Args:
            body_files_per_cam: ``{camera_index: path_to_person_npz}`` for the
                same person across cameras.
            joint_conf_threshold: Minimum ``pred_joint_confidence`` for a joint
                to contribute its 2D observation to triangulation.

        Returns:
            ``(frame_indices, R)`` where:
            - ``frame_indices`` is ``(N,)`` int32 — global frame indices where
              an orientation was successfully estimated.
            - ``R`` is ``(N, 3, 3)`` float32 — rotation matrices in cam0 space.
              Columns are the body's right / up / forward axes in world space.
        """
        # TODO: make joint selection dynamic — pick joints with highest
        # pred_joint_confidence across cameras rather than a fixed list.
        _ORIENT_JOINTS = [0, 1, 2, 16, 17]

        # Load per-camera data and build global_t → local_t lookup
        cam_data: dict[int, dict] = {}
        all_global_ts: set[int] = set()

        for k, path in body_files_per_cam.items():
            data = np.load(path, allow_pickle=False)
            if "pred_keypoints_2d" not in data.files:
                continue
            fi = data["frame_indices"]
            cam_data[k] = {
                "local_t_map": {int(gt): lt for lt, gt in enumerate(fi)},
                "kp2d": data["pred_keypoints_2d"],
                "joint_conf": (
                    data["pred_joint_confidence"]
                    if "pred_joint_confidence" in data.files
                    else None
                ),
            }
            all_global_ts.update(fi.tolist())

        frame_indices_out: list[int] = []
        R_out: list[np.ndarray] = []

        for global_t in sorted(all_global_ts):
            if global_t >= self.T:
                continue

            # Triangulate each orientation joint from all cameras that see it
            joint_3d: dict[int, np.ndarray] = {}
            for j in _ORIENT_JOINTS:
                observations: list[tuple[float, float]] = []
                proj_mats: list[np.ndarray] = []

                for k, cd in cam_data.items():
                    if global_t not in cd["local_t_map"]:
                        continue
                    if not self.cam_valid[global_t, k]:
                        continue

                    local_t = cd["local_t_map"][global_t]

                    if cd["joint_conf"] is not None:
                        if cd["joint_conf"][local_t, j] < joint_conf_threshold:
                            continue

                    kp2d_j = cd["kp2d"][local_t, j]
                    oc = self.original_coords[global_t, k]
                    os = self.original_size[global_t, k]
                    u_518, v_518 = self._orig_to_518(
                        kp2d_j, oc, float(os[0]), float(os[1])
                    )
                    if not self._in_bounds(u_518, v_518):
                        continue

                    observations.append((u_518, v_518))
                    proj_mats.append(
                        self.intrinsics[global_t, k] @ self.extrinsics[global_t, k]
                    )

                if len(observations) >= 2:
                    joint_3d[j] = self._triangulate_dlt(observations, proj_mats)

            # Need at least the hips or the shoulders to build a frame
            has_hips = (1 in joint_3d and 2 in joint_3d)
            has_shoulders = (16 in joint_3d and 17 in joint_3d)
            if not (has_hips or has_shoulders):
                continue

            R = self._build_orient_matrix(joint_3d)
            frame_indices_out.append(global_t)
            R_out.append(R)

        if not frame_indices_out:
            return np.empty(0, dtype=np.int32), np.empty((0, 3, 3), dtype=np.float32)

        return (
            np.array(frame_indices_out, dtype=np.int32),
            np.stack(R_out, axis=0),
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _triangulate_dlt(
        observations: list[tuple[float, float]],
        proj_matrices: list[np.ndarray],
    ) -> np.ndarray:
        """Linear (DLT) triangulation from N ≥ 2 camera observations.

        For each camera, the projection constraint ``x × (P @ X) = 0`` gives
        two independent linear equations.  All equations are stacked and the
        null-space solution is found via SVD.

        Args:
            observations: 2D joint positions ``(u, v)`` in 518-space per camera.
            proj_matrices: ``(3, 4)`` projection matrices ``K @ [R|t]`` per camera.

        Returns:
            ``(3,)`` world-space position.
        """
        rows = []
        for (u, v), P in zip(observations, proj_matrices):
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        A = np.stack(rows, axis=0)           # (2N, 4)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]                           # smallest singular vector
        return (X[:3] / X[3]).astype(np.float32)

    @staticmethod
    def _build_orient_matrix(joint_3d: dict[int, np.ndarray]) -> np.ndarray:
        """Build a 3×3 rotation matrix from triangulated hip/shoulder positions.

        Convention — columns are body axes in world space:
            col 0 (x): right  (left hip → right hip)
            col 1 (y): up     (mid-hips → mid-shoulders)
            col 2 (z): forward (right-hand rule, x × y)

        The three vectors are orthonormalised via Gram-Schmidt.
        """
        right_dir = np.zeros(3, dtype=np.float32)
        up_dir    = np.zeros(3, dtype=np.float32)

        if 1 in joint_3d and 2 in joint_3d:
            mid_hips = (joint_3d[1] + joint_3d[2]) / 2.0
            right_dir += joint_3d[2] - joint_3d[1]   # left_hip → right_hip
        else:
            mid_hips = joint_3d.get(0, np.zeros(3, dtype=np.float32))

        if 16 in joint_3d and 17 in joint_3d:
            mid_shoulders = (joint_3d[16] + joint_3d[17]) / 2.0
            right_dir += joint_3d[17] - joint_3d[16]  # left_shoulder → right_shoulder
        else:
            mid_shoulders = None

        if mid_shoulders is not None:
            up_dir = mid_shoulders - mid_hips
        else:
            up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Gram-Schmidt orthonormalisation: fix x first, then orthogonalise y, derive z
        x = right_dir / (np.linalg.norm(right_dir) + 1e-8)
        y = up_dir - np.dot(up_dir, x) * x
        y = y / (np.linalg.norm(y) + 1e-8)
        z = np.cross(x, y)

        return np.stack([x, y, z], axis=1).astype(np.float32)  # (3, 3)

    def apply_scale(
        self,
        scale: float,
        output_dir: str | Path | None = None,
    ) -> None:
        """Rescale VGGT depth maps and extrinsic translations and write to disk.

        The extrinsic translation column (t in [R|t]) is multiplied by ``scale``
        so that it is expressed in metres.  The depth maps are similarly rescaled.
        Files are written as ``vggt_cameras.npz`` and ``vggt_depth.npz`` in
        ``output_dir`` (defaults to ``scene_output_dir``).

        Args:
            scale: Value returned by :meth:`estimate_scale`.
            output_dir: Destination directory.  Defaults to ``scene_output_dir``.
        """
        out = Path(output_dir) if output_dir is not None else self.scene_dir

        extrinsics_scaled = self.extrinsics.copy()
        extrinsics_scaled[..., 3] *= scale  # column 3 is the translation vector

        np.savez_compressed(
            out / "vggt_cameras.npz",
            extrinsics=extrinsics_scaled,
            intrinsics=self.intrinsics,
            original_coords=self.original_coords,
            original_size=self.original_size,
            valid=self.cam_valid,
            camera_names=self.camera_names,
        )

        depth_m = self.depth_mm.astype(np.float32) / 1000.0 * scale
        depth_mm_scaled = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

        np.savez_compressed(
            out / "vggt_depth.npz",
            depth=depth_mm_scaled,
            depth_conf=self.depth_conf,
            depth_valid=self.depth_valid,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_scale_samples(
        self,
        k: int,
        body_file: Path,
        conf_threshold: float,
        min_delta_z: float,
        joint_conf_threshold: float,
    ) -> list[float]:
        """Yield s=Δz_smplx/Δz_vggt samples for one camera and one person."""
        data = np.load(body_file, allow_pickle=False)

        required = {"pred_keypoints_3d", "pred_keypoints_2d", "frame_indices"}
        if not required.issubset(data.files):
            return []

        kp3d = data["pred_keypoints_3d"]   # (T_local, J, 3) metres
        kp2d = data["pred_keypoints_2d"]   # (T_local, J, 2) or (T_local, J, 3) pixels
        frame_indices = data["frame_indices"]  # (T_local,) maps local row → global frame
        # (T_local, J) float32 in [0,1]; None if not present in this npz
        joint_conf = data["pred_joint_confidence"] if "pred_joint_confidence" in data.files else None

        samples: list[float] = []

        for local_t, global_t in enumerate(frame_indices):
            if global_t >= self.T:
                continue
            if not self.cam_valid[global_t, k]:
                continue
            if not self.depth_valid[global_t, k]:
                continue

            depth_frame = self.depth_mm[global_t, k].astype(np.float32) / 1000.0
            conf_frame  = self.depth_conf[global_t, k].astype(np.float32)

            oc = self.original_coords[global_t, k]   # [x1,y1,x2,y2] in 518-space
            os = self.original_size[global_t, k]     # [W_orig, H_orig]
            W_orig, H_orig = float(os[0]), float(os[1])

            for j_a, j_b in _LONG_BONES:
                if j_a >= kp3d.shape[1] or j_b >= kp3d.shape[1]:
                    continue

                # Skip if either joint is occluded / low-confidence in this frame
                if joint_conf is not None:
                    if (joint_conf[local_t, j_a] < joint_conf_threshold or
                            joint_conf[local_t, j_b] < joint_conf_threshold):
                        continue

                dz_smplx = float(kp3d[local_t, j_b, 2] - kp3d[local_t, j_a, 2])

                if abs(dz_smplx) < min_delta_z:
                    continue

                u_a, v_a = self._orig_to_518(kp2d[local_t, j_a], oc, W_orig, H_orig)
                u_b, v_b = self._orig_to_518(kp2d[local_t, j_b], oc, W_orig, H_orig)

                if not (self._in_bounds(u_a, v_a) and self._in_bounds(u_b, v_b)):
                    continue

                d_a = float(map_coordinates(depth_frame, [[v_a], [u_a]], order=1)[0])
                d_b = float(map_coordinates(depth_frame, [[v_b], [u_b]], order=1)[0])
                c_a = float(map_coordinates(conf_frame,  [[v_a], [u_a]], order=1)[0])
                c_b = float(map_coordinates(conf_frame,  [[v_b], [u_b]], order=1)[0])

                if c_a < conf_threshold or c_b < conf_threshold:
                    continue
                if d_a <= 0.0 or d_b <= 0.0:
                    continue

                dz_vggt = d_b - d_a

                # Reject if near-zero depth difference or opposite sign to SMPL-X
                if abs(dz_vggt) < 1e-4 or (dz_smplx * dz_vggt) <= 0:
                    continue

                s = dz_smplx / dz_vggt
                # Sanity check: physically plausible scale factors
                if 0.01 < s < 100.0:
                    samples.append(s)

        return samples

    @staticmethod
    def _orig_to_518(
        kp: np.ndarray,
        oc: np.ndarray,
        W_orig: float,
        H_orig: float,
    ) -> tuple[float, float]:
        """Map a 2D keypoint from original-image pixels to VGGT 518-space.

        Args:
            kp: Keypoint [u, v, ...] in original image pixels.
            oc: [x1, y1, x2, y2] bounding box in 518-space that the original
                image was padded/resized into.
            W_orig, H_orig: Original image dimensions in pixels.

        Returns:
            (u_518, v_518) in VGGT 518×518 pixel coordinates.
        """
        x1, y1, x2, y2 = oc
        u_518 = x1 + float(kp[0]) * (x2 - x1) / W_orig
        v_518 = y1 + float(kp[1]) * (y2 - y1) / H_orig
        return u_518, v_518

    @staticmethod
    def _in_bounds(u: float, v: float, size: int = 518) -> bool:
        return 0.0 <= u < size and 0.0 <= v < size
