"""fusion/placer.py — estimate VGGT depth scale from SMPL-X metric bone depths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as SciR


def _6d_to_aa_batch(sixd: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation (..., 6) to axis-angle (..., 3).

    Interprets the 6 values as the first two rows of R — matches the training
    convention in fusion_dataset.py.
    """
    shape = sixd.shape[:-1]
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)  # (N, 3, 3) — rows are b1, b2, b3
    aa = SciR.from_matrix(R).as_rotvec()
    return aa.reshape(shape + (3,)).astype(np.float32)


# SMPL-X 55-joint kinematic tree: (proximal, distal) pairs for long bones.
# Using large bones that have consistent metric lengths and clear depth separation.
_LONG_BONES = [
    (16, 20),  # left arm:   left shoulder → left wrist   (~9% RMS vs ~19% for humerus alone)
    (17, 21),  # right arm:  right shoulder → right wrist
    (1,  7),   # left leg:   left hip → left ankle        (~5% RMS vs ~17% for femur/tibia)
    (2,  8),   # right leg:  right hip → right ankle
]

# Maps SMPL-X joint index → MHR70 joint index.
# pred_keypoints_2d from SAM3D uses MHR70 ordering (70 joints).
# FK joints use SMPL-X ordering (55 joints).
_SMPLX_TO_MHR70 = {
    1: 9,   2: 10,   # left/right hip
    4: 11,  5: 12,   # left/right knee
    7: 13,  8: 14,   # left/right ankle
    12: 69,          # neck
    16: 5,  17: 6,   # left/right shoulder
    18: 7,  19: 8,   # left/right elbow
    20: 62, 21: 41,  # left/right wrist
}

# Subset used for PnP and Procrustes DLT — joints with lowest skeleton-definition
# mismatch between SMPL-X and MHR70 (mean error ≤ 4cm, std ≤ 0.9cm from empirical
# analysis on BBQ_001_juggle + BBQ_001_guitar).  Knees (~7cm, pose-dependent z-bias),
# neck (~4cm, std 2cm), elbows and wrists (~3-5cm, std 1.3cm) are excluded.
_SMPLX_TO_MHR70_ALIGN = {
    1: 9,   2: 10,   # left/right hip      (~2cm, std 0.4cm)
    7: 13,  8: 14,   # left/right ankle    (~3cm, std 0.7cm)
    16: 5,  17: 6,   # left/right shoulder (~3.5cm, std 0.7cm)
}


class BodyPlacer:
    """Estimate the metric scale factor of VGGT depth maps.

    VGGT depth is accurate up to an unknown global scale.  This class recovers
    that scale by comparing, for each long bone visible in a camera:

        Δz_smplx  — depth difference of the two endpoints from SMPL-X FK joints
                    in camera-oriented space (metric, metres, translation-independent).
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
        smplx_model_path: Path to SMPLX_NEUTRAL.pkl.  Required for FK-based
            scale estimation and root translation.
    """

    def __init__(self, scene_output_dir: str | Path, smplx_model_path: str | Path) -> None:
        self.scene_dir = Path(scene_output_dir)

        import smplx as smplx_lib
        smplx_path = Path(smplx_model_path)
        create_kwargs: dict = {"model_type": "smplx"}
        if smplx_path.is_file():
            create_kwargs["ext"] = smplx_path.suffix.lstrip(".")
        self._smplx_model = smplx_lib.create(
            str(smplx_model_path),
            **create_kwargs,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).eval()
        self._smplx_device = torch.device("cpu")

        cam_npz = np.load(self.scene_dir / "vggt_cameras.npz")
        # Depth maps can be large (T*K*518*518 ~ GB); memory-map so only accessed
        # slices are paged in rather than loading the full array at construction time.
        depth_npz = np.load(self.scene_dir / "vggt_depth.npz", mmap_mode="r")

        # (T, K, 3, 4) float32 — camera-from-world, OpenCV convention
        self.extrinsics = cam_npz["extrinsics"]
        # (T, K, 3, 3) float32
        self.intrinsics = cam_npz["intrinsics"]
        # (T, K, 4) float32 — [x1,y1,x2,y2] in 518-space corresponding to original image
        self.original_coords = cam_npz["original_coords"]
        # (T, K, 2) int32  — [W_orig, H_orig] of the frame before padding
        self.original_size = cam_npz["original_size"]
        # (T, K) bool — fall back to non-NaN extrinsics for files saved before the valid-flag fix
        self.cam_valid = cam_npz["valid"]
        if not self.cam_valid.any():
            self.cam_valid = ~np.isnan(self.extrinsics[:, :, 0, 0])
        # (K,) bytes
        self.camera_names = cam_npz["camera_names"]

        # (T, K, 518, 518) uint16 mm — memory-mapped
        self.depth_mm = depth_npz["depth"]
        # (T, K, 518, 518) float16 — memory-mapped
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

    def estimate_scale_per_frame(
        self,
        conf_threshold: float = 0.5,
        min_delta_z: float = 0.05,
        fused_betas_map: dict[Path, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return per-frame VGGT depth scale ``(T,)`` in metres per VGGT unit.

        For each global frame, all bone scale samples visible across cameras
        and persons are collected and their median is taken.  Frames with no
        samples receive the global median as a fallback.

        Returns:
            ``(T,)`` float32 — one scale per global frame index.

        Raises:
            RuntimeError: If no valid bone samples were found anywhere.
        """
        from collections import defaultdict
        frame_samples: dict[int, list[float]] = defaultdict(list)

        for k, cam_dir in enumerate(self._cam_dirs):
            for body_file in sorted((cam_dir / "body_data").glob("person_*.npz")):
                fused_betas = (
                    fused_betas_map.get(body_file) if fused_betas_map is not None else None
                )
                tagged = self._collect_scale_samples_tagged(
                    k, body_file, conf_threshold, min_delta_z, fused_betas,
                )
                for global_t, slist in tagged.items():
                    frame_samples[global_t].extend(slist)

        all_samples = [s for sl in frame_samples.values() for s in sl]
        if not all_samples:
            raise RuntimeError(
                "No valid bone depth samples found. "
                "Check that body_data/ and vggt_depth.npz exist and overlap in frame indices."
            )

        global_scale = float(np.median(all_samples))
        result = np.full(self.T, global_scale, dtype=np.float32)
        for global_t, slist in frame_samples.items():
            if 0 <= global_t < self.T and slist:
                result[global_t] = float(np.median(slist))

        return result

    def estimate_scale_triangulated(
        self,
        fused_pose_by_pid: dict[int, np.ndarray],
        min_cams: int = 2,
        fused_betas_map: dict[Path, np.ndarray] | None = None,
        frame_start: int = 0,
    ) -> np.ndarray:
        """Return per-frame VGGT depth scale using multi-view triangulation (metres / VGGT unit).

        For each (frame, person, bone), all cameras that observe both endpoints
        with sufficient confidence are used to triangulate the 3D joint positions
        via DLT — no depth map lookup.  Scale is then ``L_fk / L_vggt_3d``.

        This is robust to foreshortening: a bone viewed edge-on from one camera
        is correctly recovered by other cameras viewing it from a different angle.

        Args:
            min_cams: Minimum number of cameras that must observe both endpoints
                for the triangulation to be attempted.
            fused_betas_map: Optional ``{body_file: betas (10,)}`` mapping used
                to compute FK bone lengths (same semantics as in
                :meth:`estimate_scale`).
            fused_pose_by_pid: ``{pid: (T_scene, 54, 6)}`` fused body pose from
                the fusion model (6D, joints 1-54, indexed from ``frame_start``).
                Used for FK — never use raw per-camera SAM3D body_pose here.
            frame_start: Global frame index corresponding to index 0 of the
                ``fused_pose_by_pid`` arrays.

        Returns:
            ``(T,)`` float32 array — one scale per global frame index.
            Frames with no bone observations fall back to the global median.

        Raises:
            RuntimeError: If no valid triangulated bone samples were found.
        """
        from collections import defaultdict

        # ── Load body data per (cam_idx, pid) ─────────────────────────────────
        # cam_data[k][pid] = {gt_map, fk, kp2d}
        cam_data: dict[int, dict[int, dict]] = {}
        for k, cam_dir in enumerate(self._cam_dirs):
            cam_data[k] = {}
            for body_file in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(body_file.stem.split("_")[1])
                d = np.load(body_file, allow_pickle=False)
                required = {"smplx_betas", "smplx_body_pose", "pred_keypoints_2d", "frame_indices"}
                if not required.issubset(d.files):
                    continue

                fi = d["frame_indices"]
                betas = (
                    np.tile(fused_betas_map[body_file][np.newaxis], (len(fi), 1))
                    if fused_betas_map is not None and body_file in fused_betas_map
                    else d["smplx_betas"]
                )
                if pid not in fused_pose_by_pid:
                    continue
                fused_arr = fused_pose_by_pid[pid]   # (T_scene, 54, 6)
                t_fused = fi.astype(int) - frame_start
                body_pose_arr = np.zeros((len(fi), 63), dtype=np.float32)
                valid = np.where((t_fused >= 0) & (t_fused < len(fused_arr)))[0]
                if len(valid):
                    body_pose_arr[valid] = _6d_to_aa_batch(
                        fused_arr[t_fused[valid], :21]
                    ).reshape(len(valid), 63)
                fk = self._smplx_fk(betas, body_pose_arr,
                                    np.zeros((len(betas), 3), dtype=np.float32))

                gt_map = {int(gt): lt for lt, gt in enumerate(d["frame_indices"])}
                cam_data[k][pid] = {
                    "gt_map": gt_map,
                    "fk":     fk,
                    "kp2d":   d["pred_keypoints_2d"],
                }

        # ── Collect all (global_t, pid) pairs ─────────────────────────────────
        all_pids: set[int] = set()
        for k in cam_data:
            all_pids.update(cam_data[k].keys())

        global_ts_by_pid: dict[int, set[int]] = defaultdict(set)
        for k in cam_data:
            for pid, bd in cam_data[k].items():
                global_ts_by_pid[pid].update(bd["gt_map"].keys())

        frame_samples: dict[int, list[float]] = defaultdict(list)

        for pid in sorted(all_pids):
            for global_t in sorted(global_ts_by_pid[pid]):
                if global_t >= self.T:
                    continue

                for (j_a, j_b) in _LONG_BONES:
                    mhr_a = _SMPLX_TO_MHR70.get(j_a)
                    mhr_b = _SMPLX_TO_MHR70.get(j_b)
                    if mhr_a is None or mhr_b is None:
                        continue

                    pts_a: list[tuple[float, float]] = []
                    pts_b: list[tuple[float, float]] = []
                    Pmats: list[np.ndarray] = []
                    fk_lengths: list[float] = []

                    for k in range(self.K):
                        if not self.cam_valid[global_t, k]:
                            continue
                        if pid not in cam_data.get(k, {}):
                            continue
                        bd = cam_data[k][pid]
                        if global_t not in bd["gt_map"]:
                            continue
                        local_t = bd["gt_map"][global_t]

                        kp2d = bd["kp2d"]
                        if mhr_a >= kp2d.shape[1] or mhr_b >= kp2d.shape[1]:
                            continue

                        oc = self.original_coords[global_t, k]
                        os_ = self.original_size[global_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])

                        u_a, v_a = self._orig_to_vggt(kp2d[local_t, mhr_a], oc, W_orig, H_orig)
                        u_b, v_b = self._orig_to_vggt(kp2d[local_t, mhr_b], oc, W_orig, H_orig)
                        x1, y1, x2, y2 = oc
                        if not (x1 <= u_a < x2 and y1 <= v_a < y2
                                and x1 <= u_b < x2 and y1 <= v_b < y2):
                            continue

                        K_mat = self.intrinsics[global_t, k].astype(np.float64)
                        E_mat = self.extrinsics[global_t, k].astype(np.float64)
                        P = K_mat @ E_mat

                        pts_a.append((u_a, v_a))
                        pts_b.append((u_b, v_b))
                        Pmats.append(P)
                        fk_lengths.append(float(np.linalg.norm(
                            bd["fk"][local_t, j_b] - bd["fk"][local_t, j_a]
                        )))

                    if len(pts_a) < min_cams:
                        continue

                    try:
                        X_a = self._triangulate_dlt(pts_a, Pmats)
                        X_b = self._triangulate_dlt(pts_b, Pmats)
                    except Exception:
                        continue

                    L_vggt = float(np.linalg.norm(X_b - X_a))
                    if L_vggt < 1e-4:
                        continue

                    L_fk = float(np.median(fk_lengths))
                    if L_fk < 0.05:
                        continue

                    s = L_fk / L_vggt
                    if 0.1 < s < 100.0:
                        frame_samples[global_t].append(s)

        all_samples = [s for sl in frame_samples.values() for s in sl]
        if not all_samples:
            raise RuntimeError(
                "No valid triangulated bone samples found. "
                "Check that body_data/ files exist and at least two cameras share valid frames."
            )

        global_scale = float(np.median(all_samples))
        result = np.full(self.T, global_scale, dtype=np.float32)
        for global_t, slist in frame_samples.items():
            if slist:
                result[global_t] = float(np.median(slist))
        return result

    def estimate_procrustes_dlt(
        self,
        scale: float | np.ndarray,
        all_pids: set[int],
        pred_betas_by_pid: dict[int, np.ndarray],
        fused_pose_by_pid: dict[int, np.ndarray] | None = None,
        frame_start: int = 0,
        min_cams: int = 2,
        min_joints: int = 4,
    ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
        """Estimate root translation + global orient via multi-camera DLT + Procrustes.

        For each (person, frame):
          1. DLT-triangulate each SMPL-X joint across all cameras that observe it
             → world-frame 3D positions J_world[j] at metric scale.
          2. Run SMPL-X FK with the fused body_pose (if provided, else raw SAM3D
             per-camera body_pose) + pred betas, zero global_orient
             → canonical-frame joints J_can[j] rooted at the origin.
          3. Procrustes: find R, t minimising ||R @ J_can + t − J_world||²
             → global_orient matrix R and pelvis world translation t.

        Args:
            scale: Metric scale (metres per VGGT unit). Either a scalar or a
                   ``(T,)`` float32 array of per-frame scales from
                   :meth:`estimate_scale_per_frame`.
            all_pids: Ghost person IDs to process.
            pred_betas_by_pid: ``{pid: (10,)}`` shape coefficients for FK.
            fused_pose_by_pid: ``{pid: (T_scene, 54, 6)}`` fused body pose from
                the fusion model (6D, joints 1-54, indexed from ``frame_start``).
                When provided, used for FK instead of the raw per-camera SAM3D
                body_pose, giving a better canonical skeleton for Procrustes.
            frame_start: Global frame index corresponding to index 0 of the
                ``fused_pose_by_pid`` arrays. Must match the frame_start returned
                by :func:`build_fusion_tensors`.
            min_cams: Minimum cameras needed to DLT-triangulate a joint.
            min_joints: Minimum triangulated joints needed to run Procrustes.

        Returns:
            translations : ``{pid: {global_frame_idx: pelvis_world (3,)}}``
            orientations : ``{pid: {global_frame_idx: R (3,3)}}``
        """
        _JOINTS = sorted(_SMPLX_TO_MHR70_ALIGN.keys())

        # Pre-load body data once per (cam, pid) to avoid repeated file reads.
        cam_data_all: list[dict[int, dict]] = []
        for cam_dir in self._cam_dirs:
            cam_map: dict[int, dict] = {}
            for pid in sorted(all_pids):
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                if not bf.exists():
                    continue
                d = np.load(bf, allow_pickle=False)
                if not {"pred_keypoints_2d", "frame_indices", "smplx_body_pose"}.issubset(d.files):
                    continue
                fi = d["frame_indices"].astype(int)
                cam_map[pid] = {
                    "local_t": {int(g): int(l) for l, g in enumerate(fi)},
                    "kp2d":      d["pred_keypoints_2d"],
                    "body_pose": d["smplx_body_pose"],
                }
            cam_data_all.append(cam_map)

        translations: dict[int, dict[int, np.ndarray]] = {}
        orientations: dict[int, dict[int, np.ndarray]] = {}

        for pid in sorted(all_pids):
            betas = pred_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))

            all_frames: set[int] = set()
            for cm in cam_data_all:
                if pid in cm:
                    all_frames.update(cm[pid]["local_t"].keys())

            trans_out: dict[int, np.ndarray] = {}
            orient_out: dict[int, np.ndarray] = {}

            for global_t in sorted(all_frames):
                if global_t >= self.T:
                    continue

                s = float(scale[global_t]) if isinstance(scale, np.ndarray) else float(scale)

                # ── Step 1: DLT-triangulate each joint across cameras ──────────
                joint_world: dict[int, np.ndarray] = {}
                for smplx_j in _JOINTS:
                    mhr70_j = _SMPLX_TO_MHR70[smplx_j]
                    obs:   list[tuple[float, float]] = []
                    pmats: list[np.ndarray] = []

                    for k, cm in enumerate(cam_data_all):
                        if pid not in cm:
                            continue
                        if global_t not in cm[pid]["local_t"]:
                            continue
                        if not self.cam_valid[global_t, k]:
                            continue

                        local_t = cm[pid]["local_t"][global_t]
                        kp2d  = cm[pid]["kp2d"]

                        if mhr70_j >= kp2d.shape[1]:
                            continue

                        oc  = self.original_coords[global_t, k]
                        os_ = self.original_size[global_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])

                        u, v = self._orig_to_vggt(kp2d[local_t, mhr70_j], oc, W_orig, H_orig)
                        if not self._in_bounds(u, v, oc[2], oc[3]):
                            continue

                        intr = self.intrinsics[global_t, k].astype(np.float64)
                        ext  = self.extrinsics[global_t, k].astype(np.float64).copy()
                        ext[:3, 3] *= s
                        pmats.append(intr @ ext)
                        obs.append((u, v))

                    if len(obs) >= min_cams:
                        joint_world[smplx_j] = self._triangulate_dlt(obs, pmats)

                if len(joint_world) < min_joints:
                    continue

                # ── Step 2: FK canonical joint positions ──────────────────────
                # Use fused body_pose when available (better than per-camera raw).
                # fused_pose_by_pid[pid] is (T_scene, 54, 6); joints 0-20 = body.
                if fused_pose_by_pid is not None and pid in fused_pose_by_pid:
                    t_local = global_t - frame_start
                    fused_arr = fused_pose_by_pid[pid]
                    if not (0 <= t_local < len(fused_arr)):
                        continue
                    body_pose_frame = _6d_to_aa_batch(
                        fused_arr[t_local, :21]   # (21, 6) body joints 1-21
                    ).reshape(63)
                else:
                    body_pose_frame = None
                    for cm in cam_data_all:
                        if pid in cm and global_t in cm[pid]["local_t"]:
                            lt = cm[pid]["local_t"][global_t]
                            body_pose_frame = cm[pid]["body_pose"][lt]
                            break
                if body_pose_frame is None:
                    continue

                fk = self._smplx_fk(
                    betas[np.newaxis],
                    body_pose_frame[np.newaxis],
                    np.zeros((1, 3), dtype=np.float32),
                )
                J_can = fk[0]  # (55, 3) in metres

                # ── Step 3: Procrustes — R, t s.t. R @ J_can + t ≈ J_world ──
                vis = sorted(joint_world.keys())
                A = np.stack([joint_world[j] for j in vis], axis=0).astype(np.float64)
                B = np.stack([J_can[j]       for j in vis], axis=0).astype(np.float64)

                A_mean = A.mean(0)
                B_mean = B.mean(0)
                H = (B - B_mean).T @ (A - A_mean)
                U, _, Vt = np.linalg.svd(H)
                d_sign = np.linalg.det(Vt.T @ U.T)
                R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
                t = (A_mean - R.astype(np.float64) @ B_mean).astype(np.float32)

                # SMPL-X global_orient rotates around joint[0] (pelvis), so
                # pelvis world = R @ J_can[0] + t
                pelvis_world = (
                    R.astype(np.float64) @ J_can[0].astype(np.float64)
                    + t.astype(np.float64)
                ).astype(np.float32)

                trans_out[global_t] = pelvis_world
                orient_out[global_t] = R

            translations[pid] = trans_out
            orientations[pid] = orient_out

        return translations, orientations

    # ------------------------------------------------------------------
    # SMPL-X FK helper
    # ------------------------------------------------------------------

    def _smplx_fk(
        self,
        betas: np.ndarray,                          # (T_local, 10)
        body_pose: np.ndarray,                      # (T_local, 63)
        global_orient: np.ndarray,                  # (T_local, 3)
        left_hand_pose: np.ndarray | None = None,   # (T_local, 45) optional
        right_hand_pose: np.ndarray | None = None,  # (T_local, 45) optional
    ) -> np.ndarray:
        """Run SMPL-X FK and return joints in camera-oriented space.

        Uses zero translation so all positions are body-centric (origin at root).
        The global_orient rotation IS applied, so the z-axis matches camera depth.

        Returns:
            (T_local, 55, 3) float32 — first 55 SMPL-X joints in metres.
        """
        T = betas.shape[0]
        num_expr = self._smplx_model.num_expression_coeffs
        lhp = (torch.tensor(left_hand_pose,  dtype=torch.float32)
               if left_hand_pose  is not None else torch.zeros(T, 45, dtype=torch.float32))
        rhp = (torch.tensor(right_hand_pose, dtype=torch.float32)
               if right_hand_pose is not None else torch.zeros(T, 45, dtype=torch.float32))
        with torch.no_grad():
            out = self._smplx_model(
                betas=torch.tensor(betas, dtype=torch.float32),
                body_pose=torch.tensor(body_pose, dtype=torch.float32),
                global_orient=torch.tensor(global_orient, dtype=torch.float32),
                transl=torch.zeros(T, 3, dtype=torch.float32),
                jaw_pose=torch.zeros(T, 3, dtype=torch.float32),
                leye_pose=torch.zeros(T, 3, dtype=torch.float32),
                reye_pose=torch.zeros(T, 3, dtype=torch.float32),
                left_hand_pose=lhp,
                right_hand_pose=rhp,
                expression=torch.zeros(T, num_expr, dtype=torch.float32),
                return_verts=False,
            )
        return out.joints[:, :55].cpu().numpy().astype(np.float32)

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

    def apply_scale(
        self,
        scale: float,
        output_dir: str | Path | None = None,
    ) -> None:
        """Rescale VGGT depth maps and extrinsic translations and write to disk.

        The extrinsic translation column (t in [R|t]) is multiplied by ``scale``
        so that it is expressed in metres.  The depth maps are similarly rescaled.
        Files are written as ``vggt_cameras_rescaled.npz`` and
        ``vggt_depth_rescaled.npz`` in ``output_dir`` (defaults to
        ``scene_output_dir``).

        Args:
            scale: Value returned by :meth:`estimate_scale_triangulated`.
            output_dir: Destination directory.  Defaults to ``scene_output_dir``.
        """
        out = Path(output_dir) if output_dir is not None else self.scene_dir

        extrinsics_scaled = self.extrinsics.copy()
        extrinsics_scaled[..., 3] *= scale  # column 3 is the translation vector

        np.savez_compressed(
            out / "vggt_cameras_rescaled.npz",
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
            out / "vggt_depth_rescaled.npz",
            depth=depth_mm_scaled,
            depth_conf=self.depth_conf,
            depth_valid=self.depth_valid,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_scale_samples_tagged(
        self,
        k: int,
        body_file: Path,
        conf_threshold: float,
        min_delta_z: float,
        fused_betas: np.ndarray | None = None,
    ) -> dict[int, list[float]]:
        """Collect scale samples s = L_FK / L_VGGT, tagged by global frame index.

        Returns ``{global_t: [scale_samples]}`` — may be empty if no valid data.
        """
        from collections import defaultdict
        data = np.load(body_file, allow_pickle=False)

        required = {"smplx_betas", "smplx_body_pose",
                    "pred_keypoints_2d", "frame_indices"}
        if not required.issubset(data.files):
            return {}

        betas         = data["smplx_betas"]        # (T_local, 10)
        body_pose     = data["smplx_body_pose"]    # (T_local, 63)
        kp2d          = data["pred_keypoints_2d"]  # (T_local, J, 2+) original pixels
        frame_indices = data["frame_indices"]

        T_local = len(frame_indices)
        go_zero = np.zeros((T_local, 3), dtype=np.float32)

        if fused_betas is not None:
            betas_arr = np.tile(fused_betas[np.newaxis], (T_local, 1))
        else:
            betas_arr = betas
        fk_joints = self._smplx_fk(betas_arr, body_pose, go_zero)  # (T_local, 55, 3)

        result: dict[int, list[float]] = defaultdict(list)

        for local_t, global_t in enumerate(frame_indices):
            if global_t >= self.T:
                continue
            if not self.cam_valid[global_t, k]:
                continue
            if not self.depth_valid[global_t, k]:
                continue

            depth_frame = self.depth_mm[global_t, k].astype(np.float32) / 1000.0
            conf_frame  = self.depth_conf[global_t, k].astype(np.float32)

            intr = self.intrinsics[global_t, k]
            fx, fy = float(intr[0, 0]), float(intr[1, 1])
            cx, cy = float(intr[0, 2]), float(intr[1, 2])

            oc = self.original_coords[global_t, k]
            os = self.original_size[global_t, k]
            W_orig, H_orig = float(os[0]), float(os[1])

            for j_a, j_b in _LONG_BONES:
                mhr_a = _SMPLX_TO_MHR70.get(j_a)
                mhr_b = _SMPLX_TO_MHR70.get(j_b)
                if mhr_a is None or mhr_b is None:
                    continue
                if mhr_a >= kp2d.shape[1] or mhr_b >= kp2d.shape[1]:
                    continue

                u_a, v_a = self._orig_to_vggt(kp2d[local_t, mhr_a], oc, W_orig, H_orig)
                u_b, v_b = self._orig_to_vggt(kp2d[local_t, mhr_b], oc, W_orig, H_orig)

                if not (self._in_bounds(u_a, v_a, oc[2], oc[3]) and self._in_bounds(u_b, v_b, oc[2], oc[3])):
                    continue

                d_a = float(map_coordinates(depth_frame, [[v_a], [u_a]], order=1)[0])
                d_b = float(map_coordinates(depth_frame, [[v_b], [u_b]], order=1)[0])
                c_a = float(map_coordinates(conf_frame,  [[v_a], [u_a]], order=1)[0])
                c_b = float(map_coordinates(conf_frame,  [[v_b], [u_b]], order=1)[0])

                if c_a < conf_threshold or c_b < conf_threshold:
                    continue
                if d_a <= 0.0 or d_b <= 0.0:
                    continue

                P_a = np.array([(u_a - cx) / fx * d_a, (v_a - cy) / fy * d_a, d_a], dtype=np.float32)
                P_b = np.array([(u_b - cx) / fx * d_b, (v_b - cy) / fy * d_b, d_b], dtype=np.float32)
                L_vggt = float(np.linalg.norm(P_b - P_a))

                if L_vggt < 1e-4:
                    continue

                L_fk = float(np.linalg.norm(fk_joints[local_t, j_b] - fk_joints[local_t, j_a]))
                if L_fk < min_delta_z:
                    continue

                s = L_fk / L_vggt
                if 0.01 < s < 100.0:
                    result[int(global_t)].append(s)

        return result

    def _collect_scale_samples(
        self,
        k: int,
        body_file: Path,
        conf_threshold: float,
        min_delta_z: float,
        fused_betas: np.ndarray | None = None,
    ) -> list[float]:
        """Flat list of scale samples; delegates to :meth:`_collect_scale_samples_tagged`."""
        tagged = self._collect_scale_samples_tagged(
            k, body_file, conf_threshold, min_delta_z, fused_betas,
        )
        return [s for sl in tagged.values() for s in sl]

    @staticmethod
    def _orig_to_vggt(
        kp: np.ndarray,
        oc: np.ndarray,
        W_orig: float,
        H_orig: float,
    ) -> tuple[float, float]:
        """Map a 2D keypoint from original-image pixels to VGGT output space.

        Args:
            kp: Keypoint [u, v, ...] in original image pixels.
            oc: [0, 0, W_vggt, H_vggt] from vggt_cameras.npz.
            W_orig, H_orig: Original image dimensions in pixels.

        Returns:
            (u_vggt, v_vggt) in VGGT output pixel coordinates.
        """
        x1, y1, x2, y2 = oc
        u_vggt = x1 + float(kp[0]) * (x2 - x1) / W_orig
        v_vggt = y1 + float(kp[1]) * (y2 - y1) / H_orig
        return u_vggt, v_vggt

    @staticmethod
    def _in_bounds(u: float, v: float, w_max: float, h_max: float) -> bool:
        return 0.0 <= u < w_max and 0.0 <= v < h_max
