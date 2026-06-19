"""fusion/placer.py — estimate VGGT depth scale from SMPL-X metric bone depths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
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

# All candidate bone pairs for dynamic scale estimation.
# Each tuple: (mhr_a, mhr_b) — MHR70 keypoint indices.
# Bone lengths are computed via the Mapper which gives the correct SMPLX
# surface/joint position for each MHR70 keypoint (surface landmark for
# ankle/elbow, FK joint for hip/knee/wrist).
_SCALE_BONE_CANDIDATES = [
    ( 9, 13),  # L-hip  → L-ankle
    (10, 14),  # R-hip  → R-ankle
    ( 9, 11),  # L-hip  → L-knee
    (10, 12),  # R-hip  → R-knee
    (11, 13),  # L-knee → L-ankle
    (12, 14),  # R-knee → R-ankle
    ( 7, 62),  # L-elbow → L-wrist
    ( 8, 41),  # R-elbow → R-wrist
]
_N_BEST_BONES = 4
_SCALE_CONF_THR = 0.6

# Per-bone calibration factors: median L_fk / (L_vggt × gt_scale) across 62 RICH
# training scenes (compute_bone_calibration.py). Dividing L_fk by this factor
# removes the systematic MHR70 landmark offset bias from the scale estimate.
_BONE_CALIB_FACTORS: dict[tuple, float] = dict(zip(_SCALE_BONE_CANDIDATES,
    [0.97, 0.96, 0.91, 0.86, 1.03, 1.04, 0.95, 1.01]))

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

# Subset used for PnP and Procrustes DLT.
# Ankles and shoulders excluded: for sitting/occluded poses SAM3D 2D detections
# are biased by 25-30cm (ankles: occluded/folded legs; shoulders: acromion vs
# glenohumeral offset amplified by side-view depth ambiguity).
# Hips (~2cm) + neck (~4cm) give 3 non-collinear points that are robust across
# all poses (standing, sitting, bending).
_SMPLX_TO_MHR70_ALIGN = {
    1: 9,   2: 10,   # left/right hip  (~2cm, std 0.4cm)
    12: 69,          # neck            (~4cm, std 2cm)
}

# SMPL-X → COCO-17 joint mapping for ViTPose keypoints.
# Same direction as _SMPLX_TO_MHR70_ALIGN: keys = SMPL-X indices, values = COCO indices.
# Shoulders (COCO 5,6) excluded: acromion vs glenohumeral mismatch.
# Face joints (COCO 0-4) excluded: no SMPL-X body joint equivalent.
_COCO_SMPLX_ALIGN = {
    1: 11, 2: 12,   # left/right hip
    4: 13, 5: 14,   # left/right knee
    7: 15, 8: 16,   # left/right ankle
    18: 7, 19: 8,   # left/right elbow
    20: 9, 21: 10,  # left/right wrist
}

# SMPL-X → Sapiens Goliath-308 joint mapping.
# Goliath body: 0-14 (no wrists); wrists are the last joints of the hand ranges
# (41 = right wrist, 62 = left wrist).
_GOLIATH_SMPLX_ALIGN: dict[int, int] = {
    1:  9,   # left hip
    2:  10,  # right hip
    4:  11,  # left knee
    5:  12,  # right knee
    7:  13,  # left ankle
    8:  14,  # right ankle
    16: 5,   # left shoulder
    17: 6,   # right shoulder
    18: 7,   # left elbow
    19: 8,   # right elbow
    20: 62,  # left wrist  (end of left-hand range 42-62)
    21: 41,  # right wrist (end of right-hand range 21-41)
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
        crop_meta_path: Optional path to the ``crop_meta.json`` written by
            ``center_images.py`` for this scene (lives next to the centered
            images, e.g. ``<rich_root>/centered_<split>/<scene>/crop_meta.json``).
            VGGT cameras are calibrated in centered-crop pixel space, whereas
            SAM3D ``pred_keypoints_2d`` (MHR70) are in uncropped source pixels.
            When provided, the per-camera crop offset ``(off_x, off_y)`` is
            subtracted from those keypoints before projecting into VGGT space,
            removing the systematic reprojection bias on cameras whose principal
            point deviates from the image centre.  When ``None`` (no centering,
            or file absent) all offsets are ``(0, 0)`` — behaviour is unchanged.
    """

    def __init__(
        self,
        scene_output_dir: str | Path,
        smplx_model_path: str | Path,
        crop_meta_path: str | Path | None = None,
    ) -> None:
        self.scene_dir = Path(scene_output_dir)

        import sys
        _repo_root = Path(__file__).parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from mappings.mapper import Mapper
        _regressor = _repo_root / "mappings" / "mhr_smplx_regressor.npy"
        self._mapper = Mapper.load(_regressor)

        # smplx_model_path may be:
        #   str / Path           — single model for all persons (gender-agnostic)
        #   dict[int, str/Path]  — per-person models keyed by RICH person ID;
        #                          improves FK joint positions and Procrustes alignment
        if isinstance(smplx_model_path, dict):
            self._smplx_models: dict[int, object] = {
                int(pid): self._load_smplx_model(path)
                for pid, path in smplx_model_path.items()
            }
            self._smplx_model = next(iter(self._smplx_models.values()))
        else:
            self._smplx_models = {}
            self._smplx_model = self._load_smplx_model(smplx_model_path)

        self._smplx_device = torch.device("cpu")

        cam_npz = np.load(self.scene_dir / "vggt_cameras_centered.npz")

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

        # Per-camera crop offset (off_x, off_y) in SOURCE pixels, mapping the
        # uncropped SAM3D pred_keypoints_2d into the centered-crop space the
        # VGGT cameras live in:  u_crop = u_src - off_x,  v_crop = v_src - off_y.
        # Defaults to (0, 0) for every camera when no crop_meta is supplied.
        self._cam_offsets: dict[str, tuple[float, float]] = {}
        if crop_meta_path is not None:
            crop_meta_path = Path(crop_meta_path)
            if crop_meta_path.exists():
                import json
                import logging
                with open(crop_meta_path) as f:
                    _meta = json.load(f)
                for cam_name, info in _meta.get("cameras", {}).items():
                    self._cam_offsets[cam_name] = (
                        float(info.get("off_x", 0.0)),
                        float(info.get("off_y", 0.0)),
                    )
                logging.getLogger(__name__).info(
                    f"[placer] loaded crop offsets for {len(self._cam_offsets)} "
                    f"cameras from {crop_meta_path}"
                )
            else:
                import logging
                logging.getLogger(__name__).warning(
                    f"[placer] crop_meta_path {crop_meta_path} not found — "
                    f"SAM3D kp2d offset correction disabled (offsets = 0)"
                )

        # Depth is optional (legacy; not used by estimate_scale_triangulated).
        depth_path = self.scene_dir / "vggt_depth_centered.npz"
        if depth_path.exists():
            depth_npz = np.load(depth_path, mmap_mode="r")
            self.depth_mm    = depth_npz["depth"]
            self.depth_conf  = depth_npz["depth_conf"]
            self.depth_valid = depth_npz["depth_valid"]
        else:
            self.depth_mm    = None
            self.depth_conf  = None
            self.depth_valid = np.zeros(self.cam_valid.shape, dtype=bool)

        self.T, self.K = self.cam_valid.shape

        # Camera dirs sorted in the same order as the K axis.
        # Filter to only cameras present in the npz (e.g. centered VGGT may
        # exclude cameras that were missing calibration).
        _npz_names = {
            (n.decode() if isinstance(n, bytes) else n)
            for n in self.camera_names
        }
        self._cam_dirs: list[Path] = sorted(
            d for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()
            and d.name in _npz_names
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_mapanything_scale(self) -> np.ndarray | None:
        """Load per-frame scale from MapAnything preprocessing output, if available.

        Returns ``(T,)`` float32 or None when the file is absent or mismatched.
        """
        path = self.scene_dir / "mapanything_scale_centered.npy"
        if not path.exists():
            return None
        scale = np.load(path).astype(np.float32)
        if scale.shape != (self.T,):
            import logging
            logging.getLogger(__name__).warning(
                f"mapanything_scale.npy has shape {scale.shape}, expected ({self.T},) — ignoring"
            )
            return None
        return scale

    def estimate_scale_per_frame(
        self,
        conf_threshold: float = 0.5,
        min_delta_z: float = 0.05,
        fused_betas_map: dict[Path, np.ndarray] | None = None,
        frame_start: int = 0,
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
                    frame_start=frame_start,
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
            vggt_t = global_t - frame_start
            if 0 <= vggt_t < self.T and slist:
                result[vggt_t] = float(np.median(slist))

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

        # ── Pre-pass: compute mean SAM3D betas per pid across all cameras ────────
        sam3d_betas_sum: dict[int, np.ndarray] = {}
        sam3d_betas_cnt: dict[int, int] = {}
        for cam_dir in self._cam_dirs:
            for body_file in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(body_file.stem.split("_")[1])
                d_b = np.load(body_file, allow_pickle=False)
                if "smplx_betas" not in d_b.files:
                    continue
                b = d_b["smplx_betas"].mean(axis=0)   # (10,)
                sam3d_betas_sum[pid] = sam3d_betas_sum.get(pid, np.zeros(10, np.float32)) + b
                sam3d_betas_cnt[pid] = sam3d_betas_cnt.get(pid, 0) + 1
        mean_sam3d_betas: dict[int, np.ndarray] = {
            pid: sam3d_betas_sum[pid] / sam3d_betas_cnt[pid]
            for pid in sam3d_betas_sum
        }

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
                # Use mean SAM3D betas (averaged across cameras+frames) — less biased
                # than fused BetasAggregator which overestimates bone length by ~20%.
                b_mean = mean_sam3d_betas.get(pid, np.zeros(10, np.float32))
                betas = np.tile(b_mean[np.newaxis], (len(fi), 1))
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
                _, verts = self._smplx_fk(betas, body_pose_arr,
                                          np.zeros((len(betas), 3), dtype=np.float32),
                                          return_verts=True, pid=pid)
                kp_smplx = self._mapper.map(verts)  # (T_local, N_kps, 3)

                gt_map = {int(gt): lt for lt, gt in enumerate(d["frame_indices"])}
                entry: dict = {
                    "gt_map":   gt_map,
                    "kp_smplx": kp_smplx,
                    "kp2d":     d["pred_keypoints_2d"],
                }
                cam_data[k][pid] = entry

        # ── Collect all (global_t, pid) pairs ─────────────────────────────────
        all_pids: set[int] = set()
        for k in cam_data:
            all_pids.update(cam_data[k].keys())

        global_ts_by_pid: dict[int, set[int]] = defaultdict(set)
        for k in cam_data:
            for pid, bd in cam_data[k].items():
                global_ts_by_pid[pid].update(bd["gt_map"].keys())

        frame_samples: dict[int, list[float]] = defaultdict(list)
        n_bones_used: dict[int, int] = {}

        for pid in sorted(all_pids):
            for global_t in sorted(global_ts_by_pid[pid]):
                vggt_t = global_t - frame_start
                if vggt_t < 0 or vggt_t >= self.T:
                    continue

                # Score each candidate bone by how many cameras see both endpoints
                # with confidence >= _SCALE_CONF_THR.  Pick the top _N_BEST_BONES.
                bone_scores: list[tuple[int, tuple]] = []
                for bone in _SCALE_BONE_CANDIDATES:
                    mhr_a, mhr_b = bone[0], bone[1]
                    n_cams = 0
                    for k in range(self.K):
                        if not self.cam_valid[vggt_t, k]: continue
                        bd = cam_data.get(k, {}).get(pid)
                        if bd is None or global_t not in bd["gt_map"]: continue
                        lt = bd["gt_map"][global_t]
                        if mhr_a < bd["kp2d"].shape[1] and mhr_b < bd["kp2d"].shape[1]:
                            n_cams += 1
                    if n_cams >= min_cams:
                        bone_scores.append((n_cams, bone))

                bone_scores.sort(key=lambda x: x[0], reverse=True)
                selected_bones = [b for _, b in bone_scores[:_N_BEST_BONES]]
                n_bones_used[global_t] = n_bones_used.get(global_t, 0) + len(selected_bones)

                for mhr_a, mhr_b in selected_bones:
                    pts_a: list[tuple[float, float]] = []
                    pts_b: list[tuple[float, float]] = []
                    Pmats: list[np.ndarray] = []
                    fk_lengths: list[float] = []

                    for k in range(self.K):
                        if not self.cam_valid[vggt_t, k]:
                            continue
                        bd = cam_data.get(k, {}).get(pid)
                        if bd is None or global_t not in bd["gt_map"]:
                            continue
                        local_t = bd["gt_map"][global_t]

                        if mhr_a >= bd["kp2d"].shape[1] or mhr_b >= bd["kp2d"].shape[1]:
                            continue

                        oc = self.original_coords[vggt_t, k]
                        os_ = self.original_size[vggt_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])
                        x1, y1, x2, y2 = oc
                        off_x, off_y = self._cam_offset(k)

                        u_a, v_a = self._orig_to_vggt(bd["kp2d"][local_t, mhr_a], oc, W_orig, H_orig, off_x, off_y)
                        u_b, v_b = self._orig_to_vggt(bd["kp2d"][local_t, mhr_b], oc, W_orig, H_orig, off_x, off_y)
                        if not (x1 <= u_a < x2 and y1 <= v_a < y2
                                and x1 <= u_b < x2 and y1 <= v_b < y2):
                            continue

                        K_mat = self.intrinsics[vggt_t, k].astype(np.float64)
                        E_mat = self.extrinsics[vggt_t, k].astype(np.float64)
                        pts_a.append((u_a, v_a))
                        pts_b.append((u_b, v_b))
                        Pmats.append(K_mat @ E_mat)
                        fk_lengths.append(float(np.linalg.norm(
                            bd["kp_smplx"][local_t, self._mapper.index(mhr_b)]
                            - bd["kp_smplx"][local_t, self._mapper.index(mhr_a)]
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

                    bone_key = (mhr_a, mhr_b)
                    s = (L_fk / _BONE_CALIB_FACTORS.get(bone_key, 1.0)) / L_vggt
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
            vggt_t = global_t - frame_start
            if 0 <= vggt_t < self.T and slist:
                result[vggt_t] = float(np.median(slist))

        if n_bones_used:
            counts = list(n_bones_used.values())
            print(f"  [scale] bones/frame: mean={np.mean(counts):.1f}  "
                  f"min={min(counts)}  max={max(counts)}  "
                  f"frames_with_bones={len(counts)}/{self.T}")
        return result

    def estimate_procrustes_dlt_mhr(
        self,
        scale: float | np.ndarray,
        all_pids: set[int],
        pred_betas_by_pid: dict[int, np.ndarray],
        fused_pose_by_pid: dict[int, np.ndarray] | None = None,
        frame_start: int = 0,
        min_cams: int = 2,
        min_joints: int = 3,
        smooth_window: int = 15,
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
        # Pre-load body data once per (cam, pid) to avoid repeated file reads.
        # If vitpose_kps_person_<pid>.npz exists alongside body_data, use COCO
        # keypoints + per-joint confidence weights; otherwise fall back to SAM3D
        # pred_keypoints_2d (MHR70) with uniform weights.
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
                entry: dict = {
                    "local_t":  {int(g): int(l) for l, g in enumerate(fi)},
                    "body_pose": d["smplx_body_pose"],
                }
                entry["kp2d"]     = d["pred_keypoints_2d"]
                entry["kp_conf"]  = None
                entry["use_coco"] = False
                cam_map[pid] = entry
            cam_data_all.append(cam_map)

        _JOINTS = sorted(j for j in _SMPLX_TO_MHR70 if _SMPLX_TO_MHR70[j] in self._mapper._index)

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
                vggt_t = global_t - frame_start
                if vggt_t < 0 or vggt_t >= self.T:
                    continue

                s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

                # ── Step 1: DLT-triangulate each joint across cameras ──────────
                joint_world: dict[int, np.ndarray] = {}
                for smplx_j in _JOINTS:
                    joint_idx = _SMPLX_TO_MHR70[smplx_j]
                    obs:     list[tuple[float, float]] = []
                    pmats:   list[np.ndarray] = []
                    weights: list[float] = []

                    for k, cm in enumerate(cam_data_all):
                        if pid not in cm:
                            continue
                        if global_t not in cm[pid]["local_t"]:
                            continue
                        if not self.cam_valid[vggt_t, k]:
                            continue

                        local_t  = cm[pid]["local_t"][global_t]
                        kp2d     = cm[pid]["kp2d"]

                        if joint_idx >= kp2d.shape[1]:
                            continue

                        conf = 1.0

                        oc  = self.original_coords[vggt_t, k]
                        os_ = self.original_size[vggt_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])
                        off_x, off_y = self._cam_offset(k)

                        u, v = self._orig_to_vggt(kp2d[local_t, joint_idx], oc, W_orig, H_orig, off_x, off_y)
                        if not self._in_bounds(u, v, oc[2], oc[3]):
                            continue

                        intr = self.intrinsics[vggt_t, k].astype(np.float64)
                        ext  = self.extrinsics[vggt_t, k].astype(np.float64).copy()
                        ext[:3, 3] *= s
                        pmats.append(intr @ ext)
                        obs.append((u, v))
                        if k > 0:
                            cos_a = float(np.clip(ext[2, 2], -1.0, 1.0))
                            sin_w = float(np.sqrt(max(1.0 - cos_a ** 2, 0.0))) ** 2
                        else:
                            sin_w = 1.0
                        weights.append(conf * sin_w)

                    if len(obs) >= min_cams:
                        joint_world[smplx_j] = self._triangulate_dlt(obs, pmats, weights)

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

                fk, verts = self._smplx_fk(
                    betas[np.newaxis],
                    body_pose_frame[np.newaxis],
                    np.zeros((1, 3), dtype=np.float32),
                    return_verts=True,
                    pid=pid,
                )
                J_can   = fk[0]                        # (55, 3) — still needed for pelvis (joint 0)
                kp_can  = self._mapper.map(verts[0])   # (N_kps, 3) — mapper-corrected landmarks

                # ── Step 3: Procrustes — R, t s.t. R @ J_can + t ≈ J_world ──
                vis = sorted(joint_world.keys())
                A = np.stack([joint_world[j] for j in vis], axis=0).astype(np.float64)
                B = np.stack([
                    kp_can[self._mapper.index(_SMPLX_TO_MHR70[j])]
                    for j in vis
                ], axis=0).astype(np.float64)

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

        if smooth_window > 0:
            w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            for pid, frames in translations.items():
                sorted_f = sorted(frames)
                if len(sorted_f) < w:
                    continue
                traj = np.stack([frames[f] for f in sorted_f])   # (N, 3)
                traj_s = savgol_filter(traj, window_length=w, polyorder=2, axis=0)
                translations[pid] = {f: traj_s[i].astype(np.float32)
                                     for i, f in enumerate(sorted_f)}

        return translations, orientations

    def estimate_procrustes_dlt_fk(
        self,
        scale: float | np.ndarray,
        all_pids: set[int],
        pred_betas_by_pid: dict[int, np.ndarray],
        fused_pose_by_pid: dict[int, np.ndarray] | None = None,
        frame_start: int = 0,
        min_cams: int = 2,
        min_joints: int = 3,
    ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
        """Estimate root translation + orient via multi-camera DLT on SMPL-X FK projections.

        Same interface as estimate_procrustes_dlt_mhr but replaces SAM3D's
        pred_keypoints_2d (MHR70 landmarks) with SMPL-X FK joints (all 22
        body joints) projected using smplx_transl + smplx_global_orient and
        SAM3D's focal_length. Projection follows the same pipeline as M1:
        original-image space → _orig_to_vggt → DLT with K_vggt @ E_scaled.

        Returns:
            translations : {pid: {global_frame_idx: pelvis_world (3,)}}
            orientations : {pid: {global_frame_idx: R (3,3)}}
        """
        cam_data_all: list[dict[int, dict]] = []
        for cam_dir in self._cam_dirs:
            cam_map: dict[int, dict] = {}
            for pid in sorted(all_pids):
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                if not bf.exists():
                    continue
                d = np.load(bf, allow_pickle=False)
                required = {"smplx_transl", "smplx_global_orient",
                            "smplx_body_pose", "frame_indices", "focal_length"}
                if not required.issubset(d.files):
                    continue
                fi = d["frame_indices"].astype(int)
                cam_map[pid] = {
                    "local_t":      {int(g): int(l) for l, g in enumerate(fi)},
                    "transl":       d["smplx_transl"].astype(np.float64),        # (T_local, 3)
                    "orient":       d["smplx_global_orient"].astype(np.float64), # (T_local, 3) aa
                    "body_pose":    d["smplx_body_pose"],                         # (T_local, 63)
                    "focal_length": d["focal_length"].astype(np.float64),         # (T_local,) or scalar
                }
            cam_data_all.append(cam_map)

        _JOINTS = list(range(22))  # all 22 SMPL-X body joints

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
                vggt_t = global_t - frame_start
                if vggt_t < 0 or vggt_t >= self.T:
                    continue

                s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

                # ── FK canonical joints (zero orient, so Procrustes recovers R) ──
                if fused_pose_by_pid is not None and pid in fused_pose_by_pid:
                    t_local = global_t - frame_start
                    fused_arr = fused_pose_by_pid[pid]
                    if not (0 <= t_local < len(fused_arr)):
                        continue
                    body_pose_frame = _6d_to_aa_batch(
                        fused_arr[t_local, :21]
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

                J_can = self._smplx_fk(
                    betas[np.newaxis],
                    body_pose_frame[np.newaxis],
                    np.zeros((1, 3), dtype=np.float32),
                    pid=pid,
                )[0]  # (55, 3)

                # ── DLT: triangulate each SMPL-X joint from FK projections ────
                joint_world: dict[int, np.ndarray] = {}
                for smplx_j in _JOINTS:
                    obs:   list[tuple[float, float]] = []
                    pmats: list[np.ndarray] = []

                    for k, cm in enumerate(cam_data_all):
                        if pid not in cm or global_t not in cm[pid]["local_t"]:
                            continue
                        if not self.cam_valid[vggt_t, k]:
                            continue

                        lt = cm[pid]["local_t"][global_t]
                        transl = cm[pid]["transl"][lt]   # (3,) metres, camera frame
                        orient = cm[pid]["orient"][lt]   # (3,) axis-angle, camera frame
                        fl     = float(cm[pid]["focal_length"][lt])

                        R_o   = SciR.from_rotvec(orient).as_matrix()
                        # SMPL-X convention: rotation pivots around J_can[0] (pelvis),
                        # which is NOT at the origin when transl=0.
                        j_cam = (R_o @ (J_can[smplx_j] - J_can[0]).astype(np.float64)
                                 + J_can[0].astype(np.float64) + transl)
                        if j_cam[2] <= 0.0:
                            continue

                        oc  = self.original_coords[vggt_t, k]
                        os_ = self.original_size[vggt_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])

                        # Project in original image space (SAM3D convention)
                        u_orig = fl * j_cam[0] / j_cam[2] + W_orig / 2.0
                        v_orig = fl * j_cam[1] / j_cam[2] + H_orig / 2.0

                        # Convert to VGGT pixel space (same pipeline as M1)
                        u, v = self._orig_to_vggt(
                            np.array([u_orig, v_orig]), oc, W_orig, H_orig
                        )
                        if not self._in_bounds(u, v, oc[2], oc[3]):
                            continue

                        K_mat = self.intrinsics[vggt_t, k].astype(np.float64)
                        ext = self.extrinsics[vggt_t, k].astype(np.float64).copy()
                        ext[:3, 3] *= s
                        pmats.append(K_mat @ ext)
                        obs.append((u, v))

                    if len(obs) >= min_cams:
                        joint_world[smplx_j] = self._triangulate_dlt(obs, pmats)

                if len(joint_world) < min_joints:
                    continue

                # ── Procrustes: R, t s.t. R @ J_can + t ≈ J_world ───────────
                vis = sorted(joint_world.keys())
                A = np.stack([joint_world[j] for j in vis]).astype(np.float64)
                B = np.stack([J_can[j]       for j in vis]).astype(np.float64)

                A_m, B_m = A.mean(0), B.mean(0)
                H = (B - B_m).T @ (A - A_m)
                U, _, Vt = np.linalg.svd(H)
                d_sign = np.linalg.det(Vt.T @ U.T)
                R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
                t = (A_m - R.astype(np.float64) @ B_m).astype(np.float32)

                pelvis_world = (
                    R.astype(np.float64) @ J_can[0].astype(np.float64) + t.astype(np.float64)
                ).astype(np.float32)

                trans_out[global_t] = pelvis_world
                orient_out[global_t] = R

            translations[pid] = trans_out
            orientations[pid] = orient_out

        return translations, orientations

    def estimate_procrustes_dlt_sapiens(
        self,
        scale: float | np.ndarray,
        all_pids: set[int],
        pred_betas_by_pid: dict[int, np.ndarray],
        fused_pose_by_pid: dict[int, np.ndarray] | None = None,
        frame_start: int = 0,
        conf_thr: float = 0.3,
        min_cams: int = 2,
        min_joints: int = 3,
        smooth_window: int = 15,
    ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
        """Estimate root translation + orient via DLT on Sapiens Goliath-308 keypoints.

        Loads ``sapiens_centered_kps_person_{pid}.npz`` from each camera directory
        (array key ``keypoints``, shape ``(T_local, 308, 3)`` — [x, y, conf] in
        original image pixels).  Triangulates each joint with confidence-weighted
        DLT; cameras k > 0 are additionally down-weighted by sin(angle) between
        their optical axis and cam0's optical axis (baseline geometry weight).
        Then runs the same Procrustes as the other estimate_procrustes_dlt_* methods
        to recover global orientation and pelvis world position.

        Args:
            scale: Metric scale (metres per VGGT unit). Scalar or ``(T,)`` array.
            all_pids: Ghost person IDs to process.
            pred_betas_by_pid: ``{pid: (10,)}`` shape coefficients for FK.
            fused_pose_by_pid: ``{pid: (T_scene, 54, 6)}`` fused body pose (6D,
                joints 1-54, indexed from ``frame_start``).
            frame_start: Global frame index corresponding to index 0 of fused arrays.
            conf_thr: Minimum Sapiens confidence to include an observation.
            min_cams: Minimum cameras needed to triangulate a joint.
            min_joints: Minimum triangulated joints needed to run Procrustes.

        Returns:
            translations : ``{pid: {global_frame_idx: pelvis_world (3,)}}``
            orientations : ``{pid: {global_frame_idx: R (3,3)}}``
        """
        smplx_joints = sorted(_GOLIATH_SMPLX_ALIGN)
        zero_orient  = np.zeros((1, 3), dtype=np.float32)

        # ── Load Sapiens data per (cam, pid) ──────────────────────────────────
        # cam_data_all[k][pid] = {"local_t": {global_t→local_t}, "kps": (T_local,308,3)}
        cam_data_all: list[dict[int, dict]] = []
        for cam_dir in self._cam_dirs:
            cam_map: dict[int, dict] = {}
            for pid in sorted(all_pids):
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                sp = cam_dir / f"sapiens_centered_kps_person_{pid}.npz"
                if not bf.exists() or not sp.exists():
                    continue
                bd = np.load(bf, allow_pickle=False)
                if "frame_indices" not in bd.files:
                    continue
                fi = bd["frame_indices"].astype(int)
                kps = np.load(sp)["keypoints"]   # (T_local, 308, 3)
                cam_map[pid] = {
                    "local_t": {int(g): int(l) for l, g in enumerate(fi)},
                    "kps":     kps,
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

            trans_out:  dict[int, np.ndarray] = {}
            orient_out: dict[int, np.ndarray] = {}

            for global_t in sorted(all_frames):
                vggt_t = global_t - frame_start
                if vggt_t < 0 or vggt_t >= self.T:
                    continue

                s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

                # ── Step 1: DLT-triangulate each Sapiens joint ────────────────
                joint_world: dict[int, np.ndarray] = {}
                for smplx_j in smplx_joints:
                    goliath_j = _GOLIATH_SMPLX_ALIGN[smplx_j]
                    obs:     list[tuple[float, float]] = []
                    pmats:   list[np.ndarray]          = []
                    weights: list[float]               = []

                    for k, cm in enumerate(cam_data_all):
                        if pid not in cm or global_t not in cm[pid]["local_t"]:
                            continue
                        if not self.cam_valid[vggt_t, k]:
                            continue

                        lt  = cm[pid]["local_t"][global_t]
                        kp  = cm[pid]["kps"][lt, goliath_j]   # (3,) [x, y, conf]
                        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                        if conf < conf_thr:
                            continue

                        oc       = self.original_coords[vggt_t, k]
                        os_      = self.original_size[vggt_t, k]
                        W_orig, H_orig = float(os_[0]), float(os_[1])
                        u, v = self._orig_to_vggt(np.array([x, y]), oc, W_orig, H_orig)
                        if not self._in_bounds(u, v, oc[2], oc[3]):
                            continue

                        intr = self.intrinsics[vggt_t, k].astype(np.float64)
                        ext  = self.extrinsics[vggt_t, k].astype(np.float64).copy()
                        ext[:3, 3] *= s
                        pmats.append(intr @ ext)
                        obs.append((u, v))

                        # sin²(angle between cam_k and cam0 optical axes).
                        # cam0 is the world origin so its R ≈ I; ext[2,2] = R_k[2,2] = cos(θ).
                        # Down-weights cameras with a shallow baseline relative to cam0.
                        if k > 0:
                            cos_a = float(np.clip(ext[2, 2], -1.0, 1.0))
                            sin_w = float(np.sqrt(max(1.0 - cos_a ** 2, 0.0))) ** 2
                        else:
                            sin_w = 1.0
                        weights.append(conf * sin_w)

                    if len(obs) >= min_cams:
                        joint_world[smplx_j] = self._triangulate_dlt(obs, pmats, weights)

                if len(joint_world) < min_joints:
                    continue

                # ── Step 2: FK with fused body_pose ──────────────────────────
                if fused_pose_by_pid is not None and pid in fused_pose_by_pid:
                    t_local   = global_t - frame_start
                    fused_arr = fused_pose_by_pid[pid]
                    if not (0 <= t_local < len(fused_arr)):
                        continue
                    body_pose_frame = _6d_to_aa_batch(
                        fused_arr[t_local, :21]
                    ).reshape(63)
                else:
                    body_pose_frame = None
                    for k_fb, cm in enumerate(cam_data_all):
                        if pid in cm and global_t in cm[pid]["local_t"]:
                            # Sapiens files don't carry body_pose; fall back to body_data
                            bf = self._cam_dirs[k_fb] / "body_data" / f"person_{pid}.npz"
                            if bf.exists():
                                d_bp = np.load(bf, allow_pickle=False)
                                if "smplx_body_pose" in d_bp.files:
                                    lt = cm[pid]["local_t"][global_t]
                                    body_pose_frame = d_bp["smplx_body_pose"][lt]
                            break
                if body_pose_frame is None:
                    continue

                J_can = self._smplx_fk(
                    betas[np.newaxis],
                    body_pose_frame[np.newaxis],
                    zero_orient,
                    pid=pid,
                )[0]   # (55, 3)

                # ── Step 3: Procrustes — R, t s.t. R @ J_can[j] + t ≈ joint_world[j] ──
                vis = sorted(joint_world)
                A   = np.stack([joint_world[j] for j in vis]).astype(np.float64)
                B   = np.stack([J_can[j]        for j in vis]).astype(np.float64)

                A_m, B_m = A.mean(0), B.mean(0)
                H = (B - B_m).T @ (A - A_m)
                U, _, Vt = np.linalg.svd(H)
                d_sign = np.linalg.det(Vt.T @ U.T)
                R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
                t = (A_m - R.astype(np.float64) @ B_m).astype(np.float32)

                # pelvis_world = R @ J_can[0] + t
                pelvis_world = (
                    R.astype(np.float64) @ J_can[0].astype(np.float64)
                    + t.astype(np.float64)
                ).astype(np.float32)

                trans_out[global_t]  = pelvis_world
                orient_out[global_t] = R

            translations[pid] = trans_out
            orientations[pid] = orient_out

        if smooth_window > 0:
            w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            for pid, frames in translations.items():
                sorted_f = sorted(frames)
                if len(sorted_f) < w:
                    continue
                traj = np.stack([frames[f] for f in sorted_f])   # (N, 3)
                traj_s = savgol_filter(traj, window_length=w, polyorder=2, axis=0)
                translations[pid] = {f: traj_s[i].astype(np.float32)
                                     for i, f in enumerate(sorted_f)}

        return translations, orientations

    # ------------------------------------------------------------------
    # SMPL-X FK helper
    # ------------------------------------------------------------------

    @staticmethod
    def _load_smplx_model(model_path: str | Path) -> object:
        import smplx as smplx_lib
        p = Path(model_path)
        kwargs: dict = {"model_type": "smplx"}
        if p.is_file():
            kwargs["ext"] = p.suffix.lstrip(".")
        return smplx_lib.create(
            str(model_path), **kwargs,
            use_pca=False, flat_hand_mean=True, batch_size=1,
        ).eval()

    def _get_smplx_model(self, pid: int | None = None) -> object:
        if pid is not None and pid in self._smplx_models:
            return self._smplx_models[pid]
        return self._smplx_model

    def _smplx_fk(
        self,
        betas: np.ndarray,                          # (T_local, 10)
        body_pose: np.ndarray,                      # (T_local, 63)
        global_orient: np.ndarray,                  # (T_local, 3)
        left_hand_pose: np.ndarray | None = None,   # (T_local, 45) optional
        right_hand_pose: np.ndarray | None = None,  # (T_local, 45) optional
        return_verts: bool = False,
        pid: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Run SMPL-X FK and return joints (and optionally vertices).

        Uses zero translation so all positions are body-centric (origin at root).
        The global_orient rotation IS applied, so the z-axis matches camera depth.

        Returns:
            joints: (T_local, 55, 3) float32 — first 55 SMPL-X joints in metres.
            verts:  (T_local, 10475, 3) float32 — only when return_verts=True.
        """
        T = betas.shape[0]
        model = self._get_smplx_model(pid)
        num_expr = model.num_expression_coeffs
        lhp = (torch.tensor(left_hand_pose,  dtype=torch.float32)
               if left_hand_pose  is not None else torch.zeros(T, 45, dtype=torch.float32))
        rhp = (torch.tensor(right_hand_pose, dtype=torch.float32)
               if right_hand_pose is not None else torch.zeros(T, 45, dtype=torch.float32))
        with torch.no_grad():
            out = model(
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
                return_verts=return_verts,
            )
        joints = out.joints[:, :55].cpu().numpy().astype(np.float32)
        if return_verts:
            verts = out.vertices.cpu().numpy().astype(np.float32)
            return joints, verts
        return joints

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _triangulate_dlt(
        observations: list[tuple[float, float]],
        proj_matrices: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Linear (DLT) triangulation from N ≥ 2 camera observations.

        For each camera, the projection constraint ``x × (P @ X) = 0`` gives
        two independent linear equations.  All equations are stacked and the
        null-space solution is found via SVD.  Optional per-observation weights
        implement weighted least squares (each row scaled by sqrt(weight)).

        Args:
            observations: 2D joint positions ``(u, v)`` in VGGT-space per camera.
            proj_matrices: ``(3, 4)`` projection matrices ``K @ [R|t]`` per camera.
            weights: Optional per-camera confidence weights in [0, 1]. Uniform if None.

        Returns:
            ``(3,)`` world-space position.
        """
        rows = []
        for i, ((u, v), P) in enumerate(zip(observations, proj_matrices)):
            sw = np.sqrt(max(weights[i], 0.0)) if weights is not None else 1.0
            rows.append(sw * (u * P[2] - P[0]))
            rows.append(sw * (v * P[2] - P[1]))
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
        frame_start: int = 0,
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
            vggt_t = int(global_t) - frame_start
            if vggt_t < 0 or vggt_t >= self.T:
                continue
            if not self.cam_valid[vggt_t, k]:
                continue
            if not self.depth_valid[vggt_t, k]:
                continue

            depth_frame = self.depth_mm[vggt_t, k].astype(np.float32) / 1000.0
            conf_frame  = self.depth_conf[vggt_t, k].astype(np.float32)

            intr = self.intrinsics[vggt_t, k]
            fx, fy = float(intr[0, 0]), float(intr[1, 1])
            cx, cy = float(intr[0, 2]), float(intr[1, 2])

            oc = self.original_coords[vggt_t, k]
            os = self.original_size[vggt_t, k]
            W_orig, H_orig = float(os[0]), float(os[1])
            off_x, off_y = self._cam_offset(k)

            for j_a, j_b in _LONG_BONES:
                mhr_a = _SMPLX_TO_MHR70.get(j_a)
                mhr_b = _SMPLX_TO_MHR70.get(j_b)
                if mhr_a is None or mhr_b is None:
                    continue
                if mhr_a >= kp2d.shape[1] or mhr_b >= kp2d.shape[1]:
                    continue

                u_a, v_a = self._orig_to_vggt(kp2d[local_t, mhr_a], oc, W_orig, H_orig, off_x, off_y)
                u_b, v_b = self._orig_to_vggt(kp2d[local_t, mhr_b], oc, W_orig, H_orig, off_x, off_y)

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
        frame_start: int = 0,
    ) -> list[float]:
        """Flat list of scale samples; delegates to :meth:`_collect_scale_samples_tagged`."""
        tagged = self._collect_scale_samples_tagged(
            k, body_file, conf_threshold, min_delta_z, fused_betas,
            frame_start=frame_start,
        )
        return [s for sl in tagged.values() for s in sl]

    @staticmethod
    def _orig_to_vggt(
        kp: np.ndarray,
        oc: np.ndarray,
        W_orig: float,
        H_orig: float,
        off_x: float = 0.0,
        off_y: float = 0.0,
    ) -> tuple[float, float]:
        """Map a 2D keypoint from original-image pixels to VGGT output space.

        Args:
            kp: Keypoint [u, v, ...] in original (uncropped source) image pixels.
            oc: [0, 0, W_vggt, H_vggt] from vggt_cameras.npz.
            W_orig, H_orig: Centered-crop image dimensions in pixels (the frame
                the VGGT camera was actually calibrated on).
            off_x, off_y: Crop top-left offset in source pixels.  Subtracted from
                ``kp`` to move it from uncropped source space into the centered
                crop space before scaling into VGGT output space.  Default (0, 0).

        Returns:
            (u_vggt, v_vggt) in VGGT output pixel coordinates.
        """
        x1, y1, x2, y2 = oc
        u_vggt = x1 + (float(kp[0]) - off_x) * (x2 - x1) / W_orig
        v_vggt = y1 + (float(kp[1]) - off_y) * (y2 - y1) / H_orig
        return u_vggt, v_vggt

    def _cam_offset(self, k: int) -> tuple[float, float]:
        """Crop (off_x, off_y) in source pixels for camera index ``k``.

        Returns (0.0, 0.0) when no crop_meta was loaded or the camera is absent.
        """
        name = self.camera_names[k]
        if isinstance(name, bytes):
            name = name.decode()
        return self._cam_offsets.get(name, (0.0, 0.0))

    @staticmethod
    def _in_bounds(u: float, v: float, w_max: float, h_max: float) -> bool:
        return 0.0 <= u < w_max and 0.0 <= v < h_max
