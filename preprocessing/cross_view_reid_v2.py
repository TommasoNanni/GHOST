"""Cross-view person re-identification v2 — constrained rigid + per-person scale.

Algorithm per camera pair (vid_a, vid_b):

  0. DROID-SLAM (optional) corrects keypoints to world frame.  Static cameras
     fall back to abs_j = kpts_cam + pred_cam_t.

  1. Build xcorr delta-score curve over ALL person pairs weighted by appearance
     similarity × track dynamics.  Extract top-10 local-maxima as δ candidates.

  2. For each candidate δ_k run ALS (Alternating Least Squares) to fit the
     constrained model  abs_a_i ≈ λ_i · R · abs_b_i + t  where R and t are
     shared across all persons and λ_i is a per-person depth-scale scalar:
       - Seed assignment from appearance-based Hungarian.
       - ALS pass 1 → re-run Hungarian on geometric RMSE → ALS pass 2.
     Select δ* = argmin average RMSE over matched persons.

  3. Hungarian on the full N×M RMSE matrix under (R*, t*) at δ* to produce the
     final per-pair costs.  Accept pairs with RMSE < match_rmse_thr.

  4. Union-Find merges accepted pairs with same-camera conflict guard.

  5. Per-camera-pair frame offsets (δ*) are saved to cross_view_reid.json for
     downstream temporal synchronisation.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.video_dataset import Scene


class CrossVideoReidentifierV2:
    """Assign consistent global person IDs across camera views — v2 algorithm.

    Parameters
    ----------
    threshold : float
        Fallback similarity threshold when RANSAC fails for a camera pair.
    appearance_weight, shape_weight, pose_weight : float
        Weights for the hybrid appearance descriptor.
    high_app_thr : float
        Pairs with sim >= high_app_thr seed the RANSAC anchor search.
    low_app_thr : float
        When no affine is available, pairs below this threshold are rejected.
    match_rmse_thr : float
        Maximum world-frame RMSE (m) to accept a final matched pair.
        Compared against the overlap-adjusted RMSE (see overlap_penalty_k).
    overlap_penalty_k : float
        Small-sample evidence penalty. Pair/δ RMSEs are multiplied by
        (1 + k/√N) / (1 + k/√overlap_ref_frames), clamped to ≥1, where N is the
        number of overlapping frames. Short overlaps are easier to fit (same
        parameter count, fewer constraints), so their RMSE is inflated to make
        candidates with different overlap sizes comparable.
    overlap_ref_frames : int
        Overlap length at which the penalty factor is exactly 1 (no inflation).
    delta_prior_weight, delta_prior_scale : float
        Prior over the camera-pair time offset δ: candidate δs pay an evidence
        cost of delta_prior_weight metres per delta_prior_scale frames of |δ|.
        A large δ can still win, but must beat small-δ candidates by this
        margin. Set delta_prior_scale to the physically expected sync error of
        the dataset (small for hardware-synced data such as RICH).
    weak_overlap_frames, strong_reject_factor : int, float
        A geometric rejection is *final* only when backed by strong evidence:
        overlap ≥ weak_overlap_frames AND adjusted RMSE ≥ match_rmse_thr ×
        strong_reject_factor. Weak rejections (tiny/zero overlap, or RMSE in
        the marginal band) are uninformative and fall back to appearance
        similarity (threshold = `threshold`), with edge confidence capped low
        so geometric edges win Union-Find conflicts.
    min_anchor_std : float
        Minimum joint temporal std (m) for a track to count as dynamic.
        Static-only scenes fall back to best-RMSE anchor.
    ransac_anchor_thr : float
        Maximum RMSE (m) for the RANSAC anchor fit itself to be accepted.
    inlier_thr : float
        RMSE threshold (m) for counting RANSAC inliers.
    delta_search : int
        Neighbourhood half-width for δ refinement in joint_refine.
    droid_weights : str | None
        Path to droid.pth.  If None SLAM is skipped (assumes static cameras).
    droid_root : str
        Path to the DROID-SLAM repo root (for sys.path injection).
    default_intrinsics : tuple[float, float, float, float]
        Fallback (fx, fy, cx, cy) when per-camera calibration is not available.
    """

    _THRESHOLD: float = 0.4
    _APPEARANCE_WEIGHT: float = 0.5
    _SHAPE_WEIGHT: float = 0.2
    _POSE_WEIGHT: float = 0.3
    _HIGH_APP_THR: float = 0.60
    _LOW_APP_THR: float = 0.35
    _MATCH_RMSE_THR: float = 0.60
    _OVERLAP_PENALTY_K: float = 30.0
    _OVERLAP_REF_FRAMES: int = 500
    _DELTA_PRIOR_WEIGHT: float = 0.0   # disabled — consensus offsets replace the prior
    _DELTA_PRIOR_SCALE: float = 30.0
    _WEAK_OVERLAP_FRAMES: int = 100
    _STRONG_REJECT_FACTOR: float = 1.5
    _MIN_ANCHOR_STD: float = 0.05
    _RANSAC_ANCHOR_THR: float = 0.20
    _INLIER_THR: float = 0.25
    _DELTA_SEARCH: int = 2
    _DROID_WEIGHTS: str | None = None
    _DROID_ROOT: str = "/users/tnanni/ghost/DROID-SLAM"
    _DEFAULT_INTRINSICS: tuple[float, float, float, float] = (1200.0, 1200.0, 720.0, 526.0)
    _SLAM_CAMS: set[str] | None = None  # None = run SLAM on all cameras

    def __init__(
        self,
        threshold: float | None = None,
        appearance_weight: float | None = None,
        shape_weight: float | None = None,
        pose_weight: float | None = None,
        high_app_thr: float | None = None,
        low_app_thr: float | None = None,
        match_rmse_thr: float | None = None,
        overlap_penalty_k: float | None = None,
        overlap_ref_frames: int | None = None,
        delta_prior_weight: float | None = None,
        delta_prior_scale: float | None = None,
        weak_overlap_frames: int | None = None,
        strong_reject_factor: float | None = None,
        min_anchor_std: float | None = None,
        ransac_anchor_thr: float | None = None,
        inlier_thr: float | None = None,
        delta_search: int | None = None,
        droid_weights: str | None = None,
        droid_root: str | None = None,
        default_intrinsics: tuple[float, float, float, float] | None = None,
        slam_cams: set[str] | list[str] | None = None,
    ):
        self.threshold = threshold if threshold is not None else self._THRESHOLD
        self.appearance_weight = (
            appearance_weight if appearance_weight is not None else self._APPEARANCE_WEIGHT
        )
        self.shape_weight = shape_weight if shape_weight is not None else self._SHAPE_WEIGHT
        self.pose_weight = pose_weight if pose_weight is not None else self._POSE_WEIGHT
        self.high_app_thr = high_app_thr if high_app_thr is not None else self._HIGH_APP_THR
        self.low_app_thr = low_app_thr if low_app_thr is not None else self._LOW_APP_THR
        self.match_rmse_thr = (
            match_rmse_thr if match_rmse_thr is not None else self._MATCH_RMSE_THR
        )
        self.overlap_penalty_k = (
            overlap_penalty_k if overlap_penalty_k is not None else self._OVERLAP_PENALTY_K
        )
        self.overlap_ref_frames = (
            overlap_ref_frames if overlap_ref_frames is not None else self._OVERLAP_REF_FRAMES
        )
        self.delta_prior_weight = (
            delta_prior_weight if delta_prior_weight is not None else self._DELTA_PRIOR_WEIGHT
        )
        self.delta_prior_scale = (
            delta_prior_scale if delta_prior_scale is not None else self._DELTA_PRIOR_SCALE
        )
        self.weak_overlap_frames = (
            weak_overlap_frames if weak_overlap_frames is not None else self._WEAK_OVERLAP_FRAMES
        )
        self.strong_reject_factor = (
            strong_reject_factor if strong_reject_factor is not None else self._STRONG_REJECT_FACTOR
        )
        self.min_anchor_std = (
            min_anchor_std if min_anchor_std is not None else self._MIN_ANCHOR_STD
        )
        self.ransac_anchor_thr = (
            ransac_anchor_thr if ransac_anchor_thr is not None else self._RANSAC_ANCHOR_THR
        )
        self.inlier_thr = inlier_thr if inlier_thr is not None else self._INLIER_THR
        self.delta_search = delta_search if delta_search is not None else self._DELTA_SEARCH
        self.droid_weights = droid_weights if droid_weights is not None else self._DROID_WEIGHTS
        self.droid_root = droid_root if droid_root is not None else self._DROID_ROOT
        self.default_intrinsics = (
            default_intrinsics if default_intrinsics is not None else self._DEFAULT_INTRINSICS
        )
        self.slam_cams: set[str] | None = (
            set(slam_cams) if slam_cams is not None else self._SLAM_CAMS
        )

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path] | None = None,
        intrinsics_map: dict[str, np.ndarray] | None = None,
        dry_run: bool = False,
    ) -> None:
        """Assign consistent global person IDs across all camera views in a scene.

        Parameters
        ----------
        frames_dirs : dict[str, Path], optional
            Per-camera directory of extracted JPEG frames, used to run DROID-SLAM.
            Derived from scene.videos.frames_home when not provided.
        intrinsics_map : dict[str, np.ndarray], optional
            Per-camera 3×3 intrinsic matrix K.  Falls back to default_intrinsics
            when not provided.
        dry_run : bool
            If True, skip all file writes and only print the would-be remappings.
        """
        scene_id = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent

        if not dry_run and (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID v2 already done, skipping")
            return

        # Derive frames_dirs from scene.videos when not provided.
        if frames_dirs is None:
            frames_dirs = {}
            for video in scene.videos:
                if video.video_id in video_dirs and getattr(video, "frames_home", None):
                    frames_dirs[video.video_id] = video.frames_home

        return self._cross_view_reid_core(
            video_dirs=video_dirs,
            scene_id=scene_id,
            scene_dir=scene_dir,
            cross_view_reid_threshold=self.threshold,
            appearance_weight=self.appearance_weight,
            shape_weight=self.shape_weight,
            pose_weight=self.pose_weight,
            high_app_thr=self.high_app_thr,
            low_app_thr=self.low_app_thr,
            match_rmse_thr=self.match_rmse_thr,
            overlap_penalty_k=self.overlap_penalty_k,
            overlap_ref_frames=self.overlap_ref_frames,
            delta_prior_weight=self.delta_prior_weight,
            delta_prior_scale=self.delta_prior_scale,
            weak_overlap_frames=self.weak_overlap_frames,
            strong_reject_factor=self.strong_reject_factor,
            min_anchor_std=self.min_anchor_std,
            ransac_anchor_thr=self.ransac_anchor_thr,
            inlier_thr=self.inlier_thr,
            delta_search=self.delta_search,
            droid_weights=self.droid_weights,
            droid_root=self.droid_root,
            default_intrinsics=self.default_intrinsics,
            frames_dirs=frames_dirs,
            intrinsics_map=intrinsics_map or {},
            slam_cams=self.slam_cams,
            dry_run=dry_run,
        )

    # ── apply_reid_remap (identical to v1) ────────────────────────────────────

    @staticmethod
    def apply_reid_remap(
        video_dir: Path,
        id_remap: dict[int, int],
        gpu_label: str = "",
    ) -> None:
        """Rewrite mask_data.npz and json_data/*.json with re-identified IDs."""
        if not id_remap or all(k == v for k, v in id_remap.items()):
            return

        npz_path = video_dir / "mask_data.npz"
        json_dir = video_dir / "json_data"

        if npz_path.exists():
            tmp_path = npz_path.with_suffix(".tmp.npz")
            with (
                zipfile.ZipFile(str(npz_path), "r") as zf_in,
                zipfile.ZipFile(
                    str(tmp_path), "w",
                    compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                ) as zf_out,
            ):
                for name in sorted(zf_in.namelist()):
                    with zf_in.open(name) as f:
                        mask_img = np.load(io.BytesIO(f.read()))
                    total_pixels = mask_img.shape[0] * mask_img.shape[1]
                    new_mask = np.zeros_like(mask_img)
                    for old_id, new_id in id_remap.items():
                        old_region = mask_img == old_id
                        if int(old_region.sum()) > 0.80 * total_pixels:
                            logging.warning(
                                f"{gpu_label}Skipping remap {old_id}→{new_id} "
                                f"in {name}: mask covers >80% of frame"
                            )
                            continue
                        new_mask[old_region] = new_id
                    for uid in set(np.unique(mask_img)) - {0} - set(id_remap.keys()):
                        new_mask[mask_img == uid] = uid
                    buf = io.BytesIO()
                    np.save(buf, new_mask)
                    zf_out.writestr(name, buf.getvalue())
            tmp_path.replace(npz_path)

        for json_path in sorted(json_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            if "labels" in data:
                new_labels = {}
                sorted_items = sorted(
                    data["labels"].items(),
                    key=lambda kv: 0 if id_remap.get(int(kv[0]), int(kv[0])) == int(kv[0]) else 1,
                )
                for str_id, info in sorted_items:
                    old_id = int(str_id)
                    new_id = id_remap.get(old_id, old_id)
                    info["instance_id"] = new_id
                    if str(new_id) in new_labels:
                        logging.warning(
                            f"{gpu_label}Re-ID collision for id {new_id} in "
                            f"{json_path.name}: keeping original id {old_id}"
                        )
                        info["instance_id"] = old_id
                        new_labels[str(old_id)] = info
                    else:
                        new_labels[str(new_id)] = info
                data["labels"] = new_labels
            with open(json_path, "w") as f:
                json.dump(data, f)

        logging.info(
            f"{gpu_label}Re-ID segmentation remap applied in {video_dir.name}: "
            f"{id_remap}"
        )

    # ── Core ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cross_view_reid_core(
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        cross_view_reid_threshold: float,
        appearance_weight: float,
        shape_weight: float,
        pose_weight: float,
        high_app_thr: float,
        low_app_thr: float,
        match_rmse_thr: float,
        overlap_penalty_k: float,
        overlap_ref_frames: int,
        delta_prior_weight: float,
        delta_prior_scale: float,
        weak_overlap_frames: int,
        strong_reject_factor: float,
        min_anchor_std: float,
        ransac_anchor_thr: float,
        inlier_thr: float,
        delta_search: int,
        droid_weights: str | None,
        droid_root: str,
        default_intrinsics: tuple[float, float, float, float],
        frames_dirs: dict[str, Path],
        intrinsics_map: dict[str, np.ndarray],
        slam_cams: set[str] | None,
        dry_run: bool = False,
    ) -> None:

        MIN_OVERLAP = 30
        MIN_DELTA_OVERLAP_FRAMES = 100  # minimum video-level frame overlap to consider a δ candidate
        MIN_A_SINGULAR_VALUE = 0.3

        video_ids = list(video_dirs.keys())

        # ── Per-person descriptors and raw keypoints ───────────────────────────
        person_descs: dict[str, dict[int, tuple]] = {}
        person_pids: dict[str, list[int]] = {}
        # kpts3d[vid][pid] = (frames_array, kpts_T×70×3)  — corrected to world frame after SLAM
        person_kpts3d: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        # cam_t[vid][pid] = (frames_array, pred_cam_t_T×3)  — needed for SLAM scale
        person_cam_t: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / "body_data"
            gallery_path = body_dir / "appearance_gallery.npz"
            summary_path = body_dir / "body_params_summary.json"

            if not summary_path.exists():
                logging.warning(
                    f"{scene_id}/{vid_id}: no body_params_summary.json, "
                    f"skipping cross-view re-ID for this video"
                )
                continue

            with open(summary_path) as _f:
                summary = json.load(_f)
            pids = [int(k) for k in summary.get("persons", {}).keys()]
            if not pids:
                continue

            app_gallery: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                for k in gdata.files:
                    if k.endswith("_conf"):
                        continue
                    pid_key = int(k)
                    conf_key = f"{k}_conf"
                    feats = gdata[k]
                    confs = (
                        gdata[conf_key]
                        if conf_key in gdata.files
                        else np.ones(len(feats), dtype=np.float32)
                    )
                    app_gallery[pid_key] = (feats, confs)

            descs: dict[int, tuple] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue

                shape_feat: np.ndarray | None = None
                pose_feat: np.ndarray | None = None

                with np.load(str(npz_path)) as pdata:
                    # Shape (betas)
                    if "smplx_betas" in pdata:
                        bv = pdata["smplx_betas"]
                        if len(bv) > 0:
                            conf = pdata.get("pred_joint_confidence")
                            if conf is not None and len(conf) == len(bv):
                                fc = np.mean(conf, axis=-1).astype(np.float32)
                                tc = fc.sum()
                                sm = (fc[:, None] * bv).sum(0) / tc if tc > 0 else np.median(bv, 0).astype(np.float32)
                            else:
                                sm = np.median(bv, axis=0).astype(np.float32)
                            nm = np.linalg.norm(sm)
                            shape_feat = sm / nm if nm > 0 else sm

                    # Camera-frame root translation (needed for SLAM scale)
                    for key in ("pred_cam_t", "smplx_transl"):
                        if key in pdata and "frame_indices" in pdata:
                            person_cam_t.setdefault(vid_id, {})[pid] = (
                                pdata["frame_indices"].copy(),
                                pdata[key].copy(),
                            )
                            break

                    # Raw 3-D keypoints (camera space; corrected to world after SLAM)
                    if "pred_keypoints_3d" in pdata and "frame_indices" in pdata:
                        person_kpts3d.setdefault(vid_id, {})[pid] = (
                            pdata["frame_indices"].copy(),
                            pdata["pred_keypoints_3d"].copy(),
                        )

                    # Pose feature (canonical, root-relative, orient-free)
                    if "pred_keypoints_3d" in pdata and "smplx_global_orient" in pdata:
                        from scipy.spatial.transform import Rotation as _Rot
                        kps = pdata["pred_keypoints_3d"].astype(np.float32)
                        gorient = pdata["smplx_global_orient"].astype(np.float32)
                        conf_kps = pdata.get("pred_joint_confidence")
                        N = len(kps)
                        if N > 0:
                            kps_rel = kps - kps[:, 0:1, :]
                            rm = _Rot.from_rotvec(gorient).inv().as_matrix()
                            kc = np.einsum("nij,nkj->nki", rm, kps_rel).astype(np.float32)
                            pv = kc.reshape(N, -1)
                            np_norms = np.linalg.norm(pv, axis=1, keepdims=True)
                            pv = np.where(np_norms > 1e-6, pv / np_norms, pv)
                            pose_feat = pv

                app_feat = app_gallery.get(pid)
                if app_feat is None and shape_feat is None and pose_feat is None:
                    continue
                descs[pid] = (app_feat, shape_feat, pose_feat)

            if descs:
                person_descs[vid_id] = descs
                person_pids[vid_id] = sorted(descs.keys())

        active_vids = [v for v in video_ids if v in person_descs]

        if len(active_vids) < 2:
            logging.info(f"Scene {scene_id}: fewer than 2 videos with descriptors, skipping")
            return set()

        # ── SLAM: correct all cameras to world frame ───────────────────────────
        # Runs on every camera regardless of whether it is static or moving,
        # because we do not know a priori which cameras have ego-motion.

        def _quat_to_rot(q: np.ndarray) -> np.ndarray:
            qx, qy, qz, qw = q
            return np.array([
                [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
            ], dtype=np.float64)

        def _slam_image_stream(frames_dir: Path, fx, fy, cx, cy):
            import cv2, torch as _torch
            img_files = sorted(p for p in frames_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            for t, imfile in enumerate(img_files):
                image = cv2.imread(str(imfile))
                h0, w0 = image.shape[:2]
                scale = np.sqrt((384 * 512) / (h0 * w0))
                h1, w1 = int(h0 * scale), int(w0 * scale)
                image = cv2.resize(image, (w1, h1))
                image = image[:h1 - h1 % 8, :w1 - w1 % 8]
                image = _torch.as_tensor(image).permute(2, 0, 1)
                intr = _torch.tensor([fx * w1/w0, fy * h1/h0, cx * w1/w0, cy * h1/h0])
                yield t, image[None], intr

        def _run_droid_slam(frames_dir: Path, fx, fy, cx, cy, weights: str):
            import argparse, cv2, torch as _torch
            if droid_root not in sys.path:
                sys.path.insert(0, str(droid_root))
                sys.path.insert(0, str(Path(droid_root) / "droid_slam"))
            from droid import Droid

            try:
                _torch.multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            img_files = sorted(p for p in frames_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            if not img_files:
                logging.warning(f"DROID-SLAM: no images found in {frames_dir}, skipping")
                return None
            first = cv2.imread(str(img_files[0]))
            H_orig, W_orig = first.shape[:2]

            stream = list(_slam_image_stream(frames_dir, fx, fy, cx, cy))
            H_slam = stream[0][1].shape[2]
            W_slam = stream[0][1].shape[3]

            args = argparse.Namespace(
                weights=weights, image_size=[H_slam, W_slam], buffer=512,
                stereo=False, disable_vis=True, beta=0.3, filter_thresh=2.4,
                warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
                frontend_window=25, frontend_radius=2, frontend_nms=1,
                backend_thresh=22.0, backend_radius=2, backend_nms=3, upsample=False,
            )
            droid = Droid(args)
            for t, image, intrinsics in stream:
                droid.track(t, image, intrinsics=intrinsics)

            try:
                poses = droid.terminate(iter(stream))   # (N_frames, 7): [tx,ty,tz, qx,qy,qz,qw]
            except (ValueError, RuntimeError) as e:
                # Too few keyframes (e.g. static camera) — backend factor graph is empty.
                logging.warning(f"DROID-SLAM backend failed ({e}); camera may be static — skipping.")
                del droid
                import torch as _t; _t.cuda.empty_cache()
                return None

            n_kf   = droid.video.counter.value
            disps  = droid.video.disps[:n_kf].cpu().numpy()
            tstamp = droid.video.tstamp[:n_kf].cpu().numpy().astype(int)
            del droid
            import torch as _t; _t.cuda.empty_cache()

            return poses, disps, tstamp, (H_orig, W_orig), (H_slam, W_slam)

        def _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                  frame_offset, orig_hw) -> float:
            H_orig, W_orig = orig_hw
            H_disp, W_disp = disps.shape[1], disps.shape[2]
            sx = W_disp / W_orig
            sy = H_disp / H_orig
            lambdas = []
            cam_t_data = person_cam_t.get(vid_id, {})
            for pid, (frames, cam_t) in cam_t_data.items():
                frame_to_idx = {int(f): i for i, f in enumerate(frames)}
                for kf_i, t in enumerate(tstamp):
                    global_frame = int(t) + frame_offset
                    arr_idx = frame_to_idx.get(global_frame)
                    if arr_idx is None:
                        continue
                    x, y, z = cam_t[arr_idx]
                    if z < 0.5:
                        continue
                    px = int(round((fx * x / z + cx) * sx))
                    py = int(round((fy * y / z + cy) * sy))
                    if not (0 <= px < W_disp and 0 <= py < H_disp):
                        continue
                    d_inv = float(disps[kf_i, py, px])
                    if d_inv < 1e-6:
                        continue
                    lambdas.append(z * d_inv)
            if not lambdas:
                logging.warning(f"{vid_id}: no valid λ estimates, defaulting to 1.0")
                return 1.0
            lam = float(np.median(lambdas))
            logging.info(f"{vid_id}: SLAM scale λ={lam:.4f} ({len(lambdas)} samples)")
            return lam

        _slam_targets: list[str] = []
        _slam_corrected: set[str] = set()
        if droid_weights and frames_dirs:
            _slam_targets = (
                [v for v in active_vids if v in slam_cams]
                if slam_cams is not None
                else active_vids
            )
            logging.info(
                f"Scene {scene_id}: running DROID-SLAM on "
                + (str(slam_cams) if slam_cams is not None else "all cameras")
            )
            for vid_id in _slam_targets:
                fdir = frames_dirs.get(vid_id)
                if fdir is None or not Path(fdir).is_dir():
                    logging.warning(f"  {vid_id}: no frames dir for SLAM, skipping")
                    continue
                K = intrinsics_map.get(vid_id)
                fx = float(K[0, 0]) if K is not None else default_intrinsics[0]
                fy = float(K[1, 1]) if K is not None else default_intrinsics[1]
                cx = float(K[0, 2]) if K is not None else default_intrinsics[2]
                cy = float(K[1, 2]) if K is not None else default_intrinsics[3]

                result = _run_droid_slam(Path(fdir), fx, fy, cx, cy, droid_weights)
                if result is None:
                    logging.warning(f"  {vid_id}: SLAM returned no poses, skipping correction")
                    continue
                poses, disps, tstamp, orig_hw, _ = result

                # Reject SLAM result for cameras with negligible motion — DROID-SLAM
                # occasionally succeeds on static cameras with very few keyframes,
                # producing arbitrary poses.  If the translation std across all poses
                # is below 0.05 m the camera is treated as static.
                traj_std = float(np.std(poses[:, :3]))
                if traj_std < 0.05:
                    logging.warning(
                        f"  {vid_id}: SLAM poses show negligible motion "
                        f"(traj_std={traj_std:.4f} m) — treating as static, skipping correction"
                    )
                    continue

                kpts_data = person_kpts3d.get(vid_id, {})
                if not kpts_data:
                    continue
                frame_offset = int(min(frames.min() for frames, _ in kpts_data.values()))
                lam = _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                           frame_offset, orig_hw)

                # Project all keypoints to world frame:
                #   world_j = R @ kpts_cam[j] + R @ root_cam + λ * t_slam
                cam_t_data = person_cam_t.get(vid_id, {})
                for pid, (frames, kpts) in list(kpts_data.items()):
                    cam_t = cam_t_data.get(pid, (None, None))[1]
                    corrected = np.full_like(kpts, np.nan)
                    for i, frame_idx in enumerate(frames):
                        t_idx = int(frame_idx) - frame_offset
                        if t_idx < 0 or t_idx >= len(poses):
                            continue
                        pose = poses[t_idx]
                        if not np.isfinite(pose).all():
                            continue
                        R = _quat_to_rot(pose[3:])   # cam-to-world rotation
                        root_cam = cam_t[i] if cam_t is not None else np.zeros(3)
                        # world = R @ (root + joint_offsets) + λ*t
                        corrected[i] = (R @ kpts[i].T).T + R @ root_cam + lam * pose[:3]
                    person_kpts3d[vid_id][pid] = (frames, corrected)

                logging.info(f"  {vid_id}: world-frame correction applied (λ={lam:.4f})")
                _slam_corrected.add(vid_id)
        else:
            if droid_weights:
                logging.warning(
                    f"Scene {scene_id}: droid_weights set but no frames_dirs — SLAM skipped"
                )
            else:
                logging.info(
                    f"Scene {scene_id}: no droid_weights — assuming static cameras"
                )

        # ── Absolute-position correction for cameras without SLAM ──────────────
        # pred_keypoints_3d is root-relative; absolute camera-frame position =
        # pred_cam_t + kpts.  For SLAM-corrected cameras this is already done
        # inside the SLAM block.  For static cameras we apply R=I, t=0:
        #   abs_j = kpts[j] + pred_cam_t
        slam_corrected = _slam_corrected
        for vid_id in active_vids:
            if vid_id in slam_corrected:
                continue
            kpts_data = person_kpts3d.get(vid_id, {})
            cam_t_data = person_cam_t.get(vid_id, {})
            for pid, (frames, kpts) in list(kpts_data.items()):
                cam_t_entry = cam_t_data.get(pid)
                if cam_t_entry is None:
                    continue
                _, cam_t = cam_t_entry
                corrected = kpts + cam_t[:, None, :]  # broadcast over 70 joints
                person_kpts3d[vid_id][pid] = (frames, corrected)
            if kpts_data:
                logging.info(f"  {vid_id}: absolute-position correction applied (static camera)")

        # ── Union-Find with edge tracking ──────────────────────────────────────
        parent: dict[tuple, tuple] = {}
        rank_uf: dict[tuple, int] = {}
        edges: list[tuple[float, tuple, tuple]] = []

        def _find(x: tuple) -> tuple:
            if parent.setdefault(x, x) != x:
                parent[x] = _find(parent[x])
            return parent[x]

        def _union(x: tuple, y: tuple) -> None:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            if rank_uf.get(rx, 0) < rank_uf.get(ry, 0):
                rx, ry = ry, rx
            parent[ry] = rx
            if rank_uf.get(rx, 0) == rank_uf.get(ry, 0):
                rank_uf[rx] = rank_uf.get(rx, 0) + 1

        for _vid in active_vids:
            for _pid in person_pids[_vid]:
                _find((_vid, _pid))

        # ── Appearance descriptor helpers ──────────────────────────────────────

        def _xcorr_sim(fa: np.ndarray, fb: np.ndarray) -> tuple[float, int]:
            N, M = len(fa), len(fb)
            fa = fa - fa.mean(axis=0)
            fb = fb - fb.mean(axis=0)
            na = np.linalg.norm(fa, axis=1, keepdims=True)
            nb = np.linalg.norm(fb, axis=1, keepdims=True)
            fa = np.where(na > 1e-6, fa / na, fa)
            fb = np.where(nb > 1e-6, fb / nb, fb)
            S = fa @ fb.T
            best_score, best_off = -1.0, 0
            for tau in range(-(M - 1), N):
                diag = np.diagonal(S, offset=-tau)
                if len(diag) < MIN_OVERLAP:
                    continue
                score = float(diag.mean())
                if score > best_score:
                    best_score, best_off = score, tau
            return float(max(0.0, best_score)), best_off

        def _chamfer_sim(fa, ca, fb, cb) -> float:
            S = fa @ fb.T
            return 0.5 * (float(S.max(axis=1).mean()) + float(S.max(axis=0).mean()))

        def _weighted_sim_mat(
            pids_a, pids_b, descs_a, descs_b, w_app, w_shape, w_pose, vid_a="", vid_b=""
        ) -> tuple[np.ndarray, np.ndarray]:
            Na, Nb = len(pids_a), len(pids_b)
            sim_mat = np.zeros((Na, Nb), dtype=np.float32)
            wgt_mat = np.zeros((Na, Nb), dtype=np.float32)
            off_mat = np.zeros((Na, Nb), dtype=np.int32)

            for i, pa in enumerate(pids_a):
                app_a = descs_a[pa][0]
                if app_a is None:
                    continue
                feats_a, confs_a = app_a
                for j, pb in enumerate(pids_b):
                    app_b = descs_b[pb][0]
                    if app_b is None:
                        continue
                    feats_b, confs_b = app_b
                    s = _chamfer_sim(feats_a, confs_a, feats_b, confs_b)
                    sim_mat[i, j] += w_app * s
                    wgt_mat[i, j] += w_app

            sa_list = [descs_a[p][1] for p in pids_a]
            sb_list = [descs_b[p][1] for p in pids_b]
            mka = np.array([f is not None for f in sa_list], dtype=np.float32)
            mkb = np.array([f is not None for f in sb_list], dtype=np.float32)
            if mka.any() and mkb.any():
                dim = next(f for f in sa_list if f is not None).shape[0]
                zero = np.zeros(dim, dtype=np.float32)
                mat_a = np.stack([f if f is not None else zero for f in sa_list])
                mat_b = np.stack([f if f is not None else zero for f in sb_list])
                ssim = mat_a @ mat_b.T
                sw = np.outer(mka, mkb) * w_shape
                sim_mat += sw * ssim
                wgt_mat += sw

            pose_mat = np.full((Na, Nb), -1.0, dtype=np.float32)  # -1 = xcorr unavailable
            for i, pa in enumerate(pids_a):
                pose_a = descs_a[pa][2]
                if pose_a is None:
                    continue
                fva = pose_a
                for j, pb in enumerate(pids_b):
                    pose_b = descs_b[pb][2]
                    if pose_b is None:
                        continue
                    fvb = pose_b
                    s, off = _xcorr_sim(fva, fvb)
                    off_mat[i, j] = off
                    pose_mat[i, j] = s
                    logging.info(
                        f"  pose xcorr {vid_a}:P{pa} vs {vid_b}:P{pb}"
                        f"  sim={s:.3f}  offset={off:+d}"
                    )
                    sim_mat[i, j] += w_pose * s
                    wgt_mat[i, j] += w_pose

            return np.where(wgt_mat > 0, sim_mat / wgt_mat, 0.0), off_mat, pose_mat

        # ── Geometric helpers ──────────────────────────────────────────────────

        def _overlap_factor(n_frames: int) -> float:
            """Evidence inflation for short overlaps; 1.0 at ≥ overlap_ref_frames.

            A fit over few frames achieves a low RMSE more easily (same parameter
            count, fewer constraints), so its RMSE is multiplied by
            (1 + k/√N) / (1 + k/√N_ref), clamped to ≥ 1. Pairs with a full-length
            overlap keep their raw RMSE, preserving match_rmse_thr semantics.
            """
            if n_frames <= 0:
                return float("inf")
            raw = 1.0 + overlap_penalty_k / np.sqrt(float(n_frames))
            ref = 1.0 + overlap_penalty_k / np.sqrt(float(overlap_ref_frames))
            return max(raw / ref, 1.0)

        def _delta_prior_cost(delta: int) -> float:
            """Evidence cost (m) of a candidate camera-pair offset δ.

            A large δ may still win, but must beat small-δ candidates by this
            margin: delta_prior_weight metres per delta_prior_scale frames.
            """
            return delta_prior_weight * abs(int(delta)) / delta_prior_scale

        def _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta):
            """Common-frame keypoints, flat (N*70, 3), or (None, None)."""
            kb = person_kpts3d.get(vid_b, {})
            ka = person_kpts3d.get(vid_a, {})
            if pid_b not in kb or pid_a not in ka:
                return None, None
            frames_b, kpts_b = kb[pid_b]
            frames_a, kpts_a = ka[pid_a]
            fb2i = {int(f): i for i, f in enumerate(frames_b)}
            fa2i = {int(f): i for i, f in enumerate(frames_a)}
            common = sorted(f for f in fb2i if f + delta in fa2i)
            if len(common) < MIN_OVERLAP:
                return None, None
            ib = [fb2i[f] for f in common]
            ia = [fa2i[f + delta] for f in common]
            return kpts_b[ib].reshape(-1, 3), kpts_a[ia].reshape(-1, 3)

        def _track_std(vid: str, pid: int) -> float:
            kd = person_kpts3d.get(vid, {})
            if pid not in kd:
                return 0.0
            _, kpts = kd[pid]
            valid = kpts[np.isfinite(kpts).all(axis=(1, 2))]
            return float(valid.std(axis=0).mean()) if len(valid) > 0 else 0.0

        def _get_top_k_peaks(scores: dict, k: int = 10) -> list:
            """Return up to k delta values at local maxima of the scores dict."""
            if not scores:
                return [0]
            deltas = sorted(scores)
            vals = [scores[d] for d in deltas]
            peaks = []
            for i, (d, v) in enumerate(zip(deltas, vals)):
                left = vals[i - 1] if i > 0 else -1.0
                right = vals[i + 1] if i < len(vals) - 1 else -1.0
                if v >= left and v >= right and v > 0:
                    peaks.append((v, d))
            peaks.sort(reverse=True)
            if not peaks:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                return [d for d, _ in ranked[:k]]
            return [d for _, d in peaks[:k]]

        def _constrained_fit(assignment, vid_b, vid_a, delta, max_iter=10):
            """ALS for abs_a_i ≈ λ_i * (R @ abs_b_i + t) (shared R, t; per-person λ_i).

            The λ_i multiplies the full transform (rotation + translation), which is
            physically correct: monocular depth error scales the entire predicted 3D
            position, including the camera-frame translation component.

            assignment: list of (pid_b, pid_a).
            Returns (R, t, lambdas_dict, avg_rmse) or None.
            """
            pairs = []
            for pb, pa in assignment:
                src, dst = _get_aligned_flat(pb, vid_b, pa, vid_a, delta)
                if src is None:
                    continue
                valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
                if valid.sum() < MIN_OVERLAP:
                    continue
                pairs.append((pb, src[valid], dst[valid]))
            if not pairs:
                return None

            R = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            lambdas = {pb: 1.0 for pb, _, _ in pairs}

            for _ in range(max_iter):
                # Step A: fix {λ_i}, solve (R, t) via Procrustes.
                # Model: λ_i * (R @ src_i + t) = dst_i
                #   →  R @ src_i + t = dst_i / λ_i
                # So run Procrustes on (src_i, dst_i / λ_i).
                all_s = np.concatenate([src for _, src, _ in pairs])
                all_d = np.concatenate([dst / lambdas[pb] for pb, _, dst in pairs])
                mu_s = all_s.mean(0)
                mu_d = all_d.mean(0)
                H = (all_s - mu_s).T @ (all_d - mu_d)
                U, _, Vt = np.linalg.svd(H)
                d_sign = np.linalg.det(Vt.T @ U.T)
                R = Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T
                t = mu_d - R @ mu_s

                # Step B: fix (R, t), solve each λ_i via 1-D least squares.
                # λ_i = (R @ src_i + t)ᵀ dst_i / ‖R @ src_i + t‖²
                for pb, src, dst in pairs:
                    rsrc = (R @ src.T).T + t
                    num = float(np.sum(rsrc * dst))
                    den = float(np.sum(rsrc * rsrc))
                    lambdas[pb] = max(0.1, min(10.0, num / den)) if den > 1e-10 else 1.0

            rmses = []
            for pb, src, dst in pairs:
                rsrc = (R @ src.T).T + t
                pred = lambdas[pb] * rsrc
                rmses.append(float(np.sqrt(((pred - dst) ** 2).sum(1).mean())))
            return R, t, lambdas, float(np.mean(rmses))

        def _pair_rmse_with_Rt(R, t, pid_b, vid_b, pid_a, vid_a, delta) -> tuple[float, float]:
            """RMSE and optimal λ for one pair given fixed R, t.

            Model: pred = λ * (R @ src + t).
            Returns (rmse, lambda).  rmse=inf, lambda=1.0 when no common frames.
            """
            src, dst = _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta)
            if src is None:
                return float("inf"), 1.0
            valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() < MIN_OVERLAP:
                return float("inf"), 1.0
            src, dst = src[valid], dst[valid]
            rsrc = (R @ src.T).T + t
            den = float(np.sum(rsrc * rsrc))
            lam = max(0.1, min(10.0, float(np.sum(rsrc * dst)) / den)) if den > 1e-10 else 1.0
            pred = lam * rsrc
            r = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
            return (r if np.isfinite(r) else float("inf")), lam

        # ── Combined matching: xcorr peaks → ALS per candidate → consensus δ ────

        def _pair_delta_candidates(vid_a: str, vid_b: str) -> dict:
            """Stages 1–2 for one camera pair: appearance sim + scored δ candidates.

            Does NOT commit to a winning δ — all scored candidates are returned so
            that _solve_consensus_offsets can choose cycle-consistent offsets
            across the whole camera graph.
            """
            pids_a = person_pids[vid_a]
            pids_b = person_pids[vid_b]

            # Stage 1: appearance sim matrix + full xcorr delta-score curve.
            # ALL person pairs contribute, weighted by appearance sim × track std.
            sim_mat, _, xcorr_sim_mat = _weighted_sim_mat(
                pids_a, pids_b,
                person_descs[vid_a], person_descs[vid_b],
                appearance_weight, shape_weight, pose_weight,
                vid_a=vid_a, vid_b=vid_b,
            )

            from collections import defaultdict as _dd
            delta_scores: dict[int, float] = _dd(float)
            kd_a = person_kpts3d.get(vid_a, {})
            kd_b = person_kpts3d.get(vid_b, {})
            for i, pa in enumerate(pids_a):
                pose_a = person_descs[vid_a][pa][2]
                if pose_a is None or pa not in kd_a:
                    continue
                for j, pb in enumerate(pids_b):
                    pose_b = person_descs[vid_b][pb][2]
                    if pose_b is None or pb not in kd_b:
                        continue
                    w = (max(_track_std(vid_a, pa), _track_std(vid_b, pb))
                         * max(float(sim_mat[i, j]), 0.05))
                    if w < 1e-6:
                        continue
                    offset = int(kd_a[pa][0][0]) - int(kd_b[pb][0][0])
                    fva = pose_a - pose_a.mean(axis=0)
                    fvb = pose_b - pose_b.mean(axis=0)
                    na = np.linalg.norm(fva, axis=1, keepdims=True)
                    nb = np.linalg.norm(fvb, axis=1, keepdims=True)
                    fva = np.where(na > 1e-6, fva / na, fva)
                    fvb = np.where(nb > 1e-6, fvb / nb, fvb)
                    S = fva @ fvb.T
                    T_a_len, T_b_len = S.shape
                    for tau in range(-(T_b_len - 1), T_a_len):
                        diag = np.diagonal(S, offset=-tau)
                        if len(diag) < MIN_OVERLAP:
                            continue
                        score = float(diag.mean())
                        if score > 0:
                            delta_scores[offset + tau] += w * score

            delta_candidates = _get_top_k_peaks(delta_scores, k=10)

            # Filter out δ values where the two videos share fewer than MIN_DELTA_OVERLAP_FRAMES.
            # Without this, a short-overlap δ can produce a lower ALS RMSE than the true δ
            # simply because fewer frames are easier to fit.
            frames_a_all: set[int] = set()
            for _pa2 in kd_a:
                frames_a_all.update(int(_f) for _f in kd_a[_pa2][0])
            frames_b_all: set[int] = set()
            for _pb2 in kd_b:
                frames_b_all.update(int(_f) for _f in kd_b[_pb2][0])
            # Video-level overlap per candidate δ — used both for the hard filter
            # below and for the overlap penalty in the δ-selection score.
            video_overlap: dict[int, int] = {
                dk: sum(1 for _f in frames_b_all if _f + dk in frames_a_all)
                for dk in delta_candidates
            }
            filtered = [
                dk for dk in delta_candidates
                if video_overlap[dk] >= MIN_DELTA_OVERLAP_FRAMES
            ]
            if filtered:
                delta_candidates = filtered
            else:
                logging.warning(
                    f"  [{vid_a}↔{vid_b}] all δ candidates have <{MIN_DELTA_OVERLAP_FRAMES} "
                    f"overlap frames — filter skipped"
                )

            logging.info(
                f"  [{vid_a}↔{vid_b}] δ candidates ({len(delta_candidates)}): {delta_candidates}"
            )

            # Best single appearance pair — used to seed ALS pass 1.
            # Using only the most confident pair ensures (R, t) is anchored on a
            # reliable match and is not corrupted by uncertain / wrong pairs.
            # The geometric re-assignment (between pass 1 and pass 2) then places
            # all remaining persons under this clean (R, t).
            best_sim_i, best_sim_j = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
            best_seed = [(pids_b[best_sim_j], pids_a[best_sim_i])]

            # Full appearance Hungarian — kept for the appearance-only fallback.
            row_ind, col_ind = linear_sum_assignment(1.0 - sim_mat)
            app_assignment = [
                (pids_b[col_ind[k]], pids_a[row_ind[k]])
                for k in range(len(row_ind))
            ]

            # Stage 2: ALS evaluation for each candidate δ.
            # Each candidate's evidence score is
            #     rmse_k × overlap_factor(N_k)  +  delta_prior_cost(δ_k)
            # (prior currently disabled). All scored candidates are kept; the
            # cross-camera consensus solve picks the final offsets.
            pids_b_s = sorted(pids_b)
            pids_a_s = sorted(pids_a)
            cands: list[dict] = []

            for dk in delta_candidates:
                # First ALS pass: seed with only the best appearance pair so that
                # (R, t) is anchored on the most certain match alone.
                res = _constrained_fit(best_seed, vid_b, vid_a, dk)
                if res is None:
                    continue
                R_k, t_k, _, _ = res

                # Re-assignment: Hungarian on the geometric RMSE matrix.
                rm = np.full((len(pids_b_s), len(pids_a_s)), 1e6)
                for ii, pb in enumerate(pids_b_s):
                    for jj, pa in enumerate(pids_a_s):
                        r, _ = _pair_rmse_with_Rt(R_k, t_k, pb, vid_b, pa, vid_a, dk)
                        rm[ii, jj] = r if np.isfinite(r) else 1e6
                ri, ci = linear_sum_assignment(rm)
                geo_assignment = [
                    (pids_b_s[ri[k2]], pids_a_s[ci[k2]]) for k2 in range(len(ri))
                ]

                # Second ALS pass on geometric assignment.
                res2 = _constrained_fit(geo_assignment, vid_b, vid_a, dk)
                if res2 is not None:
                    R_k, t_k, _, rmse_k = res2
                else:
                    rmse_k = res[3]

                n_k = video_overlap.get(dk, 0)
                score_k = rmse_k * _overlap_factor(n_k) + _delta_prior_cost(dk)
                logging.info(
                    f"    [{vid_a}↔{vid_b}] δ={dk}  raw_rmse={rmse_k:.3f}  "
                    f"overlap={n_k}  score={score_k:.3f}"
                )
                cands.append({"delta": int(dk), "score": float(score_k),
                              "rmse": float(rmse_k), "R": R_k, "t": t_k})

            cands.sort(key=lambda c: c["score"])
            if cands:
                logging.info(
                    f"  [{vid_a}↔{vid_b}] pair-local best δ={cands[0]['delta']}  "
                    f"score={cands[0]['score']:.3f}"
                )
            return {
                "pids_a": pids_a, "pids_b": pids_b,
                "pids_a_s": pids_a_s, "pids_b_s": pids_b_s,
                "sim_mat": sim_mat, "xcorr_sim_mat": xcorr_sim_mat, "best_seed": best_seed,
                "frames_a_all": frames_a_all, "frames_b_all": frames_b_all,
                "cands": cands,
            }

        def _solve_consensus_offsets(
            pair_data: dict[tuple[str, str], dict],
        ) -> dict[tuple[str, str], int]:
            """Cycle-consistent camera-pair offsets via a Kruskal spanning forest.

            δ is a property of the camera pair (a clock offset), so all pairwise
            estimates must satisfy δ(a→c) = δ(a→b) + δ(b→c). Per-camera clocks
            are built by adding camera pairs most-confident-first (lowest best
            score); an edge only sets the offset between two previously
            unconnected components. Unreliable pairs therefore never average
            against reliable ones — they are used at most once, and only when
            nothing better connects their cameras.

            Convention: a candidate δ for pair (vid_a, vid_b) means
            frame_a = frame_b + δ, i.e. clock_a − clock_b = δ.
            """
            scored = sorted(
                (pd["cands"][0]["score"], key, pd["cands"][0]["delta"])
                for key, pd in pair_data.items() if pd["cands"]
            )
            if not scored:
                return {key: 0 for key in pair_data}

            clock: dict[str, float] = {v: 0.0 for v in active_vids}
            comp: dict[str, str] = {v: v for v in active_vids}

            def _root(v: str) -> str:
                while comp[v] != v:
                    comp[v] = comp[comp[v]]
                    v = comp[v]
                return v

            for s, (va, vb), d in scored:
                ra, rb = _root(va), _root(vb)
                if ra == rb:
                    continue  # cameras already connected by more confident pairs
                # Shift vb's whole component so that clock[va] − clock[vb] = d.
                shift = clock[va] - d - clock[vb]
                for v in active_vids:
                    if _root(v) == rb:
                        clock[v] += shift
                comp[rb] = ra
                logging.info(
                    f"    consensus tree += {va}↔{vb}  δ={d}  score={s:.3f}"
                )

            logging.info(
                f"Scene {scene_id}: consensus clocks "
                + "  ".join(f"{v}={clock[v]:+.1f}" for v in active_vids)
            )
            return {
                key: int(round(clock[key[0]] - clock[key[1]]))
                for key in pair_data
            }

        def _fit_pair_at_delta(
            vid_a: str, vid_b: str, pdata: dict, delta_bar: int
        ) -> tuple[np.ndarray | None, np.ndarray | None, float, int, float]:
            """Two-pass ALS fit of one camera pair at a fixed δ.

            Reuses the cached candidate fit when δ was already scored in stage 2.
            Returns (R, t, rmse, overlap_frames, score); R is None on failure.
            """
            pids_a_s, pids_b_s = pdata["pids_a_s"], pdata["pids_b_s"]
            n_bar = sum(1 for _f in pdata["frames_b_all"]
                        if _f + delta_bar in pdata["frames_a_all"])

            cached = next((c for c in pdata["cands"] if c["delta"] == delta_bar), None)
            if cached is not None:
                R_b, t_b, rmse_bar = cached["R"], cached["t"], cached["rmse"]
            else:
                R_b, t_b, rmse_bar = None, None, float("inf")
                res = _constrained_fit(pdata["best_seed"], vid_b, vid_a, delta_bar)
                if res is not None:
                    R_k, t_k = res[0], res[1]
                    rm = np.full((len(pids_b_s), len(pids_a_s)), 1e6)
                    for ii, pb in enumerate(pids_b_s):
                        for jj, pa in enumerate(pids_a_s):
                            r, _ = _pair_rmse_with_Rt(
                                R_k, t_k, pb, vid_b, pa, vid_a, delta_bar
                            )
                            rm[ii, jj] = r if np.isfinite(r) else 1e6
                    ri, ci = linear_sum_assignment(rm)
                    res2 = _constrained_fit(
                        [(pids_b_s[ri[k2]], pids_a_s[ci[k2]]) for k2 in range(len(ri))],
                        vid_b, vid_a, delta_bar,
                    )
                    if res2 is not None:
                        R_b, t_b, _, rmse_bar = res2
                    else:
                        R_b, t_b, _, rmse_bar = res

            score_bar = (
                rmse_bar * _overlap_factor(n_bar) + _delta_prior_cost(delta_bar)
                if np.isfinite(rmse_bar) else float("inf")
            )
            return R_b, t_b, rmse_bar, n_bar, score_bar

        def _finalize_pair(
            vid_a: str, vid_b: str, pdata: dict, delta_options: list[int]
        ) -> tuple[list[tuple[int, int, float]], int]:
            """Referee: fit at each offered offset and keep the best evidence.

            delta_options = [pair-local best δ, consensus δ̄]. The consensus can
            override a pair only by *winning on fitted evidence* — a pair is
            never forced onto an offset that scores worse than its own best,
            so the consensus cannot poison a healthy pair.
            """
            sim_mat = pdata["sim_mat"]
            pids_a, pids_b = pdata["pids_a"], pdata["pids_b"]
            pids_a_s, pids_b_s = pdata["pids_a_s"], pdata["pids_b_s"]

            # Fit all δ options; keep all results so we can retry if needed.
            all_fitted: list[tuple[float, float, np.ndarray | None, np.ndarray | None, int]] = []
            for d_opt in dict.fromkeys(int(d) for d in delta_options):
                R_o, t_o, rmse_o, n_o, score_o = _fit_pair_at_delta(
                    vid_a, vid_b, pdata, d_opt
                )
                logging.info(
                    f"  [{vid_a}↔{vid_b}] option δ={d_opt}  RMSE={rmse_o:.3f}  "
                    f"overlap={n_o}  score={score_o:.3f}"
                )
                all_fitted.append((score_o, rmse_o, R_o, t_o, d_opt))
            all_fitted.sort(key=lambda x: x[0])
            score_bar, rmse_bar, best_R, best_t, delta_bar = all_fitted[0]

            logging.info(
                f"  [{vid_a}↔{vid_b}] final δ̄={delta_bar}  RMSE={rmse_bar:.3f}  "
                f"score={score_bar:.3f}"
            )

            # Fallback to pure appearance when geometry fails at δ̄.
            if best_R is None or score_bar > match_rmse_thr * 3:
                logging.info(
                    f"  [{vid_a}↔{vid_b}] geometry failed at δ̄ — appearance-only fallback"
                )
                row_ind, col_ind = linear_sum_assignment(1.0 - sim_mat)
                return [
                    (pids_a[r], pids_b[c], float(sim_mat[r, c]))
                    for r, c in zip(row_ind, col_ind)
                    if sim_mat[r, c] >= cross_view_reid_threshold
                ], 0

            ia_idx = {pa: i for i, pa in enumerate(pids_a)}
            ib_idx = {pb: i for i, pb in enumerate(pids_b)}

            def _compute_accepted(
                R: np.ndarray, t: np.ndarray, delta: int
            ) -> tuple[list[tuple[int, int, float]], bool]:
                """Greedy + rescue acceptance under (R, t) at delta.

                Pairs are sorted by combined cost = geo_rmse + 3*(1-sim), where
                sim is the appearance+shape+pose similarity. This breaks geometric
                ties caused by spatially coincident people (ALS local minimum):
                the correct same-person pair has both lower RMSE AND higher sim,
                so it wins in the combined sort even when geometry alone fails.
                The geometric threshold (match_rmse_thr) still guards acceptance.

                Returns (accepted_pairs, has_geo_accepted).
                """
                ap: list[tuple[float, float, float, float, int, int, int]] = []
                for pb in pids_b_s:
                    for pa in pids_a_s:
                        r, lam = _pair_rmse_with_Rt(R, t, pb, vid_b, pa, vid_a, delta)
                        n_fr = 0
                        src, _ = _get_aligned_flat(pb, vid_b, pa, vid_a, delta)
                        if src is not None:
                            n_fr = len(src) // 70
                        r_adj = r * _overlap_factor(n_fr) if np.isfinite(r) else float("inf")
                        sim_val = float(sim_mat[ia_idx[pa], ib_idx[pb]])
                        r_combined = (
                            r_adj + 3.0 * (1.0 - sim_val)
                            if np.isfinite(r_adj) else float("inf")
                        )
                        logging.info(
                            f"    {vid_b}:P{pb} → {vid_a}:P{pa}  frames={n_fr}  "
                            f"RMSE={r:.3f}  adj={r_adj:.3f}  sim={sim_val:.3f}  λ={lam:.3f}"
                        )
                        ap.append((r_combined, r_adj, r, lam, n_fr, pb, pa))

                ap.sort(key=lambda x: x[0])
                ub: set[int] = set()
                ua: set[int] = set()
                acc: list[tuple[int, int, float]] = []
                has_geo_accepted: bool = False
                dc = _delta_prior_cost(delta)
                for r_combined, rmse_adj, rmse_raw, lam, n_fr, pb, pa in ap:
                    if rmse_adj >= match_rmse_thr:
                        break
                    if pb in ub or pa in ua:
                        continue
                    sim_val = float(sim_mat[ia_idx[pa], ib_idx[pb]])
                    ub.add(pb)
                    ua.add(pa)
                    has_geo_accepted = True
                    logging.info(
                        f"  {vid_a}:P{pa} ↔ {vid_b}:P{pb}  RMSE={rmse_raw:.3f} "
                        f"adj={rmse_adj:.3f} sim={sim_val:.3f} → accepted"
                    )
                    acc.append((pa, pb, float(1.0 - (rmse_adj + dc) / match_rmse_thr)))

                # Appearance rescue: weak rejects (tiny overlap or marginal RMSE)
                # fall back to appearance sim. Confidence capped at 0.25 so
                # geometric edges win Union-Find conflicts.
                resc: list[tuple[float, int, int]] = []
                for _, rmse_adj, rmse_raw, lam, n_fr, pb, pa in ap:
                    if pb in ub or pa in ua:
                        continue
                    if pa not in ia_idx or pb not in ib_idx:
                        continue
                    strong_reject = (
                        n_fr >= weak_overlap_frames
                        and rmse_adj >= match_rmse_thr * strong_reject_factor
                    )
                    if strong_reject:
                        continue
                    sim = float(sim_mat[ia_idx[pa], ib_idx[pb]])
                    if sim >= cross_view_reid_threshold:
                        resc.append((sim, pb, pa))
                resc.sort(reverse=True)
                for sim, pb, pa in resc:
                    if pb in ub or pa in ua:
                        continue
                    ub.add(pb)
                    ua.add(pa)
                    conf = 0.25 * (sim - cross_view_reid_threshold) / max(
                        1e-6, 1.0 - cross_view_reid_threshold
                    )
                    logging.info(
                        f"  {vid_a}:P{pa} ↔ {vid_b}:P{pb}  sim={sim:.3f} "
                        f"→ appearance rescue (conf={conf:.3f})"
                    )
                    acc.append((pa, pb, float(conf)))

                for _, rmse_adj, rmse_raw, lam, n_fr, pb, pa in ap:
                    if pb in ub or pa in ua:
                        continue
                    if np.isfinite(rmse_adj):
                        logging.info(
                            f"  {vid_a}:P{pa} ↔ {vid_b}:P{pb}  RMSE={rmse_raw:.3f} "
                            f"adj={rmse_adj:.3f} → rejected"
                        )

                return acc, has_geo_accepted

            # Fix B: if the chosen δ yields 0 accepted pairs (no geometry AND no
            # rescue), retry at the next-best δ option.
            accepted, _ = _compute_accepted(best_R, best_t, delta_bar)
            if not accepted and len(all_fitted) > 1:
                for score_alt, rmse_alt, R_alt, t_alt, delta_alt in all_fitted[1:]:
                    if R_alt is None:
                        continue
                    logging.info(
                        f"  [{vid_a}↔{vid_b}] 0 matches at δ={delta_bar} "
                        f"— retrying at δ={delta_alt}"
                    )
                    accepted_alt, _ = _compute_accepted(R_alt, t_alt, delta_alt)
                    if accepted_alt:
                        accepted = accepted_alt
                        delta_bar = delta_alt
                        break

            return accepted, delta_bar

        # ── All-pairs matching: candidates → consensus δ → finalize ────────────
        camera_pair_offsets: dict[str, int] = {}
        pair_data: dict[tuple[str, str], dict] = {}

        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                pair_data[(vid_a, vid_b)] = _pair_delta_candidates(vid_a, vid_b)

        consensus = _solve_consensus_offsets(pair_data)

        for (vid_a, vid_b), pdata in pair_data.items():
            delta_options = [consensus[(vid_a, vid_b)]]
            if pdata["cands"]:
                delta_options.insert(0, pdata["cands"][0]["delta"])
            matches, cam_delta = _finalize_pair(vid_a, vid_b, pdata, delta_options)
            camera_pair_offsets[f"{vid_a}→{vid_b}"] = cam_delta
            for pa, pb, sim in matches:
                edges.append((sim, (vid_a, pa), (vid_b, pb)))
                _union((vid_a, pa), (vid_b, pb))

        # ── Conflict resolution (identical to v1) ──────────────────────────────
        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _vid in active_vids:
                for _pid in person_pids[_vid]:
                    _node = (_vid, _pid)
                    comps.setdefault(_find(_node), []).append(_node)
            return comps

        def _find_path_min_edge(src: tuple, dst: tuple) -> int:
            conflict_root = _find(src)
            adj: dict[tuple, list] = {}
            for ei, (s, na, nb) in enumerate(edges):
                if _find(na) != conflict_root:
                    continue
                adj.setdefault(na, []).append((nb, ei, s))
                adj.setdefault(nb, []).append((na, ei, s))
            from collections import deque
            prev: dict[tuple, tuple | None] = {src: None}
            queue: deque = deque([src])
            while queue:
                cur = queue.popleft()
                if cur == dst:
                    break
                for nbr, ei, _ in adj.get(cur, []):
                    if nbr not in prev:
                        prev[nbr] = (cur, ei)
                        queue.append(nbr)
            else:
                return -1
            path_ei = []
            cur = dst
            while prev[cur] is not None:
                par, ei = prev[cur]
                path_ei.append(ei)
                cur = par
            return min(path_ei, key=lambda ei: edges[ei][0])

        def _resolve_conflicts() -> None:
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False
                for members in comps.values():
                    v2n: dict[str, list] = {}
                    for node in members:
                        v2n.setdefault(node[0], []).append(node)
                    for vid_id, nodes in v2n.items():
                        if len(nodes) < 2:
                            continue
                        conflict_found = True
                        worst_idx = _find_path_min_edge(nodes[0], nodes[1])
                        if worst_idx >= 0:
                            rs, rna, rnb = edges[worst_idx]
                            logging.info(
                                f"Scene {scene_id}: removed conflicting edge "
                                f"{rna[0]}/P{rna[1]} ↔ {rnb[0]}/P{rnb[1]} "
                                f"(sim={rs:.3f}) — same-video duplicate in {vid_id}"
                            )
                            edges.pop(worst_idx)
                            parent.clear(); rank_uf.clear()
                            for _v in active_vids:
                                for _p in person_pids[_v]:
                                    _find((_v, _p))
                            for _s, _na, _nb in edges:
                                _union(_na, _nb)
                        break
                if not conflict_found:
                    break

        _resolve_conflicts()

        # ── Global ID assignment ────────────────────────────────────────────────
        comps = _get_components()
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}
        pending_single: list[tuple[int, list[tuple]]] = []
        global_counter = 1
        used_global_ids: set[int] = set()
        for members in comps.values():
            if len({v for v, _ in members}) >= 2:
                gid = global_counter; global_counter += 1; used_global_ids.add(gid)
                for vid_id, pid in members:
                    if pid != gid:
                        global_remap[vid_id][pid] = gid
            else:
                pending_single.append((min(pid for _, pid in members), list(members)))

        nni = global_counter
        for _, members in sorted(pending_single):
            gid = nni; nni += 1; used_global_ids.add(gid)
            for vid_id, pid in members:
                if pid != gid:
                    global_remap[vid_id][pid] = gid

        logging.info(
            f"Scene {scene_id}: v2 → {len(comps)} global person(s), "
            f"{sum(len(m) for m in global_remap.values())} remap(s) "
            f"across {len(active_vids)} view(s)"
        )

        if dry_run:
            # Print would-be remappings without touching any files.
            print(f"\n[DRY RUN] Scene {scene_id}: {len(comps)} global persons, "
                  f"camera pair offsets: {camera_pair_offsets}")
            for vid_id, remap in global_remap.items():
                if remap:
                    print(f"  {vid_id}: would remap {remap}")
                else:
                    print(f"  {vid_id}: no remaps needed")
            return

        # ── Apply remaps ───────────────────────────────────────────────────────
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"

            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.xviewtmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    logging.warning(
                        f"{vid_id}: cross-view remap — {dst.name} already exists, discarding"
                    )
                    tmp.unlink()
                else:
                    tmp.rename(dst)

            summary_path = body_dir / "body_params_summary.json"
            if summary_path.exists():
                with open(summary_path) as _f:
                    summary = json.load(_f)
                new_persons: dict[str, object] = {}
                for str_id, info in summary.get("persons", {}).items():
                    new_persons[str(remap.get(int(str_id), int(str_id)))] = info
                summary["persons"] = new_persons
                with open(summary_path, "w") as _f:
                    json.dump(summary, _f, indent=2)

            gallery_path = body_dir / "appearance_gallery.npz"
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                new_gallery = {}
                for k in gdata.files:
                    if k.endswith("_conf"):
                        old_pid = int(k[:-5])
                        new_gallery[f"{remap.get(old_pid, old_pid)}_conf"] = gdata[k]
                    else:
                        old_pid = int(k)
                        new_gallery[str(remap.get(old_pid, old_pid))] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            with open(body_dir / "cross_view_id_mapping.json", "w") as _f:
                json.dump({str(k): v for k, v in remap.items()}, _f, indent=2)

            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)
            print(f"  {vid_id}: cross-view re-ID v2 — {len(remap)} remap(s): {remap}")

        # ── Scene-level summary ────────────────────────────────────────────────
        reid_summary = {
            "version": 2,
            "remaps": {
                vid_id: {str(k): v for k, v in remap.items()}
                for vid_id, remap in global_remap.items()
            },
            "camera_pair_offsets": camera_pair_offsets,
        }
        with open(scene_dir / "cross_view_reid.json", "w") as _f:
            json.dump(reid_summary, _f, indent=2)
